"""Dataset, split, augmentation, and sampler policy for Experiment 5."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .core import CIFAR10_MEAN, CIFAR10_STD, derive_seed


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def stratified_split(labels: Sequence[int], validation_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels_array, return_counts=True)
    if validation_size % len(classes):
        raise ValueError("Validation size must be divisible by the number of classes.")
    per_class = validation_size // len(classes)
    if np.any(counts < per_class):
        raise ValueError("A class has fewer examples than the requested validation allocation.")
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for class_id in classes:
        indices = np.flatnonzero(labels_array == class_id)
        indices = rng.permutation(indices)
        validation_parts.append(indices[:per_class])
        train_parts.append(indices[per_class:])
    train_indices = np.sort(np.concatenate(train_parts)).astype(np.int64)
    validation_indices = np.sort(np.concatenate(validation_parts)).astype(np.int64)
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Train and validation split overlap.")
    return train_indices, validation_indices


def nested_stratified_subset(
    pool_indices: np.ndarray,
    labels: Sequence[int],
    size: int,
    seed: int,
) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels_array[pool_indices])
    if size % len(classes):
        raise ValueError("Nested subset size must be divisible by class count.")
    per_class = size // len(classes)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in classes:
        candidates = pool_indices[labels_array[pool_indices] == class_id]
        selected.append(rng.permutation(candidates)[:per_class])
    return np.sort(np.concatenate(selected)).astype(np.int64)


def create_split_artifact(labels: Sequence[int], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    dataset_config = config["dataset"]
    train_indices, validation_indices = stratified_split(
        labels,
        validation_size=int(dataset_config["validation_size"]),
        seed=int(dataset_config["split_seed"]),
    )
    validation_rng = np.random.default_rng(derive_seed(int(dataset_config["split_seed"]), "validation_diagnostics"))
    permuted_validation = validation_rng.permutation(validation_indices)
    diagnostic_size = int(dataset_config["diagnostic_size"])
    calibration_size = int(dataset_config["calibration_size"])
    if diagnostic_size + calibration_size > len(validation_indices):
        raise ValueError("Diagnostic and calibration subsets must be disjoint within validation.")
    artifact: dict[str, np.ndarray] = {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "diagnostic_indices": np.sort(permuted_validation[:diagnostic_size]).astype(np.int64),
        "calibration_indices": np.sort(
            permuted_validation[diagnostic_size : diagnostic_size + calibration_size]
        ).astype(np.int64),
    }
    labels_array = np.asarray(labels, dtype=np.int64)
    for draw_index, draw_seed in enumerate(config["probe"]["label_draw_seeds"]):
        order_by_class: dict[int, np.ndarray] = {}
        draw_rng = np.random.default_rng(int(draw_seed))
        for class_id in np.unique(labels_array[train_indices]):
            candidates = train_indices[labels_array[train_indices] == class_id]
            order_by_class[int(class_id)] = draw_rng.permutation(candidates)
        for fraction in config["probe"]["label_fractions"]:
            size = int(round(float(fraction) * len(train_indices)))
            if size == len(train_indices):
                selected = train_indices.copy()
            else:
                if size % len(order_by_class):
                    raise ValueError(f"Probe subset size {size} is not class-balanced.")
                per_class = size // len(order_by_class)
                selected = np.sort(
                    np.concatenate([order_by_class[class_id][:per_class] for class_id in sorted(order_by_class)])
                ).astype(np.int64)
            fraction_key = f"{float(fraction):.2f}".replace(".", "p")
            artifact[f"probe_draw{draw_index}_{fraction_key}"] = selected
    return artifact


def validate_split_artifact(
    artifact: Mapping[str, np.ndarray], labels: Sequence[int], config: Mapping[str, Any]
) -> dict[str, Any]:
    train = np.asarray(artifact["train_indices"], dtype=np.int64)
    validation = np.asarray(artifact["validation_indices"], dtype=np.int64)
    diagnostic = np.asarray(artifact["diagnostic_indices"], dtype=np.int64)
    calibration = np.asarray(artifact["calibration_indices"], dtype=np.int64)
    dataset_config = config["dataset"]
    if len(train) != int(dataset_config["train_size"]):
        raise ValueError(f"Expected {dataset_config['train_size']} train indices, found {len(train)}")
    if len(validation) != int(dataset_config["validation_size"]):
        raise ValueError(f"Expected {dataset_config['validation_size']} validation indices, found {len(validation)}")
    if len(np.unique(np.concatenate([train, validation]))) != 50_000:
        raise ValueError("Train/validation indices are not a disjoint partition of CIFAR10 train.")
    if np.setdiff1d(diagnostic, validation).size or np.setdiff1d(calibration, validation).size:
        raise ValueError("Diagnostic/calibration indices must belong to validation.")
    if np.intersect1d(diagnostic, calibration).size:
        raise ValueError("Diagnostic and calibration validation subsets overlap.")
    labels_array = np.asarray(labels, dtype=np.int64)
    train_counts = np.bincount(labels_array[train], minlength=int(dataset_config["classes"]))
    validation_counts = np.bincount(labels_array[validation], minlength=int(dataset_config["classes"]))
    if len(set(train_counts.tolist())) != 1 or len(set(validation_counts.tolist())) != 1:
        raise ValueError("Main split is not exactly stratified.")
    return {
        "arrays": {
            name: {"size": int(len(value)), "sha256": array_sha256(np.asarray(value, dtype=np.int64))}
            for name, value in artifact.items()
        },
        "train_class_counts": train_counts.tolist(),
        "validation_class_counts": validation_counts.tolist(),
    }


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key].astype(np.int64, copy=False) for key in data.files}


@contextmanager
def local_transform_rng(seed: int) -> Iterator[None]:
    """Make torchvision's stochastic transforms a pure function of a sample key."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        seed32 = int(seed) % (2**32)
        random.seed(seed32)
        np.random.seed(seed32)
        torch.manual_seed(int(seed) % (2**63 - 1))
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def make_ssl_transform(config: Mapping[str, Any]):
    aug = config["augmentation"]
    if aug["operation_order"] != [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomApply(ColorJitter)",
        "RandomGrayscale",
        "RandomApply(GaussianBlur)",
        "RandomSolarize",
        "ToTensor",
        "Normalize",
    ]:
        raise ValueError("Unrecognized Experiment 5 augmentation operation order.")
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                int(aug["output_size"]),
                scale=tuple(map(float, aug["random_resized_crop_scale"])),
                ratio=tuple(map(float, aug["random_resized_crop_ratio"])),
                interpolation=InterpolationMode.BILINEAR,
                antialias=bool(aug["antialias"]),
            ),
            transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip_probability"])),
            transforms.RandomApply(
                [transforms.ColorJitter(*map(float, aug["color_jitter"]))],
                p=float(aug["color_jitter_probability"]),
            ),
            transforms.RandomGrayscale(p=float(aug["grayscale_probability"])),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=int(aug["gaussian_blur_kernel"]),
                        sigma=tuple(map(float, aug["gaussian_blur_sigma"])),
                    )
                ],
                p=float(aug["gaussian_blur_probability"]),
            ),
            transforms.RandomSolarize(
                threshold=float(aug["solarize_threshold"]), p=float(aug["solarize_probability"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def make_eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


class DeterministicMultiViewDataset(Dataset):
    """Apply reproducible views keyed by (source index, optimizer update)."""

    def __init__(self, base_dataset: Dataset, transform, views: int, augmentation_seed: int):
        self.base_dataset = base_dataset
        self.transform = transform
        self.views = int(views)
        self.augmentation_seed = int(augmentation_seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            source_index, update = map(int, key)
        else:
            source_index, update = int(key), 0
        image, label = self.base_dataset[source_index]
        views = []
        for view_index in range(self.views):
            seed = derive_seed(self.augmentation_seed, f"update={update}:image={source_index}:view={view_index}")
            with local_transform_rng(seed):
                views.append(self.transform(image))
        return torch.stack(views, dim=0), int(label), source_index


class FixedUpdateBatchSampler(Sampler[list[tuple[int, int]]]):
    """Deterministic complete-batch permutations indexed by optimizer update."""

    def __init__(
        self,
        source_indices: Sequence[int],
        batch_size: int,
        total_updates: int,
        start_update: int,
        data_order_seed: int,
    ):
        self.source_indices = torch.as_tensor(np.asarray(source_indices, dtype=np.int64))
        self.batch_size = int(batch_size)
        self.total_updates = int(total_updates)
        self.start_update = int(start_update)
        self.data_order_seed = int(data_order_seed)
        self.updates_per_epoch = len(self.source_indices) // self.batch_size
        if self.updates_per_epoch < 1:
            raise ValueError("Dataset is smaller than one batch.")
        if not 0 <= self.start_update <= self.total_updates:
            raise ValueError("start_update must be between zero and total_updates.")

    def __len__(self) -> int:
        return self.total_updates - self.start_update

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        first_epoch = self.start_update // self.updates_per_epoch
        last_epoch = (self.total_updates - 1) // self.updates_per_epoch if self.total_updates else -1
        for epoch in range(first_epoch, last_epoch + 1):
            generator = torch.Generator()
            generator.manual_seed(derive_seed(self.data_order_seed, f"epoch={epoch}") % (2**63 - 1))
            permutation = torch.randperm(len(self.source_indices), generator=generator)
            ordered = self.source_indices[permutation]
            for batch_in_epoch in range(self.updates_per_epoch):
                update_zero_based = epoch * self.updates_per_epoch + batch_in_epoch
                if update_zero_based < self.start_update or update_zero_based >= self.total_updates:
                    continue
                start = batch_in_epoch * self.batch_size
                indices = ordered[start : start + self.batch_size].tolist()
                yield [(int(index), update_zero_based + 1) for index in indices]


def worker_init_fn(worker_id: int) -> None:
    # Augmentation itself is stateless. This seed covers only incidental worker code.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_training_loader(
    base_dataset: Dataset,
    train_indices: Sequence[int],
    config: Mapping[str, Any],
    streams: Mapping[str, int],
    start_update: int,
    num_workers: int,
) -> DataLoader:
    training = config["training"]
    multi_view = DeterministicMultiViewDataset(
        base_dataset,
        transform=make_ssl_transform(config),
        views=int(config["augmentation"]["views"]),
        augmentation_seed=int(streams["augmentation"]),
    )
    sampler = FixedUpdateBatchSampler(
        train_indices,
        batch_size=int(training["batch_size"]),
        total_updates=int(training["total_updates"]),
        start_update=int(start_update),
        data_order_seed=int(streams["data_order"]),
    )
    return DataLoader(
        multi_view,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=worker_init_fn if int(num_workers) > 0 else None,
    )


def make_base_cifar10(data_dir: str | Path, train: bool, download: bool = False):
    return datasets.CIFAR10(root=Path(data_dir), train=train, download=download, transform=None)


def make_eval_subset(base_dataset: Dataset, indices: Sequence[int]) -> Dataset:
    class EvalDataset(Dataset):
        def __init__(self, base, selected):
            self.base = base
            self.selected = list(map(int, selected))
            self.transform = make_eval_transform()

        def __len__(self):
            return len(self.selected)

        def __getitem__(self, item):
            source_index = self.selected[item]
            image, label = self.base[source_index]
            return self.transform(image), int(label), source_index

    return EvalDataset(base_dataset, indices)


class DeterministicFakeCIFAR10(Dataset):
    """A code-path fixture; never a scientific Experiment 5 result."""

    def __init__(self, size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = (np.arange(size) % 10).astype(np.int64).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.fromarray(self.images[index]), int(self.targets[index])
