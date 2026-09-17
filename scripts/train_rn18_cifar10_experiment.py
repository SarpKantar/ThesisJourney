#!/usr/bin/env python
"""Train the third experiment: ResNet18 on CIFAR10.

The experiment compares:

1. ImageNet1k-pretrained ResNet18 fine-tuned on CIFAR10.
2. CIFAR10 LeJEPA self-supervised ResNet18, followed by fixed-epoch
   frozen-backbone linear-head training.

The script saves several checkpoints for both methods and then selects the pair
with the closest CIFAR10 test accuracy.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from torchvision.models import ResNet18_Weights, resnet18
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by runtime envs.
    raise ModuleNotFoundError(
        "This script requires PyTorch and torchvision. On this machine, use the "
        "`minitron` conda environment or run the provided Slurm job."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rn18_cifar10_common import CIFAR10_CLASSES, ensure_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEJEPA_ROOT = REPO_ROOT / "lejepa"


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_local_lejepa_components():
    package_root = LOCAL_LEJEPA_ROOT / "lejepa"
    if not package_root.exists():
        raise ModuleNotFoundError(
            "The true LeJEPA path requires the local `lejepa/` repository. "
            "Expected to find `lejepa/lejepa` inside this checkout."
        )
    prefix = "_cnn_filter_db_local_lejepa"
    for package_name, package_path in [
        (prefix, package_root),
        (f"{prefix}.univariate", package_root / "univariate"),
        (f"{prefix}.multivariate", package_root / "multivariate"),
    ]:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package
    load_module_from_path(f"{prefix}.univariate.base", package_root / "univariate" / "base.py")
    epps = load_module_from_path(f"{prefix}.univariate.epps_pulley", package_root / "univariate" / "epps_pulley.py")
    slicing = load_module_from_path(f"{prefix}.multivariate.slicing", package_root / "multivariate" / "slicing.py")
    return epps.EppsPulley, slicing.SlicingUnivariateTest


LEJEPA_EppsPulley, LEJEPA_SlicingUnivariateTest = load_local_lejepa_components()


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class MultiViewTransform:
    def __init__(self, transform, views: int):
        self.transform = transform
        self.views = views

    def __call__(self, image):
        return torch.stack([self.transform(image) for _ in range(self.views)], dim=0)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectorMLP(nn.Module):
    """Project ResNet features into the LeJEPA embedding space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_linear_head(feature_dim: int, num_classes: int, head_type: str) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(feature_dim, num_classes)
    if head_type == "layernorm_linear":
        return nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, num_classes))
    if head_type == "batchnorm_linear":
        return nn.Sequential(nn.BatchNorm1d(feature_dim), nn.Linear(feature_dim, num_classes))
    raise ValueError(f"Unknown linear head type: {head_type}")


class FrozenLinearModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        head_type: str = "linear",
    ):
        super().__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.head = make_linear_head(feature_dim, num_classes, head_type)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResNet18/CIFAR10 matched-checkpoint experiment.")
    parser.add_argument(
        "--output-dir",
        default="outputs/rn18_cifar10_experiment",
        help="Directory for checkpoints, CSV metrics, summaries, and later analysis.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/cifar10",
        help="CIFAR10 download/cache directory.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast/mixed precision.")
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Disable cuDNN while still using CUDA. Useful on nodes where cuDNN initialization fails.",
    )
    parser.add_argument("--fake-data", action="store_true", help="Use torchvision FakeData for smoke tests.")
    parser.add_argument("--fake-train-samples", type=int, default=512)
    parser.add_argument("--fake-test-samples", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--imagenet-epochs", type=int, default=20)
    parser.add_argument(
        "--imagenet-only",
        action="store_true",
        help=(
            "Train and checkpoint only the ImageNet1K-pretrained CIFAR10 fine-tune "
            "baseline, then exit before LeJEPA SSL and linear-probe training."
        ),
    )
    parser.add_argument(
        "--skip-imagenet-train",
        action="store_true",
        help="Reuse ImageNet fine-tune metrics/checkpoints from --imagenet-source-dir instead of training them.",
    )
    parser.add_argument(
        "--imagenet-source-dir",
        default="",
        help="Existing experiment directory containing imagenet_finetune_metrics.csv and ImageNet checkpoints.",
    )
    parser.add_argument("--ssl-epochs", type=int, default=40)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--imagenet-save-every", type=int, default=1)
    parser.add_argument("--ssl-save-every", type=int, default=5)
    parser.add_argument("--quick-probe-epochs", type=int, default=3)
    parser.add_argument("--quick-probe-interval", type=int, default=5)
    parser.add_argument("--ssl-min-checkpoints", type=int, default=4)
    parser.add_argument(
        "--ssl-early-stop-margin",
        type=float,
        default=0.005,
        help="Stop SSL once quick-probe accuracy exceeds best ImageNet FT accuracy by this margin.",
    )
    parser.add_argument("--finetune-lr", type=float, default=0.01)
    parser.add_argument("--linear-lr", type=float, default=0.1)
    parser.add_argument("--linear-weight-decay", type=float, default=None)
    parser.add_argument("--linear-optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument(
        "--linear-head-type",
        choices=["linear", "layernorm_linear", "batchnorm_linear"],
        default="linear",
    )
    parser.add_argument("--ssl-lr", type=float, default=5e-4)
    parser.add_argument("--ssl-weight-decay", type=float, default=5e-4)
    parser.add_argument("--ssl-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ssl-min-lr-factor", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--ssl-proj-dim", type=int, default=256)
    parser.add_argument("--ssl-hidden-dim", type=int, default=1024)
    parser.add_argument("--ssl-views", type=int, default=4)
    parser.add_argument("--ssl-global-views", type=int, default=0)
    parser.add_argument("--ssl-crop-min", type=float, default=0.45)
    parser.add_argument("--ssl-crop-max", type=float, default=1.0)
    parser.add_argument("--ssl-jitter-brightness", type=float, default=0.35)
    parser.add_argument("--ssl-jitter-contrast", type=float, default=0.35)
    parser.add_argument("--ssl-jitter-saturation", type=float, default=0.35)
    parser.add_argument("--ssl-jitter-hue", type=float, default=0.10)
    parser.add_argument("--ssl-grayscale-p", type=float, default=0.20)
    parser.add_argument("--ssl-blur-p", type=float, default=0.25)
    parser.add_argument("--ssl-solarize-p", type=float, default=0.0)
    parser.add_argument("--lejepa-lambda", type=float, default=0.05)
    parser.add_argument("--sigreg-num-slices", type=int, default=1024)
    parser.add_argument("--sigreg-t-max", type=float, default=3.0)
    parser.add_argument("--sigreg-n-points", type=int, default=17)
    parser.add_argument(
        "--ssl-amp-dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Autocast dtype for LeJEPA SSL when --amp is enabled.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny FakeData job that validates code paths quickly.",
    )
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke_test:
        return args
    args.fake_data = True
    args.fake_train_samples = min(args.fake_train_samples, 32)
    args.fake_test_samples = min(args.fake_test_samples, 32)
    args.max_train_samples = 0
    args.max_test_samples = 0
    args.imagenet_epochs = min(args.imagenet_epochs, 1)
    args.ssl_epochs = min(args.ssl_epochs, 1)
    args.linear_epochs = min(args.linear_epochs, 1)
    args.quick_probe_epochs = min(args.quick_probe_epochs, 1)
    args.quick_probe_interval = 1
    args.ssl_min_checkpoints = 1
    args.imagenet_save_every = 1
    args.ssl_save_every = 1
    args.ssl_views = min(args.ssl_views, 2)
    args.ssl_global_views = 0
    args.sigreg_num_slices = min(args.sigreg_num_slices, 16)
    args.sigreg_n_points = min(args.sigreg_n_points, 5)
    args.linear_epochs = min(args.linear_epochs, 1)
    args.batch_size = min(args.batch_size, 32)
    args.num_workers = 0
    args.device = "cpu"
    args.amp = False
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def validate_args(args: argparse.Namespace) -> None:
    if args.imagenet_only and (args.skip_imagenet_train or args.imagenet_epochs <= 0):
        raise ValueError("--imagenet-only requires ImageNet fine-tuning with --imagenet-epochs > 0.")
    if args.ssl_views < 2:
        raise ValueError("--ssl-views must be at least 2 for LeJEPA.")
    if args.ssl_global_views < 0:
        raise ValueError("--ssl-global-views must be non-negative.")
    if args.ssl_global_views > args.ssl_views:
        raise ValueError("--ssl-global-views cannot exceed --ssl-views.")
    if not 0.0 < args.ssl_crop_min <= args.ssl_crop_max <= 1.0:
        raise ValueError("--ssl-crop-min/--ssl-crop-max must satisfy 0 < min <= max <= 1.")
    for name in ["ssl_grayscale_p", "ssl_blur_p", "ssl_solarize_p"]:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if not 0.0 <= args.lejepa_lambda <= 1.0:
        raise ValueError("--lejepa-lambda must be in [0, 1].")
    if args.sigreg_n_points % 2 != 1:
        raise ValueError("--sigreg-n-points must be odd for the Epps-Pulley trapezoid rule.")
    if args.skip_imagenet_train or args.imagenet_epochs <= 0:
        if not args.imagenet_source_dir:
            raise ValueError("--skip-imagenet-train/--imagenet-epochs 0 requires --imagenet-source-dir.")


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_cuda_backend(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print("[cuda] cuDNN disabled by --disable-cudnn; CUDA kernels remain enabled.", flush=True)
        return
    try:
        probe = torch.randn(2, 3, 32, 32, device=device)
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
        _ = conv(probe)
        torch.cuda.synchronize()
        print("[cuda] cuDNN convolution probe passed.", flush=True)
    except RuntimeError as exc:
        if "cuDNN" not in str(exc):
            raise
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print(f"[cuda] cuDNN probe failed ({exc}); disabled cuDNN and continuing on CUDA.", flush=True)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def make_transforms(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    ssl_base = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(args.ssl_crop_min, args.ssl_crop_max), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        args.ssl_jitter_brightness,
                        args.ssl_jitter_contrast,
                        args.ssl_jitter_saturation,
                        args.ssl_jitter_hue,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=args.ssl_grayscale_p),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=args.ssl_blur_p),
            transforms.RandomSolarize(threshold=128, p=args.ssl_solarize_p),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, MultiViewTransform(ssl_base, views=args.ssl_views), test_transform


def maybe_subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def make_datasets(args: argparse.Namespace):
    train_transform, ssl_transform, test_transform = make_transforms(args)
    if args.fake_data:
        train_dataset = datasets.FakeData(
            size=args.fake_train_samples,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        ssl_dataset = datasets.FakeData(
            size=args.fake_train_samples,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=ssl_transform,
        )
        test_dataset = datasets.FakeData(
            size=args.fake_test_samples,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
        )
    else:
        data_dir = Path(args.data_dir)
        ensure_dir(data_dir)
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        ssl_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=ssl_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_dataset = maybe_subset(train_dataset, args.max_train_samples, args.seed)
    ssl_dataset = maybe_subset(ssl_dataset, args.max_train_samples, args.seed)
    test_dataset = maybe_subset(test_dataset, args.max_test_samples, args.seed + 1)
    return train_dataset, ssl_dataset, test_dataset


def make_loader(dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if args.num_workers else None,
        generator=generator if shuffle else None,
        persistent_workers=args.num_workers > 0,
    )


def adapt_resnet18_stem(model: nn.Module, from_imagenet: bool) -> None:
    old_conv = model.conv1
    new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if from_imagenet:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight[:, :, 2:5, 2:5])
    model.conv1 = new_conv
    model.maxpool = nn.Identity()


def build_resnet18_cifar(num_classes: int = 10, imagenet: bool = False, as_encoder: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
    model = resnet18(weights=weights)
    adapt_resnet18_stem(model, from_imagenet=imagenet)
    if as_encoder:
        model.fc = nn.Identity()
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def split_backbone_classifier_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    backbone = {}
    classifier = {}
    for key, value in model.state_dict().items():
        item = value.detach().cpu().clone()
        if key.startswith("fc."):
            classifier[key] = item
        else:
            backbone[key] = item
    return backbone, classifier


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    pred = logits.argmax(dim=1)
    return int((pred == target).sum().item()), int(target.numel())


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        correct, seen = accuracy_from_logits(logits.detach(), target)
        total_correct += correct
        total_seen += seen
        total_loss += float(loss.item()) * seen
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, target)
        correct, seen = accuracy_from_logits(logits, target)
        total_correct += correct
        total_seen += seen
        total_loss += float(loss.item()) * seen
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


@torch.no_grad()
def evaluate_with_confusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    cm = np.zeros((10, 10), dtype=np.int64)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, target)
        pred = logits.argmax(dim=1)
        for true_label, pred_label in zip(target.cpu().numpy(), pred.cpu().numpy()):
            cm[int(true_label), int(pred_label)] += 1
        correct, seen = accuracy_from_logits(logits, target)
        total_correct += correct
        total_seen += seen
        total_loss += float(loss.item()) * seen
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }, cm


def save_imagenet_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    backbone_state, classifier_state = split_backbone_classifier_state(model)
    torch.save(
        {
            "experiment": "rn18_cifar10_v3",
            "method": "imagenet_ft",
            "architecture": "torchvision_resnet18_cifar_stem",
            "dataset": "cifar10" if not args.fake_data else "fake_cifar10_shape",
            "epoch": epoch,
            "model_state": cpu_state_dict(model),
            "backbone_state": backbone_state,
            "classifier_state": classifier_state,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": (
                "Initialized from torchvision ResNet18 ImageNet1K_V1; conv1 was center-cropped "
                "from 7x7 stride-2 to a CIFAR-style 3x3 stride-1 stem."
            ),
        },
        path,
    )


def load_existing_imagenet_rows(source_dir: Path, metrics_path: Path) -> list[dict[str, Any]]:
    source_metrics = source_dir / "imagenet_finetune_metrics.csv"
    if not source_metrics.exists():
        raise FileNotFoundError(f"Missing ImageNet metrics for reuse: {source_metrics}")
    rows: list[dict[str, Any]] = []
    with source_metrics.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ckpt_path = Path(row.get("checkpoint_path", ""))
            if row.get("checkpoint_path") and not ckpt_path.exists():
                raise FileNotFoundError(f"Reused ImageNet checkpoint does not exist: {ckpt_path}")
            rows.append(dict(row))
    if not rows:
        raise RuntimeError(f"No ImageNet rows found in {source_metrics}")
    ensure_dir(metrics_path.parent)
    if source_metrics.resolve() != metrics_path.resolve():
        shutil.copyfile(source_metrics, metrics_path)
    print(
        f"[imagenet] Reusing {len(rows)} ImageNet fine-tune rows from {source_metrics}",
        flush=True,
    )
    return rows


def should_save_epoch(epoch: int, total_epochs: int, every: int) -> bool:
    if epoch == 1 or epoch == total_epochs:
        return True
    return every > 0 and epoch % every == 0


def train_imagenet_finetune(
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: Path,
    metrics_path: Path,
) -> list[dict[str, Any]]:
    print("[imagenet] Building ImageNet1k-pretrained ResNet18", flush=True)
    model = build_resnet18_cifar(num_classes=10, imagenet=not args.fake_data, as_encoder=False)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.finetune_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.imagenet_epochs, 1))
    rows: list[dict[str, Any]] = []
    fieldnames = [
        "method",
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "lr",
        "checkpoint_path",
        "elapsed_sec",
    ]
    for epoch in range(1, args.imagenet_epochs + 1):
        start = time.time()
        train_metrics = train_supervised_epoch(model, train_loader, optimizer, device, args.amp and device.type == "cuda")
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()
        ckpt_path = ""
        if should_save_epoch(epoch, args.imagenet_epochs, args.imagenet_save_every):
            ckpt = checkpoint_dir / f"imagenet_ft_epoch{epoch:03d}.pth"
            save_imagenet_checkpoint(model, ckpt, epoch, train_metrics, test_metrics, args)
            ckpt_path = str(ckpt)
        row = {
            "method": "imagenet_ft",
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "checkpoint_path": ckpt_path,
            "elapsed_sec": time.time() - start,
        }
        rows.append(row)
        append_csv_row(metrics_path, fieldnames, row)
        print(
            f"[imagenet] epoch {epoch:03d}/{args.imagenet_epochs}: "
            f"train_acc={train_metrics['accuracy']:.4f} test_acc={test_metrics['accuracy']:.4f}",
            flush=True,
        )
    return rows


def ssl_autocast(args: argparse.Namespace, device: torch.device, enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if args.ssl_amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def make_ssl_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(int(steps_per_epoch * args.ssl_epochs), 1)
    warmup_steps = max(int(steps_per_epoch * args.ssl_warmup_epochs), 1)
    min_factor = max(float(args.ssl_min_lr_factor), 0.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + 0.99 * (step + 1) / warmup_steps
        progress = min((step - warmup_steps + 1) / max(total_steps - warmup_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_sigreg(args: argparse.Namespace, device: torch.device) -> nn.Module:
    univariate_test = LEJEPA_EppsPulley(
        t_max=args.sigreg_t_max,
        n_points=args.sigreg_n_points,
    )
    sigreg = LEJEPA_SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=args.sigreg_num_slices,
        reduction="mean",
    )
    return sigreg.to(device)


def lejepa_ssl_loss(
    backbone: nn.Module,
    projector: nn.Module,
    sigreg: nn.Module,
    views: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if views.ndim != 5:
        raise ValueError(f"Expected SSL views as [batch, views, channels, height, width], got {tuple(views.shape)}")
    batch_size, num_views = int(views.size(0)), int(views.size(1))
    if num_views < 2:
        raise ValueError("LeJEPA needs at least two views per sample.")
    global_views = args.ssl_global_views if args.ssl_global_views > 0 else num_views
    global_views = min(global_views, num_views)

    flat = views.flatten(0, 1)
    embeddings = backbone(flat)
    projections = projector(embeddings).float().view(batch_size, num_views, -1).transpose(0, 1)

    centers = projections[:global_views].mean(dim=0)
    pred_loss = (projections - centers.unsqueeze(0)).square().mean()
    sigreg_loss = torch.stack([sigreg(projections[view_idx]) for view_idx in range(num_views)]).mean()
    total = (1.0 - args.lejepa_lambda) * pred_loss + args.lejepa_lambda * sigreg_loss
    metrics = {
        "loss": float(total.detach().item()),
        "prediction_loss": float(pred_loss.detach().item()),
        "sigreg_loss": float(sigreg_loss.detach().item()),
    }
    return total, metrics


def train_ssl_epoch(
    backbone: nn.Module,
    projector: nn.Module,
    sigreg: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
) -> dict[str, float]:
    backbone.train()
    projector.train()
    sigreg.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.ssl_amp_dtype == "fp16")
    total_loss = 0.0
    total_pred = 0.0
    total_sigreg = 0.0
    total_seen = 0
    for views, _ in loader:
        views = views.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with ssl_autocast(args, device, enabled=use_amp):
            loss, metrics = lejepa_ssl_loss(backbone, projector, sigreg, views, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        seen = int(views.size(0))
        total_seen += seen
        total_loss += metrics["loss"] * seen
        total_pred += metrics["prediction_loss"] * seen
        total_sigreg += metrics["sigreg_loss"] * seen
    return {
        "loss": total_loss / max(total_seen, 1),
        "prediction_loss": total_pred / max(total_seen, 1),
        "sigreg_loss": total_sigreg / max(total_seen, 1),
    }


def save_ssl_checkpoint(
    path: Path,
    epoch: int,
    backbone: nn.Module,
    projector: nn.Module,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "experiment": "rn18_cifar10_v3",
            "method": "lejepa_ssl",
            "architecture": "torchvision_resnet18_cifar_stem",
            "dataset": "cifar10_unlabeled" if not args.fake_data else "fake_cifar10_shape",
            "epoch": epoch,
            "backbone_state": cpu_state_dict(backbone),
            "projector_state": cpu_state_dict(projector),
            "ssl_loss": metrics["loss"],
            "prediction_loss": metrics["prediction_loss"],
            "sigreg_loss": metrics["sigreg_loss"],
            "lejepa_lambda": args.lejepa_lambda,
            "ssl_views": args.ssl_views,
            "ssl_global_views": args.ssl_global_views if args.ssl_global_views > 0 else args.ssl_views,
            "sigreg_num_slices": args.sigreg_num_slices,
            "sigreg_t_max": args.sigreg_t_max,
            "sigreg_n_points": args.sigreg_n_points,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": (
                "True LeJEPA objective from the supplied paper/repository: multi-view CIFAR10 "
                "prediction/invariance loss plus SIGReg with sliced Epps-Pulley normality. "
                "No target encoder, EMA, predictor, stop-gradient, prototypes, or labels are "
                "used for SSL backbone training."
            ),
        },
        path,
    )


def load_backbone_state_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("backbone_state")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain backbone_state")
    return state


def build_frozen_linear_from_backbone_state(
    backbone_state: dict[str, torch.Tensor],
    device: torch.device,
    classifier_state: dict[str, torch.Tensor] | None = None,
    head_type: str = "linear",
) -> FrozenLinearModel:
    backbone = build_resnet18_cifar(num_classes=10, imagenet=False, as_encoder=True)
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    unexpected_real = [key for key in unexpected if not key.startswith("fc.")]
    if unexpected_real:
        raise ValueError(f"Unexpected backbone keys: {unexpected_real[:5]}")
    if missing:
        raise ValueError(f"Missing backbone keys: {missing[:5]}")
    model = FrozenLinearModel(backbone, feature_dim=512, num_classes=10, head_type=head_type)
    if classifier_state:
        remapped = {
            key.replace("fc.", "head."): value
            for key, value in classifier_state.items()
        }
        model.load_state_dict(remapped, strict=False)
    return model.to(device)


def train_linear_head_from_state(
    backbone_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[FrozenLinearModel, list[dict[str, float]]]:
    model = build_frozen_linear_from_backbone_state(backbone_state, device, head_type=args.linear_head_type)
    linear_wd = args.weight_decay if args.linear_weight_decay is None else args.linear_weight_decay
    if args.linear_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.head.parameters(),
            lr=args.linear_lr,
            momentum=args.momentum,
            weight_decay=linear_wd,
            nesterov=True,
        )
    elif args.linear_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=args.linear_lr,
            weight_decay=linear_wd,
        )
    else:
        raise ValueError(f"Unknown linear optimizer: {args.linear_optimizer}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.linear_epochs, 1))
    rows: list[dict[str, float]] = []
    for epoch in range(1, args.linear_epochs + 1):
        train_metrics = train_supervised_epoch(model, train_loader, optimizer, device, args.amp and device.type == "cuda")
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()
        rows.append(
            {
                "linear_epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )
    return model, rows


def quick_probe_accuracy(
    backbone_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> float:
    old_epochs = args.linear_epochs
    args.linear_epochs = epochs
    try:
        _, rows = train_linear_head_from_state(backbone_state, args, train_loader, test_loader, device)
    finally:
        args.linear_epochs = old_epochs
    return float(rows[-1]["test_accuracy"]) if rows else float("nan")


def train_lejepa_ssl(
    args: argparse.Namespace,
    ssl_loader: DataLoader,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: Path,
    ssl_metrics_path: Path,
    quick_probe_path: Path,
    imagenet_target_accuracy: float,
) -> list[Path]:
    print("[ssl] Building true LeJEPA ResNet18 encoder", flush=True)
    backbone = build_resnet18_cifar(num_classes=10, imagenet=False, as_encoder=True).to(device)
    projector = ProjectorMLP(512, args.ssl_hidden_dim, args.ssl_proj_dim).to(device)
    sigreg = build_sigreg(args, device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=args.ssl_lr,
        weight_decay=args.ssl_weight_decay,
    )
    scheduler = make_ssl_scheduler(optimizer, args, steps_per_epoch=len(ssl_loader))
    ssl_fieldnames = [
        "method",
        "epoch",
        "ssl_loss",
        "prediction_loss",
        "sigreg_loss",
        "lejepa_lambda",
        "ssl_views",
        "sigreg_num_slices",
        "lr",
        "checkpoint_path",
        "elapsed_sec",
    ]
    quick_fieldnames = ["method", "ssl_epoch", "quick_probe_epochs", "quick_probe_accuracy", "target_accuracy"]
    saved_paths: list[Path] = []
    for epoch in range(1, args.ssl_epochs + 1):
        start = time.time()
        metrics = train_ssl_epoch(
            backbone,
            projector,
            sigreg,
            ssl_loader,
            optimizer,
            scheduler,
            device,
            args,
            args.amp and device.type == "cuda",
        )
        ckpt_path = ""
        if should_save_epoch(epoch, args.ssl_epochs, args.ssl_save_every):
            ckpt = checkpoint_dir / f"lejepa_ssl_epoch{epoch:03d}.pth"
            save_ssl_checkpoint(
                ckpt,
                epoch,
                backbone,
                projector,
                metrics,
                args,
            )
            saved_paths.append(ckpt)
            ckpt_path = str(ckpt)
        row = {
            "method": "lejepa_ssl",
            "epoch": epoch,
            "ssl_loss": metrics["loss"],
            "prediction_loss": metrics["prediction_loss"],
            "sigreg_loss": metrics["sigreg_loss"],
            "lejepa_lambda": args.lejepa_lambda,
            "ssl_views": args.ssl_views,
            "sigreg_num_slices": args.sigreg_num_slices,
            "lr": optimizer.param_groups[0]["lr"],
            "checkpoint_path": ckpt_path,
            "elapsed_sec": time.time() - start,
        }
        append_csv_row(ssl_metrics_path, ssl_fieldnames, row)
        print(
            f"[ssl] epoch {epoch:03d}/{args.ssl_epochs}: "
            f"loss={metrics['loss']:.4f} pred={metrics['prediction_loss']:.4f} "
            f"sigreg={metrics['sigreg_loss']:.4f}",
            flush=True,
        )

        do_quick_probe = (
            args.quick_probe_epochs > 0
            and (epoch == 1 or epoch == args.ssl_epochs or epoch % max(args.quick_probe_interval, 1) == 0)
        )
        if do_quick_probe:
            backbone_state = cpu_state_dict(backbone)
            probe_acc = quick_probe_accuracy(
                backbone_state,
                args,
                train_loader,
                test_loader,
                device,
                args.quick_probe_epochs,
            )
            append_csv_row(
                quick_probe_path,
                quick_fieldnames,
                {
                    "method": "lejepa_ssl_quick_probe",
                    "ssl_epoch": epoch,
                    "quick_probe_epochs": args.quick_probe_epochs,
                    "quick_probe_accuracy": probe_acc,
                    "target_accuracy": imagenet_target_accuracy,
                },
            )
            print(
                f"[ssl] quick probe at ssl_epoch={epoch:03d}: acc={probe_acc:.4f} "
                f"target={imagenet_target_accuracy:.4f}",
                flush=True,
            )
            if (
                len(saved_paths) >= args.ssl_min_checkpoints
                and probe_acc >= imagenet_target_accuracy + args.ssl_early_stop_margin
            ):
                print(
                    "[ssl] Early stopping SSL pretraining because the quick probe reached "
                    "the ImageNet target band.",
                    flush=True,
                )
                break
    return saved_paths


def train_all_linear_heads(
    args: argparse.Namespace,
    ssl_checkpoints: list[Path],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    linear_dir: Path,
    metrics_path: Path,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    fieldnames = [
        "method",
        "ssl_epoch",
        "linear_epochs",
        "linear_head_type",
        "linear_optimizer",
        "linear_lr",
        "linear_weight_decay",
        "final_train_loss",
        "final_train_accuracy",
        "final_test_loss",
        "final_test_accuracy",
        "best_test_accuracy",
        "ssl_checkpoint_path",
        "linear_checkpoint_path",
        "elapsed_sec",
    ]
    for ssl_path in ssl_checkpoints:
        start = time.time()
        ssl_checkpoint = torch.load(ssl_path, map_location="cpu")
        ssl_epoch = int(ssl_checkpoint["epoch"])
        backbone_state = ssl_checkpoint["backbone_state"]
        model, linear_rows = train_linear_head_from_state(backbone_state, args, train_loader, test_loader, device)
        final = linear_rows[-1]
        best_test = max(float(row["test_accuracy"]) for row in linear_rows)
        linear_ckpt = linear_dir / f"lejepa_linear_ssl_epoch{ssl_epoch:03d}.pth"
        linear_wd = args.weight_decay if args.linear_weight_decay is None else args.linear_weight_decay
        classifier_state = {}
        if args.linear_head_type == "linear":
            classifier_state = {
                f"fc.{key.split('.', 1)[1]}": value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if key.startswith("head.")
            }
        torch.save(
            {
                "experiment": "rn18_cifar10_v3",
                "method": "lejepa_ssl_linear",
                "architecture": "torchvision_resnet18_cifar_stem",
                "dataset": "cifar10" if not args.fake_data else "fake_cifar10_shape",
                "ssl_epoch": ssl_epoch,
                "linear_epochs": args.linear_epochs,
                "ssl_checkpoint_path": str(ssl_path),
                "backbone_state": backbone_state,
                "classifier_state": classifier_state,
                "linear_model_state": cpu_state_dict(model),
                "linear_head_type": args.linear_head_type,
                "linear_optimizer": args.linear_optimizer,
                "linear_lr": args.linear_lr,
                "linear_weight_decay": linear_wd,
                "linear_history": linear_rows,
                "final_test_accuracy": final["test_accuracy"],
                "best_test_accuracy": best_test,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": "Frozen SSL backbone with fixed-epoch linear-head training; no linear-head early stop.",
            },
            linear_ckpt,
        )
        row = {
            "method": "lejepa_ssl_linear",
            "ssl_epoch": ssl_epoch,
            "linear_epochs": args.linear_epochs,
            "linear_head_type": args.linear_head_type,
            "linear_optimizer": args.linear_optimizer,
            "linear_lr": args.linear_lr,
            "linear_weight_decay": linear_wd,
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "final_test_loss": final["test_loss"],
            "final_test_accuracy": final["test_accuracy"],
            "best_test_accuracy": best_test,
            "ssl_checkpoint_path": str(ssl_path),
            "linear_checkpoint_path": str(linear_ckpt),
            "elapsed_sec": time.time() - start,
        }
        rows_out.append(row)
        append_csv_row(metrics_path, fieldnames, row)
        print(
            f"[linear] ssl_epoch={ssl_epoch:03d}: final_acc={final['test_accuracy']:.4f} "
            f"best_acc={best_test:.4f}",
            flush=True,
        )
    return rows_out


def match_checkpoints(
    imagenet_rows: list[dict[str, Any]],
    linear_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    candidates = [row for row in imagenet_rows if row.get("checkpoint_path")]
    matches: list[dict[str, Any]] = []
    for im_row in candidates:
        for lj_row in linear_rows:
            im_acc = float(im_row["test_accuracy"])
            lj_acc = float(lj_row["final_test_accuracy"])
            matches.append(
                {
                    "imagenet_epoch": int(im_row["epoch"]),
                    "lejepa_ssl_epoch": int(lj_row["ssl_epoch"]),
                    "imagenet_accuracy": im_acc,
                    "lejepa_accuracy": lj_acc,
                    "abs_accuracy_delta": abs(im_acc - lj_acc),
                    "imagenet_checkpoint_path": im_row["checkpoint_path"],
                    "lejepa_linear_checkpoint_path": lj_row["linear_checkpoint_path"],
                    "lejepa_ssl_checkpoint_path": lj_row["ssl_checkpoint_path"],
                }
            )
    matches.sort(key=lambda row: (row["abs_accuracy_delta"], row["imagenet_epoch"], row["lejepa_ssl_epoch"]))
    for rank, row in enumerate(matches, start=1):
        row["rank"] = rank
    fieldnames = [
        "rank",
        "imagenet_epoch",
        "lejepa_ssl_epoch",
        "imagenet_accuracy",
        "lejepa_accuracy",
        "abs_accuracy_delta",
        "imagenet_checkpoint_path",
        "lejepa_linear_checkpoint_path",
        "lejepa_ssl_checkpoint_path",
    ]
    with (output_dir / "matched_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matches)
    if not matches:
        raise RuntimeError("No checkpoint matches could be computed.")
    selected = matches[0]
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_rule": "minimum absolute CIFAR10 test accuracy delta using fixed-epoch LeJEPA linear heads",
        "selected": selected,
        "top_10": matches[:10],
    }
    save_json(output_dir / "selected_pair.json", payload)
    return selected


def build_imagenet_inference_model(path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    model = build_resnet18_cifar(num_classes=10, imagenet=False, as_encoder=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device)


def build_lejepa_linear_inference_model(path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    if "linear_model_state" in checkpoint:
        model = build_frozen_linear_from_backbone_state(
            checkpoint["backbone_state"],
            device,
            head_type=checkpoint.get("linear_head_type", "linear"),
        )
        model.load_state_dict(checkpoint["linear_model_state"], strict=True)
        return model.to(device)
    return build_frozen_linear_from_backbone_state(
        checkpoint["backbone_state"],
        device,
        classifier_state=checkpoint["classifier_state"],
    )


def save_confusion_csv(path: Path, model_id: str, cm: np.ndarray) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        fieldnames = ["model_id", "true_class", *CIFAR10_CLASSES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for true_idx, class_name in enumerate(CIFAR10_CLASSES):
            row = {"model_id": model_id, "true_class": class_name}
            row.update({CIFAR10_CLASSES[pred_idx]: int(cm[true_idx, pred_idx]) for pred_idx in range(10)})
            writer.writerow(row)


def evaluate_matched_pair(
    selected: dict[str, Any],
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    confusion_path = output_dir / "matched_confusion_matrices.csv"
    if confusion_path.exists():
        confusion_path.unlink()
    rows = []
    for model_id, path_key, builder in [
        ("imagenet_ft", "imagenet_checkpoint_path", build_imagenet_inference_model),
        ("lejepa_ssl_linear", "lejepa_linear_checkpoint_path", build_lejepa_linear_inference_model),
    ]:
        model = builder(Path(selected[path_key]), device)
        metrics, cm = evaluate_with_confusion(model, test_loader, device)
        save_confusion_csv(confusion_path, model_id, cm)
        rows.append(
            {
                "model_id": model_id,
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "checkpoint_path": selected[path_key],
            }
        )
        print(f"[matched] {model_id}: test_acc={metrics['accuracy']:.4f}", flush=True)
    with (output_dir / "matched_eval_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", "test_loss", "test_accuracy", "checkpoint_path"])
        writer.writeheader()
        writer.writerows(rows)


def write_imagenet_only_summary(
    output_dir: Path,
    args: argparse.Namespace,
    imagenet_rows: list[dict[str, Any]],
) -> None:
    best = max(imagenet_rows, key=lambda row: float(row["test_accuracy"]))
    final = max(imagenet_rows, key=lambda row: int(row["epoch"]))
    checkpoint_count = sum(1 for row in imagenet_rows if row.get("checkpoint_path"))
    lines = [
        "# ResNet18 CIFAR10 ImageNet Baseline Regeneration",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "- Model: torchvision ResNet18 initialized with ImageNet1K_V1 weights.",
        "- Adaptation: CIFAR-style `3x3`, stride-1 stem with no max-pool.",
        f"- Seed: `{args.seed}`; batch size: `{args.batch_size}`; epochs: `{args.imagenet_epochs}`.",
        f"- Saved checkpoints: `{checkpoint_count}`.",
        f"- Best test accuracy: `{float(best['test_accuracy']):.4f}` at epoch `{int(best['epoch'])}`.",
        f"- Final test accuracy: `{float(final['test_accuracy']):.4f}` at epoch `{int(final['epoch'])}`.",
        "- Metrics: `imagenet_finetune_metrics.csv`; checkpoints: `checkpoints/imagenet_ft_epoch*.pth`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    selected: dict[str, Any],
    imagenet_rows: list[dict[str, Any]],
    linear_rows: list[dict[str, Any]],
    ssl_checkpoints: list[Path],
) -> None:
    best_imagenet = max(float(row["test_accuracy"]) for row in imagenet_rows)
    best_lejepa = max(float(row["final_test_accuracy"]) for row in linear_rows)
    lines = []
    lines.append("# ResNet18 CIFAR10 Third Experiment")
    lines.append("")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Model: torchvision ResNet18 with a CIFAR-style `3x3` stride-1 stem.")
    lines.append("- Dataset: CIFAR10 test accuracy is the matching metric.")
    lines.append("- Method A: ImageNet1k-pretrained ResNet18, all layers fine-tuned on CIFAR10.")
    if args.skip_imagenet_train or args.imagenet_epochs <= 0:
        lines.append(f"- Method A checkpoints were reused from `{args.imagenet_source_dir}`.")
    lines.append(
        "- Method B: true LeJEPA SSL on CIFAR10 using view-center prediction plus "
        "SIGReg/Epps-Pulley isotropic-Gaussian regularization; frozen backbone, fixed-epoch linear head."
    )
    lines.append(
        f"- LeJEPA setup: `V={args.ssl_views}`, `global_views="
        f"{args.ssl_global_views if args.ssl_global_views > 0 else args.ssl_views}`, "
        f"`lambda={args.lejepa_lambda}`, `sigreg_slices={args.sigreg_num_slices}`."
    )
    lines.append(
        f"- LeJEPA augmentation: crop scale `({args.ssl_crop_min}, {args.ssl_crop_max})`, "
        f"jitter `({args.ssl_jitter_brightness}, {args.ssl_jitter_contrast}, "
        f"{args.ssl_jitter_saturation}, {args.ssl_jitter_hue})`, "
        f"blur p `{args.ssl_blur_p}`, solarize p `{args.ssl_solarize_p}`."
    )
    linear_wd = args.weight_decay if args.linear_weight_decay is None else args.linear_weight_decay
    lines.append(
        f"- Linear probe: `{args.linear_head_type}`, optimizer `{args.linear_optimizer}`, "
        f"lr `{args.linear_lr}`, weight decay `{linear_wd}`, epochs `{args.linear_epochs}`."
    )
    lines.append("- Labels are not used for SSL backbone training; labels are used only for quick/final linear probes.")
    lines.append("- Matching: minimum absolute CIFAR10 test-accuracy gap among saved checkpoints.")
    if args.fake_data:
        lines.append("- This run used `--fake-data`; use the Slurm job for real CIFAR10 results.")
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    lines.append(f"- ImageNet FT checkpointed epochs: `{sum(1 for row in imagenet_rows if row.get('checkpoint_path'))}`")
    lines.append(f"- LeJEPA SSL checkpoints: `{len(ssl_checkpoints)}`")
    lines.append(f"- LeJEPA linear heads trained: `{len(linear_rows)}`")
    lines.append("")
    lines.append("## Accuracy")
    lines.append("")
    lines.append(f"- Best ImageNet FT test accuracy: `{best_imagenet:.4f}`")
    lines.append(f"- Best LeJEPA linear-head test accuracy: `{best_lejepa:.4f}`")
    lines.append(
        "- Selected matched pair: "
        f"ImageNet epoch `{selected['imagenet_epoch']}` acc `{selected['imagenet_accuracy']:.4f}` vs "
        f"LeJEPA SSL epoch `{selected['lejepa_ssl_epoch']}` acc `{selected['lejepa_accuracy']:.4f}` "
        f"(delta `{selected['abs_accuracy_delta']:.4f}`)."
    )
    lines.append("")
    lines.append("## Key Files")
    lines.append("")
    lines.append("- `selected_pair.json`: selected checkpoint pair and top-10 alternatives.")
    lines.append("- `matched_checkpoints.csv`: all pairwise accuracy matches.")
    lines.append("- `imagenet_finetune_metrics.csv`: supervised fine-tuning curve.")
    lines.append("- `lejepa_ssl_metrics.csv`: SSL loss and checkpoint curve.")
    lines.append("- `lejepa_linear_metrics.csv`: fixed-epoch linear-head results per SSL checkpoint.")
    lines.append("- `matched_confusion_matrices.csv`: CIFAR10 confusion matrices for the matched pair.")
    lines.append("- `lejepa_protocol.md`: exact LeJEPA adaptation used for this RN18/CIFAR10 run.")
    lines.append("")
    lines.append("Next step: run `scripts/analyze_rn18_cifar10_filters.py` on this output directory.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_lejepa_protocol(output_dir: Path, args: argparse.Namespace) -> None:
    global_views = args.ssl_global_views if args.ssl_global_views > 0 else args.ssl_views
    lines = []
    lines.append("# RN18/CIFAR10 LeJEPA Protocol")
    lines.append("")
    lines.append("This run uses the original local LeJEPA paper/repository objective, not the earlier EMA/predictor proxy.")
    lines.append("")
    lines.append("## Source Interpretation")
    lines.append("")
    lines.append("- Paper/repo objective: LeJEPA = prediction/invariance loss across views + SIGReg.")
    lines.append("- SIGReg implementation: the original local `lejepa/lejepa/univariate/epps_pulley.py` and `lejepa/lejepa/multivariate/slicing.py` source files.")
    lines.append("- For ResNet/non-ViT backbones, the paper's Algorithm 2 says to set `global_views = all_views`; this run does that unless `--ssl-global-views` overrides it.")
    lines.append("- Removed from the previous proxy: EMA target encoder, target projector, predictor MLP, normalized cosine target prediction, and stop-gradient.")
    lines.append("")
    lines.append("## CIFAR10 Adaptation")
    lines.append("")
    lines.append(f"- Each CIFAR10 image produces `{args.ssl_views}` stochastic 32x32 augmented views.")
    lines.append(f"- ResNet18 uses the same CIFAR-style 3x3 stride-1 stem as the ImageNet fine-tune baseline.")
    lines.append(f"- The encoder maps every view to a 512-d feature; a projector maps 512 -> {args.ssl_hidden_dim} -> {args.ssl_hidden_dim} -> {args.ssl_proj_dim}.")
    lines.append(f"- Prediction loss pulls every view embedding toward the per-image center of `{global_views}` global/all views.")
    lines.append(f"- SIGReg is computed separately over the minibatch for each view, then averaged across views.")
    lines.append(f"- Final SSL loss: `(1 - lambda) * prediction_loss + lambda * sigreg_loss`, with `lambda={args.lejepa_lambda}`.")
    lines.append(f"- SIGReg parameters: `{args.sigreg_num_slices}` slices, `t_max={args.sigreg_t_max}`, `{args.sigreg_n_points}` Epps-Pulley integration points.")
    lines.append(
        f"- Augmentation: RandomResizedCrop scale `({args.ssl_crop_min}, {args.ssl_crop_max})`, "
        f"ColorJitter `({args.ssl_jitter_brightness}, {args.ssl_jitter_contrast}, "
        f"{args.ssl_jitter_saturation}, {args.ssl_jitter_hue})`, grayscale p `{args.ssl_grayscale_p}`, "
        f"blur p `{args.ssl_blur_p}`, solarize p `{args.ssl_solarize_p}`."
    )
    lines.append("- Optimizer: AdamW over encoder + projector only; no labels affect SSL backbone gradients.")
    lines.append("")
    lines.append("## Evaluation")
    lines.append("")
    lines.append("- Saved SSL checkpoints are evaluated with frozen-backbone fixed-epoch linear heads.")
    linear_wd = args.weight_decay if args.linear_weight_decay is None else args.linear_weight_decay
    lines.append(
        f"- Linear probe: `{args.linear_head_type}`, `{args.linear_optimizer}`, lr `{args.linear_lr}`, "
        f"weight decay `{linear_wd}`, epochs `{args.linear_epochs}`."
    )
    lines.append("- Quick probes are monitoring only; final matching uses the fixed linear-head results saved in `lejepa_linear_metrics.csv`.")
    if args.skip_imagenet_train or args.imagenet_epochs <= 0:
        lines.append(f"- ImageNet fine-tune checkpoints are reused from `{args.imagenet_source_dir}`.")
    lines.append("- Old proxy checkpoints are intentionally ignored.")
    (output_dir / "lejepa_protocol.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = apply_smoke_defaults(parse_args())
    validate_args(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if (args.skip_imagenet_train or args.imagenet_epochs <= 0) and Path(args.imagenet_source_dir).resolve() == output_dir.resolve():
        raise ValueError("Use a fresh --output-dir when reusing ImageNet checkpoints from --imagenet-source-dir.")
    checkpoint_dir = output_dir / "checkpoints"
    linear_dir = output_dir / "linear_heads"
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(linear_dir)

    config = vars(args).copy()
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(output_dir / "config.json", config)

    device = resolve_device(args.device)
    print(f"[start] output_dir={output_dir}", flush=True)
    print(f"[start] device={device}", flush=True)
    print(f"[start] torch={torch.__version__}", flush=True)
    print(f"[start] torchvision datasets root={args.data_dir}", flush=True)
    if args.imagenet_only:
        print("[start] mode=imagenet-only; LeJEPA and linear-probe phases are disabled", flush=True)
    else:
        print(
            f"[start] true LeJEPA: views={args.ssl_views} "
            f"global_views={args.ssl_global_views if args.ssl_global_views > 0 else args.ssl_views} "
            f"lambda={args.lejepa_lambda} sigreg_slices={args.sigreg_num_slices}",
            flush=True,
        )
    configure_cuda_backend(args, device)

    train_dataset, ssl_dataset, test_dataset = make_datasets(args)
    print(
        f"[data] train={len(train_dataset)} ssl={len(ssl_dataset)} test={len(test_dataset)} "
        f"fake={args.fake_data}",
        flush=True,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True)
    ssl_loader = make_loader(ssl_dataset, args, shuffle=True)
    test_loader = make_loader(test_dataset, args, shuffle=False)

    imagenet_metrics_path = output_dir / "imagenet_finetune_metrics.csv"
    ssl_metrics_path = output_dir / "lejepa_ssl_metrics.csv"
    quick_probe_path = output_dir / "lejepa_quick_probe_metrics.csv"
    linear_metrics_path = output_dir / "lejepa_linear_metrics.csv"
    for path in [imagenet_metrics_path, ssl_metrics_path, quick_probe_path, linear_metrics_path]:
        if path.exists():
            path.unlink()

    if args.skip_imagenet_train or args.imagenet_epochs <= 0:
        imagenet_rows = load_existing_imagenet_rows(Path(args.imagenet_source_dir), imagenet_metrics_path)
    else:
        imagenet_rows = train_imagenet_finetune(
            args,
            train_loader,
            test_loader,
            device,
            checkpoint_dir,
            imagenet_metrics_path,
        )
    if args.imagenet_only:
        write_imagenet_only_summary(output_dir, args, imagenet_rows)
        best = max(imagenet_rows, key=lambda row: float(row["test_accuracy"]))
        final = max(imagenet_rows, key=lambda row: int(row["epoch"]))
        checkpoint_count = sum(1 for row in imagenet_rows if row.get("checkpoint_path"))
        print(
            "[done] ImageNet-only fine-tune complete: "
            f"epochs={len(imagenet_rows)} checkpoints={checkpoint_count} "
            f"best_epoch={int(best['epoch'])} best_acc={float(best['test_accuracy']):.4f} "
            f"final_acc={float(final['test_accuracy']):.4f}",
            flush=True,
        )
        return 0
    imagenet_target_accuracy = max(float(row["test_accuracy"]) for row in imagenet_rows)
    write_lejepa_protocol(output_dir, args)
    ssl_checkpoints = train_lejepa_ssl(
        args,
        ssl_loader,
        train_loader,
        test_loader,
        device,
        checkpoint_dir,
        ssl_metrics_path,
        quick_probe_path,
        imagenet_target_accuracy,
    )
    ssl_checkpoints = sorted(set(ssl_checkpoints))
    if not ssl_checkpoints:
        raise RuntimeError("No LeJEPA SSL checkpoints were saved.")
    linear_rows = train_all_linear_heads(
        args,
        ssl_checkpoints,
        train_loader,
        test_loader,
        device,
        linear_dir,
        linear_metrics_path,
    )
    selected = match_checkpoints(imagenet_rows, linear_rows, output_dir)
    evaluate_matched_pair(selected, test_loader, device, output_dir)
    write_summary(output_dir, args, selected, imagenet_rows, linear_rows, ssl_checkpoints)
    print(
        "[done] selected pair: "
        f"ImageNet epoch {selected['imagenet_epoch']} acc={selected['imagenet_accuracy']:.4f}; "
        f"LeJEPA SSL epoch {selected['lejepa_ssl_epoch']} acc={selected['lejepa_accuracy']:.4f}; "
        f"delta={selected['abs_accuracy_delta']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
