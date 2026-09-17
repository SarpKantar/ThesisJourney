"""Shared, testable components for Experiment 5.

The module deliberately has no dependency on the older Experiment 3 driver.  It
implements the fixed A/B/C objectives, deterministic stream policy, CIFAR model,
and checkpoint helpers used by the Experiment 5 entry points.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment5" / "main.json"
LOCAL_LEJEPA_ROOT = REPO_ROOT / "lejepa"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
ARMS = {
    "A": "alignment_only",
    "B": "alignment_plus_sigreg",
    "C": "alignment_plus_vicreg_variance_covariance",
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    dataset = config["dataset"]
    training = config["training"]
    if int(dataset["train_size"]) + int(dataset["validation_size"]) != 50_000:
        raise ValueError("Experiment 5 CIFAR10 split must cover all 50,000 official training images.")
    if int(training["complete_batches_per_epoch"]) != int(dataset["train_size"]) // int(training["batch_size"]):
        raise ValueError("complete_batches_per_epoch must equal floor(train_size / batch_size).")
    expected_updates = int(training["complete_batches_per_epoch"]) * int(training["equivalent_epochs"])
    if int(training["total_updates"]) != expected_updates:
        raise ValueError("total_updates must match complete batches times equivalent epochs.")
    checkpoints = {int(value) for value in training["checkpoint_updates"]}
    if 0 not in checkpoints or int(training["total_updates"]) not in checkpoints:
        raise ValueError("Checkpoint schedule must include initialization and the final update.")
    if int(config["sigreg"]["integration_points"]) % 2 != 1:
        raise ValueError("SIGReg integration_points must be odd.")
    if int(config["augmentation"]["views"]) < 2:
        raise ValueError("At least two views are required for alignment.")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def derive_seed(base_seed: int, stream_name: str) -> int:
    """Derive stable, non-overlapping 63-bit seeds without Python's salted hash."""

    payload = f"experiment5:{int(base_seed)}:{stream_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def seed_streams(base_seed: int, names: Sequence[str]) -> dict[str, int]:
    result = {name: derive_seed(base_seed, name) for name in names}
    if len(set(result.values())) != len(result):
        raise RuntimeError("Derived RNG stream seeds unexpectedly collided.")
    return result


def set_global_seed(seed: int) -> None:
    seed32 = int(seed) % (2**32)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(int(seed) % (2**63 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) % (2**63 - 1))
    torch.backends.cudnn.benchmark = False


@contextmanager
def deterministic_rng(seed: int) -> Iterator[None]:
    """Temporarily seed Python/NumPy/Torch, restoring every state afterward."""

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


def capture_rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_local_lejepa_components():
    package_root = LOCAL_LEJEPA_ROOT / "lejepa"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Missing pinned LeJEPA source at {package_root}")
    prefix = "_cnn_filter_db_exp5_lejepa"
    for package_name, package_path in [
        (prefix, package_root),
        (f"{prefix}.univariate", package_root / "univariate"),
        (f"{prefix}.multivariate", package_root / "multivariate"),
    ]:
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            sys.modules[package_name] = package
    _load_module(f"{prefix}.univariate.base", package_root / "univariate" / "base.py")
    epps = _load_module(f"{prefix}.univariate.epps_pulley", package_root / "univariate" / "epps_pulley.py")
    slicing = _load_module(f"{prefix}.multivariate.slicing", package_root / "multivariate" / "slicing.py")
    return epps.EppsPulley, slicing.SlicingUnivariateTest


def git_metadata(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()

    try:
        status = run("status", "--porcelain")
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def assert_pinned_lejepa(config: Mapping[str, Any], allow_dirty: bool = False) -> dict[str, Any]:
    metadata = git_metadata(LOCAL_LEJEPA_ROOT)
    expected = str(config["sigreg"]["source_commit"])
    if metadata["commit"] != expected:
        raise RuntimeError(
            f"LeJEPA source mismatch: expected {expected}, found {metadata['commit']}. "
            "Do not run Experiment 5 against an unrecorded implementation."
        )
    meaningful_status = [line for line in metadata["status"] if "__pycache__" not in line and not line.endswith(".pyc")]
    metadata["meaningful_dirty_paths"] = meaningful_status
    if meaningful_status and not allow_dirty:
        raise RuntimeError(
            "Pinned LeJEPA checkout has source changes. Pass --allow-dirty-lejepa only for an explicitly recorded audit run."
        )
    return metadata


class ProjectorMLP(nn.Module):
    """The fixed 512 -> 2048 -> 2048 -> 64 Experiment 5 projector."""

    def __init__(self, dims: Sequence[int], biases: Sequence[bool] = (False, False, True)):
        super().__init__()
        if len(dims) != 4:
            raise ValueError(f"Expected four projector dimensions, got {dims}")
        if len(biases) != 3:
            raise ValueError(f"Expected three projector bias flags, got {biases}")
        in_dim, hidden1, hidden2, out_dim = map(int, dims)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1, bias=bool(biases[0])),
            nn.BatchNorm1d(hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2, bias=bool(biases[1])),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
            nn.Linear(hidden2, out_dim, bias=bool(biases[2])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone() -> nn.Module:
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    return model


def build_model_pair(config: Mapping[str, Any], initialization_seed: int, device: torch.device):
    """Build exactly paired initial states for each independently launched arm."""

    with deterministic_rng(initialization_seed):
        backbone = build_backbone()
        projector = ProjectorMLP(
            config["model"]["projector_dims"], config["model"]["projector_linear_biases"]
        )
    return backbone.to(device), projector.to(device)


def all_view_alignment(projections: torch.Tensor) -> torch.Tensor:
    """Mean squared view-to-center alignment with gradients through the center.

    Args:
        projections: float tensor shaped [views, batch, dimension].
    """

    if projections.ndim != 3 or projections.shape[0] < 2:
        raise ValueError(f"Expected [V,B,D] with V>=2, got {tuple(projections.shape)}")
    center = projections.mean(dim=0)
    return (projections - center.unsqueeze(0)).square().mean()


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("off_diagonal requires a square matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_variance_covariance(
    projections: torch.Tensor,
    variance_target: float = 1.0,
    variance_epsilon: float = 1e-4,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Experiment 5 arm C regularizer, averaged over views.

    Covariance uses the sample denominator B-1 and its squared off-diagonal
    entries are summed then divided by d, exactly as preregistered in the plan.
    """

    if projections.ndim != 3:
        raise ValueError(f"Expected [V,B,D], got {tuple(projections.shape)}")
    _, batch_size, dimension = projections.shape
    if batch_size < 2:
        raise ValueError("VICReg covariance requires at least two independent images.")
    float_z = projections.float()
    variance_terms = []
    covariance_terms = []
    for view in float_z:
        centered = view - view.mean(dim=0, keepdim=True)
        variance = centered.square().sum(dim=0) / (batch_size - 1)
        variance_term = torch.relu(variance_target - torch.sqrt(variance + variance_epsilon)).mean()
        covariance = centered.T @ centered / (batch_size - 1)
        covariance_term = off_diagonal(covariance).square().sum() / dimension
        variance_terms.append(variance_term)
        covariance_terms.append(covariance_term)
    variance_mean = torch.stack(variance_terms).mean()
    covariance_mean = torch.stack(covariance_terms).mean()
    total = variance_weight * variance_mean + covariance_weight * covariance_mean
    return total, {"variance": variance_mean, "covariance": covariance_mean}


def soft_orthogonality(matrix: torch.Tensor) -> torch.Tensor:
    """Dimension-normalized SO on the smaller Gram matrix."""

    if matrix.ndim < 2:
        raise ValueError("Soft orthogonality requires a matrix or convolution tensor.")
    a = matrix.float().reshape(matrix.shape[0], -1)
    if a.shape[0] <= a.shape[1]:
        gram = a @ a.T
    else:
        gram = a.T @ a
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - identity).square().sum() / gram.shape[0]


def model_soft_orthogonality(modules: Sequence[nn.Module]) -> torch.Tensor:
    penalties = []
    for module in modules:
        for child in module.modules():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                penalties.append(soft_orthogonality(child.weight))
    if not penalties:
        raise ValueError("No Conv2d/Linear matrices found for soft orthogonality.")
    return torch.stack(penalties).mean()


def build_sigreg(config: Mapping[str, Any], device: torch.device, direction_seed: int) -> nn.Module:
    EppsPulley, SlicingUnivariateTest = load_local_lejepa_components()
    sig_config = config["sigreg"]
    univariate = EppsPulley(
        t_max=float(sig_config["integration_bound"]),
        n_points=int(sig_config["integration_points"]),
        integration=str(sig_config["integration"]),
    )
    sigreg = SlicingUnivariateTest(
        univariate_test=univariate,
        num_slices=int(sig_config["num_slices"]),
        reduction=str(sig_config["slice_reduction"]),
    ).to(device)
    sigreg.global_step.fill_(int(direction_seed) % (2**31 - 1))
    return sigreg


def sigreg_over_views(sigreg: nn.Module, projections: torch.Tensor) -> torch.Tensor:
    if projections.dtype != torch.float32:
        projections = projections.float()
    return torch.stack([sigreg(view) for view in projections]).mean()


def objective(
    arm: str,
    projections: torch.Tensor,
    config: Mapping[str, Any],
    sigreg: nn.Module | None = None,
    coefficient: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    arm = arm.upper()
    if arm not in ARMS:
        raise ValueError(f"Unknown Experiment 5 arm {arm!r}; choose from {sorted(ARMS)}")
    alignment = all_view_alignment(projections.float())
    align_coefficient = float(config["training"]["alignment_coefficient"])
    zero = alignment.new_zeros(())
    sigreg_value = zero
    variance = zero
    covariance = zero
    regularizer = zero
    if arm == "B":
        if sigreg is None:
            raise ValueError("Arm B requires the pinned SIGReg module.")
        regularizer = sigreg_over_views(sigreg, projections)
        sigreg_value = regularizer
        coefficient = float(config["training"]["sigreg_coefficient"] if coefficient is None else coefficient)
    elif arm == "C":
        regularizer, components = vicreg_variance_covariance(
            projections,
            variance_target=float(config["vicreg_comparator"]["variance_target"]),
            variance_epsilon=float(config["vicreg_comparator"]["variance_epsilon"]),
            variance_weight=float(config["vicreg_comparator"]["variance_weight"]),
            covariance_weight=float(config["vicreg_comparator"]["covariance_weight"]),
        )
        variance = components["variance"]
        covariance = components["covariance"]
        configured = config["training"]["vc_coefficient"]
        if coefficient is None and configured is None:
            raise ValueError("Arm C coefficient is unset. Run coefficient calibration, then pass --coefficient.")
        coefficient = float(configured if coefficient is None else coefficient)
    else:
        coefficient = 0.0
    total = align_coefficient * alignment + float(coefficient) * regularizer
    return total, {
        "alignment": alignment,
        "regularizer": regularizer,
        "sigreg": sigreg_value,
        "vc_variance": variance,
        "vc_covariance": covariance,
        "coefficient": alignment.new_tensor(float(coefficient)),
    }


def optimizer_parameter_groups(
    modules: Sequence[nn.Module], matrix_weight_decay: float
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Decay only Conv/Linear weight matrices; never biases or normalization."""

    decay_ids: set[int] = set()
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    prefixes: dict[int, str] = {}
    for index, root in enumerate(modules):
        root_prefix = "backbone" if index == 0 else f"module_{index}"
        prefixes[id(root)] = root_prefix
        for module_name, child in root.named_modules():
            if isinstance(child, (nn.Conv2d, nn.Linear)) and child.weight is not None:
                decay_ids.add(id(child.weight))

    decay_params = []
    no_decay_params = []
    seen: set[int] = set()
    for index, root in enumerate(modules):
        root_prefix = "backbone" if index == 0 else ("projector" if index == 1 else f"module_{index}")
        for name, parameter in root.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            qualified = f"{root_prefix}.{name}"
            if id(parameter) in decay_ids:
                decay_params.append(parameter)
                decay_names.append(qualified)
            else:
                no_decay_params.append(parameter)
                no_decay_names.append(qualified)
    groups = [
        {"params": decay_params, "weight_decay": float(matrix_weight_decay), "group_name": "matrix_decay"},
        {"params": no_decay_params, "weight_decay": 0.0, "group_name": "bias_norm_no_decay"},
    ]
    return groups, {"matrix_decay": decay_names, "bias_norm_no_decay": no_decay_names}


def learning_rate_for_update(update: int, config: Mapping[str, Any]) -> float:
    """Learning rate used by one-indexed optimizer update ``update``."""

    training = config["training"]
    update = int(update)
    total = int(training["total_updates"])
    warmup = int(training["warmup_updates"])
    base = float(training["learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    if not 1 <= update <= total:
        raise ValueError(f"update must be in [1,{total}], got {update}")
    if update <= warmup:
        return base * update / warmup
    progress = (update - warmup) / max(total - warmup, 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def autocast_context(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.replace("\n", " ")}
    for name in ["torch", "torchvision", "numpy", "pandas", "PIL"]:
        try:
            module = __import__(name)
            result[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            result[name] = None
    result["cuda_runtime"] = torch.version.cuda
    result["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
    return result


def atomic_json_dump(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(row)) + "\n")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
