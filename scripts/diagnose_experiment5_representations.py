#!/usr/bin/env python
"""Extract stage/projector features and representation diagnostics for Experiment 5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import (
    DEFAULT_CONFIG,
    ProjectorMLP,
    atomic_json_dump,
    build_backbone,
    config_sha256,
    derive_seed,
    load_config,
    resolve_device,
    sha256_file,
)
from experiment5.data import (
    DeterministicMultiViewDataset,
    load_split,
    make_base_cifar10,
    make_eval_subset,
    make_ssl_transform,
)


LOCATIONS = [
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "backbone_y",
    "projector_linear1",
    "projector_bn1",
    "projector_hidden1",
    "projector_linear2",
    "projector_bn2",
    "projector_hidden2",
    "final_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split-file", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/protocol/cifar10_split_indices.npz")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    return parser.parse_args()


class FeatureCollector:
    def __init__(self, backbone, projector):
        self.current: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, module in [
            ("stage1", backbone.layer1),
            ("stage2", backbone.layer2),
            ("stage3", backbone.layer3),
            ("stage4", backbone.layer4),
            ("projector_linear1", projector.net[0]),
            ("projector_bn1", projector.net[1]),
            ("projector_hidden1", projector.net[2]),
            ("projector_linear2", projector.net[3]),
            ("projector_bn2", projector.net[4]),
            ("projector_hidden2", projector.net[5]),
        ]:
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def store(_module, _inputs, output):
            if output.ndim == 4:
                output = F.adaptive_avg_pool2d(output, 1).flatten(1)
            self.current[name] = output.detach()

        return store

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def extract_features(backbone, projector, loader, device: torch.device, multi_view: bool):
    outputs = {location: [] for location in LOCATIONS}
    source_indices = []
    collector = FeatureCollector(backbone, projector)
    try:
        for batch in loader:
            images, _labels, indices = batch
            if multi_view:
                batch_size, views = images.shape[:2]
                flat = images.flatten(0, 1).to(device, non_blocking=True)
            else:
                batch_size, views = images.shape[0], 1
                flat = images.to(device, non_blocking=True)
            collector.current.clear()
            backbone_y = backbone(flat)
            final_z = projector(backbone_y)
            current = {
                **collector.current,
                "backbone_y": backbone_y.detach(),
                "final_z": final_z.detach(),
            }
            for location in LOCATIONS:
                value = current[location].float().cpu()
                if multi_view:
                    value = value.view(batch_size, views, -1).transpose(0, 1).contiguous()
                outputs[location].append(value)
            source_indices.append(indices.cpu())
    finally:
        collector.close()
    if multi_view:
        merged = {location: torch.cat(parts, dim=1) for location, parts in outputs.items()}
    else:
        merged = {location: torch.cat(parts, dim=0) for location, parts in outputs.items()}
    return merged, torch.cat(source_indices).numpy()


def entropy_effective_rank(weights: torch.Tensor) -> float:
    weights = weights.double().clamp_min(0)
    total = weights.sum()
    if float(total) <= 0:
        return 0.0
    probabilities = weights / total
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def representation_metrics(
    features: torch.Tensor, analysis_device: torch.device
) -> tuple[dict[str, float], np.ndarray]:
    values = features.to(analysis_device).double()
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    covariance_rank = entropy_effective_rank(eigenvalues)
    singular_values = torch.linalg.svdvals(values)
    rankme = entropy_effective_rank(singular_values)
    coordinate_std = torch.sqrt(torch.diag(covariance).clamp_min(0))
    dimension = values.shape[1]
    return (
        {
            "samples": int(values.shape[0]),
            "dimension": int(dimension),
            "total_variance": float(torch.trace(covariance)),
            "mean_coordinate_std": float(coordinate_std.mean()),
            "minimum_coordinate_std": float(coordinate_std.min()),
            "maximum_coordinate_std": float(coordinate_std.max()),
            "covariance_energy_effective_rank": covariance_rank,
            "normalized_covariance_energy_effective_rank": covariance_rank / max(dimension, 1),
            "rankme": rankme,
            "normalized_rankme": rankme / max(min(values.shape), 1),
            "mean_l2_norm": float(torch.linalg.vector_norm(values, dim=1).mean()),
        },
        eigenvalues.cpu().numpy(),
    )


def lidar_metrics(
    multi_view_features: torch.Tensor,
    delta: float,
    epsilon: float,
    analysis_device: torch.device,
):
    """LiDAR equations (1)-(2), using images as surrogate classes and views as samples."""

    values = multi_view_features.transpose(0, 1).to(analysis_device).double()  # [images, views, dimension]
    n_images, n_views, dimension = values.shape
    class_means = values.mean(dim=1)
    global_mean = class_means.mean(dim=0, keepdim=True)
    between_centered = class_means - global_mean
    between = between_centered.T @ between_centered / max(n_images - 1, 1)
    within_centered = values - class_means[:, None, :]
    flat_within = within_centered.reshape(-1, dimension)
    within = flat_within.T @ flat_within / max(n_images * (n_views - 1), 1)
    within = within + delta * torch.eye(dimension, dtype=within.dtype, device=within.device)
    eigenvalues_w, eigenvectors_w = torch.linalg.eigh(within)
    inverse_sqrt = eigenvectors_w @ torch.diag(eigenvalues_w.clamp_min(epsilon).rsqrt()) @ eigenvectors_w.T
    lidar_matrix = inverse_sqrt @ between @ inverse_sqrt
    lidar_matrix = 0.5 * (lidar_matrix + lidar_matrix.T)
    eigenvalues = torch.linalg.eigvalsh(lidar_matrix).clamp_min(0).flip(0)
    lidar_rank = entropy_effective_rank(eigenvalues + epsilon)
    return {
        "lidar": lidar_rank,
        "normalized_lidar": lidar_rank / max(min(dimension, n_images - 1), 1),
        "between_trace": float(torch.trace(between)),
        "within_trace_before_delta": float(torch.trace(within) - delta * dimension),
        "lidar_top_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) else 0.0,
        "surrogate_classes": int(n_images),
        "views_per_class": int(n_views),
        "delta": float(delta),
    }, eigenvalues.cpu().numpy()


def sliced_gaussian_discrepancy(
    values: torch.Tensor,
    slices: int,
    points: int,
    bound: float,
    seed: int,
    chunk_size: int = 256,
) -> float:
    values = values.float()
    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, bound, points)
    phi = torch.exp(-0.5 * t.square())
    dt = bound / (points - 1)
    weights = torch.full((points,), 2.0 * dt)
    weights[[0, -1]] = dt
    weights *= phi
    total = 0.0
    consumed = 0
    for start in range(0, slices, chunk_size):
        count = min(chunk_size, slices - start)
        directions = torch.randn((values.shape[1], count), generator=generator)
        directions /= directions.norm(dim=0, keepdim=True)
        projected = values @ directions
        arguments = projected.unsqueeze(-1) * t
        real = torch.cos(arguments).mean(dim=0)
        imaginary = torch.sin(arguments).mean(dim=0)
        per_slice = (((real - phi).square() + imaginary.square()) @ weights) * len(values)
        total += float(per_slice.sum())
        consumed += count
    return total / consumed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != config["experiment"] or checkpoint.get("config_sha256") != config_sha256(config):
        raise ValueError("Checkpoint and diagnostic protocol do not match.")
    split = load_split(args.split_file)
    diagnostic_indices = split["diagnostic_indices"]
    if len(diagnostic_indices) != int(config["representation_diagnostics"]["validation_images"]):
        raise ValueError("Diagnostic subset size does not match config.")
    base_dataset = make_base_cifar10(args.data_dir, train=True, download=False)
    eval_dataset = make_eval_subset(base_dataset, diagnostic_indices)
    augmentation_seed = int(config["representation_diagnostics"]["augmentation_seed"])
    multi_dataset = DeterministicMultiViewDataset(
        base_dataset,
        make_ssl_transform(config),
        int(config["representation_diagnostics"]["augmented_views"]),
        augmentation_seed,
    )
    multi_subset = Subset(multi_dataset, diagnostic_indices.tolist())
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(eval_dataset, **loader_kwargs)
    multi_loader = DataLoader(multi_subset, **loader_kwargs)
    backbone = build_backbone().to(device)
    projector = ProjectorMLP(
        config["model"]["projector_dims"], config["model"]["projector_linear_biases"]
    ).to(device)
    backbone.load_state_dict(checkpoint["backbone_state"], strict=True)
    projector.load_state_dict(checkpoint["projector_state"], strict=True)
    backbone.eval()
    projector.eval()
    eval_features, eval_ids = extract_features(backbone, projector, eval_loader, device, multi_view=False)
    view_features, view_ids = extract_features(backbone, projector, multi_loader, device, multi_view=True)
    if not np.array_equal(eval_ids, view_ids) or not np.array_equal(eval_ids, diagnostic_indices):
        raise RuntimeError("Feature extraction changed the fixed diagnostic image order.")

    diagnostic = config["representation_diagnostics"]
    metric_rows = []
    spectra: dict[str, np.ndarray] = {}
    for location in LOCATIONS:
        basic, covariance_spectrum = representation_metrics(eval_features[location], device)
        lidar, lidar_spectrum = lidar_metrics(
            view_features[location],
            delta=float(diagnostic["lidar_within_covariance_delta"]),
            epsilon=float(diagnostic["lidar_entropy_epsilon"]),
            analysis_device=device,
        )
        views = view_features[location]
        center = views.mean(dim=0)
        alignment = float((views - center.unsqueeze(0)).square().mean())
        active_fraction = float((eval_features[location] != 0).float().mean())
        positive_fraction = float((eval_features[location] > 0).float().mean())
        metric_rows.append(
            {
                "experiment": checkpoint["experiment"],
                "arm": checkpoint["arm"],
                "seed": checkpoint["seed"],
                "update": checkpoint["update"],
                "location": location,
                **basic,
                **lidar,
                "paired_view_alignment_mse": alignment,
                "active_unit_fraction": active_fraction,
                "positive_unit_fraction": positive_fraction,
            }
        )
        spectra[f"{location}_covariance_eigenvalues"] = covariance_spectrum
        spectra[f"{location}_lidar_eigenvalues"] = lidar_spectrum

    z = eval_features["final_z"]
    gaussian_generator = torch.Generator().manual_seed(int(diagnostic["gaussian_simulation_seed"]))
    gaussian = torch.randn(z.shape, generator=gaussian_generator)
    z_discrepancy = sliced_gaussian_discrepancy(
        z,
        int(diagnostic["gaussian_test_slices"]),
        int(diagnostic["gaussian_test_points"]),
        float(diagnostic["gaussian_test_bound"]),
        int(diagnostic["gaussian_test_direction_seed"]),
    )
    gaussian_discrepancy = sliced_gaussian_discrepancy(
        gaussian,
        int(diagnostic["gaussian_test_slices"]),
        int(diagnostic["gaussian_test_points"]),
        float(diagnostic["gaussian_test_bound"]),
        int(diagnostic["gaussian_test_direction_seed"]),
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent.parent / "diagnostics" / checkpoint_path.stem / "representations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "representation_metrics.csv", metric_rows)
    np.savez_compressed(output_dir / "representation_spectra.npz", **spectra)
    summary = {
        "experiment": checkpoint["experiment"],
        "arm": checkpoint["arm"],
        "seed": checkpoint["seed"],
        "update": checkpoint["update"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_file_sha256": sha256_file(args.split_file),
        "diagnostic_image_ids_sha256": __import__("hashlib").sha256(
            np.ascontiguousarray(eval_ids, dtype=np.int64).tobytes()
        ).hexdigest(),
        "locations": LOCATIONS,
        "final_z_sliced_gaussian_discrepancy": z_discrepancy,
        "matched_size_gaussian_discrepancy": gaussian_discrepancy,
        "lidar_convention": {
            "source": "Thilak et al., ICLR 2024, equations (1)-(2)",
            "url": "https://arxiv.org/abs/2312.04000",
            "surrogate_classes": "the fixed validation image IDs",
            "samples_per_class": int(diagnostic["augmented_views"]),
            "within_covariance": "unbiased denominator n_images*(views-1)",
            "between_covariance": "unbiased denominator n_images-1",
            "delta": float(diagnostic["lidar_within_covariance_delta"]),
            "primary_source": "https://arxiv.org/abs/2312.04000",
        },
        "notes": [
            "CNN stage outputs are global-average pooled per image.",
            "Projector linear, post-BN, and post-GELU tensors are recorded separately for pre/post-normalization checks.",
            "RankMe uses normalized singular values; covariance effective rank uses covariance eigenvalues.",
            "The Gaussian discrepancy uses directions independent from training and matched-size Gaussian simulation context.",
            "A low finite-slice discrepancy is not proof of multivariate Gaussianity.",
        ],
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
