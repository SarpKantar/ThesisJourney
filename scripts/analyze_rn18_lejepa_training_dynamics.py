#!/usr/bin/env python
"""Analyze LeJEPA ResNet18 filter dynamics across saved SSL checkpoints.

The analysis projects every checkpoint into one externally fitted, full CNN
Filter DB PCA basis. It never fits or updates a PCA basis locally. Histogram
edges are fixed once from the union of all checkpoint projections so drift
values remain comparable across training time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import analyze_rn18_cifar10_kernel_distributions as kernel_analysis
import rn18_cifar10_common as common


EXPECTED_EPOCHS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
EXPECTED_FULL_DB_FILTERS = 1_464_797_156
EXPECTED_PREPROCESSING_ID = "cnn_filter_db_paper_float16_maxabs_v1"
CHECKPOINT_RE = re.compile(r"lejepa_ssl_epoch(\d{3})\.pth$")
STAGE_ORDER = ("layer1", "layer2", "layer3", "layer4")
DEFAULT_EXPERIMENT_DIR = (
    "outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong"
)
DEFAULT_PCA_BASIS = (
    "outputs/(4thEXP)rn18_cifar10_filter_dynamics/"
    "original_cnn_filter_db_pca/cnn_filter_db_full_pca.npz"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/(4thEXP)rn18_cifar10_filter_dynamics/lejepa_training_dynamics"
)

DESCRIPTORS = (
    "raw_l2",
    "bn_folded_l2",
    "normalized_dc_gain",
    "center_surround",
    "high_frequency_ratio",
    "orientation_bias",
    "orientation_anisotropy",
    "center_energy_ratio",
    "roughness",
)
WEIGHT_REPRESENTATIONS = (
    "raw_weight",
    "bn_folded_weight",
    "shape_normalized_weight",
)
DESCRIPTOR_LABELS = {
    **kernel_analysis.DESCRIPTOR_LABELS,
    "raw_weight": "Raw coefficient",
    "bn_folded_weight": "BN-folded coefficient",
    "shape_normalized_weight": "Shape-normalized coefficient",
    "bn_abs_scale": "Absolute BN scale",
}
PLOT_COLORS = [
    (0, 114, 178),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (230, 159, 0),
    (86, 180, 233),
    (213, 94, 0),
    (0, 0, 0),
    (117, 112, 179),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze all canonical LeJEPA SSL checkpoints in a fixed, external "
            "full-CNN-Filter-DB PCA basis."
        )
    )
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--pca-basis",
        default=DEFAULT_PCA_BASIS,
        help=(
            "External full CNN Filter DB PCA NPZ. The file must contain strict "
            "full-dataset provenance; this script never fits PCA."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=70)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--expected-epochs",
        default=",".join(str(epoch) for epoch in EXPECTED_EPOCHS),
        help="Exact required checkpoint epochs. Extra and missing checkpoints fail validation.",
    )
    parser.add_argument(
        "--milestone-epochs",
        default="1,30,60,90,120",
        help="Saved epochs included in ridge and c0/c1 milestone figures.",
    )
    parser.add_argument(
        "--include-stem",
        action="store_true",
        help="Include CIFAR conv1. The default excludes it to match paper-aligned Exp3 analysis.",
    )
    parser.add_argument("--max-density-points", type=int, default=250_000)
    parser.add_argument(
        "--descriptor-max-samples",
        type=int,
        default=200_000,
        help="Maximum deterministic sample used only for descriptor quantiles.",
    )
    return parser.parse_args()


def parse_epoch_list(text: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive epochs")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate epochs: {values}")
    return values


def scalar_from_npz(data: Any, key: str) -> Any:
    if key not in data.files:
        raise ValueError(f"PCA artifact is missing required provenance key: {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"PCA artifact key {key!r} must be scalar, got shape {value.shape}")
    return value.reshape(()).item()


def json_value(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, np.generic):
            item = item.item()
        return item
    if array.size <= 100:
        return array.tolist()
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def load_reference_pca(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing external full-DB PCA artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        required_arrays = (
            "components",
            "mean",
            "explained_variance",
            "explained_variance_ratio",
            "covariance",
        )
        missing = [key for key in required_arrays if key not in data.files]
        if missing:
            raise ValueError(f"PCA artifact is missing arrays: {missing}")
        components = np.asarray(data["components"], dtype=np.float64)
        mean = np.asarray(data["mean"], dtype=np.float64)
        explained_variance = np.asarray(data["explained_variance"], dtype=np.float64)
        ratio = np.asarray(data["explained_variance_ratio"], dtype=np.float64)
        covariance = np.asarray(data["covariance"], dtype=np.float64)
        n_samples = int(scalar_from_npz(data, "n_samples"))
        full_count = int(scalar_from_npz(data, "full_dataset_filter_count"))
        n_features = int(scalar_from_npz(data, "n_features"))
        is_full = bool(scalar_from_npz(data, "is_full_dataset_fit"))
        start = int(scalar_from_npz(data, "filter_index_start"))
        stop = int(scalar_from_npz(data, "filter_index_stop_exclusive"))
        preprocessing_id = str(scalar_from_npz(data, "preprocessing_id"))
        mean_validation = bool(scalar_from_npz(data, "published_mean_validation_passed"))
        dataset_shape = tuple(
            int(value) for value in np.asarray(data["dataset_filters_shape"]).reshape(-1)
        )
        source_doi = str(scalar_from_npz(data, "source_dataset_doi"))
        source_md5 = str(scalar_from_npz(data, "source_compressed_md5"))
        metadata = {
            key: json_value(data[key])
            for key in data.files
            if key not in {"components", "mean", "explained_variance", "explained_variance_ratio", "singular_values", "covariance"}
        }

    if (
        components.shape != (9, 9)
        or mean.shape != (9,)
        or explained_variance.shape != (9,)
        or ratio.shape != (9,)
        or covariance.shape != (9, 9)
    ):
        raise ValueError(
            "External PCA must have components=(9,9), covariance=(9,9), and "
            "9-element mean/variance/ratio arrays; got "
            f"{components.shape}, {covariance.shape}, {mean.shape}, "
            f"{explained_variance.shape}, {ratio.shape}"
        )
    for name, array in (
        ("components", components),
        ("mean", mean),
        ("explained_variance", explained_variance),
        ("ratio", ratio),
        ("covariance", covariance),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"External PCA {name} contains non-finite values")
    if n_samples != EXPECTED_FULL_DB_FILTERS or full_count != EXPECTED_FULL_DB_FILTERS:
        raise ValueError(
            "PCA artifact is not the required full CNN Filter DB fit: "
            f"n_samples={n_samples}, full_dataset_filter_count={full_count}"
        )
    if not is_full or n_features != 9 or start != 0 or stop != EXPECTED_FULL_DB_FILTERS:
        raise ValueError(
            "PCA full-dataset provenance is inconsistent: "
            f"is_full={is_full}, n_features={n_features}, range=[{start},{stop})"
        )
    if preprocessing_id != EXPECTED_PREPROCESSING_ID:
        raise ValueError(
            f"Expected preprocessing_id={EXPECTED_PREPROCESSING_ID!r}, got {preprocessing_id!r}"
        )
    if dataset_shape != (EXPECTED_FULL_DB_FILTERS, 3, 3):
        raise ValueError(f"Unexpected source filter tensor shape: {dataset_shape}")
    if source_doi != "10.5281/zenodo.6371680" or len(source_md5) != 32:
        raise ValueError(
            "PCA source-dataset provenance is incomplete or unexpected: "
            f"doi={source_doi!r}, compressed_md5={source_md5!r}"
        )
    gram_error = float(np.max(np.abs(components @ components.T - np.eye(9))))
    if gram_error > 1e-6:
        raise ValueError(f"PCA components are not orthonormal; max error={gram_error:.3g}")
    if np.any(ratio < 0) or not np.isclose(float(ratio.sum()), 1.0, atol=1e-6):
        raise ValueError(f"Invalid explained variance ratios; sum={float(ratio.sum()):.12f}")
    covariance_symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    if covariance_symmetry_error > 1e-9:
        raise ValueError(
            "PCA covariance is not symmetric; max error="
            f"{covariance_symmetry_error:.3g}"
        )
    reconstructed = components.T @ np.diag(explained_variance) @ components
    covariance_reconstruction_error = float(np.max(np.abs(covariance - reconstructed)))
    if covariance_reconstruction_error > 1e-8:
        raise ValueError(
            "PCA covariance/eigendecomposition mismatch; max error="
            f"{covariance_reconstruction_error:.3g}"
        )
    warning = None
    if not mean_validation:
        maximum_error = float(metadata.get("published_mean_max_abs_error", math.nan))
        tolerance = float(metadata.get("published_mean_tolerance", math.nan))
        warning = (
            "PUBLISHED-MEAN DISCREPANCY: the independently accumulated mean of the "
            "released full CNN Filter DB does not match the mean vector printed in the "
            f"paper supplement (max abs error={maximum_error:.9g}, "
            f"artifact tolerance={tolerance:.9g}). The complete released-data fit is "
            "accepted because all full-range and numerical provenance checks pass; the "
            "discrepancy must remain visible in downstream interpretation."
        )
        print(f"[warning] {warning}", flush=True)
    metadata.update(
        {
            "validated_n_samples": n_samples,
            "validated_preprocessing_id": preprocessing_id,
            "consumer_orthonormality_max_abs_error": gram_error,
            "consumer_explained_variance_ratio_sum": float(ratio.sum()),
            "consumer_covariance_symmetry_max_abs_error": covariance_symmetry_error,
            "consumer_covariance_reconstruction_max_abs_error": covariance_reconstruction_error,
            "consumer_published_mean_warning": warning,
            "consumer_artifact_accepted": True,
        }
    )
    return {
        "components": components,
        "mean": mean,
        "explained_variance": explained_variance,
        "explained_variance_ratio": ratio.astype(np.float64),
    }, metadata


def discover_checkpoints(
    experiment_dir: Path,
    expected_epochs: tuple[int, ...],
) -> list[tuple[int, Path]]:
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    discovered: dict[int, Path] = {}
    unexpected_names = []
    for path in sorted(checkpoint_dir.glob("lejepa_ssl_epoch*.pth")):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            unexpected_names.append(path.name)
            continue
        epoch = int(match.group(1))
        if epoch in discovered:
            raise ValueError(f"Duplicate checkpoint epoch {epoch}: {discovered[epoch]} and {path}")
        discovered[epoch] = path
    if unexpected_names:
        raise ValueError(f"Unexpected LeJEPA checkpoint filenames: {unexpected_names}")
    actual = tuple(sorted(discovered))
    if actual != tuple(expected_epochs):
        missing = sorted(set(expected_epochs) - set(actual))
        extra = sorted(set(actual) - set(expected_epochs))
        raise ValueError(
            f"Checkpoint epoch set mismatch. expected={expected_epochs}, actual={actual}, "
            f"missing={missing}, extra={extra}"
        )
    return [(epoch, discovered[epoch]) for epoch in expected_epochs]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def reference_scale_filters(x: np.ndarray) -> np.ndarray:
    """Reproduce the full-DB artifact's float16-then-maxabs preprocessing."""
    x16 = x.astype(np.float16)
    denominator = np.abs(x16).max(axis=1)
    denominator = np.where(denominator == 0, np.float16(1.0), denominator)[:, None]
    return (x16 / denominator).astype(np.float32)


def fixed_edges(z_by_epoch: dict[int, np.ndarray], bins: int) -> list[np.ndarray]:
    edges = []
    for component in range(9):
        low = min(float(z[:, component].min()) for z in z_by_epoch.values())
        high = max(float(z[:, component].max()) for z in z_by_epoch.values())
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"Non-finite PCA range for component {component}")
        if low == high:
            low -= 0.5
            high += 0.5
        pad = (high - low) * 0.001
        edges.append(np.linspace(low - pad, high + pad, bins + 1, dtype=np.float64))
    return edges


def probabilities_for_z(z: np.ndarray, edges: Sequence[np.ndarray]) -> np.ndarray:
    return np.vstack(
        [common.probability_histogram(z[:, component], edges[component]) for component in range(9)]
    )


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    token = "|".join(parts).encode("utf-8")
    return np.random.default_rng(seed + zlib.crc32(token))


def sample_values(values: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if maximum <= 0 or flat.size <= maximum:
        return flat.astype(np.float64, copy=False)
    indices = rng.choice(flat.size, size=maximum, replace=False)
    return flat[indices].astype(np.float64, copy=False)


def summarize_values(
    values: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        raise ValueError("Cannot summarize an empty value array")
    sample = sample_values(flat, max_samples, rng)
    mean = float(np.mean(flat, dtype=np.float64))
    mean_square = float(np.mean(np.square(flat, dtype=np.float64), dtype=np.float64))
    variance = max(mean_square - mean * mean, 0.0)
    q05, median, q95 = np.quantile(sample, [0.05, 0.50, 0.95])
    return {
        "count": int(flat.size),
        "sample_count": int(sample.size),
        "mean": mean,
        "std": math.sqrt(variance),
        "rms": math.sqrt(max(mean_square, 0.0)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "minimum": float(np.min(flat)),
        "q05": float(q05),
        "median": float(median),
        "q95": float(q95),
        "maximum": float(np.max(flat)),
    }


def scope_definitions(layers: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = [
        {"scope_type": "global", "scope": "global", "layer_indices": list(range(len(layers)))}
    ]
    for stage in STAGE_ORDER:
        indices = [index for index, layer in enumerate(layers) if layer["stage"] == stage]
        if indices:
            scopes.append({"scope_type": "stage", "scope": stage, "layer_indices": indices})
    for index, layer in enumerate(layers):
        scopes.append(
            {
                "scope_type": "layer",
                "scope": layer["layer_key"],
                "stage": layer["stage"],
                "layer_order": layer["layer_order"],
                "conv_depth_norm": layer["conv_depth_norm"],
                "layer_indices": [index],
            }
        )
    return scopes


def concatenate_layer_field(
    layers: Sequence[dict[str, Any]],
    indices: Sequence[int],
    field: str,
    descriptor: str | None = None,
) -> np.ndarray:
    arrays = [
        layers[index]["descriptors"][descriptor]
        if descriptor is not None
        else layers[index][field]
        for index in indices
    ]
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


def build_layer_payloads(
    state: dict[str, torch.Tensor],
    slices: Sequence[common.LayerSlice],
) -> list[dict[str, Any]]:
    items = common.sorted_resnet18_filter_items(
        state,
        include_stem=any(layer_slice.stage == "stem" for layer_slice in slices),
    )
    if len(items) != len(slices):
        raise ValueError(f"Layer extraction mismatch: {len(items)} tensors vs {len(slices)} slices")
    layers = []
    for layer_slice, (layer_key, tensor) in zip(slices, items):
        if layer_key != layer_slice.layer_key:
            raise ValueError(f"Layer ordering mismatch: {layer_key} vs {layer_slice.layer_key}")
        tensor_np = common.tensor_to_numpy(tensor)
        raw = tensor_np.reshape(-1, 9).astype(np.float32, copy=False)
        factor = kernel_analysis.batchnorm_factor(state, layer_key, int(tensor_np.shape[0]))
        folded_tensor = tensor_np * factor[:, None, None, None]
        folded = folded_tensor.reshape(-1, 9).astype(np.float32, copy=False)
        normalized = common.scale_filters(raw).astype(np.float32, copy=False)
        layers.append(
            {
                "layer_key": layer_key,
                "stage": layer_slice.stage,
                "layer_order": layer_slice.layer_order,
                "conv_depth_norm": layer_slice.conv_depth_norm,
                "raw": raw.copy(),
                "bn_folded": folded.copy(),
                "shape_normalized": normalized.copy(),
                "bn_factor": factor.copy(),
                "descriptors": kernel_analysis.kernel_descriptors(raw, folded, normalized),
            }
        )
    return layers


def descriptor_rows_for_epoch(
    epoch: int,
    layers: Sequence[dict[str, Any]],
    max_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    scopes = scope_definitions(layers)
    for scope in scopes:
        prefix = {
            "epoch": epoch,
            **{key: value for key, value in scope.items() if key != "layer_indices"},
        }
        indices = scope["layer_indices"]
        representation_fields = {
            "raw_weight": "raw",
            "bn_folded_weight": "bn_folded",
            "shape_normalized_weight": "shape_normalized",
        }
        for metric, field in representation_fields.items():
            values = concatenate_layer_field(layers, indices, field).reshape(-1)
            rows.append(
                {
                    **prefix,
                    "category": "weight_representation",
                    "metric": metric,
                    **summarize_values(
                        values,
                        max_samples,
                        stable_rng(seed, str(epoch), scope["scope_type"], scope["scope"], metric),
                    ),
                }
            )
        for descriptor in DESCRIPTORS:
            values = concatenate_layer_field(layers, indices, "descriptors", descriptor=descriptor)
            rows.append(
                {
                    **prefix,
                    "category": "kernel_descriptor",
                    "metric": descriptor,
                    **summarize_values(
                        values,
                        max_samples,
                        stable_rng(seed, str(epoch), scope["scope_type"], scope["scope"], descriptor),
                    ),
                }
            )
        bn_values = np.concatenate([layers[index]["bn_factor"] for index in indices])
        rows.append(
            {
                **prefix,
                "category": "batchnorm",
                "metric": "bn_abs_scale",
                **summarize_values(
                    np.abs(bn_values),
                    max_samples,
                    stable_rng(seed, str(epoch), scope["scope_type"], scope["scope"], "bn_abs_scale"),
                ),
            }
        )

        if scope["scope_type"] not in {"global", "stage"}:
            continue
        for representation, field in (
            ("raw", "raw"),
            ("bn_folded", "bn_folded"),
            ("shape_normalized", "shape_normalized"),
        ):
            kernels = concatenate_layer_field(layers, indices, field)
            energy = np.square(kernels, dtype=np.float64).mean(axis=0)
            energy /= max(float(energy.sum()), 1e-12)
            for position, value in enumerate(energy):
                energy_rows.append(
                    {
                        **prefix,
                        "representation": representation,
                        "row": position // 3,
                        "column": position % 3,
                        "position": position,
                        "normalized_energy": float(value),
                    }
                )
    return rows, energy_rows


def draw_rectangular_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    path: Path,
    value_format: str = ".3f",
) -> None:
    rows, columns = matrix.shape
    cell_w = max(56, min(104, 900 // max(columns, 1)))
    cell_h = 38
    left, top, right, bottom = 128, 104, 34, 190
    width = left + columns * cell_w + right
    height = top + rows * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(23, bold=True)
    label_font = common.get_font(11)
    value_font = common.get_font(9, bold=True)
    draw.text((22, 20), title, font=title_font, fill=(20, 20, 20))
    finite = matrix[np.isfinite(matrix)]
    maximum = max(float(finite.max()) if finite.size else 0.0, 1e-12)
    for row_index, label in enumerate(row_labels):
        y0 = top + row_index * cell_h
        draw.text((12, y0 + 11), str(label), font=label_font, fill=(20, 20, 20))
        for column_index in range(columns):
            value = float(matrix[row_index, column_index])
            x0 = left + column_index * cell_w
            if np.isfinite(value):
                color = common.heat_color(value / maximum)
                text_value = format(value, value_format)
            else:
                color = (235, 235, 235)
                text_value = "NA"
            draw.rectangle(
                [x0, y0, x0 + cell_w, y0 + cell_h],
                fill=color,
                outline=(225, 225, 225),
            )
            text_color = (255, 255, 255) if common.luminance(color) < 95 else (20, 20, 20)
            common.draw_centered(
                draw,
                (x0 + cell_w / 2, y0 + cell_h / 2),
                text_value,
                value_font,
                text_color,
            )
    for column_index, label in enumerate(column_labels):
        x0 = left + column_index * cell_w
        text_w, text_h = common.text_size(draw, str(label), label_font)
        label_image = Image.new("RGBA", (text_w + 8, text_h + 8), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((4, 4), str(label), font=label_font, fill=(20, 20, 20))
        label_image = label_image.rotate(90, expand=True)
        image.paste(
            label_image,
            (
                int(x0 + cell_w / 2 - label_image.width / 2),
                top + rows * cell_h + 12,
            ),
            label_image,
        )
    image.save(path)


def draw_panel_grid(
    panels: Sequence[dict[str, Any]],
    path: Path,
    title: str,
    columns: int = 2,
) -> None:
    rows = math.ceil(len(panels) / columns)
    cell_w, cell_h = 510, 330
    left, top, right, bottom = 72, 92, 24, 58
    width = left + columns * cell_w + right
    height = top + rows * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(13, bold=True)
    small_font = common.get_font(10)
    draw.text((24, 22), title, font=title_font, fill=(20, 20, 20))
    for panel_index, panel in enumerate(panels):
        row_index, column_index = divmod(panel_index, columns)
        x0 = left + column_index * cell_w
        y0 = top + row_index * cell_h
        plot_w, plot_h = cell_w - 82, cell_h - 90
        series = panel["series"]
        all_x = [float(value) for item in series for value in item["x"]]
        all_y = [
            float(value)
            for item in series
            for value in item["y"]
            if np.isfinite(float(value))
        ]
        if not all_x or not all_y:
            continue
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        if x_min == x_max:
            x_max = x_min + 1.0
        if panel.get("zero_floor", False):
            y_min = min(0.0, y_min)
        padding = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
        y_min -= padding
        y_max += padding
        if y_min == y_max:
            y_max = y_min + 1.0

        def x_map(value: float) -> float:
            return x0 + (value - x_min) / (x_max - x_min) * plot_w

        def y_map(value: float) -> float:
            return y0 + (y_max - value) / (y_max - y_min) * plot_h

        common.draw_centered(draw, (x0 + plot_w / 2, y0 - 19), panel["title"], label_font)
        draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(65, 65, 65))
        for grid_index in range(5):
            grid_y = y0 + grid_index / 4 * plot_h
            draw.line([(x0, grid_y), (x0 + plot_w, grid_y)], fill=(235, 235, 235), width=1)
            grid_value = y_max - grid_index / 4 * (y_max - y_min)
            draw.text((x0 - 63, grid_y - 6), f"{grid_value:.3g}", font=small_font, fill=(70, 70, 70))
        for item_index, item in enumerate(series):
            color = item.get("color", PLOT_COLORS[item_index % len(PLOT_COLORS)])
            points = [
                (x_map(float(x)), y_map(float(y)))
                for x, y in zip(item["x"], item["y"])
                if np.isfinite(float(y))
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x_value, y_value in points:
                draw.ellipse(
                    [x_value - 2.5, y_value - 2.5, x_value + 2.5, y_value + 2.5],
                    fill=color,
                )
        draw.text((x0, y0 + plot_h + 8), f"{x_min:g}", font=small_font, fill=(70, 70, 70))
        right_label = f"{x_max:g}"
        right_width, _ = common.text_size(draw, right_label, small_font)
        draw.text(
            (x0 + plot_w - right_width, y0 + plot_h + 8),
            right_label,
            font=small_font,
            fill=(70, 70, 70),
        )
        legend_x = x0
        legend_y = y0 + plot_h + 35
        for item_index, item in enumerate(series):
            color = item.get("color", PLOT_COLORS[item_index % len(PLOT_COLORS)])
            draw.line([(legend_x, legend_y), (legend_x + 20, legend_y)], fill=color, width=3)
            draw.text((legend_x + 25, legend_y - 6), item["label"], font=small_font, fill=(30, 30, 30))
            legend_x += 28 + common.text_size(draw, item["label"], small_font)[0] + 18
    image.save(path)


def draw_milestone_ridges(
    milestone_epochs: Sequence[int],
    probabilities: dict[int, np.ndarray],
    path: Path,
) -> None:
    rows, columns = len(milestone_epochs), 9
    cell_w, cell_h = 132, 92
    left, top, right, bottom = 122, 92, 26, 42
    width = left + columns * cell_w + right
    height = top + rows * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(13, bold=True)
    draw.text((24, 22), "LeJEPA Milestone PCA Ridges (Fixed Full-DB Basis and Edges)", font=title_font, fill=(20, 20, 20))
    for component in range(columns):
        common.draw_centered(draw, (left + component * cell_w + cell_w / 2, top - 22), f"c{component}", label_font)
    for row_index, epoch in enumerate(milestone_epochs):
        y0 = top + row_index * cell_h
        draw.text((18, y0 + 38), f"epoch {epoch}", font=label_font, fill=(20, 20, 20))
        color = PLOT_COLORS[row_index % len(PLOT_COLORS)]
        for component in range(columns):
            x0 = left + component * cell_w
            values = common.smooth_histogram(probabilities[epoch][component], passes=2)
            values /= max(float(values.max()), 1e-12)
            baseline = y0 + cell_h - 14
            points = [(x0 + 4, baseline)]
            for index, value in enumerate(values):
                x_value = x0 + 4 + index / max(len(values) - 1, 1) * (cell_w - 8)
                y_value = baseline - float(value) * (cell_h - 23)
                points.append((x_value, y_value))
            points.append((x0 + cell_w - 4, baseline))
            draw.polygon(points, fill=color)
            draw.line(points[1:-1], fill=(245, 245, 245), width=1)
            draw.line([(x0 + 4, baseline), (x0 + cell_w - 4, baseline)], fill=(70, 70, 70))
    image.save(path)


def draw_c0_c1_densities(
    milestone_epochs: Sequence[int],
    z_by_epoch: dict[int, np.ndarray],
    edges: Sequence[np.ndarray],
    max_points: int,
    seed: int,
    path: Path,
) -> None:
    columns = min(3, len(milestone_epochs))
    rows = math.ceil(len(milestone_epochs) / columns)
    panel = 310
    gap = 58
    left, top, right, bottom = 68, 102, 32, 52
    width = left + columns * panel + max(columns - 1, 0) * gap + right
    height = top + rows * panel + max(rows - 1, 0) * gap + bottom
    matrices = []
    for epoch in milestone_epochs:
        z = z_by_epoch[epoch]
        if max_points > 0 and len(z) > max_points:
            indices = stable_rng(seed, "density", str(epoch)).choice(len(z), size=max_points, replace=False)
            z = z[indices]
        histogram, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=(edges[0], edges[1]))
        matrices.append(np.log1p(histogram.T))
    maximum = max(float(matrix.max()) for matrix in matrices)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(13, bold=True)
    draw.text((24, 22), "LeJEPA c0/c1 Density Through Training", font=title_font, fill=(20, 20, 20))
    draw.text((24, 56), "Shared full-DB PCA axes, fixed union edges, shared log-density color scale", font=common.get_font(12), fill=(80, 80, 80))
    for panel_index, (epoch, matrix) in enumerate(zip(milestone_epochs, matrices)):
        row_index, column_index = divmod(panel_index, columns)
        x0 = left + column_index * (panel + gap)
        y0 = top + row_index * (panel + gap)
        common.draw_centered(draw, (x0 + panel / 2, y0 - 20), f"epoch {epoch}", label_font)
        bins_y, bins_x = matrix.shape
        for y_index in range(bins_y):
            for x_index in range(bins_x):
                value = float(matrix[y_index, x_index]) / max(maximum, 1e-12)
                color = common.density_color(value)
                bx0 = x0 + x_index / bins_x * panel
                bx1 = x0 + (x_index + 1) / bins_x * panel
                by0 = y0 + (bins_y - 1 - y_index) / bins_y * panel
                by1 = y0 + (bins_y - y_index) / bins_y * panel
                draw.rectangle([bx0, by0, bx1 + 1, by1 + 1], fill=color)
        draw.rectangle([x0, y0, x0 + panel, y0 + panel], outline=(50, 50, 50), width=2)
        common.draw_centered(draw, (x0 + panel / 2, y0 + panel + 20), "c0", label_font)
        draw.text((x0 - 32, y0 + panel / 2 - 7), "c1", font=label_font, fill=(20, 20, 20))
    image.save(path)


def draw_initial_final_residuals(
    initial_epoch: int,
    final_epoch: int,
    probabilities: dict[int, np.ndarray],
    edges: Sequence[np.ndarray],
    path: Path,
) -> None:
    columns, rows = 3, 3
    cell_w, cell_h = 410, 265
    left, top, right, bottom = 68, 98, 24, 42
    width = left + columns * cell_w + right
    height = top + rows * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(13, bold=True)
    small_font = common.get_font(10)
    draw.text((24, 22), f"PCA Histogram Residual: Epoch {final_epoch} - Epoch {initial_epoch}", font=title_font, fill=(20, 20, 20))
    for component in range(9):
        row_index, column_index = divmod(component, columns)
        x0 = left + column_index * cell_w
        y0 = top + row_index * cell_h
        plot_w, plot_h = cell_w - 58, cell_h - 52
        residual = probabilities[final_epoch][component] - probabilities[initial_epoch][component]
        max_abs = max(float(np.abs(residual).max()), 1e-12)
        zero_y = y0 + plot_h / 2
        draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(65, 65, 65))
        draw.line([(x0, zero_y), (x0 + plot_w, zero_y)], fill=(90, 90, 90), width=1)
        points = []
        for index, value in enumerate(residual):
            x_value = x0 + index / max(len(residual) - 1, 1) * plot_w
            y_value = zero_y - float(value) / max_abs * (plot_h / 2 - 7)
            points.append((x_value, y_value))
        if len(points) > 1:
            draw.line(points, fill=(160, 20, 55), width=3)
        common.draw_centered(draw, (x0 + plot_w / 2, y0 - 17), f"c{component}", label_font)
        draw.text((x0, y0 + plot_h + 6), f"{edges[component][0]:.2f}", font=small_font, fill=(70, 70, 70))
        right_text = f"{edges[component][-1]:.2f}"
        text_width, _ = common.text_size(draw, right_text, small_font)
        draw.text((x0 + plot_w - text_width, y0 + plot_h + 6), right_text, font=small_font, fill=(70, 70, 70))
        draw.text((x0 + 4, y0 + 3), f"±{max_abs:.2g}", font=small_font, fill=(70, 70, 70))
    image.save(path)


def draw_spatial_energy_trajectories(
    energy_df: pd.DataFrame,
    epochs: Sequence[int],
    path: Path,
) -> None:
    panels = []
    position_labels = ["TL", "TC", "TR", "ML", "C", "MR", "BL", "BC", "BR"]
    for representation in ("raw", "bn_folded", "shape_normalized"):
        subset = energy_df[
            (energy_df["scope_type"] == "global")
            & (energy_df["representation"] == representation)
        ]
        series = []
        for position in range(9):
            position_df = subset[subset["position"] == position].sort_values("epoch")
            series.append(
                {
                    "x": position_df["epoch"].astype(float).tolist(),
                    "y": position_df["normalized_energy"].astype(float).tolist(),
                    "label": position_labels[position],
                    "color": PLOT_COLORS[position],
                }
            )
        panels.append(
            {
                "title": representation.replace("_", " ").title(),
                "series": series,
                "zero_floor": True,
            }
        )
    draw_panel_grid(panels, path, "Global 3x3 Spatial-Energy Trajectories", columns=1)


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted, locally produced checkpoint without changing its contents."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only keyword.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    return checkpoint


def project_reference_pca(x: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    scaled = reference_scale_filters(x)
    projected = (
        (scaled.astype(np.float64, copy=False) - pca["mean"])
        @ pca["components"].T
    )
    if not np.isfinite(projected).all():
        raise ValueError("Reference-PCA projection produced non-finite coefficients")
    return projected.astype(np.float32)


def slices_signature(slices: Sequence[common.LayerSlice]) -> list[tuple[Any, ...]]:
    return [
        (
            layer_slice.layer_key,
            layer_slice.stage,
            layer_slice.layer_order,
            layer_slice.start,
            layer_slice.end,
            layer_slice.shape,
        )
        for layer_slice in slices
    ]


def pca_scope_definitions(
    slices: Sequence[common.LayerSlice],
) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = [
        {"scope_type": "global", "scope": "global", "slice_indices": list(range(len(slices)))}
    ]
    stage_order = (("stem",) if any(item.stage == "stem" for item in slices) else ()) + STAGE_ORDER
    for stage in stage_order:
        indices = [index for index, item in enumerate(slices) if item.stage == stage]
        if indices:
            scopes.append({"scope_type": "stage", "scope": stage, "slice_indices": indices})
    for index, item in enumerate(slices):
        scopes.append(
            {
                "scope_type": "layer",
                "scope": item.layer_key,
                "stage": item.stage,
                "layer_order": item.layer_order,
                "conv_depth_norm": item.conv_depth_norm,
                "slice_indices": [index],
            }
        )
    return scopes


def subset_projection(
    z: np.ndarray,
    slices: Sequence[common.LayerSlice],
    indices: Sequence[int],
) -> np.ndarray:
    if len(indices) == len(slices):
        return z
    chunks = [z[slices[index].start : slices[index].end] for index in indices]
    if len(chunks) == 1:
        return chunks[0]
    return np.vstack(chunks)


def build_scope_probabilities(
    z_by_epoch: dict[int, np.ndarray],
    slices: Sequence[common.LayerSlice],
    scopes: Sequence[dict[str, Any]],
    edges: Sequence[np.ndarray],
) -> dict[tuple[str, str], dict[int, np.ndarray]]:
    result: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for scope in scopes:
        key = (str(scope["scope_type"]), str(scope["scope"]))
        result[key] = {}
        for epoch, z in z_by_epoch.items():
            values = subset_projection(z, slices, scope["slice_indices"])
            result[key][epoch] = probabilities_for_z(values, edges)
    return result


def drift_between(
    probabilities: dict[int, np.ndarray],
    epoch_a: int,
    epoch_b: int,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    return common.drift_from_probs(
        probabilities[epoch_a], probabilities[epoch_b], weights
    )


def short_layer_label(layer_key: str) -> str:
    match = re.fullmatch(r"layer(\d+)\.(\d+)\.conv(\d+)\.weight", layer_key)
    if match:
        return f"L{match.group(1)}.{match.group(2)}.c{match.group(3)}"
    return layer_key.replace(".weight", "")


def milestone_histogram_rows_and_panels(
    samples: dict[str, dict[int, np.ndarray]],
    milestone_epochs: Sequence[int],
    metrics: Sequence[str],
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    for metric in metrics:
        available = [samples[metric][epoch] for epoch in milestone_epochs]
        pooled = np.concatenate(available)
        low, high = (float(value) for value in np.quantile(pooled, [0.001, 0.999]))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low, high = float(np.min(pooled)), float(np.max(pooled))
        if low == high:
            low -= 0.5
            high += 0.5
        edges = np.linspace(low, high, bins + 1, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        series = []
        for epoch_index, epoch in enumerate(milestone_epochs):
            values = np.clip(samples[metric][epoch], low, high)
            probability = common.probability_histogram(values, edges)
            smoothed = common.smooth_histogram(probability, passes=2)
            series.append(
                {
                    "x": centers.tolist(),
                    "y": smoothed.tolist(),
                    "label": f"e{epoch}",
                    "color": PLOT_COLORS[epoch_index % len(PLOT_COLORS)],
                }
            )
            for bin_index, (left, right, center, value) in enumerate(
                zip(edges[:-1], edges[1:], centers, probability)
            ):
                rows.append(
                    {
                        "metric": metric,
                        "epoch": epoch,
                        "bin_index": bin_index,
                        "bin_left": float(left),
                        "bin_right": float(right),
                        "bin_center": float(center),
                        "probability": float(value),
                    }
                )
        panels.append(
            {
                "title": DESCRIPTOR_LABELS.get(metric, metric.replace("_", " ").title()),
                "series": series,
                "zero_floor": True,
            }
        )
    return rows, panels


def load_training_context(
    experiment_dir: Path,
    expected_epochs: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    source_metadata: dict[str, Any] = {}
    specifications = {
        "ssl": ("lejepa_ssl_metrics.csv", "epoch"),
        "quick_probe": ("lejepa_quick_probe_metrics.csv", "ssl_epoch"),
        "fixed_probe": ("lejepa_linear_metrics.csv", "ssl_epoch"),
    }
    for name, (filename, epoch_column) in specifications.items():
        path = experiment_dir / filename
        source_metadata[name] = {"path": str(path), "available": path.is_file()}
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if epoch_column not in frame.columns or frame[epoch_column].duplicated().any():
            raise ValueError(f"Invalid epoch column in training-context file: {path}")
        frames[name] = frame.copy()
        source_metadata[name].update(
            {"rows": int(len(frame)), "sha256": sha256_file(path)}
        )

    context = pd.DataFrame({"epoch": list(expected_epochs)})
    if "ssl" in frames:
        ssl = frames["ssl"]
        actual_ssl_epochs = tuple(int(value) for value in ssl["epoch"].tolist())
        maximum = max(expected_epochs)
        if actual_ssl_epochs != tuple(range(1, maximum + 1)):
            raise ValueError(
                "SSL metric epochs must be complete from 1 through the final checkpoint; "
                f"got {actual_ssl_epochs[:3]}...{actual_ssl_epochs[-3:]}"
            )
        columns = [
            column
            for column in ("epoch", "ssl_loss", "prediction_loss", "sigreg_loss", "lr")
            if column in ssl.columns
        ]
        context = context.merge(ssl[columns], on="epoch", how="left", validate="one_to_one")
    if "quick_probe" in frames:
        quick = frames["quick_probe"].rename(columns={"ssl_epoch": "epoch"})
        columns = [
            column
            for column in ("epoch", "quick_probe_accuracy", "target_accuracy")
            if column in quick.columns
        ]
        context = context.merge(quick[columns], on="epoch", how="left", validate="one_to_one")
    if "fixed_probe" in frames:
        fixed = frames["fixed_probe"].rename(columns={"ssl_epoch": "epoch"})
        rename = {
            "final_test_accuracy": "fixed_probe_final_accuracy",
            "best_test_accuracy": "fixed_probe_best_accuracy",
        }
        fixed = fixed.rename(columns=rename)
        columns = [
            column
            for column in (
                "epoch",
                "fixed_probe_final_accuracy",
                "fixed_probe_best_accuracy",
            )
            if column in fixed.columns
        ]
        context = context.merge(fixed[columns], on="epoch", how="left", validate="one_to_one")
    return context, frames, source_metadata


def draw_training_context(
    frames: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    panels: list[dict[str, Any]] = []
    if "ssl" in frames:
        frame = frames["ssl"]
        loss_series = []
        for index, (column, label) in enumerate(
            (("ssl_loss", "SSL"), ("prediction_loss", "prediction"))
        ):
            if column in frame:
                loss_series.append(
                    {
                        "x": frame["epoch"].tolist(),
                        "y": frame[column].tolist(),
                        "label": label,
                        "color": PLOT_COLORS[index],
                    }
                )
        if loss_series:
            panels.append({"title": "SSL and prediction loss", "series": loss_series})
        if "sigreg_loss" in frame:
            panels.append(
                {
                    "title": "SIGReg loss",
                    "series": [
                        {
                            "x": frame["epoch"].tolist(),
                            "y": frame["sigreg_loss"].tolist(),
                            "label": "SIGReg",
                            "color": PLOT_COLORS[2],
                        }
                    ],
                }
            )
    probe_series = []
    if "quick_probe" in frames:
        frame = frames["quick_probe"]
        probe_series.append(
            {
                "x": frame["ssl_epoch"].tolist(),
                "y": frame["quick_probe_accuracy"].tolist(),
                "label": "quick probe",
                "color": PLOT_COLORS[4],
            }
        )
    if "fixed_probe" in frames:
        frame = frames["fixed_probe"]
        for column, label, color in (
            ("final_test_accuracy", "fixed final", PLOT_COLORS[1]),
            ("best_test_accuracy", "fixed best", PLOT_COLORS[2]),
        ):
            if column in frame:
                probe_series.append(
                    {
                        "x": frame["ssl_epoch"].tolist(),
                        "y": frame[column].tolist(),
                        "label": label,
                        "color": color,
                    }
                )
    if probe_series:
        panels.append(
            {"title": "Frozen-feature probe accuracy", "series": probe_series, "zero_floor": True}
        )
    if panels:
        draw_panel_grid(panels, path, "Available LeJEPA Training Context", columns=2)


def write_summary(
    path: Path,
    epochs: Sequence[int],
    milestone_epochs: Sequence[int],
    include_stem: bool,
    bins: int,
    pca_path: Path,
    pca_sha256: str,
    pca_metadata: dict[str, Any],
    global_trajectory: pd.DataFrame,
    stage_trajectory: pd.DataFrame,
    layer_trajectory: pd.DataFrame,
    quality_df: pd.DataFrame,
    context_df: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    first, final = epochs[0], epochs[-1]
    final_global = global_trajectory[global_trajectory["epoch"] == final].iloc[0]
    previous_max = global_trajectory.sort_values("drift_from_previous", ascending=False).iloc[0]
    final_stage = stage_trajectory[stage_trajectory["epoch"] == final].sort_values(
        "drift_from_epoch1", ascending=False
    )
    final_layers = layer_trajectory[layer_trajectory["epoch"] == final].sort_values(
        "drift_from_epoch1", ascending=False
    )
    warning = pca_metadata.get("consumer_published_mean_warning")
    lines = ["# LeJEPA Training-Dynamics Analysis (Full CNN Filter DB PCA)", ""]
    if warning:
        lines.extend(
            [
                "> **IMPORTANT PCA PROVENANCE WARNING**",
                ">",
                f"> {warning}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope and invariants",
            "",
            f"- Exactly `{len(epochs)}` SSL backbone checkpoints were analyzed: `{', '.join(map(str, epochs))}`.",
            f"- Milestone distribution figures use epochs `{', '.join(map(str, milestone_epochs))}`.",
            f"- The CIFAR stem is `{'included' if include_stem else 'excluded'}`; the default therefore covers the 16 residual-block 3x3 convolutions.",
            "- No checkpoint was trained, updated, selected, or interpolated by this analysis.",
            "- Every target kernel is cast to float16 first and then divided by its own float16 maximum absolute coefficient, exactly matching the external artifact preprocessing.",
            f"- PCA histogram drift uses `{bins}` bins and one component-wise edge set computed once from the union of every analyzed checkpoint. The same edges are used globally, by stage, and by layer.",
            "- Drift is the explained-variance-weighted sum of component-wise symmetric KL divergences. It is symmetric, non-negative, and descriptive rather than causal.",
            "- Entropy and sparsity are local raw-layer diagnostics and do not depend on the external PCA basis. Sparsity here retains the legacy 1% of layer peak convention.",
            "",
            "## External PCA provenance",
            "",
            f"- Artifact: `{pca_path}`",
            f"- SHA256: `{pca_sha256}`",
            f"- Full released-dataset filter count: `{pca_metadata['validated_n_samples']:,}`",
            f"- Preprocessing ID: `{pca_metadata['validated_preprocessing_id']}`",
            f"- Consumer orthonormality error: `{pca_metadata['consumer_orthonormality_max_abs_error']:.3e}`",
            f"- Consumer covariance reconstruction error: `{pca_metadata['consumer_covariance_reconstruction_max_abs_error']:.3e}`",
            "",
            "## Headline numerical observations",
            "",
            f"- Global epoch-{first} to epoch-{final} drift: `{float(final_global['drift_from_epoch1']):.6f}`.",
            f"- Largest adjacent-saved-checkpoint global drift occurs at epoch `{int(previous_max['epoch'])}` relative to epoch `{int(previous_max['previous_epoch'])}`: `{float(previous_max['drift_from_previous']):.6f}`.",
        ]
    )
    if not final_stage.empty:
        row = final_stage.iloc[0]
        lines.append(
            f"- Stage with the largest epoch-{first} to epoch-{final} drift: `{row['stage']}` (`{float(row['drift_from_epoch1']):.6f}`)."
        )
    if not final_layers.empty:
        row = final_layers.iloc[0]
        lines.append(
            f"- Layer with the largest epoch-{first} to epoch-{final} drift: `{row['layer_key']}` (`{float(row['drift_from_epoch1']):.6f}`)."
        )
    first_quality = quality_df[quality_df["epoch"] == first]
    final_quality = quality_df[quality_df["epoch"] == final]
    lines.extend(
        [
            f"- Mean layer entropy changes from `{first_quality['entropy_H'].mean():.6f}` to `{final_quality['entropy_H'].mean():.6f}`; mean 1%-peak sparsity changes from `{first_quality['sparsity_S'].mean():.6f}` to `{final_quality['sparsity_S'].mean():.6f}`.",
            "- Loss and probe curves are context only: their co-movement with filter statistics does not establish that a particular filter change caused accuracy or loss changes.",
            "",
            "## Output map",
            "",
            "- `global_pairwise_epoch_drift.csv` and `figures/global_pairwise_epoch_drift_heatmap.png`: all checkpoint pairs.",
            "- `figures/global_pairwise_epoch_drift_chart.png`: full line-chart counterpart, one panel per reference epoch.",
            "- `global_drift_trajectories.csv`, `stage_drift_trajectories.csv`, and `layer_drift_trajectories.csv`: change from epoch 1, the previous saved checkpoint, and epoch 120.",
            "- `global_component_drift_contributions.csv` and `pca_component_statistics.csv`: component-level change and coefficient statistics.",
            "- `layer_quality_by_epoch.csv`: local entropy, H/TH, sparsity, and raw-weight statistics.",
            "- `distribution_summary_by_epoch.csv`: raw, BN-folded, shape-normalized, BN-scale, and descriptor summaries for global/stage/layer scopes.",
            "- `spatial_energy_by_epoch.csv`: 3x3 position energy for global and stage scopes.",
            "- `milestone_scalar_histograms.csv`: fixed milestone histogram values behind the raw/BN/shape and descriptor figures.",
            "- `dynamics_training_context.csv`: checkpoint-aligned SSL loss and frozen-probe metrics where available.",
            "- `analysis_provenance.json`, `pca_basis_provenance.json`, and `validation_checks.csv`: reproducibility and validation records.",
            "",
            "## Interpretation limits",
            "",
            "The external basis supplies a stable coordinate system, not a universal semantic labeling of individual components. PCA component signs are conventional, histogram KL values depend on binning, and summary-statistic trajectories can hide multimodal redistribution. Read the heatmaps, chart counterparts, milestone distributions, raw/BN/shape controls, and task metrics together.",
            "",
            f"Analysis elapsed time: `{elapsed_seconds:.1f}` seconds.",
        ]
    )
    if context_df.empty:
        lines.append("\nNo training-context CSV was available.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.bins < 4:
        raise ValueError("--bins must be at least 4")
    if args.max_density_points < 0 or args.descriptor_max_samples < 0:
        raise ValueError("Sampling limits must be non-negative")

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    pca_path = Path(args.pca_basis).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    figure_dir = output_dir / "figures"
    common.ensure_dir(output_dir)
    common.ensure_dir(figure_dir)

    epochs = parse_epoch_list(args.expected_epochs, "--expected-epochs")
    if tuple(sorted(epochs)) != epochs:
        raise ValueError("--expected-epochs must be strictly increasing")
    milestones = parse_epoch_list(args.milestone_epochs, "--milestone-epochs")
    unknown_milestones = sorted(set(milestones) - set(epochs))
    if unknown_milestones:
        raise ValueError(f"Milestones are not checkpoint epochs: {unknown_milestones}")

    checkpoints = discover_checkpoints(experiment_dir, epochs)
    pca, pca_metadata = load_reference_pca(pca_path)
    pca_sha256 = sha256_file(pca_path)
    weights = pca["explained_variance_ratio"]
    print(
        f"[validated] full-DB PCA {pca_sha256[:12]} with "
        f"{int(pca_metadata['validated_n_samples']):,} source filters",
        flush=True,
    )

    validation_rows: list[dict[str, Any]] = [
        {"check": "canonical_checkpoint_epoch_set_exact", "passed": True, "detail": str(epochs)},
        {"check": "pca_full_dataset_provenance", "passed": True, "detail": str(pca_path)},
        {
            "check": "pca_numerical_consistency",
            "passed": True,
            "detail": (
                f"orth={pca_metadata['consumer_orthonormality_max_abs_error']:.3e}; "
                f"cov={pca_metadata['consumer_covariance_reconstruction_max_abs_error']:.3e}"
            ),
        },
        {
            "check": "published_mean_matches_supplement",
            "passed": bool(pca_metadata.get("published_mean_validation_passed", False)),
            "detail": str(pca_metadata.get("consumer_published_mean_warning") or "passed"),
            "severity": "warning",
        },
    ]

    z_by_epoch: dict[int, np.ndarray] = {}
    descriptor_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    component_stat_rows: list[dict[str, Any]] = []
    checkpoint_provenance: list[dict[str, Any]] = []
    milestone_samples: dict[str, dict[int, np.ndarray]] = {
        metric: {} for metric in (*WEIGHT_REPRESENTATIONS, *DESCRIPTORS, "bn_abs_scale")
    }
    canonical_slices: list[common.LayerSlice] | None = None
    canonical_signature: list[tuple[Any, ...]] | None = None

    for epoch, checkpoint_path in checkpoints:
        print(f"[checkpoint] epoch {epoch}: {checkpoint_path}", flush=True)
        checkpoint = load_torch_checkpoint(checkpoint_path)
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        if checkpoint_epoch != epoch:
            raise ValueError(
                f"Checkpoint epoch metadata mismatch: filename={epoch}, metadata={checkpoint_epoch}"
            )
        state = common.load_backbone_state(checkpoint)
        raw_bank, slices, local_quality = common.extract_filter_bank(
            state, f"lejepa_ssl_epoch{epoch:03d}", include_stem=args.include_stem
        )
        signature = slices_signature(slices)
        if canonical_signature is None:
            canonical_signature = signature
            canonical_slices = slices
        elif signature != canonical_signature:
            raise ValueError(f"Layer schema changed at epoch {epoch}")
        z = project_reference_pca(raw_bank, pca)
        z_by_epoch[epoch] = z
        del raw_bank

        for row in local_quality:
            quality_rows.append({"epoch": epoch, **row})
        for component in range(9):
            stats = summarize_values(
                z[:, component],
                args.descriptor_max_samples,
                stable_rng(args.seed, "pca-component", str(epoch), str(component)),
            )
            component_stat_rows.append(
                {
                    "epoch": epoch,
                    "component": component,
                    "full_db_explained_variance_ratio": float(weights[component]),
                    **stats,
                }
            )

        assert canonical_slices is not None
        layers = build_layer_payloads(state, canonical_slices)
        epoch_descriptor_rows, epoch_energy_rows = descriptor_rows_for_epoch(
            epoch, layers, args.descriptor_max_samples, args.seed
        )
        descriptor_rows.extend(epoch_descriptor_rows)
        energy_rows.extend(epoch_energy_rows)
        if epoch in milestones:
            all_indices = list(range(len(layers)))
            for metric, field in (
                ("raw_weight", "raw"),
                ("bn_folded_weight", "bn_folded"),
                ("shape_normalized_weight", "shape_normalized"),
            ):
                values = concatenate_layer_field(layers, all_indices, field).reshape(-1)
                milestone_samples[metric][epoch] = sample_values(
                    values,
                    args.descriptor_max_samples,
                    stable_rng(args.seed, "milestone", metric, str(epoch)),
                ).astype(np.float32)
            for descriptor in DESCRIPTORS:
                values = concatenate_layer_field(
                    layers, all_indices, "descriptors", descriptor=descriptor
                )
                milestone_samples[descriptor][epoch] = sample_values(
                    values,
                    args.descriptor_max_samples,
                    stable_rng(args.seed, "milestone", descriptor, str(epoch)),
                ).astype(np.float32)
            bn_values = np.concatenate([layer["bn_factor"] for layer in layers])
            milestone_samples["bn_abs_scale"][epoch] = sample_values(
                np.abs(bn_values),
                args.descriptor_max_samples,
                stable_rng(args.seed, "milestone", "bn_abs_scale", str(epoch)),
            ).astype(np.float32)

        checkpoint_provenance.append(
            {
                "epoch": epoch,
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
                "file_size_bytes": checkpoint_path.stat().st_size,
                "metadata_method": checkpoint.get("method"),
                "metadata_architecture": checkpoint.get("architecture"),
                "metadata_dataset": checkpoint.get("dataset"),
                "metadata_ssl_loss": checkpoint.get("ssl_loss"),
                "num_projected_filters": int(len(z)),
            }
        )
        del layers, state, checkpoint

    assert canonical_slices is not None
    validation_rows.append(
        {
            "check": "checkpoint_layer_schema_identical",
            "passed": True,
            "detail": f"{len(canonical_slices)} layers; {len(z_by_epoch[epochs[0]]):,} filters/checkpoint",
        }
    )
    edges = fixed_edges(z_by_epoch, args.bins)
    edge_rows = [
        {
            "component": component,
            "edge_index": edge_index,
            "edge": float(value),
        }
        for component, component_edges in enumerate(edges)
        for edge_index, value in enumerate(component_edges)
    ]
    pd.DataFrame(edge_rows).to_csv(output_dir / "fixed_pca_histogram_edges.csv", index=False)
    scopes = pca_scope_definitions(canonical_slices)
    probabilities_by_scope = build_scope_probabilities(
        z_by_epoch, canonical_slices, scopes, edges
    )
    global_probabilities = probabilities_by_scope[("global", "global")]
    probability_error = max(
        abs(float(probability.sum()) - 1.0)
        for scope_probabilities in probabilities_by_scope.values()
        for probability_matrix in scope_probabilities.values()
        for probability in probability_matrix
    )
    if probability_error > 1e-10:
        raise ValueError(f"Histogram probabilities do not sum to one: {probability_error}")
    validation_rows.append(
        {
            "check": "fixed_union_histogram_probabilities",
            "passed": True,
            "detail": f"max probability-sum error={probability_error:.3e}",
        }
    )

    pairwise_matrix = np.zeros((len(epochs), len(epochs)), dtype=np.float64)
    pairwise_rows: list[dict[str, Any]] = []
    pairwise_component_rows: list[dict[str, Any]] = []
    for index_a, epoch_a in enumerate(epochs):
        for index_b, epoch_b in enumerate(epochs):
            total, contributions = drift_between(
                global_probabilities, epoch_a, epoch_b, weights
            )
            pairwise_matrix[index_a, index_b] = total
            pairwise_rows.append(
                {"epoch_a": epoch_a, "epoch_b": epoch_b, "drift_D": total}
            )
            for component, contribution in enumerate(contributions):
                pairwise_component_rows.append(
                    {
                        "epoch_a": epoch_a,
                        "epoch_b": epoch_b,
                        "component": component,
                        "full_db_explained_variance_ratio": float(weights[component]),
                        "weighted_symmetric_kl": float(contribution),
                    }
                )
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "global_pairwise_epoch_drift.csv", index=False
    )
    pd.DataFrame(pairwise_component_rows).to_csv(
        output_dir / "global_pairwise_component_contributions.csv", index=False
    )

    global_trajectory_rows: list[dict[str, Any]] = []
    global_component_rows: list[dict[str, Any]] = []
    stage_trajectory_rows: list[dict[str, Any]] = []
    layer_trajectory_rows: list[dict[str, Any]] = []
    first_epoch, final_epoch = epochs[0], epochs[-1]
    for epoch_index, epoch in enumerate(epochs):
        previous_epoch = epochs[max(epoch_index - 1, 0)]
        comparisons = {
            "from_epoch1": first_epoch,
            "from_previous": previous_epoch,
            "to_final": final_epoch,
        }
        totals: dict[str, float] = {}
        for comparison, reference_epoch in comparisons.items():
            total, contributions = drift_between(
                global_probabilities, reference_epoch, epoch, weights
            )
            totals[comparison] = total
            for component, contribution in enumerate(contributions):
                global_component_rows.append(
                    {
                        "epoch": epoch,
                        "comparison": comparison,
                        "reference_epoch": reference_epoch,
                        "component": component,
                        "full_db_explained_variance_ratio": float(weights[component]),
                        "weighted_symmetric_kl": float(contribution),
                        "global_drift_D": total,
                    }
                )
        global_trajectory_rows.append(
            {
                "epoch": epoch,
                "previous_epoch": previous_epoch,
                "drift_from_epoch1": totals["from_epoch1"],
                "drift_from_previous": totals["from_previous"],
                "drift_to_final": totals["to_final"],
            }
        )
        for scope in scopes:
            if scope["scope_type"] not in {"stage", "layer"}:
                continue
            scope_probabilities = probabilities_by_scope[
                (str(scope["scope_type"]), str(scope["scope"]))
            ]
            values = {
                name: drift_between(scope_probabilities, reference, epoch, weights)[0]
                for name, reference in comparisons.items()
            }
            row = {
                "epoch": epoch,
                "previous_epoch": previous_epoch,
                "drift_from_epoch1": values["from_epoch1"],
                "drift_from_previous": values["from_previous"],
                "drift_to_final": values["to_final"],
            }
            if scope["scope_type"] == "stage":
                stage_trajectory_rows.append({"stage": scope["scope"], **row})
            else:
                layer_trajectory_rows.append(
                    {
                        "layer_key": scope["scope"],
                        "stage": scope["stage"],
                        "layer_order": scope["layer_order"],
                        "conv_depth_norm": scope["conv_depth_norm"],
                        **row,
                    }
                )

    global_trajectory_df = pd.DataFrame(global_trajectory_rows)
    stage_trajectory_df = pd.DataFrame(stage_trajectory_rows)
    layer_trajectory_df = pd.DataFrame(layer_trajectory_rows)
    global_trajectory_df.to_csv(output_dir / "global_drift_trajectories.csv", index=False)
    stage_trajectory_df.to_csv(output_dir / "stage_drift_trajectories.csv", index=False)
    layer_trajectory_df.to_csv(output_dir / "layer_drift_trajectories.csv", index=False)
    pd.DataFrame(global_component_rows).to_csv(
        output_dir / "global_component_drift_contributions.csv", index=False
    )

    quality_df = pd.DataFrame(quality_rows)
    descriptor_df = pd.DataFrame(descriptor_rows)
    energy_df = pd.DataFrame(energy_rows)
    component_stats_df = pd.DataFrame(component_stat_rows)
    quality_df.to_csv(output_dir / "layer_quality_by_epoch.csv", index=False)
    descriptor_df.to_csv(output_dir / "distribution_summary_by_epoch.csv", index=False)
    energy_df.to_csv(output_dir / "spatial_energy_by_epoch.csv", index=False)
    component_stats_df.to_csv(output_dir / "pca_component_statistics.csv", index=False)

    quality_aggregate_rows = []
    for epoch in epochs:
        subset = quality_df[quality_df["epoch"] == epoch]
        filter_weights = subset["num_filters"].to_numpy(dtype=np.float64)
        for metric in ("entropy_H", "H_over_TH", "sparsity_S"):
            values = subset[metric].to_numpy(dtype=np.float64)
            quality_aggregate_rows.append(
                {
                    "epoch": epoch,
                    "metric": metric,
                    "mean_across_layers": float(values.mean()),
                    "weighted_mean_by_filter_count": float(np.average(values, weights=filter_weights)),
                    "median_across_layers": float(np.median(values)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    quality_aggregate_df = pd.DataFrame(quality_aggregate_rows)
    quality_aggregate_df.to_csv(output_dir / "quality_aggregate_by_epoch.csv", index=False)

    histogram_rows: list[dict[str, Any]] = []
    weight_hist_rows, weight_panels = milestone_histogram_rows_and_panels(
        milestone_samples,
        milestones,
        (*WEIGHT_REPRESENTATIONS, "bn_abs_scale"),
        args.bins,
    )
    descriptor_hist_rows, descriptor_panels = milestone_histogram_rows_and_panels(
        milestone_samples, milestones, DESCRIPTORS, args.bins
    )
    histogram_rows.extend(weight_hist_rows)
    histogram_rows.extend(descriptor_hist_rows)
    pd.DataFrame(histogram_rows).to_csv(
        output_dir / "milestone_scalar_histograms.csv", index=False
    )

    context_df, context_frames, context_sources = load_training_context(
        experiment_dir, epochs
    )
    context_df.to_csv(output_dir / "dynamics_training_context.csv", index=False)

    print("[figures] rendering dynamics suite", flush=True)
    labels = [f"e{epoch}" for epoch in epochs]
    common.draw_heatmap(
        pairwise_matrix,
        labels,
        "LeJEPA Pairwise Drift in the Full CNN Filter DB PCA Basis",
        figure_dir / "global_pairwise_epoch_drift_heatmap.png",
        vmin=0.0,
        vmax=float(pairwise_matrix.max()),
    )
    pairwise_panels = [
        {
            "title": f"Reference epoch {epoch}",
            "series": [
                {
                    "x": list(epochs),
                    "y": pairwise_matrix[index].tolist(),
                    "label": f"D(e{epoch}, epoch)",
                    "color": PLOT_COLORS[index % len(PLOT_COLORS)],
                }
            ],
            "zero_floor": True,
        }
        for index, epoch in enumerate(epochs)
    ]
    draw_panel_grid(
        pairwise_panels,
        figure_dir / "global_pairwise_epoch_drift_chart.png",
        "Pairwise Epoch Drift: Line-Chart Counterpart to the Heatmap",
        columns=3,
    )
    common.draw_line_series(
        [
            {
                "x": global_trajectory_df["epoch"].tolist(),
                "y": global_trajectory_df[column].tolist(),
                "label": label,
                "color": PLOT_COLORS[index],
            }
            for index, (column, label) in enumerate(
                (
                    ("drift_from_epoch1", "from epoch 1"),
                    ("drift_from_previous", "from previous saved"),
                    ("drift_to_final", "to epoch 120"),
                )
            )
        ],
        "Global LeJEPA Filter-Distribution Drift",
        "SSL epoch",
        "weighted symmetric KL (D)",
        figure_dir / "global_drift_trajectories.png",
        y_min=0.0,
    )
    common.draw_eigenfilters(
        pca["components"],
        weights,
        figure_dir / "full_cnn_filter_db_eigenfilters.png",
        "Full CNN Filter DB PCA Directions (sign-canonicalized)",
    )
    draw_milestone_ridges(
        milestones,
        global_probabilities,
        figure_dir / "milestone_pca_ridges.png",
    )
    draw_c0_c1_densities(
        milestones,
        z_by_epoch,
        edges,
        args.max_density_points,
        args.seed,
        figure_dir / "milestone_c0_c1_density.png",
    )
    draw_initial_final_residuals(
        first_epoch,
        final_epoch,
        global_probabilities,
        edges,
        figure_dir / "initial_final_pca_residuals.png",
    )

    component_drift_df = pd.DataFrame(global_component_rows)
    component_panels = []
    comparison_labels = {
        "from_epoch1": "from e1",
        "from_previous": "from previous",
        "to_final": "to final",
    }
    for component in range(9):
        series = []
        for comparison_index, comparison in enumerate(comparison_labels):
            subset = component_drift_df[
                (component_drift_df["component"] == component)
                & (component_drift_df["comparison"] == comparison)
            ].sort_values("epoch")
            series.append(
                {
                    "x": subset["epoch"].tolist(),
                    "y": subset["weighted_symmetric_kl"].tolist(),
                    "label": comparison_labels[comparison],
                    "color": PLOT_COLORS[comparison_index],
                }
            )
        component_panels.append(
            {"title": f"component c{component}", "series": series, "zero_floor": True}
        )
    draw_panel_grid(
        component_panels,
        figure_dir / "component_drift_dynamics.png",
        "Full-DB PCA Component Drift Contributions",
        columns=3,
    )
    component_stat_panels = []
    for component in range(9):
        subset = component_stats_df[component_stats_df["component"] == component].sort_values(
            "epoch"
        )
        component_stat_panels.append(
            {
                "title": f"coefficient c{component}",
                "series": [
                    {
                        "x": subset["epoch"].tolist(),
                        "y": subset["mean"].tolist(),
                        "label": "mean",
                        "color": PLOT_COLORS[0],
                    },
                    {
                        "x": subset["epoch"].tolist(),
                        "y": subset["std"].tolist(),
                        "label": "std",
                        "color": PLOT_COLORS[1],
                    },
                ],
            }
        )
    draw_panel_grid(
        component_stat_panels,
        figure_dir / "pca_component_statistics.png",
        "PCA Coefficient Mean and Standard Deviation",
        columns=3,
    )

    stage_panels = []
    for stage in stage_trajectory_df["stage"].drop_duplicates().tolist():
        subset = stage_trajectory_df[stage_trajectory_df["stage"] == stage].sort_values("epoch")
        stage_panels.append(
            {
                "title": stage,
                "series": [
                    {
                        "x": subset["epoch"].tolist(),
                        "y": subset[column].tolist(),
                        "label": label,
                        "color": PLOT_COLORS[index],
                    }
                    for index, (column, label) in enumerate(
                        (
                            ("drift_from_epoch1", "from e1"),
                            ("drift_from_previous", "from previous"),
                            ("drift_to_final", "to final"),
                        )
                    )
                ],
                "zero_floor": True,
            }
        )
    draw_panel_grid(
        stage_panels,
        figure_dir / "stage_drift_trajectories.png",
        "Stage-Level Filter-Distribution Drift",
        columns=2,
    )
    layer_panels = []
    for stage in layer_trajectory_df["stage"].drop_duplicates().tolist():
        for column, label in (
            ("drift_from_epoch1", "from epoch 1"),
            ("drift_from_previous", "from previous saved"),
            ("drift_to_final", "to final"),
        ):
            subset = layer_trajectory_df[layer_trajectory_df["stage"] == stage]
            series = []
            for layer_index, layer_key in enumerate(subset["layer_key"].drop_duplicates()):
                layer_df = subset[subset["layer_key"] == layer_key].sort_values("epoch")
                series.append(
                    {
                        "x": layer_df["epoch"].tolist(),
                        "y": layer_df[column].tolist(),
                        "label": short_layer_label(layer_key),
                        "color": PLOT_COLORS[layer_index],
                    }
                )
            layer_panels.append(
                {"title": f"{stage}: {label}", "series": series, "zero_floor": True}
            )
    draw_panel_grid(
        layer_panels,
        figure_dir / "layer_drift_trajectories_chart.png",
        "Layer-Level Filter-Distribution Drift",
        columns=2,
    )
    layer_order = (
        layer_trajectory_df[["layer_key", "layer_order"]]
        .drop_duplicates()
        .sort_values("layer_order")["layer_key"]
        .tolist()
    )
    for column, filename, title in (
        ("drift_from_epoch1", "layer_drift_from_epoch1_heatmap.png", "Layer Drift from Epoch 1"),
        ("drift_from_previous", "layer_drift_from_previous_heatmap.png", "Layer Drift from Previous Saved Checkpoint"),
        ("drift_to_final", "layer_drift_to_final_heatmap.png", "Layer Drift to Final Epoch"),
    ):
        matrix = np.asarray(
            [
                layer_trajectory_df[layer_trajectory_df["layer_key"] == layer_key]
                .sort_values("epoch")[column]
                .to_numpy(dtype=float)
                for layer_key in layer_order
            ]
        )
        draw_rectangular_heatmap(
            matrix,
            [short_layer_label(value) for value in layer_order],
            labels,
            title,
            figure_dir / filename,
            value_format=".3f",
        )

    quality_layer_order = (
        quality_df[["layer_key", "layer_order"]]
        .drop_duplicates()
        .sort_values("layer_order")["layer_key"]
        .tolist()
    )
    for metric, filename, title in (
        ("entropy_H", "entropy_by_layer_epoch_heatmap.png", "Local Layer Entropy Through SSL Training"),
        ("H_over_TH", "entropy_over_random_threshold_heatmap.png", "Layer H/TH Through SSL Training"),
        ("sparsity_S", "sparsity_by_layer_epoch_heatmap.png", "Layer Sparsity at 1% of Layer Peak"),
    ):
        matrix = np.asarray(
            [
                quality_df[quality_df["layer_key"] == layer_key]
                .sort_values("epoch")[metric]
                .to_numpy(dtype=float)
                for layer_key in quality_layer_order
            ]
        )
        draw_rectangular_heatmap(
            matrix,
            [short_layer_label(value) for value in quality_layer_order],
            labels,
            title,
            figure_dir / filename,
            value_format=".3f",
        )
    quality_panels = []
    for metric, title in (
        ("entropy_H", "Mean local entropy"),
        ("H_over_TH", "Mean H/TH"),
        ("sparsity_S", "Mean 1%-peak sparsity"),
    ):
        subset = quality_aggregate_df[quality_aggregate_df["metric"] == metric]
        quality_panels.append(
            {
                "title": title,
                "series": [
                    {
                        "x": subset["epoch"].tolist(),
                        "y": subset[column].tolist(),
                        "label": label,
                        "color": PLOT_COLORS[index],
                    }
                    for index, (column, label) in enumerate(
                        (
                            ("mean_across_layers", "layer mean"),
                            ("weighted_mean_by_filter_count", "filter-weighted"),
                            ("median_across_layers", "layer median"),
                        )
                    )
                ],
                "zero_floor": True,
            }
        )
    draw_panel_grid(
        quality_panels,
        figure_dir / "entropy_sparsity_trajectories.png",
        "Entropy and Sparsity Dynamics (PCA-Independent)",
        columns=1,
    )

    global_summary = descriptor_df[descriptor_df["scope_type"] == "global"]
    descriptor_panels_out = []
    for descriptor in DESCRIPTORS:
        subset = global_summary[global_summary["metric"] == descriptor].sort_values("epoch")
        descriptor_panels_out.append(
            {
                "title": DESCRIPTOR_LABELS[descriptor],
                "series": [
                    {
                        "x": subset["epoch"].tolist(),
                        "y": subset["mean"].tolist(),
                        "label": "mean",
                        "color": PLOT_COLORS[0],
                    },
                    {
                        "x": subset["epoch"].tolist(),
                        "y": subset["median"].tolist(),
                        "label": "median",
                        "color": PLOT_COLORS[1],
                    },
                ],
            }
        )
    draw_panel_grid(
        descriptor_panels_out,
        figure_dir / "global_kernel_descriptor_trajectories.png",
        "Global Kernel-Descriptor Trajectories",
        columns=3,
    )
    representation_panels = []
    for representation in WEIGHT_REPRESENTATIONS:
        subset = global_summary[global_summary["metric"] == representation].sort_values("epoch")
        representation_panels.append(
            {
                "title": DESCRIPTOR_LABELS[representation],
                "series": [
                    {
                        "x": subset["epoch"].tolist(),
                        "y": subset[column].tolist(),
                        "label": label,
                        "color": PLOT_COLORS[index],
                    }
                    for index, (column, label) in enumerate(
                        (("mean", "mean"), ("std", "std"), ("rms", "RMS"))
                    )
                ],
            }
        )
    draw_panel_grid(
        representation_panels,
        figure_dir / "raw_bn_shape_weight_trajectories.png",
        "Raw, BN-Folded, and Shape-Normalized Weight Dynamics",
        columns=1,
    )
    bn_subset = descriptor_df[descriptor_df["metric"] == "bn_abs_scale"]
    bn_series = []
    for index, (scope_type, scope_name) in enumerate(
        [("global", "global")] + [("stage", stage) for stage in STAGE_ORDER]
    ):
        subset = bn_subset[
            (bn_subset["scope_type"] == scope_type) & (bn_subset["scope"] == scope_name)
        ].sort_values("epoch")
        if subset.empty:
            continue
        bn_series.append(
            {
                "x": subset["epoch"].tolist(),
                "y": subset["mean"].tolist(),
                "label": scope_name,
                "color": PLOT_COLORS[index],
            }
        )
    common.draw_line_series(
        bn_series,
        "Absolute BatchNorm Scale Through SSL Training",
        "SSL epoch",
        "mean |gamma / sqrt(running variance + eps)|",
        figure_dir / "batchnorm_scale_trajectories.png",
        y_min=0.0,
    )
    draw_spatial_energy_trajectories(
        energy_df, epochs, figure_dir / "spatial_energy_trajectories.png"
    )
    draw_panel_grid(
        weight_panels,
        figure_dir / "milestone_weight_distributions.png",
        "Milestone Raw, BN-Folded, Shape, and BN-Scale Distributions",
        columns=2,
    )
    draw_panel_grid(
        descriptor_panels,
        figure_dir / "milestone_descriptor_distributions.png",
        "Milestone Kernel-Descriptor Distributions",
        columns=3,
    )
    draw_training_context(context_frames, figure_dir / "loss_probe_context.png")

    validation_rows.extend(
        [
            {
                "check": "pairwise_drift_symmetric",
                "passed": bool(np.allclose(pairwise_matrix, pairwise_matrix.T, atol=1e-12)),
                "detail": f"max error={float(np.max(np.abs(pairwise_matrix - pairwise_matrix.T))):.3e}",
            },
            {
                "check": "pairwise_drift_diagonal_zero",
                "passed": bool(np.allclose(np.diag(pairwise_matrix), 0.0, atol=1e-12)),
                "detail": f"max diagonal={float(np.max(np.abs(np.diag(pairwise_matrix)))):.3e}",
            },
            {
                "check": "all_core_outputs_finite",
                "passed": bool(
                    np.isfinite(pairwise_matrix).all()
                    and np.isfinite(global_trajectory_df.select_dtypes(include=[np.number])).all().all()
                    and np.isfinite(stage_trajectory_df.select_dtypes(include=[np.number])).all().all()
                    and np.isfinite(layer_trajectory_df.select_dtypes(include=[np.number])).all().all()
                ),
                "detail": "pairwise/global/stage/layer drift tables",
            },
        ]
    )
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(output_dir / "validation_checks.csv", index=False)
    required_pass = validation_df[
        validation_df.get("severity", pd.Series(index=validation_df.index, dtype=object)).fillna("error")
        != "warning"
    ]
    if not bool(required_pass["passed"].all()):
        raise ValueError("One or more required validation checks failed")

    provenance = {
        "analysis": "rn18_lejepa_training_dynamics",
        "no_training_performed": True,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "expected_epochs": list(epochs),
        "milestone_epochs": list(milestones),
        "checkpoint_set_validation": "exact; missing and extra canonical SSL files are fatal",
        "include_stem": bool(args.include_stem),
        "bins": int(args.bins),
        "histogram_edge_protocol": (
            "one component-wise min/max edge set from the union of all checkpoint "
            "projections; reused for global, stage, and layer scopes"
        ),
        "target_pca_preprocessing": (
            "cast each target filter to float16, float16 per-row max-abs normalization, "
            "then center/project with the external float64 PCA arrays"
        ),
        "local_descriptor_shape_normalization": (
            "float32 per-kernel max-abs normalization retained for PCA-independent "
            "Experiment-3-compatible descriptor controls"
        ),
        "pca_basis": {
            "path": str(pca_path),
            "sha256": pca_sha256,
            "metadata": pca_metadata,
        },
        "checkpoints": checkpoint_provenance,
        "layer_schema": [
            {
                "layer_key": item.layer_key,
                "stage": item.stage,
                "layer_order": item.layer_order,
                "conv_depth_norm": item.conv_depth_norm,
                "start": item.start,
                "end": item.end,
                "shape": list(item.shape),
            }
            for item in canonical_slices
        ],
        "training_context_sources": context_sources,
        "sampling": {
            "seed": int(args.seed),
            "max_density_points": int(args.max_density_points),
            "descriptor_max_samples": int(args.descriptor_max_samples),
            "sampling_is_only_for_quantiles_and_display_histograms": True,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    (output_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "pca_basis_provenance.json").write_text(
        json.dumps(provenance["pca_basis"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(
        output_dir / "summary.md",
        epochs,
        milestones,
        args.include_stem,
        args.bins,
        pca_path,
        pca_sha256,
        pca_metadata,
        global_trajectory_df,
        stage_trajectory_df,
        layer_trajectory_df,
        quality_df,
        context_df,
        time.time() - started,
    )
    print(
        f"[done] wrote {len(list(figure_dir.glob('*.png')))} figures and analysis tables "
        f"to {output_dir} in {time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
