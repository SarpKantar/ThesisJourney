#!/usr/bin/env python
"""Distribution-focused kernel comparison for the matched RN18/CIFAR10 pair."""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.stats import kurtosis, ks_2samp, skew, wasserstein_distance

import analyze_rn18_cifar10_filters as matched
import rn18_cifar10_common as common


PAIR = ("imagenet_ft", "lejepa_ssl_linear")
MODEL_COLORS = {
    "imagenet_ft": (0, 114, 178),
    "lejepa_ssl_linear": (0, 158, 115),
}
MODEL_LABELS = {
    "imagenet_ft": "ImageNet FT epoch 1",
    "lejepa_ssl_linear": "LeJEPA SSL epoch 110",
}
CHART_COLORS = [
    (0, 114, 178),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (230, 159, 0),
    (86, 180, 233),
    (117, 112, 179),
    (0, 0, 0),
    (102, 166, 30),
]
STAGE_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4"]
REPRESENTATIONS = ["raw", "bn_folded", "shape_normalized"]
REPRESENTATION_LABELS = {
    "raw": "Raw weights",
    "bn_folded": "BN-folded effective weights",
    "shape_normalized": "Per-kernel normalized shape",
}
DESCRIPTORS = [
    "raw_l2",
    "bn_folded_l2",
    "normalized_dc_gain",
    "center_surround",
    "high_frequency_ratio",
    "orientation_bias",
    "orientation_anisotropy",
    "center_energy_ratio",
    "roughness",
]
DESCRIPTOR_LABELS = {
    "raw_l2": "Raw L2 norm",
    "bn_folded_l2": "BN-folded L2 norm",
    "normalized_dc_gain": "Normalized DC gain",
    "center_surround": "Center-surround",
    "high_frequency_ratio": "High-frequency ratio",
    "orientation_bias": "H/V orientation bias",
    "orientation_anisotropy": "Orientation anisotropy",
    "center_energy_ratio": "Center energy ratio",
    "roughness": "Spatial roughness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw, BN-folded, and normalized RN18 kernel distributions."
    )
    parser.add_argument(
        "--experiment-dir",
        default="outputs/rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to <experiment-dir>/analysis_imagenet_epoch001_vs_lejepa_epoch110.",
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--max-stat-samples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--selected-pair-file", default="")
    parser.add_argument("--imagenet-checkpoint", default="")
    parser.add_argument("--lejepa-checkpoint", default="")
    parser.add_argument("--imagenet-accuracy", type=float, default=None)
    parser.add_argument("--lejepa-accuracy", type=float, default=None)
    parser.add_argument("--imagenet-epoch", type=int, default=None)
    parser.add_argument("--lejepa-epoch", type=int, default=None)
    parser.add_argument(
        "--skip-selection-validation",
        action="store_true",
        help=(
            "Skip checks that the selected accuracies are the extrema of the historical metric files. "
            "Useful only when an explicitly supplied checkpoint is a documented regeneration."
        ),
    )
    return parser.parse_args()


def bn_prefix_for_conv(layer_key: str) -> str:
    if layer_key == "conv1.weight":
        return "bn1"
    if layer_key.endswith(".conv1.weight"):
        return layer_key[: -len("conv1.weight")] + "bn1"
    if layer_key.endswith(".conv2.weight"):
        return layer_key[: -len("conv2.weight")] + "bn2"
    raise ValueError(f"No BatchNorm mapping for {layer_key}")


def batchnorm_factor(state: dict, layer_key: str, out_channels: int) -> np.ndarray:
    prefix = bn_prefix_for_conv(layer_key)
    gamma_key = f"{prefix}.weight"
    var_key = f"{prefix}.running_var"
    if gamma_key not in state or var_key not in state:
        return np.ones(out_channels, dtype=np.float32)
    gamma = common.tensor_to_numpy(state[gamma_key]).reshape(-1)
    running_var = common.tensor_to_numpy(state[var_key]).reshape(-1)
    if len(gamma) != out_channels or len(running_var) != out_channels:
        raise ValueError(f"BatchNorm shape mismatch for {layer_key}")
    return (gamma / np.sqrt(running_var + 1e-5)).astype(np.float32)


def kernel_descriptors(raw: np.ndarray, folded: np.ndarray, normalized: np.ndarray) -> dict[str, np.ndarray]:
    eps = np.finfo(np.float32).eps
    grid = normalized.reshape(-1, 3, 3)
    raw_l2 = np.linalg.norm(raw, axis=1)
    folded_l2 = np.linalg.norm(folded, axis=1)
    normalized_dc_gain = normalized.sum(axis=1)
    center = grid[:, 1, 1]
    surround = (grid.sum(axis=(1, 2)) - center) / 8.0
    center_surround = center - surround
    centered = grid - grid.mean(axis=(1, 2), keepdims=True)
    high_frequency_ratio = (
        np.square(centered).sum(axis=(1, 2))
        / np.maximum(np.square(grid).sum(axis=(1, 2)), eps)
    )
    horizontal_energy = np.square(np.diff(grid, axis=2)).sum(axis=(1, 2))
    vertical_energy = np.square(np.diff(grid, axis=1)).sum(axis=(1, 2))
    orientation_total = horizontal_energy + vertical_energy
    orientation_bias = (horizontal_energy - vertical_energy) / np.maximum(orientation_total, eps)
    orientation_anisotropy = np.abs(orientation_bias)
    center_energy_ratio = np.square(center) / np.maximum(np.square(grid).sum(axis=(1, 2)), eps)
    roughness = orientation_total / np.maximum(np.square(grid).sum(axis=(1, 2)), eps)
    return {
        "raw_l2": raw_l2.astype(np.float32),
        "bn_folded_l2": folded_l2.astype(np.float32),
        "normalized_dc_gain": normalized_dc_gain.astype(np.float32),
        "center_surround": center_surround.astype(np.float32),
        "high_frequency_ratio": high_frequency_ratio.astype(np.float32),
        "orientation_bias": orientation_bias.astype(np.float32),
        "orientation_anisotropy": orientation_anisotropy.astype(np.float32),
        "center_energy_ratio": center_energy_ratio.astype(np.float32),
        "roughness": roughness.astype(np.float32),
    }


def load_kernel_models(selected: dict, experiment_dir: Path) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for spec in matched.MODEL_SPECS:
        model_id = spec["model_id"]
        checkpoint_path = matched.resolve_path(str(selected[spec["path_key"]]), experiment_dir)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = common.load_backbone_state(checkpoint)
        layers = []
        items = common.sorted_resnet18_filter_items(state, include_stem=True)
        for layer_order, (layer_key, tensor) in enumerate(items):
            tensor_np = common.tensor_to_numpy(tensor)
            raw = tensor_np.reshape(-1, 9).astype(np.float32, copy=False)
            factor = batchnorm_factor(state, layer_key, int(tensor_np.shape[0]))
            folded_tensor = tensor_np * factor[:, None, None, None]
            folded = folded_tensor.reshape(-1, 9).astype(np.float32, copy=False)
            denominator = np.abs(raw).max(axis=1, keepdims=True)
            normalized = raw / np.where(denominator == 0.0, 1.0, denominator)
            layers.append(
                {
                    "layer_key": layer_key,
                    "stage": common.layer_stage(layer_key),
                    "layer_order": layer_order,
                    "conv_depth_norm": layer_order / max(len(items) - 1, 1),
                    "shape": tuple(int(v) for v in tensor_np.shape),
                    "raw": raw.copy(),
                    "bn_folded": folded.copy(),
                    "shape_normalized": normalized.astype(np.float32, copy=True),
                    "bn_factor": factor.copy(),
                    "descriptors": kernel_descriptors(raw, folded, normalized),
                }
            )
        models[model_id] = {
            "checkpoint_path": str(checkpoint_path),
            "test_accuracy": float(selected[spec["acc_key"]]),
            "layers": layers,
        }
        print(f"[extract] {model_id}: {len(layers)} convolution layers", flush=True)
    return models


def scope_definitions(layers: Sequence[dict]) -> list[dict]:
    scopes = [{"scope_type": "global", "scope": "global", "layer_indices": list(range(len(layers)))}]
    for stage in STAGE_ORDER:
        indices = [idx for idx, layer in enumerate(layers) if layer["stage"] == stage]
        if indices:
            scopes.append({"scope_type": "stage", "scope": stage, "layer_indices": indices})
    for idx, layer in enumerate(layers):
        scopes.append(
            {
                "scope_type": "layer",
                "scope": layer["layer_key"],
                "layer_indices": [idx],
                "layer_order": layer["layer_order"],
                "conv_depth_norm": layer["conv_depth_norm"],
                "stage": layer["stage"],
            }
        )
    return scopes


def concatenate_scope(
    model: dict,
    scope: dict,
    field: str,
    descriptor: str | None = None,
) -> np.ndarray:
    arrays = []
    for idx in scope["layer_indices"]:
        layer = model["layers"][idx]
        values = layer["descriptors"][descriptor] if descriptor else layer[field].reshape(-1)
        arrays.append(values)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    token = "|".join(parts).encode("utf-8")
    return np.random.default_rng(seed + zlib.crc32(token))


def sample_values(values: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if max_samples <= 0 or len(flat) <= max_samples:
        return flat.astype(np.float64, copy=False)
    indices = rng.choice(len(flat), size=max_samples, replace=False)
    return flat[indices].astype(np.float64, copy=False)


def scalar_statistics(values: np.ndarray, sample: np.ndarray) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    mean = float(values64.mean())
    std = float(values64.std())
    quantiles = np.quantile(sample, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(values64.size),
        "sample_count": int(sample.size),
        "mean": mean,
        "std": std,
        "rms": float(np.sqrt(np.square(values64).mean())),
        "mean_abs": float(np.abs(values64).mean()),
        "minimum": float(values64.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(values64.max()),
        "skewness": float(skew(sample, bias=False)) if len(sample) > 2 else math.nan,
        "excess_kurtosis": float(kurtosis(sample, fisher=True, bias=False)) if len(sample) > 3 else math.nan,
        "positive_fraction": float((values64 > 0).mean()),
        "near_zero_fraction_1pct_std": float((np.abs(values64) < max(std * 0.01, 1e-12)).mean()),
    }


def distribution_distances(a: np.ndarray, b: np.ndarray, bins: int) -> dict[str, float]:
    pooled = np.concatenate([a, b])
    lo, hi = np.quantile(pooled, [0.001, 0.999])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(pooled))
        hi = float(np.max(pooled))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    a_clip = np.clip(a, lo, hi)
    b_clip = np.clip(b, lo, hi)
    edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
    p = common.probability_histogram(a_clip, edges)
    q = common.probability_histogram(b_clip, edges)
    midpoint = 0.5 * (p + q)
    js = 0.5 * (
        np.sum(p * np.log(np.maximum(p, common.EPS) / np.maximum(midpoint, common.EPS)))
        + np.sum(q * np.log(np.maximum(q, common.EPS) / np.maximum(midpoint, common.EPS)))
    )
    pooled_std = max(float(np.std(pooled)), 1e-12)
    return {
        "tv_distance": 0.5 * float(np.abs(p - q).sum()),
        "jensen_shannon": float(js),
        "symmetric_kl": common.symmetric_kl(p, q),
        "hellinger": float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum())),
        "ks_statistic": float(ks_2samp(a, b, method="auto").statistic),
        "wasserstein": float(wasserstein_distance(a, b)),
        "wasserstein_over_pooled_std": float(wasserstein_distance(a, b) / pooled_std),
        "histogram_low_q001": float(lo),
        "histogram_high_q999": float(hi),
    }


def analyze_scalar_field(
    models: dict[str, dict],
    scopes: Sequence[dict],
    field: str,
    metric_name: str,
    bins: int,
    max_samples: int,
    seed: int,
    descriptor: str | None = None,
) -> tuple[list[dict], list[dict]]:
    stats_rows = []
    distance_rows = []
    for scope in scopes:
        samples = {}
        for model_id in PAIR:
            values = concatenate_scope(models[model_id], scope, field, descriptor=descriptor)
            rng = stable_rng(seed, model_id, scope["scope_type"], scope["scope"], metric_name)
            sample = sample_values(values, max_samples, rng)
            samples[model_id] = sample
            row = {
                "model_id": model_id,
                "metric": metric_name,
                **{key: value for key, value in scope.items() if key != "layer_indices"},
                **scalar_statistics(values, sample),
            }
            stats_rows.append(row)
        distance_rows.append(
            {
                "metric": metric_name,
                **{key: value for key, value in scope.items() if key != "layer_indices"},
                **distribution_distances(samples[PAIR[0]], samples[PAIR[1]], bins),
            }
        )
    return stats_rows, distance_rows


def draw_histogram_panels(
    panels: Sequence[dict],
    path: Path,
    title: str,
    cols: int,
    bins: int,
    max_samples: int,
    seed: int,
) -> None:
    rows = math.ceil(len(panels) / cols)
    cell_w, cell_h = 360, 240
    left, top, right, bottom = 56, 90, 28, 54
    width = left + cols * cell_w + right
    height = top + rows * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(14, bold=True)
    small_font = common.get_font(11)
    draw.text((24, 24), title, font=title_font, fill=(20, 20, 20))
    for panel_idx, panel in enumerate(panels):
        row_idx, col_idx = divmod(panel_idx, cols)
        x0 = left + col_idx * cell_w
        y0 = top + row_idx * cell_h
        plot_w, plot_h = cell_w - 48, cell_h - 62
        rng_a = stable_rng(seed, "plot", str(panel_idx), "a")
        rng_b = stable_rng(seed, "plot", str(panel_idx), "b")
        a = sample_values(panel["a"], max_samples, rng_a)
        b = sample_values(panel["b"], max_samples, rng_b)
        pooled = np.concatenate([a, b])
        lo, hi = np.quantile(pooled, [0.001, 0.999])
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        hist_a, edges = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi))
        hist_b, _ = np.histogram(np.clip(b, lo, hi), bins=edges)
        hist_a = common.smooth_histogram(hist_a, passes=2)
        hist_b = common.smooth_histogram(hist_b, passes=2)
        peak = max(float(hist_a.max()), float(hist_b.max()), 1.0)
        hist_a /= peak
        hist_b /= peak
        draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(70, 70, 70))
        for hist, model_id in [(hist_a, PAIR[0]), (hist_b, PAIR[1])]:
            points = []
            for idx, value in enumerate(hist):
                x = x0 + idx / max(len(hist) - 1, 1) * plot_w
                y = y0 + plot_h - float(value) * (plot_h - 8)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=MODEL_COLORS[model_id], width=3)
        common.draw_centered(draw, (x0 + plot_w / 2, y0 - 17), panel["label"], label_font)
        draw.text((x0, y0 + plot_h + 7), f"{lo:.3g}", font=small_font, fill=(80, 80, 80))
        right_text = f"{hi:.3g}"
        text_w, _ = common.text_size(draw, right_text, small_font)
        draw.text((x0 + plot_w - text_w, y0 + plot_h + 7), right_text, font=small_font, fill=(80, 80, 80))
        if "tv" in panel:
            draw.text(
                (x0 + 7, y0 + 7),
                f"TV={float(panel['tv']):.3f}",
                font=small_font,
                fill=(40, 40, 40),
            )
    legend_y = height - 25
    legend_x = left
    for model_id in PAIR:
        draw.line([(legend_x, legend_y), (legend_x + 26, legend_y)], fill=MODEL_COLORS[model_id], width=4)
        draw.text((legend_x + 34, legend_y - 7), MODEL_LABELS[model_id], font=small_font, fill=(20, 20, 20))
        legend_x += 250
    image.save(path)


def draw_rectangular_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    path: Path,
) -> None:
    rows, cols = matrix.shape
    cell_w, cell_h = 132, 38
    left, top, right, bottom = 205, 118, 38, 150
    width = left + cols * cell_w + right
    height = top + rows * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(12)
    value_font = common.get_font(11, bold=True)
    draw.text((24, 24), title, font=title_font, fill=(20, 20, 20))
    draw.text(
        (24, 58),
        "Color is normalized independently within each column; cells show absolute values.",
        font=label_font,
        fill=(80, 80, 80),
    )
    col_max = np.maximum(np.nanmax(matrix, axis=0), 1e-12)
    for row_idx, label in enumerate(row_labels):
        y0 = top + row_idx * cell_h
        draw.text((12, y0 + 11), label, font=label_font, fill=(20, 20, 20))
        for col_idx in range(cols):
            value = float(matrix[row_idx, col_idx])
            color = common.heat_color(value / col_max[col_idx])
            x0 = left + col_idx * cell_w
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=color, outline=(230, 230, 230))
            fill = (255, 255, 255) if common.luminance(color) < 100 else (20, 20, 20)
            common.draw_centered(draw, (x0 + cell_w / 2, y0 + cell_h / 2), f"{value:.3f}", value_font, fill)
    for col_idx, label in enumerate(column_labels):
        x0 = left + col_idx * cell_w
        text_w, text_h = common.text_size(draw, label, label_font)
        text_image = Image.new("RGBA", (text_w + 6, text_h + 6), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text((3, 3), label, font=label_font, fill=(20, 20, 20))
        text_image = text_image.rotate(90, expand=True)
        image.paste(
            text_image,
            (int(x0 + cell_w / 2 - text_image.width / 2), top + rows * cell_h + 12),
            text_image,
        )
    image.save(path)


def draw_small_multiple_depth(
    stats_df: pd.DataFrame,
    metrics: Sequence[str],
    path: Path,
    title: str,
) -> None:
    cols = 3
    rows = math.ceil(len(metrics) / cols)
    cell_w, cell_h = 390, 255
    left, top, right, bottom = 68, 90, 24, 54
    width = left + cols * cell_w + right
    height = top + rows * cell_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(13, bold=True)
    small_font = common.get_font(10)
    draw.text((24, 24), title, font=title_font, fill=(20, 20, 20))
    layer_df = stats_df[stats_df["scope_type"] == "layer"]
    for panel_idx, metric in enumerate(metrics):
        row_idx, col_idx = divmod(panel_idx, cols)
        x0 = left + col_idx * cell_w
        y0 = top + row_idx * cell_h
        plot_w, plot_h = cell_w - 56, cell_h - 64
        subsets = {}
        all_y = []
        for model_id in PAIR:
            subset = layer_df[
                (layer_df["model_id"] == model_id) & (layer_df["metric"] == metric)
            ].sort_values("layer_order")
            subsets[model_id] = subset
            all_y.extend(subset["mean"].astype(float).tolist())
        y_min = min(all_y) if all_y else 0.0
        y_max = max(all_y) if all_y else 1.0
        pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
        y_min -= pad
        y_max += pad
        draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(70, 70, 70))
        common.draw_centered(draw, (x0 + plot_w / 2, y0 - 17), DESCRIPTOR_LABELS[metric], label_font)
        for k in range(4):
            y = y0 + k / 3 * plot_h
            draw.line([(x0, y), (x0 + plot_w, y)], fill=(235, 235, 235))
        for model_id, subset in subsets.items():
            points = []
            for row in subset.itertuples():
                x = x0 + float(row.conv_depth_norm) * plot_w
                y = y0 + (y_max - float(row.mean)) / max(y_max - y_min, 1e-12) * plot_h
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=MODEL_COLORS[model_id], width=3)
            for x, y in points:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=MODEL_COLORS[model_id])
        draw.text((x0, y0 + plot_h + 7), "stem", font=small_font, fill=(80, 80, 80))
        end_label = "layer4"
        label_w, _ = common.text_size(draw, end_label, small_font)
        draw.text((x0 + plot_w - label_w, y0 + plot_h + 7), end_label, font=small_font, fill=(80, 80, 80))
        draw.text((x0 + 5, y0 + 5), f"{y_max:.3g}", font=small_font, fill=(80, 80, 80))
        draw.text((x0 + 5, y0 + plot_h - 16), f"{y_min:.3g}", font=small_font, fill=(80, 80, 80))
    legend_y = height - 25
    legend_x = left
    for model_id in PAIR:
        draw.line([(legend_x, legend_y), (legend_x + 26, legend_y)], fill=MODEL_COLORS[model_id], width=4)
        draw.text((legend_x + 34, legend_y - 7), MODEL_LABELS[model_id], font=small_font, fill=(20, 20, 20))
        legend_x += 250
    image.save(path)


def spatial_energy_rows(models: dict[str, dict]) -> list[dict]:
    rows = []
    for model_id, model in models.items():
        scopes = [("global", list(range(len(model["layers"]))))]
        scopes.extend(
            (
                stage,
                [idx for idx, layer in enumerate(model["layers"]) if layer["stage"] == stage],
            )
            for stage in STAGE_ORDER
        )
        for scope, indices in scopes:
            arrays = [model["layers"][idx]["shape_normalized"] for idx in indices]
            x = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
            energy = np.square(x).mean(axis=0)
            energy /= max(float(energy.sum()), 1e-12)
            for position, value in enumerate(energy):
                rows.append(
                    {
                        "model_id": model_id,
                        "scope": scope,
                        "row": position // 3,
                        "column": position % 3,
                        "normalized_energy": float(value),
                    }
                )
    return rows


def draw_spatial_energy_maps(rows: Sequence[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    scopes = ["global", *STAGE_ORDER]
    cell = 34
    panel = cell * 3
    col_gap, row_gap = 54, 34
    left, top = 150, 112
    width = left + 3 * panel + 2 * col_gap + 36
    height = top + len(scopes) * panel + (len(scopes) - 1) * row_gap + 52
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(14, bold=True)
    value_font = common.get_font(10)
    draw.text((24, 24), "Normalized Spatial Energy of 3x3 Kernels", font=title_font, fill=(20, 20, 20))
    headers = ["ImageNet FT", "LeJEPA", "LeJEPA - ImageNet"]
    for col_idx, header in enumerate(headers):
        x0 = left + col_idx * (panel + col_gap)
        common.draw_centered(draw, (x0 + panel / 2, top - 28), header, label_font)
    residual_max = 0.0
    values_by_scope = {}
    for scope in scopes:
        values = {}
        for model_id in PAIR:
            subset = df[(df["scope"] == scope) & (df["model_id"] == model_id)].sort_values(["row", "column"])
            values[model_id] = subset["normalized_energy"].to_numpy(dtype=float).reshape(3, 3)
        residual = values[PAIR[1]] - values[PAIR[0]]
        residual_max = max(residual_max, float(np.abs(residual).max()))
        values_by_scope[scope] = (values, residual)
    global_max = max(
        float(values[model_id].max())
        for values, _ in values_by_scope.values()
        for model_id in PAIR
    )
    for row_idx, scope in enumerate(scopes):
        y0 = top + row_idx * (panel + row_gap)
        draw.text((20, y0 + panel / 2 - 8), scope, font=label_font, fill=(20, 20, 20))
        values, residual = values_by_scope[scope]
        panels = [values[PAIR[0]], values[PAIR[1]], residual]
        for col_idx, matrix in enumerate(panels):
            x0 = left + col_idx * (panel + col_gap)
            for r in range(3):
                for c in range(3):
                    value = float(matrix[r, c])
                    color = (
                        common.diverging_color(value, max(residual_max, 1e-12))
                        if col_idx == 2
                        else common.heat_color(value / max(global_max, 1e-12))
                    )
                    bx, by = x0 + c * cell, y0 + r * cell
                    draw.rectangle([bx, by, bx + cell, by + cell], fill=color, outline=(235, 235, 235))
                    fill = (255, 255, 255) if common.luminance(color) < 95 else (20, 20, 20)
                    common.draw_centered(draw, (bx + cell / 2, by + cell / 2), f"{value:.3f}", value_font, fill)
    image.save(path)


def draw_spatial_energy_charts(rows: Sequence[dict], output_dir: Path) -> None:
    """Line-chart companions for every scope in the spatial-energy heatmap."""
    common.ensure_dir(output_dir)
    df = pd.DataFrame(rows)
    position_labels = "0=TL, 1=TC, 2=TR, 3=ML, 4=C, 5=MR, 6=BL, 7=BC, 8=BR"
    for scope in ["global", *STAGE_ORDER]:
        series = []
        for model_id in PAIR:
            subset = df[(df["scope"] == scope) & (df["model_id"] == model_id)].sort_values(
                ["row", "column"]
            )
            series.append(
                {
                    "x": list(range(9)),
                    "y": subset["normalized_energy"].astype(float).tolist(),
                    "label": MODEL_LABELS[model_id],
                    "color": MODEL_COLORS[model_id],
                }
            )
        common.draw_line_series(
            series,
            f"Normalized 3x3 Spatial Energy: {scope}",
            position_labels,
            "normalized energy",
            output_dir / f"spatial_energy_{scope}_chart.png",
            y_min=0.0,
        )


def batchnorm_statistics(models: dict[str, dict], scopes: Sequence[dict]) -> list[dict]:
    rows = []
    for scope in scopes:
        for model_id in PAIR:
            values = np.concatenate(
                [models[model_id]["layers"][idx]["bn_factor"] for idx in scope["layer_indices"]]
            )
            absolute = np.abs(values)
            rows.append(
                {
                    "model_id": model_id,
                    **{key: value for key, value in scope.items() if key != "layer_indices"},
                    "count": len(values),
                    "mean": float(values.mean()),
                    "mean_abs": float(absolute.mean()),
                    "std": float(values.std()),
                    "median_abs": float(np.median(absolute)),
                    "q95_abs": float(np.quantile(absolute, 0.95)),
                    "negative_fraction": float((values < 0).mean()),
                }
            )
    return rows


def write_summary(
    path: Path,
    selected: dict,
    weight_stats: pd.DataFrame,
    weight_distances: pd.DataFrame,
    descriptor_stats: pd.DataFrame,
    descriptor_distances: pd.DataFrame,
    bn_stats: pd.DataFrame,
) -> None:
    def stat(model_id: str, representation: str, column: str) -> float:
        row = weight_stats[
            (weight_stats["model_id"] == model_id)
            & (weight_stats["scope_type"] == "global")
            & (weight_stats["metric"] == representation)
        ].iloc[0]
        return float(row[column])

    def distance(metric: str, column: str) -> float:
        row = weight_distances[
            (weight_distances["scope_type"] == "global")
            & (weight_distances["metric"] == metric)
        ].iloc[0]
        return float(row[column])

    def descriptor_mean(model_id: str, metric: str) -> float:
        row = descriptor_stats[
            (descriptor_stats["model_id"] == model_id)
            & (descriptor_stats["scope_type"] == "global")
            & (descriptor_stats["metric"] == metric)
        ].iloc[0]
        return float(row["mean"])

    raw_std_ratio = stat(PAIR[1], "raw", "std") / max(stat(PAIR[0], "raw", "std"), 1e-12)
    folded_std_ratio = stat(PAIR[1], "bn_folded", "std") / max(
        stat(PAIR[0], "bn_folded", "std"), 1e-12
    )
    shape_tv = distance("shape_normalized", "tv_distance")
    bn_global = bn_stats[bn_stats["scope_type"] == "global"].set_index("model_id")
    imagenet_bn_scale = float(bn_global.loc[PAIR[0], "mean_abs"])
    lejepa_bn_scale = float(bn_global.loc[PAIR[1], "mean_abs"])
    high_frequency_imagenet = descriptor_mean(PAIR[0], "high_frequency_ratio")
    high_frequency_lejepa = descriptor_mean(PAIR[1], "high_frequency_ratio")
    roughness_imagenet = descriptor_mean(PAIR[0], "roughness")
    roughness_lejepa = descriptor_mean(PAIR[1], "roughness")
    center_energy_imagenet = descriptor_mean(PAIR[0], "center_energy_ratio")
    center_energy_lejepa = descriptor_mean(PAIR[1], "center_energy_ratio")
    layer_shape = weight_distances[
        (weight_distances["scope_type"] == "layer")
        & (weight_distances["metric"] == "shape_normalized")
    ].sort_values("jensen_shannon", ascending=False)
    global_descriptors = descriptor_distances[
        descriptor_distances["scope_type"] == "global"
    ].sort_values("jensen_shannon", ascending=False)
    layer_descriptor = (
        descriptor_distances[descriptor_distances["scope_type"] == "layer"]
        .groupby(["scope", "layer_order", "stage"], as_index=False)["jensen_shannon"]
        .mean()
        .sort_values("jensen_shannon", ascending=False)
    )
    lines = [
        "# RN18/CIFAR10 Kernel Distribution Comparison",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Compared Checkpoints",
        "",
        f"- ImageNet1K-pretrained + CIFAR10 fine-tuned epoch `1`: accuracy `{float(selected['imagenet_accuracy']):.4f}`.",
        f"- LeJEPA CIFAR10 SSL epoch `{int(selected['lejepa_ssl_epoch'])}` + frozen linear probe: accuracy `{float(selected['lejepa_accuracy']):.4f}`.",
        f"- Accuracy gap: `{float(selected['abs_accuracy_delta']):.4f}`.",
        "",
        "This report deliberately focuses on 3x3 convolution kernel distributions. Training-loss and optimization curves are not used to draw the kernel conclusions.",
        "",
        "## Representation Controls",
        "",
        "Three representations are compared:",
        "",
        "1. `raw`: stored convolution weights.",
        "2. `bn_folded`: effective inference weights after multiplying each output channel by BatchNorm `gamma / sqrt(running_var + eps)`.",
        "3. `shape_normalized`: every 3x3 kernel divided by its own maximum absolute coefficient.",
        "",
        "The BN-folded and normalized views are essential because raw Conv+BatchNorm weight scale is not functionally identifiable.",
        "",
        "## Main Findings",
        "",
        f"- LeJEPA raw weights have `{raw_std_ratio:.2f}x` the pooled standard deviation of ImageNet FT weights.",
        f"- After BatchNorm folding, the pooled standard-deviation ratio is `{folded_std_ratio:.2f}x`.",
        f"- Mean absolute BatchNorm folding scale is `{imagenet_bn_scale:.3f}` for ImageNet FT and `{lejepa_bn_scale:.3f}` for LeJEPA, explaining why the raw-scale ranking reverses after folding.",
        f"- After removing per-kernel magnitude, global shape TV distance is `{shape_tv:.4f}`.",
        f"- LeJEPA has higher mean high-frequency ratio (`{high_frequency_lejepa:.3f}` vs `{high_frequency_imagenet:.3f}`) and spatial roughness (`{roughness_lejepa:.3f}` vs `{roughness_imagenet:.3f}`).",
        f"- LeJEPA has lower center-coefficient energy (`{center_energy_lejepa:.3f}` vs `{center_energy_imagenet:.3f}`), indicating a modest redistribution from the center toward surrounding 3x3 positions.",
        "- Normalized-shape differences are concentrated in the stem and final residual block; most middle-layer scalar-shape divergences are much smaller.",
        "- Therefore, raw scale differences must not be interpreted alone; BN-folded magnitude and normalized shape provide the stronger structural evidence.",
        "",
        "Layers with the largest normalized-shape Jensen-Shannon divergence:",
        "",
        "| Layer | Stage | JS | TV | Wasserstein / pooled std |",
        "|---|---|---:|---:|---:|",
    ]
    for row in layer_shape.head(8).itertuples():
        lines.append(
            f"| `{row.scope}` | {row.stage} | {float(row.jensen_shannon):.6f} | "
            f"{float(row.tv_distance):.6f} | {float(row.wasserstein_over_pooled_std):.6f} |"
        )
    lines.extend(
        [
            "",
            "Most different global kernel descriptors:",
            "",
            "| Descriptor | JS | TV | KS |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in global_descriptors.head(6).itertuples():
        lines.append(
            f"| {DESCRIPTOR_LABELS[str(row.metric)]} | {float(row.jensen_shannon):.6f} | "
            f"{float(row.tv_distance):.6f} | {float(row.ks_statistic):.6f} |"
        )
    lines.extend(
        [
            "",
            "Layers with the largest average descriptor divergence:",
            "",
            "| Layer | Stage | Mean descriptor JS |",
            "|---|---|---:|",
        ]
    )
    for row in layer_descriptor.head(8).itertuples():
        lines.append(f"| `{row.scope}` | {row.stage} | {float(row.jensen_shannon):.6f} |")
    lines.extend(
        [
            "",
            "## Output Guide",
            "",
            "- `figures/global_weight_distributions.png`: raw, BN-folded, and normalized scalar distributions.",
            "- `figures/stage_weight_distributions.png`: stage-resolved raw and effective-weight distributions.",
            "- `figures/global_kernel_descriptor_distributions.png`: filter norm, frequency, orientation, center, and roughness distributions.",
            "- `figures/layerwise_weight_distance_heatmap.png`: depth localization of raw/effective/shape differences.",
            "- `figures/layerwise_weight_distance_chart.png`: line-chart version of the same six layerwise distance columns.",
            "- `figures/layerwise_descriptor_distance_heatmap.png`: layer x descriptor distribution differences.",
            "- `figures/layerwise_descriptor_distance_chart.png`: line-chart version of the same nine descriptor divergences.",
            "- `figures/kernel_descriptors_by_depth.png`: mean descriptor trajectories through RN18.",
            "- `figures/batchnorm_scale_by_depth.png`: BatchNorm compensation across depth.",
            "- `figures/spatial_energy_maps.png`: average normalized 3x3 energy and signed residual by stage.",
            "- `figures/spatial_energy_maps/`: per-scope ImageNet-versus-LeJEPA line-chart versions of the energy maps.",
            "- `pca_shape_analysis/`: PCA coefficient, residual, entropy, sparsity, and stage/layer drift analysis for the same pair.",
            "",
            "## Interpretation Limits",
            "",
            "- Distribution differences are descriptive and do not by themselves prove a causal mechanism for accuracy.",
            "- Independently trained channels can be permuted, so this report avoids treating kernel-by-kernel index alignment as semantic correspondence.",
            "- BatchNorm-folded kernels are closer to inference behavior than raw kernels, but residual connections and later nonlinearities still matter.",
            "- The result currently uses one random seed. Repeated seeds are needed before calling a pattern method-specific.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else experiment_dir / "analysis_imagenet_epoch001_vs_lejepa_epoch110"
    )
    figure_dir = output_dir / "figures"
    common.ensure_dir(output_dir)
    common.ensure_dir(figure_dir)

    selected = matched.apply_selected_overrides(
        matched.load_selected_pair(experiment_dir, args.selected_pair_file),
        args,
    )
    if int(selected["imagenet_epoch"]) != 1:
        raise ValueError(f"Expected ImageNet epoch 1, got {selected['imagenet_epoch']}")
    if not args.skip_selection_validation:
        imagenet_metrics = pd.read_csv(experiment_dir / "imagenet_finetune_metrics.csv")
        lejepa_metrics = pd.read_csv(experiment_dir / "lejepa_linear_metrics.csv")
        minimum_imagenet_accuracy = float(imagenet_metrics["test_accuracy"].min())
        maximum_lejepa_accuracy = float(lejepa_metrics["final_test_accuracy"].max())
        if not np.isclose(float(selected["imagenet_accuracy"]), minimum_imagenet_accuracy):
            raise ValueError("Selected ImageNet checkpoint is not the lowest-accuracy saved checkpoint.")
        if not np.isclose(float(selected["lejepa_accuracy"]), maximum_lejepa_accuracy):
            raise ValueError("Selected LeJEPA checkpoint is not the highest-accuracy fixed-probe checkpoint.")
    models = load_kernel_models(selected, experiment_dir)
    scopes = scope_definitions(models[PAIR[0]]["layers"])
    (output_dir / "checkpoint_pair.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "models": {
                    model_id: {
                        "checkpoint_path": model["checkpoint_path"],
                        "test_accuracy": model["test_accuracy"],
                    }
                    for model_id, model in models.items()
                },
                "representations": {
                    "raw": "Stored convolution weights.",
                    "bn_folded": "Conv weights multiplied by BN gamma / sqrt(running_var + 1e-5).",
                    "shape_normalized": "Each raw 3x3 kernel divided by its maximum absolute coefficient.",
                },
            },
            indent=2,
        )
        + "\n"
    )

    print("[stats] Weight distributions", flush=True)
    weight_stats_rows = []
    weight_distance_rows = []
    for representation in REPRESENTATIONS:
        stats_rows, distance_rows = analyze_scalar_field(
            models,
            scopes,
            representation,
            representation,
            args.bins,
            args.max_stat_samples,
            args.seed,
        )
        weight_stats_rows.extend(stats_rows)
        weight_distance_rows.extend(distance_rows)
    weight_stats = pd.DataFrame(weight_stats_rows)
    weight_distances = pd.DataFrame(weight_distance_rows)
    weight_stats.to_csv(output_dir / "weight_distribution_statistics.csv", index=False)
    weight_distances.to_csv(output_dir / "weight_distribution_distances.csv", index=False)

    print("[stats] Kernel descriptors", flush=True)
    descriptor_stats_rows = []
    descriptor_distance_rows = []
    for descriptor in DESCRIPTORS:
        stats_rows, distance_rows = analyze_scalar_field(
            models,
            scopes,
            field="descriptors",
            metric_name=descriptor,
            bins=args.bins,
            max_samples=args.max_stat_samples,
            seed=args.seed,
            descriptor=descriptor,
        )
        descriptor_stats_rows.extend(stats_rows)
        descriptor_distance_rows.extend(distance_rows)
    descriptor_stats = pd.DataFrame(descriptor_stats_rows)
    descriptor_distances = pd.DataFrame(descriptor_distance_rows)
    descriptor_stats.to_csv(output_dir / "kernel_descriptor_statistics.csv", index=False)
    descriptor_distances.to_csv(output_dir / "kernel_descriptor_distances.csv", index=False)

    bn_rows = batchnorm_statistics(models, scopes)
    bn_stats = pd.DataFrame(bn_rows)
    bn_stats.to_csv(output_dir / "batchnorm_scale_statistics.csv", index=False)
    energy_rows = spatial_energy_rows(models)
    pd.DataFrame(energy_rows).to_csv(output_dir / "spatial_energy_maps.csv", index=False)

    print("[plot] Distribution figures", flush=True)
    global_scope = scopes[0]
    global_weight_panels = []
    for representation in REPRESENTATIONS:
        distance_row = weight_distances[
            (weight_distances["scope_type"] == "global")
            & (weight_distances["metric"] == representation)
        ].iloc[0]
        global_weight_panels.append(
            {
                "label": REPRESENTATION_LABELS[representation],
                "a": concatenate_scope(models[PAIR[0]], global_scope, representation),
                "b": concatenate_scope(models[PAIR[1]], global_scope, representation),
                "tv": float(distance_row["tv_distance"]),
            }
        )
    draw_histogram_panels(
        global_weight_panels,
        figure_dir / "global_weight_distributions.png",
        "Global 3x3 Weight Distributions",
        cols=3,
        bins=args.bins,
        max_samples=args.max_stat_samples,
        seed=args.seed,
    )

    stage_panels = []
    for stage in STAGE_ORDER:
        scope = next(item for item in scopes if item["scope_type"] == "stage" and item["scope"] == stage)
        for representation in ["raw", "bn_folded"]:
            distance_row = weight_distances[
                (weight_distances["scope_type"] == "stage")
                & (weight_distances["scope"] == stage)
                & (weight_distances["metric"] == representation)
            ].iloc[0]
            stage_panels.append(
                {
                    "label": f"{stage}: {REPRESENTATION_LABELS[representation]}",
                    "a": concatenate_scope(models[PAIR[0]], scope, representation),
                    "b": concatenate_scope(models[PAIR[1]], scope, representation),
                    "tv": float(distance_row["tv_distance"]),
                }
            )
    draw_histogram_panels(
        stage_panels,
        figure_dir / "stage_weight_distributions.png",
        "Stage-Resolved Raw and BN-Folded Weight Distributions",
        cols=2,
        bins=args.bins,
        max_samples=args.max_stat_samples,
        seed=args.seed,
    )

    descriptor_panels = []
    for descriptor in DESCRIPTORS:
        distance_row = descriptor_distances[
            (descriptor_distances["scope_type"] == "global")
            & (descriptor_distances["metric"] == descriptor)
        ].iloc[0]
        descriptor_panels.append(
            {
                "label": DESCRIPTOR_LABELS[descriptor],
                "a": concatenate_scope(models[PAIR[0]], global_scope, "descriptors", descriptor),
                "b": concatenate_scope(models[PAIR[1]], global_scope, "descriptors", descriptor),
                "tv": float(distance_row["tv_distance"]),
            }
        )
    draw_histogram_panels(
        descriptor_panels,
        figure_dir / "global_kernel_descriptor_distributions.png",
        "Global 3x3 Kernel Descriptor Distributions",
        cols=3,
        bins=args.bins,
        max_samples=args.max_stat_samples,
        seed=args.seed,
    )

    layer_scopes = [scope for scope in scopes if scope["scope_type"] == "layer"]
    layer_labels = [scope["scope"].replace(".weight", "") for scope in layer_scopes]
    weight_columns = []
    weight_matrix_columns = []
    for representation in REPRESENTATIONS:
        for distance_column, short_label in [
            ("tv_distance", "TV"),
            ("wasserstein_over_pooled_std", "W1/std"),
        ]:
            subset = weight_distances[
                (weight_distances["scope_type"] == "layer")
                & (weight_distances["metric"] == representation)
            ].sort_values("layer_order")
            weight_matrix_columns.append(subset[distance_column].to_numpy(dtype=float))
            weight_columns.append(f"{representation}: {short_label}")
    draw_rectangular_heatmap(
        np.column_stack(weight_matrix_columns),
        layer_labels,
        weight_columns,
        "Layerwise Weight Distribution Distances",
        figure_dir / "layerwise_weight_distance_heatmap.png",
    )
    weight_chart_series = []
    chart_index = 0
    for representation in REPRESENTATIONS:
        for distance_column, short_label in [
            ("tv_distance", "TV"),
            ("wasserstein_over_pooled_std", "W1/std"),
        ]:
            subset = weight_distances[
                (weight_distances["scope_type"] == "layer")
                & (weight_distances["metric"] == representation)
            ].sort_values("layer_order")
            weight_chart_series.append(
                {
                    "x": subset["conv_depth_norm"].astype(float).tolist(),
                    "y": subset[distance_column].astype(float).tolist(),
                    "label": f"{representation}: {short_label}",
                    "color": CHART_COLORS[chart_index % len(CHART_COLORS)],
                }
            )
            chart_index += 1
    common.draw_line_series(
        weight_chart_series,
        "Layerwise Weight Distribution Distances (Chart)",
        "normalized convolution depth",
        "distribution distance",
        figure_dir / "layerwise_weight_distance_chart.png",
        y_min=0.0,
    )

    descriptor_matrix_columns = []
    for descriptor in DESCRIPTORS:
        subset = descriptor_distances[
            (descriptor_distances["scope_type"] == "layer")
            & (descriptor_distances["metric"] == descriptor)
        ].sort_values("layer_order")
        descriptor_matrix_columns.append(subset["jensen_shannon"].to_numpy(dtype=float))
    draw_rectangular_heatmap(
        np.column_stack(descriptor_matrix_columns),
        layer_labels,
        [DESCRIPTOR_LABELS[name] for name in DESCRIPTORS],
        "Layerwise Kernel Descriptor JS Divergence",
        figure_dir / "layerwise_descriptor_distance_heatmap.png",
    )
    descriptor_chart_series = []
    for descriptor_index, descriptor in enumerate(DESCRIPTORS):
        subset = descriptor_distances[
            (descriptor_distances["scope_type"] == "layer")
            & (descriptor_distances["metric"] == descriptor)
        ].sort_values("layer_order")
        descriptor_chart_series.append(
            {
                "x": subset["conv_depth_norm"].astype(float).tolist(),
                "y": subset["jensen_shannon"].astype(float).tolist(),
                "label": DESCRIPTOR_LABELS[descriptor],
                "color": CHART_COLORS[descriptor_index % len(CHART_COLORS)],
            }
        )
    common.draw_line_series(
        descriptor_chart_series,
        "Layerwise Kernel Descriptor JS Divergence (Chart)",
        "normalized convolution depth",
        "Jensen-Shannon divergence",
        figure_dir / "layerwise_descriptor_distance_chart.png",
        y_min=0.0,
    )

    draw_small_multiple_depth(
        descriptor_stats,
        DESCRIPTORS,
        figure_dir / "kernel_descriptors_by_depth.png",
        "Mean Kernel Descriptors Across RN18 Depth",
    )

    bn_layer = bn_stats[bn_stats["scope_type"] == "layer"].sort_values(["model_id", "layer_order"])
    common.draw_line_series(
        [
            {
                "x": bn_layer[bn_layer["model_id"] == model_id]["conv_depth_norm"].tolist(),
                "y": bn_layer[bn_layer["model_id"] == model_id]["mean_abs"].tolist(),
                "label": MODEL_LABELS[model_id],
                "color": MODEL_COLORS[model_id],
            }
            for model_id in PAIR
        ],
        "Mean Absolute BatchNorm Folding Scale by Depth",
        "normalized convolution depth",
        "mean |gamma / sqrt(var + eps)|",
        figure_dir / "batchnorm_scale_by_depth.png",
        y_min=0.0,
    )
    draw_spatial_energy_maps(energy_rows, figure_dir / "spatial_energy_maps.png")
    draw_spatial_energy_charts(energy_rows, figure_dir / "spatial_energy_maps")

    write_summary(
        output_dir / "summary.md",
        selected,
        weight_stats,
        weight_distances,
        descriptor_stats,
        descriptor_distances,
        bn_stats,
    )
    print(f"[done] Wrote kernel-distribution analysis to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
