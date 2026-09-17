#!/usr/bin/env python
"""Signed residual analysis: LeJepa SSL minus ImageNet FT RN50 filters.

This is a follow-up to ``analyze_rn50_filters.py``. It keeps the same RN50
3x3-filter extraction and PCA basis, but focuses only on where the two close
models differ directionally.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:
    torch = None
from PIL import Image, ImageDraw

import analyze_rn50_filters as base


MODEL_A = "imagenet_ft"
MODEL_B = "lejepa_ssl"
PAIR_SPECS = [spec for spec in base.MODEL_SPECS if spec["model_id"] in {MODEL_A, MODEL_B}]
DISPLAY = {spec["model_id"]: spec["display"] for spec in PAIR_SPECS}
COLORS = {spec["model_id"]: spec["color"] for spec in PAIR_SPECS}
EPS = np.finfo(np.float64).eps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze signed residuals between LeJepa SSL and ImageNet FT RN50 filters."
    )
    parser.add_argument("--model-dir", default="models", help="Directory with RN50 .pth files.")
    parser.add_argument(
        "--source-analysis-dir",
        default="outputs/rn50_filter_analysis",
        help="Previous RN50 analysis directory containing pca_target_rn50.npz.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rn50_lejepa_imagenet_residuals",
        help="Directory where residual CSVs and figures will be written.",
    )
    parser.add_argument("--bins", type=int, default=70, help="Histogram bins for residuals.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--max-density-points",
        type=int,
        default=600_000,
        help="Maximum points per model for c0/c1 residual density figure.",
    )
    parser.add_argument(
        "--top-layer-components",
        type=int,
        default=8,
        help="How many strongest layer/component residuals to summarize in plots.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate CSV-based figures and summary from existing output CSVs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pca(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PCA basis: {path}")
    data = np.load(path)
    required = ["components", "mean", "explained_variance_ratio"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    return {
        "components": data["components"].astype(np.float32),
        "mean": data["mean"].astype(np.float32),
        "weights": data["explained_variance_ratio"].astype(np.float64),
    }


def project_models(model_infos: dict[str, dict], pca_data: dict[str, np.ndarray]) -> None:
    components = pca_data["components"]
    mean = pca_data["mean"]
    for model_id, info in model_infos.items():
        scaled = base.scale_filters(info["X"]).astype(np.float32, copy=False)
        info["X_scaled"] = scaled
        info["Z"] = ((scaled - mean) @ components.T).astype(np.float32, copy=False)
        print(f"[pca] Projected {DISPLAY[model_id]}: {info['Z'].shape}", flush=True)


def component_edges(a: np.ndarray, b: np.ndarray, bins: int) -> np.ndarray:
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite PCA coefficient range")
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = (hi - lo) * 0.001
    return np.linspace(lo - pad, hi + pad, bins + 1, dtype=np.float64)


def probability_hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges)
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total == 0.0:
        return np.zeros_like(hist)
    return hist / total


def summarize_values(
    a_values: np.ndarray,
    b_values: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    weight: float,
) -> dict[str, float]:
    residual = p_b - p_a
    tv_distance = 0.5 * float(np.abs(residual).sum())
    p_safe = np.where(p_a == 0, EPS, p_a)
    q_safe = np.where(p_b == 0, EPS, p_b)
    kl_sym = float(base.scipy_entropy(p_safe, q_safe) + base.scipy_entropy(q_safe, p_safe))
    return {
        "mean_imagenet": float(np.mean(a_values)),
        "mean_lejepa": float(np.mean(b_values)),
        "mean_delta_lejepa_minus_imagenet": float(np.mean(b_values) - np.mean(a_values)),
        "std_imagenet": float(np.std(a_values)),
        "std_lejepa": float(np.std(b_values)),
        "std_delta_lejepa_minus_imagenet": float(np.std(b_values) - np.std(a_values)),
        "median_delta_lejepa_minus_imagenet": float(np.median(b_values) - np.median(a_values)),
        "q05_delta_lejepa_minus_imagenet": float(np.quantile(b_values, 0.05) - np.quantile(a_values, 0.05)),
        "q95_delta_lejepa_minus_imagenet": float(np.quantile(b_values, 0.95) - np.quantile(a_values, 0.95)),
        "positive_residual_mass": float(np.maximum(residual, 0).sum()),
        "negative_residual_mass": float(-np.minimum(residual, 0).sum()),
        "tv_distance": tv_distance,
        "weighted_tv_distance": float(weight * tv_distance),
        "symmetric_kl": kl_sym,
        "weighted_symmetric_kl": float(weight * kl_sym),
    }


def residual_histogram_rows(
    z_a: np.ndarray,
    z_b: np.ndarray,
    weights: np.ndarray,
    bins: int,
    prefix: dict,
) -> tuple[list[dict], list[dict]]:
    hist_rows: list[dict] = []
    summary_rows: list[dict] = []
    for component in range(z_a.shape[1]):
        a_values = z_a[:, component]
        b_values = z_b[:, component]
        edges = component_edges(a_values, b_values, bins)
        p_a = probability_hist(a_values, edges)
        p_b = probability_hist(b_values, edges)
        residual = p_b - p_a
        centers = (edges[:-1] + edges[1:]) / 2.0
        row = dict(prefix)
        row.update(
            {
                "component": component,
                "component_weight": float(weights[component]),
                **summarize_values(a_values, b_values, p_a, p_b, float(weights[component])),
            }
        )
        max_pos = int(np.argmax(residual))
        max_neg = int(np.argmin(residual))
        row.update(
            {
                "max_positive_bin_center": float(centers[max_pos]),
                "max_positive_residual": float(residual[max_pos]),
                "max_negative_bin_center": float(centers[max_neg]),
                "max_negative_residual": float(residual[max_neg]),
            }
        )
        summary_rows.append(row)
        for idx in range(bins):
            hist_row = dict(prefix)
            hist_row.update(
                {
                    "component": component,
                    "bin_index": idx,
                    "bin_left": float(edges[idx]),
                    "bin_right": float(edges[idx + 1]),
                    "bin_center": float(centers[idx]),
                    "prob_imagenet": float(p_a[idx]),
                    "prob_lejepa": float(p_b[idx]),
                    "residual_lejepa_minus_imagenet": float(residual[idx]),
                }
            )
            hist_rows.append(hist_row)
    return hist_rows, summary_rows


def layer_label(layer_key: str) -> str:
    parts = layer_key.split(".")
    if len(parts) >= 2 and parts[0].startswith("layer"):
        return f"l{parts[0][-1]}.{parts[1]}"
    return layer_key


def get_layer_pair(model_infos: dict[str, dict], layer_order: int) -> tuple[np.ndarray, np.ndarray, base.LayerSlice]:
    z_a, layer_slice = base.subset_by_layer(model_infos[MODEL_A], layer_order)
    z_b, _ = base.subset_by_layer(model_infos[MODEL_B], layer_order)
    return z_a, z_b, layer_slice


def stage_groups(model_infos: dict[str, dict], stage: str) -> tuple[np.ndarray, np.ndarray]:
    return base.subset_by_stage(model_infos[MODEL_A], stage), base.subset_by_stage(model_infos[MODEL_B], stage)


def quality_residual_rows(quality_rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(quality_rows)
    out: list[dict] = []
    metric_names = [
        "entropy_H",
        "H_over_TH",
        "sparsity_S",
        "mean_abs_weight",
        "std_weight",
        "abs_peak_weight",
    ]
    for layer_order in sorted(df["layer_order"].unique()):
        a = df[(df["model_id"] == MODEL_A) & (df["layer_order"] == layer_order)].iloc[0]
        b = df[(df["model_id"] == MODEL_B) & (df["layer_order"] == layer_order)].iloc[0]
        row = {
            "layer_order": int(layer_order),
            "layer_key": a["layer_key"],
            "layer_label": layer_label(a["layer_key"]),
            "stage": a["stage"],
            "conv_depth_norm": float(a["conv_depth_norm"]),
            "num_filters": int(a["num_filters"]),
            "random_threshold_TH": float(a["random_threshold_TH"]),
        }
        for metric in metric_names:
            row[f"{metric}_imagenet"] = float(a[metric])
            row[f"{metric}_lejepa"] = float(b[metric])
            row[f"{metric}_delta_lejepa_minus_imagenet"] = float(b[metric]) - float(a[metric])
        out.append(row)
    return out


def read_source_pair_drift(source_dir: Path) -> float | None:
    path = source_dir / "drift_global_pairs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    match = df[(df["group_a"] == MODEL_A) & (df["group_b"] == MODEL_B)]
    if match.empty:
        return None
    return float(match.iloc[0]["drift_D"])


def draw_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    path: Path,
    signed: bool = False,
) -> None:
    cell_w = 82
    cell_h = 42
    left = 168
    top = 94
    right = 34
    bottom = 56
    width = left + matrix.shape[1] * cell_w + right
    height = top + matrix.shape[0] * cell_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(14)
    small_font = base.get_font(12)
    value_font = base.get_font(12, bold=True)
    draw.text((22, 22), title, font=title_font, fill=(20, 20, 20))
    max_abs = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    max_value = float(np.max(matrix)) if matrix.size else 1.0
    max_value = max(max_value, 1e-12)

    for c, label in enumerate(col_labels):
        base.draw_centered(draw, (left + c * cell_w + cell_w / 2, top - 22), label, label_font)
    for r, label in enumerate(row_labels):
        draw.text((20, top + r * cell_h + cell_h / 2 - 8), label, font=label_font, fill=(20, 20, 20))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = float(matrix[r, c])
            if signed:
                color = base.diverging_color(value, max_abs)
            else:
                color = base.heat_color(value / max_value)
            x0 = left + c * cell_w
            y0 = top + r * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=color, outline=(220, 220, 220))
            fill = (255, 255, 255) if base.luminance(color) < 100 else (20, 20, 20)
            base.draw_centered(draw, (x0 + cell_w / 2, y0 + cell_h / 2), f"{value:.3f}", value_font, fill)

    if signed:
        draw.text((left, height - 34), f"blue: ImageNet FT larger, red: LeJepa SSL larger; max |delta|={max_abs:.4f}", font=small_font, fill=(80, 80, 80))
    else:
        draw.text((left, height - 34), f"larger values indicate stronger residual distribution difference; max={max_value:.4f}", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_component_summary(summary_df: pd.DataFrame, path: Path) -> None:
    width, height = 980, 520
    left = 76
    top = 78
    panel_gap = 64
    panel_w = (width - left - 48 - panel_gap) // 2
    panel_h = 350
    bottom = top + panel_h
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(14)
    small_font = base.get_font(12)
    draw.text((24, 24), "Global LeJepa - ImageNet Residual Summary by PCA Component", font=title_font, fill=(20, 20, 20))

    panels = [
        ("tv_distance", "TV distance", False),
        ("mean_delta_lejepa_minus_imagenet", "signed mean delta", True),
    ]
    for panel_idx, (metric, label, signed) in enumerate(panels):
        x0 = left + panel_idx * (panel_w + panel_gap)
        values = summary_df.sort_values("component")[metric].to_numpy(dtype=float)
        ymax = float(np.max(np.abs(values))) if signed else float(np.max(values))
        ymax = max(ymax, 1e-12)
        y_min = -ymax if signed else 0.0
        y_max = ymax
        zero_y = top + (y_max - 0.0) / (y_max - y_min) * panel_h
        draw.rectangle([x0, top, x0 + panel_w, bottom], outline=(50, 50, 50), width=1)
        for k in range(5):
            y = top + k / 4 * panel_h
            draw.line([(x0, y), (x0 + panel_w, y)], fill=(235, 235, 235))
        if signed:
            draw.line([(x0, zero_y), (x0 + panel_w, zero_y)], fill=(80, 80, 80), width=2)
        bar_gap = 7
        bar_w = (panel_w - 28 - 8 * bar_gap) / 9
        for component, value in enumerate(values):
            bx = x0 + 14 + component * (bar_w + bar_gap)
            by = top + (y_max - value) / (y_max - y_min) * panel_h
            if signed:
                color = (210, 60, 55) if value >= 0 else (55, 85, 210)
                draw.rectangle([bx, min(by, zero_y), bx + bar_w, max(by, zero_y)], fill=color)
            else:
                color = base.heat_color(value / ymax)
                draw.rectangle([bx, by, bx + bar_w, bottom], fill=color)
            base.draw_centered(draw, (bx + bar_w / 2, bottom + 16), str(component), small_font)
        base.draw_centered(draw, (x0 + panel_w / 2, top - 22), label, label_font)
    draw.text((left, height - 34), "Positive signed delta means larger LeJepa SSL coefficient mean.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_residual_hist_panels(
    hist_df: pd.DataFrame,
    top_df: pd.DataFrame,
    path: Path,
    max_panels: int = 8,
) -> None:
    selected = top_df.head(max_panels).copy()
    cols = 4
    rows = int(math.ceil(len(selected) / cols))
    panel_w = 260
    panel_h = 170
    left = 62
    top = 78
    gap_x = 32
    gap_y = 54
    width = left * 2 + cols * panel_w + (cols - 1) * gap_x
    height = top + rows * panel_h + (rows - 1) * gap_y + 58
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(13, bold=True)
    small_font = base.get_font(11)
    draw.text((24, 24), "Top Layer/Component Signed Histogram Residuals", font=title_font, fill=(20, 20, 20))

    for idx, row in enumerate(selected.itertuples(index=False)):
        r = idx // cols
        c = idx % cols
        x0 = left + c * (panel_w + gap_x)
        y0 = top + r * (panel_h + gap_y)
        subset = hist_df[
            (hist_df["layer_order"] == row.layer_order)
            & (hist_df["component"] == row.component)
        ].sort_values("bin_index")
        residual = subset["residual_lejepa_minus_imagenet"].to_numpy(dtype=float)
        centers = subset["bin_center"].to_numpy(dtype=float)
        max_abs = max(float(np.max(np.abs(residual))), 1e-12)
        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline=(50, 50, 50), width=1)
        zero_y = y0 + panel_h / 2
        draw.line([(x0, zero_y), (x0 + panel_w, zero_y)], fill=(70, 70, 70), width=1)
        for bin_idx, value in enumerate(residual):
            x = x0 + bin_idx / max(len(residual), 1) * panel_w
            bw = max(1, panel_w / max(len(residual), 1))
            y = zero_y - value / max_abs * (panel_h / 2 - 18)
            color = (210, 60, 55) if value >= 0 else (55, 85, 210)
            draw.rectangle([x, min(y, zero_y), x + bw, max(y, zero_y)], fill=color)
        label = f"{row.layer_label} c{int(row.component)} TV={row.tv_distance:.3f}"
        base.draw_centered(draw, (x0 + panel_w / 2, y0 - 18), label, label_font)
        draw.text((x0, y0 + panel_h + 8), f"x [{centers[0]:.2f}, {centers[-1]:.2f}]", font=small_font, fill=(80, 80, 80))
    draw.text((left, height - 32), "Red: LeJepa overrepresented. Blue: ImageNet overrepresented.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_quality_delta(quality_df: pd.DataFrame, path: Path) -> None:
    width, height = 980, 560
    left = 82
    right = 44
    top = 72
    bottom = 76
    panel_gap = 64
    panel_w = (width - left - right - panel_gap) / 2
    panel_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(14)
    small_font = base.get_font(12)
    draw.text((24, 24), "Layer Quality Deltas: LeJepa SSL - ImageNet FT", font=title_font, fill=(20, 20, 20))

    panels = [
        ("entropy_H_delta_lejepa_minus_imagenet", "entropy H delta"),
        ("sparsity_S_delta_lejepa_minus_imagenet", "sparsity S delta"),
    ]
    x_values = quality_df["conv_depth_norm"].to_numpy(dtype=float)
    for idx, (metric, label) in enumerate(panels):
        x0 = left + idx * (panel_w + panel_gap)
        y_values = quality_df[metric].to_numpy(dtype=float)
        y_abs = max(float(np.max(np.abs(y_values))), 1e-12)
        y_min, y_max = -1.08 * y_abs, 1.08 * y_abs
        def xmap(x: float) -> float:
            return x0 + x * panel_w
        def ymap(y: float) -> float:
            return top + (y_max - y) / (y_max - y_min) * panel_h
        draw.rectangle([x0, top, x0 + panel_w, top + panel_h], outline=(50, 50, 50), width=1)
        zero_y = ymap(0.0)
        draw.line([(x0, zero_y), (x0 + panel_w, zero_y)], fill=(80, 80, 80), width=2)
        for k in range(6):
            x = x0 + k / 5 * panel_w
            draw.line([(x, top), (x, top + panel_h)], fill=(238, 238, 238))
            base.draw_centered(draw, (x, top + panel_h + 22), f"{k/5:.1f}", small_font)
        points = [(xmap(float(x)), ymap(float(y))) for x, y in zip(x_values, y_values)]
        draw.line(points, fill=(0, 130, 105), width=3)
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 130, 105), outline="white")
        base.draw_centered(draw, (x0 + panel_w / 2, top - 22), label, label_font)
    draw.text((left, height - 34), "Positive values indicate larger LeJepa layer metric.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_c0_c1_residual_density(
    model_infos: dict[str, dict],
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    bins = 220
    groups: dict[str, np.ndarray] = {}
    for model_id in [MODEL_A, MODEL_B]:
        z = model_infos[model_id]["Z"][:, :2]
        if 0 < max_points < len(z):
            idx = rng.choice(len(z), size=max_points, replace=False)
            z = z[idx]
        groups[model_id] = z
    all_z = np.vstack(list(groups.values()))
    x_range = (float(all_z[:, 0].min()), float(all_z[:, 0].max()))
    y_range = (float(all_z[:, 1].min()), float(all_z[:, 1].max()))

    hists = {}
    probs = {}
    for model_id, z in groups.items():
        hist, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=bins, range=[x_range, y_range])
        hists[model_id] = hist.T
        probs[model_id] = hists[model_id] / max(float(hists[model_id].sum()), 1.0)
    residual = probs[MODEL_B] - probs[MODEL_A]
    max_abs = max(float(np.max(np.abs(residual))), 1e-12)

    panel = 300
    gap = 34
    left = 44
    top = 86
    width = left * 2 + 3 * panel + 2 * gap
    height = top + panel + 66
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(15, bold=True)
    small_font = base.get_font(12)
    draw.text((24, 24), "c0/c1 Density Residual: LeJepa SSL - ImageNet FT", font=title_font, fill=(20, 20, 20))

    panels = [
        (MODEL_A, "ImageNet FT density"),
        (MODEL_B, "LeJepa SSL density"),
        ("residual", "Residual"),
    ]
    for idx, (key, label) in enumerate(panels):
        if key == "residual":
            arr = np.zeros((bins, bins, 3), dtype=np.uint8)
            for level in range(256):
                lo = -1.0 + 2.0 * level / 256
                hi = -1.0 + 2.0 * (level + 1) / 256
                mask = (residual / max_abs >= lo) & (residual / max_abs < hi)
                arr[mask] = base.diverging_color((lo + hi) / 2, 1.0)
        else:
            density = np.log1p(hists[key])
            if density.max() > 0:
                density = density / density.max()
            arr = np.zeros((bins, bins, 3), dtype=np.uint8)
            for level in range(256):
                mask = (density >= level / 256) & (density < (level + 1) / 256)
                arr[mask] = base.density_color(level / 255)
        panel_img = Image.fromarray(np.flipud(arr), mode="RGB").resize((panel, panel), Image.Resampling.BILINEAR)
        x0 = left + idx * (panel + gap)
        img.paste(panel_img, (x0, top))
        draw.rectangle([x0, top, x0 + panel, top + panel], outline=(40, 40, 40), width=1)
        base.draw_centered(draw, (x0 + panel / 2, top - 22), label, label_font)
    draw.text((left, height - 34), f"x: c0 [{x_range[0]:.2f}, {x_range[1]:.2f}], y: c1 [{y_range[0]:.2f}, {y_range[1]:.2f}]", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_global_residual_histograms(
    hist_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    path: Path,
) -> None:
    selected = summary_df.sort_values("component")
    cols = 3
    rows = int(math.ceil(len(selected) / cols))
    panel_w = 286
    panel_h = 154
    left = 62
    top = 86
    gap_x = 36
    gap_y = 58
    width = left * 2 + cols * panel_w + (cols - 1) * gap_x
    height = top + rows * panel_h + (rows - 1) * gap_y + 58
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(13, bold=True)
    small_font = base.get_font(11)
    draw.text((24, 24), "Global Signed Residual Histograms by PCA Component", font=title_font, fill=(20, 20, 20))

    for idx, row in enumerate(selected.itertuples(index=False)):
        r = idx // cols
        c = idx % cols
        x0 = left + c * (panel_w + gap_x)
        y0 = top + r * (panel_h + gap_y)
        component = int(row.component)
        subset = hist_df[hist_df["component"] == component].sort_values("bin_index")
        residual = subset["residual_lejepa_minus_imagenet"].to_numpy(dtype=float)
        centers = subset["bin_center"].to_numpy(dtype=float)
        max_abs = max(float(np.max(np.abs(residual))), 1e-12)
        zero_y = y0 + panel_h / 2
        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline=(50, 50, 50), width=1)
        draw.line([(x0, zero_y), (x0 + panel_w, zero_y)], fill=(70, 70, 70), width=1)
        for bin_idx, value in enumerate(residual):
            x = x0 + bin_idx / max(len(residual), 1) * panel_w
            bw = max(1, panel_w / max(len(residual), 1))
            y = zero_y - value / max_abs * (panel_h / 2 - 16)
            color = (210, 60, 55) if value >= 0 else (55, 85, 210)
            draw.rectangle([x, min(y, zero_y), x + bw, max(y, zero_y)], fill=color)
        label = f"c{component} TV={row.tv_distance:.3f} wTV={row.weighted_tv_distance:.3f}"
        base.draw_centered(draw, (x0 + panel_w / 2, y0 - 18), label, label_font)
        draw.text((x0, y0 + panel_h + 8), f"x [{centers[0]:.2f}, {centers[-1]:.2f}]", font=small_font, fill=(80, 80, 80))
    draw.text((left, height - 32), "Red: LeJepa overrepresented. Blue: ImageNet overrepresented.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_stage_c0_residual_histograms(
    hist_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    path: Path,
) -> None:
    stages = ["layer1", "layer2", "layer3", "layer4"]
    panel_w = 280
    panel_h = 170
    left = 62
    top = 86
    gap_x = 34
    width = left * 2 + len(stages) * panel_w + (len(stages) - 1) * gap_x
    height = top + panel_h + 64
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(13, bold=True)
    small_font = base.get_font(11)
    draw.text((24, 24), "Stage-Level c0 Signed Residual Histograms", font=title_font, fill=(20, 20, 20))

    for idx, stage in enumerate(stages):
        x0 = left + idx * (panel_w + gap_x)
        y0 = top
        subset = hist_df[(hist_df["stage"] == stage) & (hist_df["component"] == 0)].sort_values("bin_index")
        summary = summary_df[(summary_df["stage"] == stage) & (summary_df["component"] == 0)].iloc[0]
        residual = subset["residual_lejepa_minus_imagenet"].to_numpy(dtype=float)
        centers = subset["bin_center"].to_numpy(dtype=float)
        max_abs = max(float(np.max(np.abs(residual))), 1e-12)
        zero_y = y0 + panel_h / 2
        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline=(50, 50, 50), width=1)
        draw.line([(x0, zero_y), (x0 + panel_w, zero_y)], fill=(70, 70, 70), width=1)
        for bin_idx, value in enumerate(residual):
            x = x0 + bin_idx / max(len(residual), 1) * panel_w
            bw = max(1, panel_w / max(len(residual), 1))
            y = zero_y - value / max_abs * (panel_h / 2 - 16)
            color = (210, 60, 55) if value >= 0 else (55, 85, 210)
            draw.rectangle([x, min(y, zero_y), x + bw, max(y, zero_y)], fill=color)
        label = f"{stage} c0 TV={float(summary.tv_distance):.3f}"
        base.draw_centered(draw, (x0 + panel_w / 2, y0 - 18), label, label_font)
        draw.text((x0, y0 + panel_h + 8), f"x [{centers[0]:.2f}, {centers[-1]:.2f}]", font=small_font, fill=(80, 80, 80))
    draw.text((left, height - 32), "Red: LeJepa overrepresented. Blue: ImageNet overrepresented.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_layer_depth_profiles(
    layer_summary: pd.DataFrame,
    layer_component_summary: pd.DataFrame,
    path: Path,
) -> None:
    width, height = 1080, 720
    left = 72
    right = 36
    top = 88
    bottom = 70
    gap_x = 62
    gap_y = 68
    panel_w = (width - left - right - gap_x) / 2
    panel_h = (height - top - bottom - gap_y) / 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(14, bold=True)
    small_font = base.get_font(11)
    draw.text((24, 24), "Layer Residual Profiles Across Depth", font=title_font, fill=(20, 20, 20))

    layer_df = layer_summary.sort_values("layer_order").copy()
    c0_df = layer_component_summary[layer_component_summary["component"] == 0].sort_values("layer_order").copy()
    layer_df = layer_df.merge(
        c0_df[
            [
                "layer_order",
                "weighted_tv_distance",
                "tv_distance",
                "mean_delta_lejepa_minus_imagenet",
            ]
        ],
        on="layer_order",
        suffixes=("", "_c0"),
    )
    x_values = layer_df["layer_order"].to_numpy(dtype=float)
    x_min, x_max = 0.0, 15.0

    panels = [
        ("drift_D", "layer drift D", False, (0, 114, 178)),
        ("weighted_tv_distance", "sum weighted TV", False, (213, 94, 0)),
        ("weighted_tv_distance_c0", "c0 weighted TV", False, (0, 158, 115)),
        ("mean_delta_lejepa_minus_imagenet", "c0 signed mean delta", True, (120, 90, 170)),
    ]

    for idx, (metric, label, signed, color) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x0 = left + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        y_values = layer_df[metric].to_numpy(dtype=float)
        if signed:
            y_abs = max(float(np.max(np.abs(y_values))), 1e-12)
            y_min, y_max = -1.08 * y_abs, 1.08 * y_abs
        else:
            y_min, y_max = 0.0, max(float(np.max(y_values)) * 1.08, 1e-12)

        def xmap(x: float) -> float:
            return x0 + (x - x_min) / (x_max - x_min) * panel_w

        def ymap(y: float) -> float:
            return y0 + (y_max - y) / (y_max - y_min) * panel_h

        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline=(50, 50, 50), width=1)
        for boundary in [2.5, 6.5, 12.5]:
            x = xmap(boundary)
            draw.line([(x, y0), (x, y0 + panel_h)], fill=(220, 220, 220), width=1)
        for k in range(4):
            y = y0 + k / 3 * panel_h
            draw.line([(x0, y), (x0 + panel_w, y)], fill=(238, 238, 238))
        if signed:
            zero_y = ymap(0.0)
            draw.line([(x0, zero_y), (x0 + panel_w, zero_y)], fill=(80, 80, 80), width=2)
        points = [(xmap(float(x)), ymap(float(y))) for x, y in zip(x_values, y_values)]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline="white")
        for tick in [0, 5, 10, 15]:
            x = xmap(float(tick))
            draw.line([(x, y0 + panel_h), (x, y0 + panel_h + 5)], fill=(40, 40, 40))
            base.draw_centered(draw, (x, y0 + panel_h + 20), str(tick), small_font)
        base.draw_centered(draw, (x0 + panel_w / 2, y0 - 20), label, label_font)
    draw.text((left, height - 34), "Vertical guides mark stage boundaries after layer1, layer2, and layer3.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_layer_component_bubble_map(
    layer_component_summary: pd.DataFrame,
    path: Path,
) -> None:
    layer_df = layer_component_summary.sort_values(["layer_order", "component"]).copy()
    layer_labels = (
        layer_df[["layer_order", "layer_label"]]
        .drop_duplicates()
        .sort_values("layer_order")["layer_label"]
        .tolist()
    )
    components = [f"c{i}" for i in range(9)]
    cell_w = 58
    cell_h = 48
    left = 92
    top = 96
    right = 94
    bottom = 104
    width = left + len(layer_labels) * cell_w + right
    height = top + len(components) * cell_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(12)
    small_font = base.get_font(11)
    draw.text((24, 24), "Layer x Component Residual Bubble Map", font=title_font, fill=(20, 20, 20))
    draw.text((24, 52), "bubble size: weighted TV; color: signed mean delta", font=small_font, fill=(80, 80, 80))

    max_wtv = max(float(layer_df["weighted_tv_distance"].max()), 1e-12)
    max_abs_delta = max(float(np.max(np.abs(layer_df["mean_delta_lejepa_minus_imagenet"]))), 1e-12)
    for c, label in enumerate(layer_labels):
        x = left + c * cell_w + cell_w / 2
        base.draw_centered(draw, (x, top - 18), label, label_font)
    for r, label in enumerate(components):
        y = top + r * cell_h + cell_h / 2
        draw.text((24, y - 8), label, font=label_font, fill=(20, 20, 20))
    for c in range(len(layer_labels) + 1):
        x = left + c * cell_w
        draw.line([(x, top), (x, top + len(components) * cell_h)], fill=(225, 225, 225))
    for r in range(len(components) + 1):
        y = top + r * cell_h
        draw.line([(left, y), (left + len(layer_labels) * cell_w, y)], fill=(225, 225, 225))

    for row in layer_df.itertuples(index=False):
        cx = left + int(row.layer_order) * cell_w + cell_w / 2
        cy = top + int(row.component) * cell_h + cell_h / 2
        radius = 2.0 + 19.0 * math.sqrt(float(row.weighted_tv_distance) / max_wtv)
        color = base.diverging_color(float(row.mean_delta_lejepa_minus_imagenet), max_abs_delta)
        outline = (70, 70, 70) if radius > 8 else (110, 110, 110)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline=outline)

    legend_x = left
    legend_y = height - 64
    for label, frac in [("small", 0.25), ("medium", 0.55), ("max", 1.0)]:
        radius = 2.0 + 19.0 * math.sqrt(frac)
        cx = legend_x
        cy = legend_y
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(245, 245, 245), outline=(80, 80, 80))
        draw.text((cx + 26, cy - 8), label, font=small_font, fill=(80, 80, 80))
        legend_x += 106
    draw.text((left + 360, height - 72), f"blue: ImageNet larger mean; red: LeJepa larger mean; max |delta|={max_abs_delta:.4f}", font=small_font, fill=(80, 80, 80))
    draw.text((left + 360, height - 48), f"largest weighted TV={max_wtv:.4f}", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_quality_delta_heatmap_all(
    quality_df: pd.DataFrame,
    path: Path,
) -> None:
    metric_specs = [
        ("entropy_H_delta_lejepa_minus_imagenet", "H"),
        ("H_over_TH_delta_lejepa_minus_imagenet", "H/TH"),
        ("sparsity_S_delta_lejepa_minus_imagenet", "S"),
        ("mean_abs_weight_delta_lejepa_minus_imagenet", "mean|w|"),
        ("std_weight_delta_lejepa_minus_imagenet", "std(w)"),
        ("abs_peak_weight_delta_lejepa_minus_imagenet", "max|w|"),
    ]
    df = quality_df.sort_values("layer_order")
    row_labels = df["layer_label"].tolist()
    matrix = df[[col for col, _ in metric_specs]].to_numpy(dtype=float)
    col_labels = [label for _, label in metric_specs]
    cell_w = 116
    cell_h = 42
    left = 168
    top = 94
    right = 40
    bottom = 58
    width = left + len(col_labels) * cell_w + right
    height = top + len(row_labels) * cell_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(13)
    small_font = base.get_font(11)
    value_font = base.get_font(11, bold=True)
    draw.text((22, 22), "Layer Quality Delta Heatmap: LeJepa - ImageNet", font=title_font, fill=(20, 20, 20))
    for c, label in enumerate(col_labels):
        base.draw_centered(draw, (left + c * cell_w + cell_w / 2, top - 22), label, label_font)
    for r, label in enumerate(row_labels):
        draw.text((20, top + r * cell_h + cell_h / 2 - 8), label, font=label_font, fill=(20, 20, 20))
    col_max_abs = np.maximum(np.max(np.abs(matrix), axis=0), 1e-12)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = float(matrix[r, c])
            color = base.diverging_color(value, float(col_max_abs[c]))
            x0 = left + c * cell_w
            y0 = top + r * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=color, outline=(220, 220, 220))
            fill = (255, 255, 255) if base.luminance(color) < 100 else (20, 20, 20)
            base.draw_centered(draw, (x0 + cell_w / 2, y0 + cell_h / 2), f"{value:.3f}", value_font, fill)
    draw.text((left, height - 34), "Colors are scaled separately per metric column; red means LeJepa larger.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_global_tv_kl_comparison(
    global_summary: pd.DataFrame,
    kl_df: pd.DataFrame,
    path: Path,
) -> None:
    merged = global_summary[["component", "weighted_tv_distance", "tv_distance"]].merge(
        kl_df[["component", "weighted_symmetric_kl"]],
        on="component",
        how="left",
    ).sort_values("component")
    width, height = 980, 520
    left = 76
    top = 80
    panel_gap = 66
    panel_w = (width - left - 50 - panel_gap) / 2
    panel_h = 340
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = base.get_font(24, bold=True)
    label_font = base.get_font(14, bold=True)
    small_font = base.get_font(11)
    draw.text((24, 24), "Global Component Residual Metrics: Weighted TV vs KL", font=title_font, fill=(20, 20, 20))

    panels = [
        ("weighted_tv_distance", "weighted TV", (213, 94, 0)),
        ("weighted_symmetric_kl", "weighted symmetric KL", (0, 114, 178)),
    ]
    for idx, (metric, label, color) in enumerate(panels):
        x0 = left + idx * (panel_w + panel_gap)
        values = merged[metric].fillna(0.0).to_numpy(dtype=float)
        ymax = max(float(values.max()) * 1.08, 1e-12)
        draw.rectangle([x0, top, x0 + panel_w, top + panel_h], outline=(50, 50, 50), width=1)
        for k in range(5):
            y = top + k / 4 * panel_h
            draw.line([(x0, y), (x0 + panel_w, y)], fill=(238, 238, 238))
        bar_gap = 8
        bar_w = (panel_w - 28 - 8 * bar_gap) / 9
        for component, value in enumerate(values):
            bx = x0 + 14 + component * (bar_w + bar_gap)
            by = top + (ymax - value) / ymax * panel_h
            draw.rectangle([bx, by, bx + bar_w, top + panel_h], fill=color)
            base.draw_centered(draw, (bx + bar_w / 2, top + panel_h + 17), str(component), small_font)
        base.draw_centered(draw, (x0 + panel_w / 2, top - 22), label, label_font)
    draw.text((left, height - 34), "Both metrics are weighted by PCA explained variance; component numbers are shown on x-axis.", font=small_font, fill=(80, 80, 80))
    img.save(path)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV for figures-only mode: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_residual_figures(
    figure_dir: Path,
    global_summary: pd.DataFrame,
    global_hist: pd.DataFrame,
    stage_summary: pd.DataFrame,
    stage_hist: pd.DataFrame,
    layer_component_summary: pd.DataFrame,
    layer_hist: pd.DataFrame,
    layer_summary: pd.DataFrame,
    quality_delta: pd.DataFrame,
    global_kl_contribs: pd.DataFrame,
    top_layer_components_count: int,
    max_density_points: int,
    seed: int,
    model_infos: dict[str, dict] | None = None,
) -> list[Path]:
    ensure_dir(figure_dir)
    layer_labels = [
        row.layer_label
        for row in layer_summary.sort_values("layer_order").itertuples(index=False)
    ]
    components = [f"c{i}" for i in range(9)]
    weighted_tv_matrix = np.zeros((16, 9), dtype=np.float64)
    mean_delta_matrix = np.zeros((16, 9), dtype=np.float64)
    for row in layer_component_summary.itertuples(index=False):
        weighted_tv_matrix[int(row.layer_order), int(row.component)] = float(row.weighted_tv_distance)
        mean_delta_matrix[int(row.layer_order), int(row.component)] = float(row.mean_delta_lejepa_minus_imagenet)

    stage_labels = ["layer1", "layer2", "layer3", "layer4"]
    stage_weighted_tv = np.zeros((len(stage_labels), 9), dtype=np.float64)
    stage_mean_delta = np.zeros((len(stage_labels), 9), dtype=np.float64)
    stage_index = {stage: idx for idx, stage in enumerate(stage_labels)}
    for row in stage_summary.itertuples(index=False):
        stage_weighted_tv[stage_index[row.stage], int(row.component)] = float(row.weighted_tv_distance)
        stage_mean_delta[stage_index[row.stage], int(row.component)] = float(row.mean_delta_lejepa_minus_imagenet)

    figures = [
        figure_dir / "global_component_residual_summary.png",
        figure_dir / "layer_component_weighted_tv_heatmap.png",
        figure_dir / "layer_component_mean_delta_heatmap.png",
        figure_dir / "top_layer_component_residual_histograms.png",
        figure_dir / "layer_quality_delta_by_depth.png",
    ]
    c0_c1_path = figure_dir / "c0_c1_residual_density.png"
    extra_figures = [
        figure_dir / "global_component_residual_histograms.png",
        figure_dir / "stage_component_weighted_tv_heatmap.png",
        figure_dir / "stage_component_mean_delta_heatmap.png",
        figure_dir / "stage_c0_residual_histograms.png",
        figure_dir / "layer_residual_depth_profiles.png",
        figure_dir / "layer_component_residual_bubble_map.png",
        figure_dir / "layer_quality_delta_heatmap.png",
    ]
    if not global_kl_contribs.empty:
        extra_figures.append(figure_dir / "global_component_tv_kl_comparison.png")

    draw_component_summary(global_summary, figures[0])
    draw_heatmap(
        weighted_tv_matrix,
        layer_labels,
        components,
        "Layer x Component Residual Magnitude: weighted TV",
        figures[1],
        signed=False,
    )
    draw_heatmap(
        mean_delta_matrix,
        layer_labels,
        components,
        "Layer x Component Signed Mean Delta: LeJepa - ImageNet",
        figures[2],
        signed=True,
    )
    top_layer_components = layer_component_summary.sort_values(
        "weighted_tv_distance",
        ascending=False,
    ).head(top_layer_components_count)
    draw_residual_hist_panels(
        layer_hist,
        top_layer_components,
        figures[3],
        max_panels=top_layer_components_count,
    )
    draw_quality_delta(quality_delta, figures[4])
    if model_infos is not None:
        draw_c0_c1_residual_density(
            model_infos,
            c0_c1_path,
            max_points=max_density_points,
            seed=seed + 202,
        )
        figures.append(c0_c1_path)
    elif c0_c1_path.exists():
        figures.append(c0_c1_path)

    draw_global_residual_histograms(global_hist, global_summary, extra_figures[0])
    draw_heatmap(
        stage_weighted_tv,
        stage_labels,
        components,
        "Stage x Component Residual Magnitude: weighted TV",
        extra_figures[1],
        signed=False,
    )
    draw_heatmap(
        stage_mean_delta,
        stage_labels,
        components,
        "Stage x Component Signed Mean Delta: LeJepa - ImageNet",
        extra_figures[2],
        signed=True,
    )
    draw_stage_c0_residual_histograms(stage_hist, stage_summary, extra_figures[3])
    draw_layer_depth_profiles(layer_summary, layer_component_summary, extra_figures[4])
    draw_layer_component_bubble_map(layer_component_summary, extra_figures[5])
    draw_quality_delta_heatmap_all(quality_delta, extra_figures[6])
    if not global_kl_contribs.empty:
        draw_global_tv_kl_comparison(global_summary, global_kl_contribs, extra_figures[7])
    figures.extend(extra_figures)
    return figures


def save_summary(
    path: Path,
    global_drift: float,
    source_global_drift: float | None,
    global_summary: pd.DataFrame,
    stage_summary: pd.DataFrame,
    layer_component_summary: pd.DataFrame,
    layer_residual_summary: pd.DataFrame,
    quality_df: pd.DataFrame,
    global_kl_contribs: pd.DataFrame,
    source_pca: Path,
    figures: Sequence[Path],
) -> None:
    top_components = global_summary.sort_values("weighted_tv_distance", ascending=False).head(5)
    top_kl_components = (
        global_kl_contribs.sort_values("weighted_symmetric_kl", ascending=False).head(5)
        if not global_kl_contribs.empty
        else pd.DataFrame()
    )
    top_stage_components = stage_summary.sort_values("weighted_tv_distance", ascending=False).head(8)
    top_layer_components = layer_component_summary.sort_values("weighted_tv_distance", ascending=False).head(10)
    top_layer_drifts = layer_residual_summary.sort_values("drift_D", ascending=False).head(8)
    c0_layers = layer_component_summary[layer_component_summary["component"] == 0].sort_values("layer_order")
    c0_positive = c0_layers.sort_values("mean_delta_lejepa_minus_imagenet", ascending=False).head(5)
    c0_negative = c0_layers.sort_values("mean_delta_lejepa_minus_imagenet", ascending=True).head(5)
    stage_c0 = stage_summary[stage_summary["component"] == 0].copy()
    stage_c0["stage_order"] = stage_c0["stage"].map({"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4})
    stage_c0 = stage_c0.sort_values("stage_order")
    entropy_delta = quality_df["entropy_H_delta_lejepa_minus_imagenet"]
    sparsity_delta = quality_df["sparsity_S_delta_lejepa_minus_imagenet"]
    quality_stage = (
        quality_df.groupby("stage", as_index=False)
        .agg(
            mean_entropy_delta=("entropy_H_delta_lejepa_minus_imagenet", "mean"),
            mean_h_over_th_delta=("H_over_TH_delta_lejepa_minus_imagenet", "mean"),
            mean_sparsity_delta=("sparsity_S_delta_lejepa_minus_imagenet", "mean"),
            mean_abs_weight_delta=("mean_abs_weight_delta_lejepa_minus_imagenet", "mean"),
            mean_std_weight_delta=("std_weight_delta_lejepa_minus_imagenet", "mean"),
            mean_peak_delta=("abs_peak_weight_delta_lejepa_minus_imagenet", "mean"),
        )
        .sort_values("stage")
    )
    top_entropy_pos = quality_df.sort_values("entropy_H_delta_lejepa_minus_imagenet", ascending=False).head(5)
    top_entropy_neg = quality_df.sort_values("entropy_H_delta_lejepa_minus_imagenet", ascending=True).head(5)
    top_sparsity_pos = quality_df.sort_values("sparsity_S_delta_lejepa_minus_imagenet", ascending=False).head(5)
    top_sparsity_neg = quality_df.sort_values("sparsity_S_delta_lejepa_minus_imagenet", ascending=True).head(5)
    all_layer_weighted_tv = float(layer_component_summary["weighted_tv_distance"].sum())
    c0_layer_weighted_tv = float(c0_layers["weighted_tv_distance"].sum())
    c0_share = c0_layer_weighted_tv / max(all_layer_weighted_tv, 1e-12)
    global_c0 = global_summary[global_summary["component"] == 0].iloc[0]

    lines: list[str] = []
    lines.append("# LeJepa SSL minus ImageNet FT Residual Analysis")
    lines.append("")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"PCA basis: `{source_pca}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Signed residual is defined as:")
    lines.append("")
    lines.append("```text")
    lines.append("LeJepa SSL - ImageNet FT")
    lines.append("```")
    lines.append("")
    lines.append("Both models are compared after the same per-filter max-absolute normalization and projection into the same RN50 PCA basis used by the first three-model experiment.")
    lines.append("This is important: `c0`, `c1`, and the other PCA components mean the same filter directions here as they did in the previous global drift analysis.")
    lines.append("")
    lines.append("The analysis focuses only on RN50 `3x3` bottleneck convolution filters. `1x1` and `7x7` kernels are not mixed into this space.")
    lines.append("")
    lines.append("## Metric Guide")
    lines.append("")
    lines.append("- `TV`: total variation distance between two normalized histograms for one PCA coefficient. `0` means identical distributions; larger values mean more probability mass has moved.")
    lines.append("- `Weighted TV`: `TV * PCA explained-variance ratio`. This favors residuals in PCA directions that explain more of the overall RN50 filter variation.")
    lines.append("- `Mean delta`: `mean(LeJepa coefficient) - mean(ImageNet coefficient)`. Positive means LeJepa is shifted toward the positive side of that PCA axis.")
    lines.append("- `Max + bin`: PCA coefficient bin where LeJepa is most overrepresented.")
    lines.append("- `Max - bin`: PCA coefficient bin where ImageNet is most overrepresented.")
    lines.append("- `Symmetric KL`: a histogram-distribution divergence used for the pair drift score; unlike signed residuals, it is not directional.")
    lines.append("")
    lines.append("## Main Numbers")
    lines.append("")
    lines.append(f"- Pair drift with residual-analysis two-model histogram range: `{global_drift:.6f}`")
    if source_global_drift is not None:
        lines.append(f"- Pair drift from previous three-model analysis: `{source_global_drift:.6f}`")
    lines.append(f"- Largest global component residual by weighted TV: `c{int(top_components.iloc[0]['component'])}` with weighted TV `{top_components.iloc[0]['weighted_tv_distance']:.6f}`")
    lines.append(f"- Global `c0` TV: `{global_c0.tv_distance:.6f}`; global `c0` weighted TV: `{global_c0.weighted_tv_distance:.6f}`; global `c0` mean delta: `{global_c0.mean_delta_lejepa_minus_imagenet:.6f}`")
    lines.append(f"- Share of layer/component weighted TV carried by `c0`: `{c0_share:.3f}`")
    lines.append(f"- Mean entropy delta across layers: `{entropy_delta.mean():.6f}`")
    lines.append(f"- Mean sparsity delta across layers: `{sparsity_delta.mean():.6f}`")
    lines.append("")
    lines.append("## Main Interpretation")
    lines.append("")
    lines.append("LeJepa SSL and ImageNet FT remain globally close in the pooled RN50 `3x3` filter space. The pair drift is still small, and it matches the previous three-model analysis when the same histogram/PCA setup is used.")
    lines.append("")
    lines.append("The residual analysis changes the question from whether the two models are globally far apart to where, specifically, the two close distributions disagree. The answer is that the disagreement is structured rather than uniform.")
    lines.append("")
    lines.append("`c0` is the dominant layer/component residual direction after PCA weighting. The top layer/component residuals are all in `c0`, and they occur in selected early, middle/deep, and late layers rather than being spread evenly across all depths.")
    lines.append("")
    lines.append("The signed mean-delta pattern is also depth-dependent: early and late layers tend to move toward the positive side of `c0`, while several middle/deep layers move toward the negative side.")
    lines.append("")
    lines.append("## Global Component Residuals")
    lines.append("")
    lines.append("| Component | TV | Weighted TV | Mean delta | Max + bin | Max + residual | Max - bin | Max - residual |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in top_components.itertuples(index=False):
        lines.append(
            f"| {int(row.component)} | {row.tv_distance:.6f} | {row.weighted_tv_distance:.6f} | "
            f"{row.mean_delta_lejepa_minus_imagenet:.6f} | {row.max_positive_bin_center:.4f} | "
            f"{row.max_positive_residual:.6f} | {row.max_negative_bin_center:.4f} | {row.max_negative_residual:.6f} |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- `c0` is the strongest global component after PCA weighting. This means the main residual is aligned with the most important RN50 filter direction learned by PCA.")
    lines.append("- Some components, such as `c3`, can have larger unweighted TV than `c0`, but lower PCA explained-variance weight. Weighted TV prevents a less central PCA direction from dominating the interpretation.")
    lines.append("- `Max + bin` and `Max - bin` are locations on the PCA coefficient axis, not residual magnitudes. The residual magnitudes are shown in the adjacent residual columns.")
    lines.append("")
    if not top_kl_components.empty:
        lines.append("### KL Contribution Check")
        lines.append("")
        lines.append("| Component | Component weight | Weighted symmetric KL |")
        lines.append("|---:|---:|---:|")
        for row in top_kl_components.itertuples(index=False):
            lines.append(f"| {int(row.component)} | {row.component_weight:.6f} | {row.weighted_symmetric_kl:.6f} |")
        lines.append("")
        lines.append("Weighted KL is a useful cross-check because it comes from the original drift metric. The KL ranking is not identical to weighted TV because TV and KL emphasize distribution differences differently, but both help identify which PCA components contribute most to the LeJepa/ImageNet separation.")
        lines.append("")
    lines.append("## Stage-Level Residuals")
    lines.append("")
    lines.append("The stage-level view asks whether residuals are concentrated in broad RN50 depth regions.")
    lines.append("")
    lines.append("| Stage | c0 TV | c0 Weighted TV | c0 Mean delta | Max + bin | Max - bin |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in stage_c0.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.tv_distance:.6f} | {row.weighted_tv_distance:.6f} | "
            f"{row.mean_delta_lejepa_minus_imagenet:.6f} | {row.max_positive_bin_center:.4f} | {row.max_negative_bin_center:.4f} |"
        )
    lines.append("")
    lines.append("Strongest stage/component residuals:")
    lines.append("")
    lines.append("| Stage | Component | TV | Weighted TV | Mean delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in top_stage_components.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {int(row.component)} | {row.tv_distance:.6f} | "
            f"{row.weighted_tv_distance:.6f} | {row.mean_delta_lejepa_minus_imagenet:.6f} |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Stage summaries preserve the main `c0` story but smooth over individual block-level behavior.")
    lines.append("- They are useful for seeing broad depth trends, while the layer/component table below identifies the exact bottleneck blocks carrying the residual.")
    lines.append("")
    lines.append("## Layer/Component Residuals")
    lines.append("")
    lines.append("| Layer | Stage | Component | TV | Weighted TV | Mean delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in top_layer_components.itertuples(index=False):
        lines.append(
            f"| `{row.layer_key}` | {row.stage} | {int(row.component)} | {row.tv_distance:.6f} | "
            f"{row.weighted_tv_distance:.6f} | {row.mean_delta_lejepa_minus_imagenet:.6f} |"
        )
    lines.append("")
    lines.append("By layer drift, the largest residual differences are:")
    lines.append("")
    lines.append("| Layer | Stage | Drift D | Sum weighted TV | Max component weighted TV |")
    lines.append("|---|---|---:|---:|---:|")
    for row in top_layer_drifts.itertuples(index=False):
        lines.append(
            f"| `{row.layer_key}` | {row.stage} | {row.drift_D:.6f} | "
            f"{row.weighted_tv_distance:.6f} | {row.max_component_weighted_tv:.6f} |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- The strongest layer/component residuals are concentrated in `c0`.")
    lines.append("- The most prominent layers are not a single contiguous run. They include early `layer1`, selected `layer2`, selected `layer3`, and the first late block `layer4.0`.")
    lines.append("- This supports a depth-dependent residual signature rather than a simple global offset between the two models.")
    lines.append("")
    lines.append("## Signed Direction Across Depth")
    lines.append("")
    lines.append("The signed `c0` mean delta shows whether LeJepa shifts toward the positive or negative side of the main PCA direction at each depth.")
    lines.append("")
    lines.append("Largest positive `c0` mean shifts:")
    lines.append("")
    lines.append("| Layer | Stage | c0 mean delta | c0 Weighted TV |")
    lines.append("|---|---|---:|---:|")
    for row in c0_positive.itertuples(index=False):
        lines.append(f"| `{row.layer_key}` | {row.stage} | {row.mean_delta_lejepa_minus_imagenet:.6f} | {row.weighted_tv_distance:.6f} |")
    lines.append("")
    lines.append("Largest negative `c0` mean shifts:")
    lines.append("")
    lines.append("| Layer | Stage | c0 mean delta | c0 Weighted TV |")
    lines.append("|---|---|---:|---:|")
    for row in c0_negative.itertuples(index=False):
        lines.append(f"| `{row.layer_key}` | {row.stage} | {row.mean_delta_lejepa_minus_imagenet:.6f} | {row.weighted_tv_distance:.6f} |")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Positive values mean LeJepa has a larger average coefficient than ImageNet along that PCA direction.")
    lines.append("- Negative values mean ImageNet is larger on average, or equivalently LeJepa has shifted toward the negative side of the PCA axis.")
    lines.append("- The sign change across depth matters: LeJepa is not merely shifted in one global direction. It reorganizes `c0` usage differently at different depths.")
    lines.append("- Mean delta should be read together with TV/weighted TV. A layer can have a small mean delta but still a large TV if the distribution shape changes without moving the average much.")
    lines.append("")
    lines.append("## Layer Quality Residuals")
    lines.append("")
    lines.append("Layer-quality residuals compare non-PCA layer statistics. Positive deltas mean LeJepa is larger than ImageNet.")
    lines.append("")
    lines.append("Stage-level quality deltas:")
    lines.append("")
    lines.append("| Stage | Mean entropy delta | Mean H/TH delta | Mean sparsity delta | Mean abs weight delta | Mean std weight delta | Mean peak delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in quality_stage.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.mean_entropy_delta:.6f} | {row.mean_h_over_th_delta:.6f} | "
            f"{row.mean_sparsity_delta:.6f} | {row.mean_abs_weight_delta:.6f} | "
            f"{row.mean_std_weight_delta:.6f} | {row.mean_peak_delta:.6f} |"
        )
    lines.append("")
    lines.append("Largest positive entropy deltas:")
    lines.append("")
    lines.append("| Layer | Stage | ImageNet H | LeJepa H | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for row in top_entropy_pos.itertuples(index=False):
        lines.append(f"| `{row.layer_key}` | {row.stage} | {row.entropy_H_imagenet:.6f} | {row.entropy_H_lejepa:.6f} | {row.entropy_H_delta_lejepa_minus_imagenet:.6f} |")
    lines.append("")
    lines.append("Largest negative entropy deltas:")
    lines.append("")
    lines.append("| Layer | Stage | ImageNet H | LeJepa H | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for row in top_entropy_neg.itertuples(index=False):
        lines.append(f"| `{row.layer_key}` | {row.stage} | {row.entropy_H_imagenet:.6f} | {row.entropy_H_lejepa:.6f} | {row.entropy_H_delta_lejepa_minus_imagenet:.6f} |")
    lines.append("")
    lines.append("Largest positive sparsity deltas:")
    lines.append("")
    lines.append("| Layer | Stage | ImageNet S | LeJepa S | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for row in top_sparsity_pos.itertuples(index=False):
        lines.append(f"| `{row.layer_key}` | {row.stage} | {row.sparsity_S_imagenet:.6f} | {row.sparsity_S_lejepa:.6f} | {row.sparsity_S_delta_lejepa_minus_imagenet:.6f} |")
    lines.append("")
    lines.append("Largest negative sparsity deltas:")
    lines.append("")
    lines.append("| Layer | Stage | ImageNet S | LeJepa S | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for row in top_sparsity_neg.itertuples(index=False):
        lines.append(f"| `{row.layer_key}` | {row.stage} | {row.sparsity_S_imagenet:.6f} | {row.sparsity_S_lejepa:.6f} | {row.sparsity_S_delta_lejepa_minus_imagenet:.6f} |")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- The clearest quality-level contrast is sparsity allocation.")
    lines.append("- ImageNet FT is much more sparse in the first bottleneck, especially `layer1.0`.")
    lines.append("- LeJepa SSL becomes much more sparse in late `layer4` blocks.")
    lines.append("- In `layer4.0`, LeJepa has both higher sparsity and higher entropy. That combination argues against reading late-layer sparsity as simple collapse; it may indicate more near-zero filters while the active structure remains more diverse.")
    lines.append("- LeJepa has lower `mean abs weight`, `std weight`, and peak absolute weight across all layers in the raw-weight quality metrics. Those raw-weight deltas should not be mixed with the PCA normalization story; they describe magnitude/scale behavior before per-filter max-absolute normalization.")
    lines.append("")
    lines.append("## Figure Guide")
    lines.append("")
    figure_notes = {
        "global_component_residual_summary.png": "Bar summary of global TV and signed mean delta by PCA component.",
        "global_component_residual_histograms.png": "Signed residual histograms for all nine global PCA components.",
        "global_component_tv_kl_comparison.png": "Side-by-side global component ranking by weighted TV and weighted symmetric KL.",
        "stage_component_weighted_tv_heatmap.png": "Stage x component residual magnitude using weighted TV.",
        "stage_component_mean_delta_heatmap.png": "Stage x component signed mean shift.",
        "stage_c0_residual_histograms.png": "Stage-level signed residual histogram for the dominant `c0` direction.",
        "layer_component_weighted_tv_heatmap.png": "Layer x component residual magnitude using weighted TV.",
        "layer_component_mean_delta_heatmap.png": "Layer x component signed mean shift.",
        "top_layer_component_residual_histograms.png": "Signed residual histograms for the strongest layer/component pairs.",
        "layer_residual_depth_profiles.png": "Depth curves for drift, summed weighted TV, `c0` weighted TV, and signed `c0` mean delta.",
        "layer_component_residual_bubble_map.png": "Bubble size shows weighted TV; color shows signed mean delta.",
        "layer_quality_delta_by_depth.png": "Depth curves for entropy and sparsity deltas.",
        "layer_quality_delta_heatmap.png": "All layer-quality metric deltas in one heatmap.",
        "c0_c1_residual_density.png": "2D density comparison and signed residual in the first two PCA coordinates.",
    }
    for figure in figures:
        note = figure_notes.get(Path(figure).name, "Generated residual figure.")
        lines.append(f"- `{figure}`: {note}")
    lines.append("")
    lines.append("## Scientific Takeaway")
    lines.append("")
    lines.append("1. LeJepa SSL and ImageNet FT are globally close in RN50 `3x3` filter space.")
    lines.append("")
    lines.append(f"   The global drift is `{global_drift:.6f}`, so this experiment does not support a claim that LeJepa occupies a completely different broad filter distribution from ImageNet fine-tuning.")
    lines.append("")
    lines.append("2. The residual is still meaningful.")
    lines.append("")
    lines.append("   The difference is concentrated in specific PCA directions and specific depths, especially `c0`, rather than being uniformly spread across the network.")
    lines.append("")
    lines.append("3. The residual is depth-dependent.")
    lines.append("")
    lines.append("   Early, selected middle/deep, and late blocks carry different signs and magnitudes of the `c0` shift. This is more informative than a single pooled drift score.")
    lines.append("")
    lines.append("4. Quality metrics show an early-vs-late redistribution.")
    lines.append("")
    lines.append("   ImageNet is much sparser in the first bottleneck, while LeJepa is much sparser in late `layer4` blocks. Late-layer LeJepa sparsity is paired with higher entropy in key layers.")
    lines.append("")
    lines.append("## Suggested Meeting Explanation")
    lines.append("")
    lines.append("The residual analysis shows that LeJepa SSL is not globally far from ImageNet fine-tuning in RN50 `3x3` filter space. The two models share a similar broad filter basis. However, when we subtract the ImageNet distribution from the LeJepa distribution, the remaining signal is structured: it is strongest in PCA component `c0` and appears in specific depths rather than everywhere. LeJepa changes how filter mass is allocated along the main RN50 filter direction, with signs that change across depth. It also reallocates sparsity: ImageNet is much sparser in the first bottleneck, while LeJepa is much sparser in late `layer4` blocks. So the best interpretation is not that LeJepa learned a completely different filter universe, but that it imposes a layer-specific residual signature on top of an ImageNet-like RN50 filter basis.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- The PCA basis is local to the compared RN50 models from the first experiment. A reference basis from CNN Filter DB would test whether this residual pattern is unusual relative to a broader model population.")
    lines.append("- Histogram-based metrics depend on binning. The current analysis uses the configured bin count consistently, but a bin-count sensitivity check would make the result stronger.")
    lines.append("- Global pooled filter counts are dominated by later, wider layers. Layer-level and stage-level figures are therefore essential for interpreting depth structure.")
    lines.append("- Signed mean deltas do not capture full distribution-shape changes. They should be interpreted together with TV, weighted TV, and the residual histograms.")
    lines.append("")
    lines.append("## Recommended Follow-Ups")
    lines.append("")
    lines.append("1. Inspect actual filters from the strongest residual bins, especially `layer3.3 c0`, `layer3.2 c0`, `layer4.0 c0`, and `layer1.0 c0`.")
    lines.append("2. Repeat the residual analysis with several histogram bin counts to check robustness.")
    lines.append("3. Add a reference PCA basis from CNN Filter DB or same-family RN50 ImageNet/classification models.")
    lines.append("4. Compare LeJepa-vs-ImageNet residuals against scratch-vs-ImageNet residuals at the same layer/component level.")
    lines.append("5. Add bootstrap confidence intervals for TV and mean-delta estimates, especially for early layers with fewer filters.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for figure in figures:
        lines.append(f"- `{figure}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    if torch is not None:
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    elif not args.figures_only:
        raise ModuleNotFoundError("PyTorch is required unless --figures-only is used.")
    base.MODEL_SPECS = PAIR_SPECS

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    source_dir = Path(args.source_analysis_dir)
    pca_path = source_dir / "pca_target_rn50.npz"
    print(f"[start] Output directory: {output_dir}", flush=True)
    print(f"[start] PCA basis: {pca_path}", flush=True)

    if args.figures_only:
        print("[plot] Loading existing residual CSVs", flush=True)
        global_summary = read_required_csv(output_dir / "global_component_residual_summary.csv")
        global_hist = read_required_csv(output_dir / "global_component_residual_histograms.csv")
        stage_summary = read_required_csv(output_dir / "stage_component_residual_summary.csv")
        stage_hist = read_required_csv(output_dir / "stage_component_residual_histograms.csv")
        layer_component_summary = read_required_csv(output_dir / "layer_component_residual_summary.csv")
        layer_hist = read_required_csv(output_dir / "layer_component_residual_histograms.csv")
        layer_summary = read_required_csv(output_dir / "layer_residual_summary.csv")
        quality_delta = read_required_csv(output_dir / "layer_quality_residuals.csv")
        pair_drift = read_required_csv(output_dir / "pair_drift.csv")
        global_kl_contribs = read_optional_csv(output_dir / "global_component_kl_contributions.csv")
        global_drift = float(pair_drift.iloc[0]["residual_analysis_pair_range_drift_D"])
        source_value = pair_drift.iloc[0].get("source_three_model_global_drift_D")
        source_global_drift = None if pd.isna(source_value) else float(source_value)

        figures = write_residual_figures(
            figure_dir=figure_dir,
            global_summary=global_summary,
            global_hist=global_hist,
            stage_summary=stage_summary,
            stage_hist=stage_hist,
            layer_component_summary=layer_component_summary,
            layer_hist=layer_hist,
            layer_summary=layer_summary,
            quality_delta=quality_delta,
            global_kl_contribs=global_kl_contribs,
            top_layer_components_count=args.top_layer_components,
            max_density_points=args.max_density_points,
            seed=args.seed,
            model_infos=None,
        )
        save_summary(
            output_dir / "summary.md",
            global_drift,
            source_global_drift,
            global_summary,
            stage_summary,
            layer_component_summary,
            layer_summary,
            quality_delta,
            global_kl_contribs,
            pca_path,
            figures,
        )
        print("[done] Figures regenerated from existing CSVs", flush=True)
        print(f"[done] Summary: {output_dir / 'summary.md'}", flush=True)
        print(f"[done] Figures: {figure_dir}", flush=True)
        return 0

    pca_data = load_pca(pca_path)

    print("[extract] Loading ImageNet FT and LeJepa SSL checkpoints", flush=True)
    model_infos, quality_rows = base.extract_models(Path(args.model_dir))
    group_ids = [MODEL_A, MODEL_B]
    project_models(model_infos, pca_data)
    weights = pca_data["weights"]

    print("[global] Computing signed residual histograms", flush=True)
    z_a = model_infos[MODEL_A]["Z"]
    z_b = model_infos[MODEL_B]["Z"]
    global_hist_rows, global_summary_rows = residual_histogram_rows(
        z_a,
        z_b,
        weights,
        args.bins,
        {"scope": "global"},
    )
    global_summary = pd.DataFrame(global_summary_rows)
    global_hist = pd.DataFrame(global_hist_rows)
    global_summary.to_csv(output_dir / "global_component_residual_summary.csv", index=False)
    global_hist.to_csv(output_dir / "global_component_residual_histograms.csv", index=False)

    global_matrix, global_contribs, _ = base.compute_drift(
        {MODEL_A: z_a, MODEL_B: z_b},
        weights,
        bins=args.bins,
    )
    global_drift = float(global_matrix[0, 1])
    source_global_drift = read_source_pair_drift(source_dir)
    pd.DataFrame(
        [
            {
                "group_a": MODEL_A,
                "group_b": MODEL_B,
                "residual_analysis_pair_range_drift_D": global_drift,
                "source_three_model_global_drift_D": source_global_drift,
                "definition": "weighted_symmetric_kl",
            }
        ]
    ).to_csv(output_dir / "pair_drift.csv", index=False)
    global_kl_contribs = pd.DataFrame(
        [
            {
                "component": component,
                "component_weight": float(weights[component]),
                "weighted_symmetric_kl": float(global_contribs[(MODEL_A, MODEL_B)][component]),
            }
            for component in range(9)
        ]
    )
    global_kl_contribs.to_csv(output_dir / "global_component_kl_contributions.csv", index=False)

    print("[stage] Computing stage/component residual summaries", flush=True)
    stage_hist_rows: list[dict] = []
    stage_summary_rows: list[dict] = []
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        stage_a, stage_b = stage_groups(model_infos, stage)
        hist_rows, summary_rows = residual_histogram_rows(
            stage_a,
            stage_b,
            weights,
            args.bins,
            {"scope": "stage", "stage": stage},
        )
        stage_hist_rows.extend(hist_rows)
        stage_summary_rows.extend(summary_rows)
    stage_summary = pd.DataFrame(stage_summary_rows)
    stage_hist = pd.DataFrame(stage_hist_rows)
    stage_summary.to_csv(output_dir / "stage_component_residual_summary.csv", index=False)
    stage_hist.to_csv(output_dir / "stage_component_residual_histograms.csv", index=False)

    print("[layer] Computing layer/component residual summaries", flush=True)
    layer_hist_rows: list[dict] = []
    layer_component_summary_rows: list[dict] = []
    layer_drift_rows: list[dict] = []
    for layer_order in range(16):
        layer_a, layer_b, layer_slice = get_layer_pair(model_infos, layer_order)
        prefix = {
            "scope": "layer",
            "layer_order": layer_order,
            "layer_key": layer_slice.layer_key,
            "layer_label": layer_label(layer_slice.layer_key),
            "stage": layer_slice.stage,
            "conv_depth_norm": layer_slice.conv_depth_norm,
            "num_filters": int(layer_a.shape[0]),
        }
        hist_rows, summary_rows = residual_histogram_rows(
            layer_a,
            layer_b,
            weights,
            args.bins,
            prefix,
        )
        layer_hist_rows.extend(hist_rows)
        layer_component_summary_rows.extend(summary_rows)
        matrix, _, _ = base.compute_drift(
            {MODEL_A: layer_a, MODEL_B: layer_b},
            weights,
            bins=args.bins,
        )
        layer_drift_rows.append(
            {
                "layer_order": layer_order,
                "layer_key": layer_slice.layer_key,
                "layer_label": layer_label(layer_slice.layer_key),
                "stage": layer_slice.stage,
                "conv_depth_norm": layer_slice.conv_depth_norm,
                "num_filters": int(layer_a.shape[0]),
                "drift_D": float(matrix[0, 1]),
            }
        )
    layer_hist = pd.DataFrame(layer_hist_rows)
    layer_component_summary = pd.DataFrame(layer_component_summary_rows)
    layer_drift = pd.DataFrame(layer_drift_rows)
    layer_component_summary.to_csv(output_dir / "layer_component_residual_summary.csv", index=False)
    layer_hist.to_csv(output_dir / "layer_component_residual_histograms.csv", index=False)
    layer_drift.to_csv(output_dir / "layer_residual_drift.csv", index=False)

    layer_summary = (
        layer_component_summary.groupby(
            ["layer_order", "layer_key", "layer_label", "stage", "conv_depth_norm", "num_filters"],
            as_index=False,
        )
        .agg(
            tv_distance=("tv_distance", "mean"),
            weighted_tv_distance=("weighted_tv_distance", "sum"),
            mean_abs_delta=("mean_delta_lejepa_minus_imagenet", lambda x: float(np.mean(np.abs(x)))),
            max_component_tv=("tv_distance", "max"),
            max_component_weighted_tv=("weighted_tv_distance", "max"),
        )
        .merge(layer_drift, on=["layer_order", "layer_key", "layer_label", "stage", "conv_depth_norm", "num_filters"])
    )
    layer_summary.to_csv(output_dir / "layer_residual_summary.csv", index=False)

    print("[quality] Computing layer-quality residuals", flush=True)
    quality_delta = pd.DataFrame(quality_residual_rows(quality_rows))
    quality_delta.to_csv(output_dir / "layer_quality_residuals.csv", index=False)

    print("[plot] Writing residual figures", flush=True)
    figures = write_residual_figures(
        figure_dir=figure_dir,
        global_summary=global_summary,
        global_hist=global_hist,
        stage_summary=stage_summary,
        stage_hist=stage_hist,
        layer_component_summary=layer_component_summary,
        layer_hist=layer_hist,
        layer_summary=layer_summary,
        quality_delta=quality_delta,
        global_kl_contribs=global_kl_contribs,
        top_layer_components_count=args.top_layer_components,
        max_density_points=args.max_density_points,
        seed=args.seed,
        model_infos=model_infos,
    )

    save_summary(
        output_dir / "summary.md",
        global_drift,
        source_global_drift,
        global_summary,
        stage_summary,
        layer_component_summary,
        layer_summary,
        quality_delta,
        global_kl_contribs,
        pca_path,
        figures,
    )

    print("[done] Residual analysis complete", flush=True)
    print(f"[done] Summary: {output_dir / 'summary.md'}", flush=True)
    print(f"[done] Figures: {figure_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
