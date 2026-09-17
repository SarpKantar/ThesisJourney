#!/usr/bin/env python
"""Analyze 3x3 filters from the three local RN50 checkpoints.

The script intentionally uses Pillow for plotting instead of matplotlib because
the available Torch environment on this machine does not include matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:  # Allows CSV-only plotting helpers to import this module.
    torch = None
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA


MODEL_SPECS = [
    {
        "model_id": "scratch_ft",
        "display": "Scratch FT",
        "file": "rn50_rand_ft_backbone.pth",
        "recipe": "trained_from_scratch_on_target_dataset",
        "color": (213, 94, 0),
    },
    {
        "model_id": "imagenet_ft",
        "display": "ImageNet FT",
        "file": "rn50_imagenet_ft_backbone.pth",
        "recipe": "imagenet_pretrained_then_target_finetuned",
        "color": (0, 114, 178),
    },
    {
        "model_id": "lejepa_ssl",
        "display": "LeJepa SSL",
        "file": "rn50_lejepa_ssl_backbone.pth",
        "recipe": "lejepa_ssl_backbone_plus_target_logreg",
        "color": (0, 158, 115),
    },
]

PAIR_ORDER = [
    ("scratch_ft", "imagenet_ft"),
    ("scratch_ft", "lejepa_ssl"),
    ("imagenet_ft", "lejepa_ssl"),
]

LAYER_RE = re.compile(r"layer(\d+)\.(\d+)\.conv2\.weight$")
EPS = np.finfo(np.float32).eps


@dataclass
class LayerSlice:
    model_id: str
    layer_key: str
    stage: str
    layer_order: int
    conv_depth_norm: float
    start: int
    end: int
    shape: Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 3x3 RN50 filters with CNN Filter DB style metrics."
    )
    parser.add_argument("--model-dir", default="models", help="Directory with RN50 .pth files.")
    parser.add_argument(
        "--output-dir",
        default="outputs/rn50_filter_analysis",
        help="Directory where CSVs and figures will be written.",
    )
    parser.add_argument("--bins", type=int, default=70, help="Histogram bins for KL drift.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=30,
        help="Number of global bootstrap repetitions. Use 0 to disable.",
    )
    parser.add_argument(
        "--bootstrap-sample-size",
        type=int,
        default=100_000,
        help="Filters sampled per model in each bootstrap repetition.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=600_000,
        help="Maximum points per model for c0/c1 density plots.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    w, h = text_size(draw, text, font)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def lerp_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def heat_color(t: float) -> Tuple[int, int, int]:
    stops = [
        (0.00, (0, 0, 0)),
        (0.18, (70, 0, 0)),
        (0.45, (190, 0, 0)),
        (0.70, (255, 120, 0)),
        (0.88, (255, 230, 0)),
        (1.00, (255, 255, 255)),
    ]
    t = max(0.0, min(1.0, float(t)))
    for (ta, ca), (tb, cb) in zip(stops[:-1], stops[1:]):
        if ta <= t <= tb:
            return lerp_color(ca, cb, (t - ta) / max(tb - ta, 1e-12))
    return stops[-1][1]


def density_color(t: float) -> Tuple[int, int, int]:
    return heat_color(t)


def diverging_color(value: float, max_abs: float) -> Tuple[int, int, int]:
    if max_abs <= 0:
        return (245, 245, 245)
    x = max(-1.0, min(1.0, value / max_abs))
    if x < 0:
        return lerp_color((45, 65, 230), (245, 245, 245), x + 1.0)
    return lerp_color((245, 245, 245), (220, 35, 35), x)


def luminance(rgb: Tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def load_checkpoint(path: Path) -> dict:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to load RN50 checkpoints.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.load(path, map_location="cpu")


def sort_conv3_items(state_dict: dict) -> List[Tuple[str, torch.Tensor]]:
    items = []
    for key, value in state_dict.items():
        if not hasattr(value, "shape") or len(value.shape) != 4:
            continue
        shape = tuple(int(v) for v in value.shape)
        if shape[-2:] != (3, 3):
            continue
        match = LAYER_RE.match(key)
        if not match:
            continue
        stage = int(match.group(1))
        block = int(match.group(2))
        items.append((stage, block, key, value))
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(key, value) for _, _, key, value in items]


def scale_filters(x: np.ndarray) -> np.ndarray:
    den = np.abs(x).max(axis=1)
    den = np.where(den == 0, 1, den)[:, None]
    return x / den


def random_entropy_threshold(n: int) -> float:
    return 1.26 / (1.0 + math.exp(-0.89 * (math.log2(max(n, 1)) - 2.30))) - 0.31


def layer_quality(x: np.ndarray) -> Dict[str, float]:
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular_values**2 / max(x.shape[0] - 1, 1)
    if float(variance.sum()) == 0.0:
        variance_ratio = np.ones_like(variance) / len(variance)
    else:
        variance_ratio = variance / variance.sum()
    entropy_h = float(scipy_entropy(variance_ratio, base=10))

    threshold = float(np.abs(x).max() / 100.0)
    sparsity = float((np.abs(x) < threshold).all(axis=1).mean())
    return {
        "entropy_H": entropy_h,
        "sparsity_S": sparsity,
        "sparsity_threshold": threshold,
        "mean_abs_weight": float(np.abs(x).mean()),
        "std_weight": float(x.std()),
        "abs_peak_weight": float(np.abs(x).max()),
    }


def extract_models(model_dir: Path) -> Tuple[Dict[str, dict], List[dict]]:
    model_infos: Dict[str, dict] = {}
    layer_rows: List[dict] = []

    for spec in MODEL_SPECS:
        ckpt_path = model_dir / spec["file"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        print(f"[extract] Loading {ckpt_path}", flush=True)
        checkpoint = load_checkpoint(ckpt_path)
        state_dict = checkpoint.get("backbone_state")
        if not isinstance(state_dict, dict):
            raise ValueError(f"{ckpt_path} does not contain a backbone_state dictionary")

        conv3_items = sort_conv3_items(state_dict)
        if len(conv3_items) != 16:
            raise ValueError(f"Expected 16 RN50 3x3 conv2 tensors in {ckpt_path}, got {len(conv3_items)}")

        arrays = []
        slices: List[LayerSlice] = []
        cursor = 0
        for layer_order, (key, tensor) in enumerate(conv3_items):
            raw = tensor.detach().cpu().float().numpy()
            x = raw.reshape(-1, 9).astype(np.float32, copy=False)
            n = int(x.shape[0])
            start, end = cursor, cursor + n
            cursor = end
            stage = key.split(".")[0]
            conv_depth_norm = layer_order / max(len(conv3_items) - 1, 1)
            quality = layer_quality(x)
            threshold = random_entropy_threshold(n)

            arrays.append(x.copy())
            slices.append(
                LayerSlice(
                    model_id=spec["model_id"],
                    layer_key=key,
                    stage=stage,
                    layer_order=layer_order,
                    conv_depth_norm=conv_depth_norm,
                    start=start,
                    end=end,
                    shape=tuple(int(v) for v in raw.shape),
                )
            )
            layer_rows.append(
                {
                    "model_id": spec["model_id"],
                    "display": spec["display"],
                    "recipe": spec["recipe"],
                    "checkpoint_file": spec["file"],
                    "backbone_metadata": checkpoint.get("backbone", ""),
                    "target_dataset": checkpoint.get("dataset", ""),
                    "best_bacc": checkpoint.get("best_bacc", ""),
                    "epochs": checkpoint.get("epochs", ""),
                    "layer_key": key,
                    "stage": stage,
                    "layer_order": layer_order,
                    "conv_depth_norm": conv_depth_norm,
                    "shape": "x".join(str(v) for v in raw.shape),
                    "num_filters": n,
                    "random_threshold_TH": threshold,
                    "H_over_TH": quality["entropy_H"] / threshold if threshold else np.nan,
                    **quality,
                }
            )

        x_model = np.vstack(arrays).astype(np.float32, copy=False)
        model_infos[spec["model_id"]] = {
            "spec": spec,
            "checkpoint": checkpoint,
            "X": x_model,
            "slices": slices,
            "display": spec["display"],
            "color": spec["color"],
        }
        print(
            f"[extract] {spec['display']}: {x_model.shape[0]:,} filters from {len(slices)} layers",
            flush=True,
        )

    return model_infos, layer_rows


def histogram_probabilities(z: np.ndarray, x_range: Tuple[float, float], bins: int) -> np.ndarray:
    probs = np.zeros((z.shape[1], bins), dtype=np.float64)
    for component in range(z.shape[1]):
        hist, _ = np.histogram(z[:, component], bins=bins, range=x_range)
        hist = hist.astype(np.float64)
        hist[hist == 0] = EPS
        probs[component] = hist / hist.sum()
    return probs


def drift_from_probs(p: np.ndarray, q: np.ndarray, weights: np.ndarray) -> Tuple[float, np.ndarray]:
    contributions = np.zeros(p.shape[0], dtype=np.float64)
    for component in range(p.shape[0]):
        contributions[component] = weights[component] * (
            scipy_entropy(p[component], q[component]) + scipy_entropy(q[component], p[component])
        )
    return float(contributions.sum()), contributions


def compute_drift(
    groups: Dict[str, np.ndarray],
    weights: np.ndarray,
    bins: int,
    x_range: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray], Tuple[float, float]]:
    group_ids = list(groups.keys())
    if x_range is None:
        lo = min(float(np.min(groups[g])) for g in group_ids)
        hi = max(float(np.max(groups[g])) for g in group_ids)
        if lo == hi:
            hi = lo + 1.0
        x_range = (lo, hi)

    probabilities = {gid: histogram_probabilities(groups[gid], x_range, bins) for gid in group_ids}
    matrix = np.zeros((len(group_ids), len(group_ids)), dtype=np.float64)
    contributions: Dict[Tuple[str, str], np.ndarray] = {}
    for i, a in enumerate(group_ids):
        for j, b in enumerate(group_ids):
            total, contrib = drift_from_probs(probabilities[a], probabilities[b], weights)
            matrix[i, j] = total
            contributions[(a, b)] = contrib
    return matrix, contributions, x_range


def matrix_to_rows(matrix: np.ndarray, group_ids: Sequence[str], prefix: dict | None = None) -> List[dict]:
    rows = []
    for i, a in enumerate(group_ids):
        for j, b in enumerate(group_ids):
            row = dict(prefix or {})
            row.update({"group_a": a, "group_b": b, "drift_D": float(matrix[i, j])})
            rows.append(row)
    return rows


def write_matrix_csv(path: Path, matrix: np.ndarray, group_ids: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(group_ids))
        for gid, row in zip(group_ids, matrix):
            writer.writerow([gid] + [f"{value:.10f}" for value in row])


def subset_by_stage(info: dict, stage: str) -> np.ndarray:
    chunks = [info["Z"][s.start : s.end] for s in info["slices"] if s.stage == stage]
    return np.vstack(chunks)


def subset_by_layer(info: dict, layer_order: int) -> Tuple[np.ndarray, LayerSlice]:
    for layer_slice in info["slices"]:
        if layer_slice.layer_order == layer_order:
            return info["Z"][layer_slice.start : layer_slice.end], layer_slice
    raise KeyError(layer_order)


def sample_rows(z: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size <= 0 or sample_size >= len(z):
        return z
    idx = rng.choice(len(z), size=sample_size, replace=False)
    return z[idx]


def bootstrap_global(
    model_infos: Dict[str, dict],
    weights: np.ndarray,
    bins: int,
    reps: int,
    sample_size: int,
    x_range: Tuple[float, float],
    seed: int,
) -> List[dict]:
    if reps <= 0:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        groups = {
            model_id: sample_rows(info["Z"], sample_size, rng)
            for model_id, info in model_infos.items()
        }
        matrix, _, _ = compute_drift(groups, weights, bins, x_range=x_range)
        group_ids = list(groups.keys())
        for a, b in PAIR_ORDER:
            ia, ib = group_ids.index(a), group_ids.index(b)
            rows.append({"rep": rep, "group_a": a, "group_b": b, "drift_D": float(matrix[ia, ib])})
    return rows


def summarize_bootstrap(rows: List[dict]) -> List[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for (a, b), group in df.groupby(["group_a", "group_b"]):
        values = group["drift_D"].to_numpy()
        out.append(
            {
                "group_a": a,
                "group_b": b,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "p025": float(np.quantile(values, 0.025)),
                "p500": float(np.quantile(values, 0.500)),
                "p975": float(np.quantile(values, 0.975)),
                "n_reps": int(len(values)),
            }
        )
    return out


def draw_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    n = len(labels)
    cell = 112
    left = 178
    top = 92
    right = 52
    bottom = 158
    width = left + n * cell + right
    height = top + n * cell + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(15)
    value_font = get_font(18, bold=True)

    draw.text((24, 24), title, font=title_font, fill=(20, 20, 20))
    if vmin is None:
        vmin = float(np.min(matrix))
    if vmax is None:
        vmax = float(np.max(matrix))
    if vmax == vmin:
        vmax = vmin + 1.0

    for i in range(n):
        y = top + i * cell
        draw.text((20, y + cell / 2 - 8), labels[i], font=label_font, fill=(20, 20, 20))
        x = left + i * cell
        text = labels[i]
        tw, th = text_size(draw, text, label_font)
        txt = Image.new("RGBA", (tw + 4, th + 4), (255, 255, 255, 0))
        tdraw = ImageDraw.Draw(txt)
        tdraw.text((2, 2), text, font=label_font, fill=(20, 20, 20))
        txt = txt.rotate(90, expand=True)
        img.paste(txt, (int(x + cell / 2 - txt.width / 2), top + n * cell + 12), txt)

    for i in range(n):
        for j in range(n):
            value = float(matrix[i, j])
            t = (value - vmin) / (vmax - vmin)
            color = heat_color(t)
            x0 = left + j * cell
            y0 = top + i * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color, outline=(210, 210, 210))
            fill = (255, 255, 255) if luminance(color) < 100 else (20, 20, 20)
            draw_centered(draw, (x0 + cell / 2, y0 + cell / 2), f"{value:.3f}", value_font, fill)

    img.save(path)


def draw_eigenfilters(components: np.ndarray, variance_ratio: np.ndarray, path: Path) -> None:
    cell = 34
    gap = 20
    panel_w = cell * 3
    panel_h = cell * 3
    top = 76
    left = 28
    bottom = 46
    width = left * 2 + 9 * panel_w + 8 * gap
    height = top + panel_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(23, bold=True)
    label_font = get_font(19, bold=True)
    small_font = get_font(16)
    draw.text((24, 20), "Target RN50 PCA Eigenfilters", font=title_font, fill=(20, 20, 20))
    max_abs = float(np.abs(components).max())
    for component in range(9):
        x0 = left + component * (panel_w + gap)
        draw_centered(draw, (x0 + panel_w / 2, 52), f"v{component}", label_font)
        values = components[component].reshape(3, 3)
        for r in range(3):
            for c in range(3):
                color = diverging_color(float(values[r, c]), max_abs)
                x = x0 + c * cell
                y = top + r * cell
                draw.rectangle([x, y, x + cell, y + cell], fill=color, outline=(235, 235, 235))
        draw_centered(draw, (x0 + panel_w / 2, top + panel_h + 24), f"{variance_ratio[component]:.2f}", small_font)
    img.save(path)


def smooth_histogram(values: np.ndarray, passes: int = 3) -> np.ndarray:
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kernel = kernel / kernel.sum()
    out = values.astype(np.float64)
    for _ in range(passes):
        out = np.convolve(out, kernel, mode="same")
    return out


def draw_ridge_plot(
    groups: Dict[str, np.ndarray],
    labels: Dict[str, str],
    colors: Dict[str, Tuple[int, int, int]],
    x_range: Tuple[float, float],
    bins: int,
    path: Path,
) -> None:
    group_ids = list(groups.keys())
    rows = len(group_ids)
    cols = 9
    cell_w = 128
    cell_h = 92
    left = 130
    top = 78
    width = left + cols * cell_w + 30
    height = top + rows * cell_h + 58
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(15)
    small_font = get_font(13)
    draw.text((24, 24), "PCA Coefficient Ridge Plot", font=title_font, fill=(20, 20, 20))

    for c in range(cols):
        draw_centered(draw, (left + c * cell_w + cell_w / 2, top - 22), f"c{c}", label_font)
    for r, gid in enumerate(group_ids):
        y0 = top + r * cell_h
        draw.text((18, y0 + cell_h / 2 - 8), labels[gid], font=label_font, fill=(20, 20, 20))
        for c in range(cols):
            x0 = left + c * cell_w
            values = groups[gid][:, c]
            hist, _ = np.histogram(values, bins=bins, range=x_range)
            hist = smooth_histogram(hist)
            if hist.max() > 0:
                hist = hist / hist.max()
            baseline = y0 + cell_h - 20
            max_h = cell_h - 24
            points = [(x0, baseline)]
            for idx, h in enumerate(hist):
                x = x0 + int(round(idx / max(len(hist) - 1, 1) * (cell_w - 10))) + 5
                y = baseline - int(round(h * max_h))
                points.append((x, y))
            points.append((x0 + cell_w - 5, baseline))
            draw.polygon(points, fill=colors[gid])
            draw.line(points[1:-1], fill=(245, 245, 245), width=1)
            draw.line([(x0 + 5, baseline), (x0 + cell_w - 5, baseline)], fill=(60, 60, 60), width=1)

    draw.text((left, height - 34), f"Shared x-range: [{x_range[0]:.2f}, {x_range[1]:.2f}]", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_line_chart(
    rows: Sequence[dict],
    value_key: str,
    title: str,
    ylabel: str,
    path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width, height = 980, 560
    left, right, top, bottom = 92, 210, 70, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(15)
    small_font = get_font(13)
    draw.text((24, 24), title, font=title_font, fill=(20, 20, 20))

    values = [float(row[value_key]) for row in rows if row[value_key] == row[value_key]]
    if y_min is None:
        y_min = min(values) if values else 0.0
    if y_max is None:
        y_max = max(values) if values else 1.0
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    y_min -= pad
    y_max += pad
    if y_max == y_min:
        y_max = y_min + 1.0

    def xmap(x: float) -> float:
        return left + x * plot_w

    def ymap(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    for k in range(6):
        t = k / 5
        y = top + t * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(230, 230, 230))
        value = y_max - t * (y_max - y_min)
        draw.text((20, y - 8), f"{value:.3f}", font=small_font, fill=(80, 80, 80))
    for k in range(6):
        x = left + k / 5 * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill=(235, 235, 235))
        draw_centered(draw, (x, top + plot_h + 24), f"{k/5:.1f}", small_font)

    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(40, 40, 40), width=1)
    draw_centered(draw, (left + plot_w / 2, height - 24), "normalized convolution depth", label_font)
    draw.text((18, top - 28), ylabel, font=label_font, fill=(20, 20, 20))

    row_df = pd.DataFrame(rows)
    for spec in MODEL_SPECS:
        gid = spec["model_id"]
        group = row_df[row_df["model_id"] == gid].sort_values("conv_depth_norm")
        if group.empty:
            continue
        points = [(xmap(float(r.conv_depth_norm)), ymap(float(getattr(r, value_key)))) for r in group.itertuples()]
        draw.line(points, fill=spec["color"], width=3)
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=spec["color"], outline="white")
        lx = left + plot_w + 28
        ly = top + 24 + MODEL_SPECS.index(spec) * 28
        draw.line([(lx, ly), (lx + 28, ly)], fill=spec["color"], width=4)
        draw.text((lx + 36, ly - 8), spec["display"], font=label_font, fill=(20, 20, 20))

    img.save(path)


def draw_pair_line_chart(rows: Sequence[dict], path: Path) -> None:
    width, height = 980, 560
    left, right, top, bottom = 92, 240, 70, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(15)
    small_font = get_font(13)
    draw.text((24, 24), "Layer-Level Drift by Depth", font=title_font, fill=(20, 20, 20))
    df = pd.DataFrame(rows)
    y_min = 0.0
    y_max = float(df["drift_D"].max()) if not df.empty else 1.0
    y_max = y_max * 1.10 if y_max else 1.0

    def xmap(x: float) -> float:
        return left + x * plot_w

    def ymap(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    for k in range(6):
        y = top + k / 5 * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(230, 230, 230))
        value = y_max - k / 5 * (y_max - y_min)
        draw.text((20, y - 8), f"{value:.3f}", font=small_font, fill=(80, 80, 80))
    for k in range(6):
        x = left + k / 5 * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill=(235, 235, 235))
        draw_centered(draw, (x, top + plot_h + 24), f"{k/5:.1f}", small_font)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(40, 40, 40), width=1)
    draw_centered(draw, (left + plot_w / 2, height - 24), "normalized convolution depth", label_font)
    draw.text((18, top - 28), "drift D", font=label_font, fill=(20, 20, 20))

    pair_colors = {
        "scratch_ft vs imagenet_ft": (170, 65, 20),
        "scratch_ft vs lejepa_ssl": (120, 90, 25),
        "imagenet_ft vs lejepa_ssl": (0, 120, 160),
    }
    for idx, (a, b) in enumerate(PAIR_ORDER):
        pair = f"{a} vs {b}"
        group = df[(df["group_a"] == a) & (df["group_b"] == b)].sort_values("conv_depth_norm")
        if group.empty:
            continue
        points = [(xmap(float(r.conv_depth_norm)), ymap(float(r.drift_D))) for r in group.itertuples()]
        color = pair_colors[pair]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline="white")
        lx = left + plot_w + 28
        ly = top + 24 + idx * 32
        draw.line([(lx, ly), (lx + 28, ly)], fill=color, width=4)
        label = f"{model_display(a)} vs {model_display(b)}"
        draw.text((lx + 36, ly - 8), label, font=label_font, fill=(20, 20, 20))

    img.save(path)


def draw_scatter_density(
    model_infos: Dict[str, dict],
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    bins = 220
    groups = {}
    for model_id, info in model_infos.items():
        z = sample_rows(info["Z"][:, :2], max_points, rng)
        groups[model_id] = z
    all_z = np.vstack(list(groups.values()))
    x_range = (float(all_z[:, 0].min()), float(all_z[:, 0].max()))
    y_range = (float(all_z[:, 1].min()), float(all_z[:, 1].max()))

    panel = 300
    gap = 28
    left = 42
    top = 82
    width = left * 2 + 3 * panel + 2 * gap
    height = top + panel + 64
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(16, bold=True)
    small_font = get_font(13)
    draw.text((24, 24), "c0/c1 Density Scatter", font=title_font, fill=(20, 20, 20))

    for idx, spec in enumerate(MODEL_SPECS):
        gid = spec["model_id"]
        z = groups[gid]
        hist, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=bins, range=[x_range, y_range])
        density = np.log1p(hist.T)
        if density.max() > 0:
            density = density / density.max()
        arr = np.zeros((bins, bins, 3), dtype=np.uint8)
        for level in range(256):
            mask = (density >= level / 256) & (density < (level + 1) / 256)
            arr[mask] = density_color(level / 255)
        density_img = Image.fromarray(np.flipud(arr), mode="RGB").resize((panel, panel), Image.Resampling.BILINEAR)
        x0 = left + idx * (panel + gap)
        y0 = top
        img.paste(density_img, (x0, y0))
        draw.rectangle([x0, y0, x0 + panel, y0 + panel], outline=(40, 40, 40), width=1)
        draw_centered(draw, (x0 + panel / 2, y0 - 22), spec["display"], label_font)
    draw.text((left, height - 34), f"x: c0 [{x_range[0]:.2f}, {x_range[1]:.2f}], y: c1 [{y_range[0]:.2f}, {y_range[1]:.2f}]", font=small_font, fill=(80, 80, 80))
    img.save(path)


def draw_component_bars(contributions: Dict[Tuple[str, str], np.ndarray], path: Path) -> None:
    width, height = 980, 560
    left, right, top, bottom = 78, 40, 84, 68
    gap = 34
    panel_w = (width - left - right - 2 * gap) / 3
    panel_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(15)
    small_font = get_font(12)
    draw.text((24, 24), "Global Drift Contributions by PCA Component", font=title_font, fill=(20, 20, 20))
    max_value = max(float(contributions[pair].max()) for pair in PAIR_ORDER)
    max_value = max(max_value, 1e-12)
    colors = [(170, 65, 20), (120, 90, 25), (0, 120, 160)]
    for idx, pair in enumerate(PAIR_ORDER):
        a, b = pair
        values = contributions[pair]
        x0 = left + idx * (panel_w + gap)
        y0 = top
        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline=(50, 50, 50), width=1)
        for k in range(5):
            y = y0 + k / 4 * panel_h
            draw.line([(x0, y), (x0 + panel_w, y)], fill=(235, 235, 235))
        bar_gap = 4
        bar_w = (panel_w - 28 - 8 * bar_gap) / 9
        for component, value in enumerate(values):
            h = float(value) / max_value * (panel_h - 42)
            bx = x0 + 14 + component * (bar_w + bar_gap)
            by = y0 + panel_h - 24 - h
            draw.rectangle([bx, by, bx + bar_w, y0 + panel_h - 24], fill=colors[idx])
            draw_centered(draw, (bx + bar_w / 2, y0 + panel_h - 10), str(component), small_font)
        title = f"{model_display(a)} vs {model_display(b)}"
        draw_centered(draw, (x0 + panel_w / 2, y0 - 24), title, label_font)
    img.save(path)


def model_display(model_id: str) -> str:
    for spec in MODEL_SPECS:
        if spec["model_id"] == model_id:
            return spec["display"]
    return model_id


def save_summary(
    path: Path,
    model_infos: Dict[str, dict],
    global_matrix: np.ndarray,
    group_ids: Sequence[str],
    stage_rows: List[dict],
    quality_rows: List[dict],
    explained_variance_ratio: np.ndarray,
) -> None:
    lines = []
    lines.append("# RN50 Filter Analysis Summary")
    lines.append("")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append("| Model | Backbone metadata | Dataset | Best bacc | 3x3 filters |")
    lines.append("|---|---|---:|---:|---:|")
    for spec in MODEL_SPECS:
        info = model_infos[spec["model_id"]]
        ckpt = info["checkpoint"]
        lines.append(
            f"| {spec['display']} | {ckpt.get('backbone', '')} | {ckpt.get('dataset', '')} | "
            f"{float(ckpt.get('best_bacc', float('nan'))):.4f} | {len(info['X']):,} |"
        )
    lines.append("")
    lines.append("## PCA")
    lines.append("")
    lines.append("Explained variance ratio:")
    lines.append("")
    lines.append("```text")
    lines.append(str([round(float(v), 6) for v in explained_variance_ratio]))
    lines.append("```")
    lines.append("")
    lines.append("## Global Drift")
    lines.append("")
    lines.append("| A | B | Drift D |")
    lines.append("|---|---|---:|")
    for a, b in PAIR_ORDER:
        ia, ib = group_ids.index(a), group_ids.index(b)
        lines.append(f"| {model_display(a)} | {model_display(b)} | {global_matrix[ia, ib]:.6f} |")
    lines.append("")
    lines.append("## Stage Drift")
    lines.append("")
    lines.append("| Stage | A | B | Drift D |")
    lines.append("|---|---|---|---:|")
    for row in stage_rows:
        if row["group_a"] == row["group_b"]:
            continue
        if (row["group_a"], row["group_b"]) not in PAIR_ORDER:
            continue
        lines.append(
            f"| {row['stage']} | {model_display(row['group_a'])} | {model_display(row['group_b'])} | "
            f"{row['drift_D']:.6f} |"
        )
    lines.append("")
    lines.append("## Quality Flags")
    lines.append("")
    lines.append("Flags are heuristic: `S > 0.01`, `H/TH > 0.95`, or `H < 0.50`.")
    lines.append("")
    lines.append("| Model | Layer | H | S | H/TH | Reason |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in quality_rows:
        reasons = []
        if float(row["sparsity_S"]) > 0.01:
            reasons.append("sparse")
        if float(row["H_over_TH"]) > 0.95:
            reasons.append("near_random_threshold")
        if float(row["entropy_H"]) < 0.50:
            reasons.append("low_diversity")
        if reasons:
            lines.append(
                f"| {model_display(row['model_id'])} | {row['layer_key']} | {float(row['entropy_H']):.4f} | "
                f"{float(row['sparsity_S']):.4f} | {float(row['H_over_TH']):.4f} | {', '.join(reasons)} |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to load RN50 checkpoints.")
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    print(f"[start] Output directory: {output_dir}", flush=True)
    model_infos, layer_rows = extract_models(Path(args.model_dir))
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(output_dir / "local_layer_quality.csv", index=False)

    print("[pca] Fitting target RN50 PCA basis", flush=True)
    group_ids = [spec["model_id"] for spec in MODEL_SPECS]
    all_x = np.vstack([model_infos[gid]["X"] for gid in group_ids]).astype(np.float32, copy=False)
    all_scaled = scale_filters(all_x)
    pca = PCA(n_components=9, svd_solver="full", random_state=args.seed)
    all_z = pca.fit_transform(all_scaled).astype(np.float32, copy=False)
    np.savez_compressed(
        output_dir / "pca_target_rn50.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
    )

    cursor = 0
    for gid in group_ids:
        n = len(model_infos[gid]["X"])
        model_infos[gid]["Z"] = all_z[cursor : cursor + n]
        cursor += n
    del all_x, all_scaled, all_z

    weights = pca.explained_variance_ratio_.astype(np.float64)
    labels = {spec["model_id"]: spec["display"] for spec in MODEL_SPECS}
    colors = {spec["model_id"]: spec["color"] for spec in MODEL_SPECS}

    print("[drift] Computing global drift", flush=True)
    global_groups = {gid: model_infos[gid]["Z"] for gid in group_ids}
    global_matrix, global_contribs_all, global_range = compute_drift(
        global_groups, weights, bins=args.bins
    )
    global_contribs = {pair: global_contribs_all[pair] for pair in PAIR_ORDER}
    write_matrix_csv(output_dir / "drift_global_target_basis.csv", global_matrix, group_ids)
    pd.DataFrame(matrix_to_rows(global_matrix, group_ids)).to_csv(
        output_dir / "drift_global_pairs.csv", index=False
    )
    contribution_rows = []
    for (a, b), values in global_contribs.items():
        for component, value in enumerate(values):
            contribution_rows.append(
                {
                    "group_a": a,
                    "group_b": b,
                    "component": component,
                    "weighted_contribution": float(value),
                    "component_weight": float(weights[component]),
                }
            )
    pd.DataFrame(contribution_rows).to_csv(output_dir / "drift_component_contributions.csv", index=False)

    print("[drift] Computing stage drift", flush=True)
    stage_rows: List[dict] = []
    stage_matrices = {}
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        stage_groups = {gid: subset_by_stage(model_infos[gid], stage) for gid in group_ids}
        matrix, _, _ = compute_drift(stage_groups, weights, bins=args.bins)
        stage_matrices[stage] = matrix
        stage_rows.extend(matrix_to_rows(matrix, group_ids, prefix={"stage": stage}))
        write_matrix_csv(output_dir / f"drift_stage_{stage}.csv", matrix, group_ids)
    pd.DataFrame(stage_rows).to_csv(output_dir / "drift_stage_target_basis.csv", index=False)

    print("[drift] Computing layer drift", flush=True)
    layer_rows_out: List[dict] = []
    for layer_order in range(16):
        layer_groups = {}
        layer_info = None
        for gid in group_ids:
            z, layer_slice = subset_by_layer(model_infos[gid], layer_order)
            layer_groups[gid] = z
            layer_info = layer_slice
        matrix, _, _ = compute_drift(layer_groups, weights, bins=args.bins)
        assert layer_info is not None
        for row in matrix_to_rows(
            matrix,
            group_ids,
            prefix={
                "layer_order": layer_order,
                "layer_key": layer_info.layer_key,
                "stage": layer_info.stage,
                "conv_depth_norm": layer_info.conv_depth_norm,
            },
        ):
            layer_rows_out.append(row)
    pd.DataFrame(layer_rows_out).to_csv(output_dir / "drift_layer_target_basis.csv", index=False)

    print("[bootstrap] Computing global bootstrap CIs", flush=True)
    bootstrap_rows = bootstrap_global(
        model_infos,
        weights,
        bins=args.bins,
        reps=args.bootstrap_reps,
        sample_size=args.bootstrap_sample_size,
        x_range=global_range,
        seed=args.seed + 101,
    )
    if bootstrap_rows:
        pd.DataFrame(bootstrap_rows).to_csv(output_dir / "bootstrap_global_samples.csv", index=False)
        pd.DataFrame(summarize_bootstrap(bootstrap_rows)).to_csv(
            output_dir / "bootstrap_global_confidence_intervals.csv", index=False
        )

    print("[plot] Writing figures", flush=True)
    display_labels = [labels[gid] for gid in group_ids]
    draw_eigenfilters(
        pca.components_,
        pca.explained_variance_ratio_,
        figure_dir / "pca_eigenfilters_target_basis.png",
    )
    draw_heatmap(
        global_matrix,
        display_labels,
        "Global KL Drift: all RN50 3x3 filters",
        figure_dir / "drift_global_heatmap.png",
    )
    stage_vmax = max(float(m.max()) for m in stage_matrices.values())
    for stage, matrix in stage_matrices.items():
        draw_heatmap(
            matrix,
            display_labels,
            f"Stage KL Drift: {stage}",
            figure_dir / f"drift_stage_{stage}_heatmap.png",
            vmin=0.0,
            vmax=stage_vmax,
        )
    draw_line_chart(
        layer_rows,
        "entropy_H",
        "Layer Entropy H by Depth",
        "entropy H",
        figure_dir / "entropy_by_depth.png",
        y_min=0.0,
        y_max=max(1.0, float(layer_df["entropy_H"].max())),
    )
    draw_line_chart(
        layer_rows,
        "sparsity_S",
        "Layer Sparsity S by Depth",
        "sparsity S",
        figure_dir / "sparsity_by_depth.png",
        y_min=0.0,
        y_max=max(0.1, float(layer_df["sparsity_S"].max())),
    )
    draw_line_chart(
        layer_rows,
        "H_over_TH",
        "Entropy Relative to Random Threshold",
        "H / TH",
        figure_dir / "entropy_over_random_threshold_by_depth.png",
        y_min=0.0,
        y_max=max(1.05, float(layer_df["H_over_TH"].max())),
    )
    draw_ridge_plot(
        global_groups,
        labels,
        colors,
        global_range,
        bins=args.bins,
        path=figure_dir / "ridge_global_three_models.png",
    )
    draw_scatter_density(
        model_infos,
        figure_dir / "scatter_c0_c1_by_model.png",
        max_points=args.max_scatter_points,
        seed=args.seed + 202,
    )
    draw_component_bars(global_contribs, figure_dir / "component_drift_contributions.png")
    draw_pair_line_chart(layer_rows_out, figure_dir / "drift_layer_pairs_by_depth.png")

    save_summary(
        output_dir / "summary.md",
        model_infos,
        global_matrix,
        group_ids,
        stage_rows,
        layer_rows,
        pca.explained_variance_ratio_,
    )

    print("[done] Analysis complete", flush=True)
    print(f"[done] Summary: {output_dir / 'summary.md'}", flush=True)
    print(f"[done] Figures: {figure_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
