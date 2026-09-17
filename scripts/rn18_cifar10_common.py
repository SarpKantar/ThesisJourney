#!/usr/bin/env python
"""Shared helpers for the ResNet18/CIFAR10 third experiment."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EPS = np.finfo(np.float64).eps
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

RESNET18_FILTER_RE = re.compile(r"layer(\d+)\.(\d+)\.conv([12])\.weight$")


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
        (0.00, (247, 247, 247)),
        (0.20, (209, 229, 240)),
        (0.42, (103, 169, 207)),
        (0.65, (33, 102, 172)),
        (0.83, (178, 24, 43)),
        (1.00, (103, 0, 31)),
    ]
    t = max(0.0, min(1.0, float(t)))
    for (ta, ca), (tb, cb) in zip(stops[:-1], stops[1:]):
        if ta <= t <= tb:
            return lerp_color(ca, cb, (t - ta) / max(tb - ta, 1e-12))
    return stops[-1][1]


def density_color(t: float) -> Tuple[int, int, int]:
    stops = [
        (0.00, (255, 255, 255)),
        (0.22, (229, 245, 249)),
        (0.45, (153, 216, 201)),
        (0.68, (44, 162, 95)),
        (0.88, (35, 83, 141)),
        (1.00, (8, 29, 88)),
    ]
    t = max(0.0, min(1.0, float(t)))
    for (ta, ca), (tb, cb) in zip(stops[:-1], stops[1:]):
        if ta <= t <= tb:
            return lerp_color(ca, cb, (t - ta) / max(tb - ta, 1e-12))
    return stops[-1][1]


def diverging_color(value: float, max_abs: float) -> Tuple[int, int, int]:
    if max_abs <= 0:
        return (245, 245, 245)
    x = max(-1.0, min(1.0, value / max_abs))
    if x < 0:
        return lerp_color((49, 54, 149), (247, 247, 247), x + 1.0)
    return lerp_color((247, 247, 247), (165, 0, 38), x)


def luminance(rgb: Tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def save_rows_csv(path: Path, rows: Sequence[dict]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


def load_backbone_state(checkpoint: dict) -> dict:
    state = checkpoint.get("backbone_state")
    if isinstance(state, dict):
        return state
    state = checkpoint.get("model_state")
    if isinstance(state, dict):
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("fc.") and not key.startswith("module.fc.")
        }
    raise ValueError("Checkpoint does not contain backbone_state or model_state")


def sorted_resnet18_filter_items(state_dict: dict, include_stem: bool = True) -> list[tuple[str, object]]:
    items: list[tuple[int, int, int, str, object]] = []
    if include_stem and "conv1.weight" in state_dict:
        tensor = state_dict["conv1.weight"]
        if hasattr(tensor, "shape") and tuple(int(v) for v in tensor.shape[-2:]) == (3, 3):
            items.append((0, 0, 0, "conv1.weight", tensor))
    for key, value in state_dict.items():
        if not hasattr(value, "shape") or len(value.shape) != 4:
            continue
        shape = tuple(int(v) for v in value.shape)
        if shape[-2:] != (3, 3):
            continue
        match = RESNET18_FILTER_RE.match(key)
        if not match:
            continue
        stage = int(match.group(1))
        block = int(match.group(2))
        conv = int(match.group(3))
        items.append((stage, block, conv, key, value))
    items.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [(key, value) for _, _, _, key, value in items]


def layer_stage(layer_key: str) -> str:
    if layer_key == "conv1.weight":
        return "stem"
    return layer_key.split(".")[0]


def tensor_to_numpy(tensor: object) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def extract_filter_bank(
    state_dict: dict,
    model_id: str,
    include_stem: bool = True,
) -> tuple[np.ndarray, list[LayerSlice], list[dict]]:
    arrays = []
    slices: list[LayerSlice] = []
    rows: list[dict] = []
    cursor = 0
    items = sorted_resnet18_filter_items(state_dict, include_stem=include_stem)
    if not items:
        raise ValueError(f"No ResNet18 3x3 convolution filters found for {model_id}")
    for layer_order, (key, tensor) in enumerate(items):
        raw = tensor_to_numpy(tensor)
        x = raw.reshape(-1, 9).astype(np.float32, copy=False)
        n = int(x.shape[0])
        start, end = cursor, cursor + n
        cursor = end
        conv_depth_norm = layer_order / max(len(items) - 1, 1)
        quality = layer_quality(x)
        threshold = random_entropy_threshold(n)
        stage = layer_stage(key)
        arrays.append(x.copy())
        slices.append(
            LayerSlice(
                model_id=model_id,
                layer_key=key,
                stage=stage,
                layer_order=layer_order,
                conv_depth_norm=conv_depth_norm,
                start=start,
                end=end,
                shape=tuple(int(v) for v in raw.shape),
            )
        )
        rows.append(
            {
                "model_id": model_id,
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
    return np.vstack(arrays).astype(np.float32, copy=False), slices, rows


def scale_filters(x: np.ndarray) -> np.ndarray:
    den = np.abs(x).max(axis=1)
    den = np.where(den == 0, 1.0, den)[:, None]
    return x / den


def random_entropy_threshold(n: int) -> float:
    return 1.26 / (1.0 + math.exp(-0.89 * (math.log2(max(n, 1)) - 2.30))) - 0.31


def entropy_base10(probabilities: np.ndarray) -> float:
    p = probabilities.astype(np.float64, copy=False)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log10(p)).sum())


def layer_quality(x: np.ndarray) -> dict[str, float]:
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular_values**2 / max(x.shape[0] - 1, 1)
    if float(variance.sum()) == 0.0:
        variance_ratio = np.ones_like(variance) / len(variance)
    else:
        variance_ratio = variance / variance.sum()
    threshold = float(np.abs(x).max() / 100.0)
    return {
        "entropy_H": entropy_base10(variance_ratio),
        "sparsity_S": float((np.abs(x) < threshold).all(axis=1).mean()),
        "sparsity_threshold": threshold,
        "mean_abs_weight": float(np.abs(x).mean()),
        "std_weight": float(x.std()),
        "abs_peak_weight": float(np.abs(x).max()),
    }


def fit_pca(x: np.ndarray, n_components: int = 9) -> dict[str, np.ndarray]:
    x64 = x.astype(np.float64, copy=False)
    mean = x64.mean(axis=0)
    centered = x64 - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = singular_values**2 / max(x64.shape[0] - 1, 1)
    total = float(explained_variance.sum())
    if total == 0.0:
        explained_ratio = np.zeros_like(explained_variance)
    else:
        explained_ratio = explained_variance / total
    components = vt[:n_components]
    z = centered @ components.T
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance": explained_variance[:n_components].astype(np.float32),
        "explained_variance_ratio": explained_ratio[:n_components].astype(np.float32),
        "singular_values": singular_values[:n_components].astype(np.float32),
        "z": z.astype(np.float32),
    }


def project_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32, copy=False) - mean.astype(np.float32)) @ components.T).astype(
        np.float32,
        copy=False,
    )


def probability_histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges)
    hist = hist.astype(np.float64)
    hist[hist == 0] = EPS
    return hist / hist.sum()


def component_edges(groups: Iterable[np.ndarray], bins: int) -> np.ndarray:
    arrays = list(groups)
    lo = min(float(np.min(x)) for x in arrays)
    hi = max(float(np.max(x)) for x in arrays)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite PCA coefficient range")
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = (hi - lo) * 0.001
    return np.linspace(lo - pad, hi + pad, bins + 1, dtype=np.float64)


def histogram_probabilities(z: np.ndarray, edges_by_component: Sequence[np.ndarray]) -> np.ndarray:
    probs = np.zeros((z.shape[1], len(edges_by_component[0]) - 1), dtype=np.float64)
    for component in range(z.shape[1]):
        probs[component] = probability_histogram(z[:, component], edges_by_component[component])
    return probs


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = np.where(p <= 0, EPS, p)
    q_safe = np.where(q <= 0, EPS, q)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)) + np.sum(q_safe * np.log(q_safe / p_safe)))


def drift_from_probs(p: np.ndarray, q: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    contributions = np.zeros(p.shape[0], dtype=np.float64)
    for component in range(p.shape[0]):
        contributions[component] = float(weights[component]) * symmetric_kl(p[component], q[component])
    return float(contributions.sum()), contributions


def compute_drift(groups: dict[str, np.ndarray], weights: np.ndarray, bins: int) -> tuple[np.ndarray, dict]:
    group_ids = list(groups.keys())
    edges_by_component = [
        component_edges([groups[gid][:, component] for gid in group_ids], bins)
        for component in range(next(iter(groups.values())).shape[1])
    ]
    probabilities = {
        gid: histogram_probabilities(groups[gid], edges_by_component)
        for gid in group_ids
    }
    matrix = np.zeros((len(group_ids), len(group_ids)), dtype=np.float64)
    contributions = {}
    for i, a in enumerate(group_ids):
        for j, b in enumerate(group_ids):
            total, contrib = drift_from_probs(probabilities[a], probabilities[b], weights)
            matrix[i, j] = total
            contributions[(a, b)] = contrib
    return matrix, contributions


def matrix_to_rows(matrix: np.ndarray, group_ids: Sequence[str], prefix: dict | None = None) -> list[dict]:
    rows = []
    for i, a in enumerate(group_ids):
        for j, b in enumerate(group_ids):
            row = dict(prefix or {})
            row.update({"group_a": a, "group_b": b, "drift_D": float(matrix[i, j])})
            rows.append(row)
    return rows


def subset_by_stage(info: dict, stage: str) -> np.ndarray:
    chunks = [info["Z"][s.start : s.end] for s in info["slices"] if s.stage == stage]
    if not chunks:
        raise KeyError(stage)
    return np.vstack(chunks)


def subset_by_layer(info: dict, layer_order: int) -> tuple[np.ndarray, LayerSlice]:
    for layer_slice in info["slices"]:
        if layer_slice.layer_order == layer_order:
            return info["Z"][layer_slice.start : layer_slice.end], layer_slice
    raise KeyError(layer_order)


def sample_rows(z: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size <= 0 or sample_size >= len(z):
        return z
    idx = rng.choice(len(z), size=sample_size, replace=False)
    return z[idx]


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
    right = 56
    bottom = 148
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
    for i, label in enumerate(labels):
        y = top + i * cell
        draw.text((20, y + cell / 2 - 8), label, font=label_font, fill=(20, 20, 20))
        x = left + i * cell
        tw, th = text_size(draw, label, label_font)
        txt = Image.new("RGBA", (tw + 4, th + 4), (255, 255, 255, 0))
        tdraw = ImageDraw.Draw(txt)
        tdraw.text((2, 2), label, font=label_font, fill=(20, 20, 20))
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


def draw_eigenfilters(components: np.ndarray, variance_ratio: np.ndarray, path: Path, title: str) -> None:
    cell = 34
    gap = 20
    panel_w = cell * 3
    panel_h = cell * 3
    top = 76
    left = 28
    bottom = 46
    width = left * 2 + components.shape[0] * panel_w + max(components.shape[0] - 1, 0) * gap
    height = top + panel_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(23, bold=True)
    label_font = get_font(19, bold=True)
    small_font = get_font(16)
    draw.text((24, 20), title, font=title_font, fill=(20, 20, 20))
    max_abs = float(np.abs(components).max())
    for component in range(components.shape[0]):
        x0 = left + component * (panel_w + gap)
        draw_centered(draw, (x0 + panel_w / 2, 52), f"c{component}", label_font)
        values = components[component].reshape(3, 3)
        for r in range(3):
            for c in range(3):
                color = diverging_color(float(values[r, c]), max_abs)
                x = x0 + c * cell
                y = top + r * cell
                draw.rectangle([x, y, x + cell, y + cell], fill=color, outline=(235, 235, 235))
        draw_centered(
            draw,
            (x0 + panel_w / 2, top + panel_h + 24),
            f"{variance_ratio[component]:.2f}",
            small_font,
        )
    img.save(path)


def smooth_histogram(values: np.ndarray, passes: int = 3) -> np.ndarray:
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kernel = kernel / kernel.sum()
    out = values.astype(np.float64)
    for _ in range(passes):
        out = np.convolve(out, kernel, mode="same")
    return out


def draw_ridge_plot(
    groups: dict[str, np.ndarray],
    labels: dict[str, str],
    colors: dict[str, Tuple[int, int, int]],
    bins: int,
    path: Path,
    title: str,
) -> None:
    group_ids = list(groups.keys())
    cols = min(9, next(iter(groups.values())).shape[1])
    rows = len(group_ids)
    all_z = np.vstack([groups[gid][:, :cols] for gid in group_ids])
    x_range = (float(all_z.min()), float(all_z.max()))
    cell_w = 128
    cell_h = 92
    left = 150
    top = 78
    width = left + cols * cell_w + 30
    height = top + rows * cell_h + 58
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(15)
    small_font = get_font(13)
    draw.text((24, 24), title, font=title_font, fill=(20, 20, 20))
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
    draw.text(
        (left, height - 34),
        f"Shared coefficient range: [{x_range[0]:.2f}, {x_range[1]:.2f}]",
        font=small_font,
        fill=(80, 80, 80),
    )
    img.save(path)


def draw_line_series(
    series: Sequence[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width, height = 1040, 600
    left, right, top, bottom = 96, 250, 76, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(24, bold=True)
    label_font = get_font(15)
    small_font = get_font(13)
    draw.text((24, 24), title, font=title_font, fill=(20, 20, 20))
    all_x = [float(x) for item in series for x in item["x"]]
    all_y = [float(y) for item in series for y in item["y"] if np.isfinite(float(y))]
    x_min = min(all_x) if all_x else 0.0
    x_max = max(all_x) if all_x else 1.0
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_min is None:
        y_min = min(all_y) if all_y else 0.0
    if y_max is None:
        y_max = max(all_y) if all_y else 1.0
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    y_min -= y_pad
    y_max += y_pad
    if y_max == y_min:
        y_max = y_min + 1.0

    def xmap(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def ymap(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    for k in range(6):
        y = top + k / 5 * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(230, 230, 230))
        value = y_max - k / 5 * (y_max - y_min)
        draw.text((22, y - 8), f"{value:.3f}", font=small_font, fill=(80, 80, 80))
    for k in range(6):
        x = left + k / 5 * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill=(235, 235, 235))
        value = x_min + k / 5 * (x_max - x_min)
        draw_centered(draw, (x, top + plot_h + 24), f"{value:.1f}", small_font)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(40, 40, 40), width=1)
    draw_centered(draw, (left + plot_w / 2, height - 24), xlabel, label_font)
    draw.text((20, top - 30), ylabel, font=label_font, fill=(20, 20, 20))
    for idx, item in enumerate(series):
        xs = [float(v) for v in item["x"]]
        ys = [float(v) for v in item["y"]]
        points = [(xmap(x), ymap(y)) for x, y in zip(xs, ys) if np.isfinite(y)]
        color = item["color"]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline="white")
        lx = left + plot_w + 28
        ly = top + 24 + idx * 30
        draw.line([(lx, ly), (lx + 28, ly)], fill=color, width=4)
        draw.text((lx + 36, ly - 8), item["label"], font=label_font, fill=(20, 20, 20))
    img.save(path)
