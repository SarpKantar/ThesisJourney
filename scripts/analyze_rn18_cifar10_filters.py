#!/usr/bin/env python
"""Analyze matched ResNet18/CIFAR10 checkpoints from the third experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by runtime envs.
    raise ModuleNotFoundError(
        "This script requires PyTorch to load checkpoints. Use the `minitron` "
        "conda environment or the provided Slurm job."
    ) from exc

from PIL import Image, ImageDraw

import rn18_cifar10_common as common


MODEL_SPECS = [
    {
        "model_id": "imagenet_ft",
        "display": "ImageNet FT",
        "color": (0, 114, 178),
        "path_key": "imagenet_checkpoint_path",
        "acc_key": "imagenet_accuracy",
    },
    {
        "model_id": "lejepa_ssl_linear",
        "display": "LeJEPA SSL + Linear",
        "color": (0, 158, 115),
        "path_key": "lejepa_linear_checkpoint_path",
        "acc_key": "lejepa_accuracy",
    },
]

PAIR = ("imagenet_ft", "lejepa_ssl_linear")
STAGE_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4"]
EXPECTED_FULL_DB_FILTERS = 1_464_797_156
FULL_DB_PREPROCESSING_ID = "cnn_filter_db_paper_float16_maxabs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RN18/CIFAR10 matched checkpoints.")
    parser.add_argument(
        "--experiment-dir",
        default="outputs/rn18_cifar10_experiment",
        help="Directory produced by train_rn18_cifar10_experiment.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Analysis output directory. Defaults to <experiment-dir>/analysis.",
    )
    parser.add_argument("--bins", type=int, default=70)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-density-points", type=int, default=400_000)
    parser.add_argument(
        "--pca-basis",
        default="",
        help=(
            "Optional reference PCA .npz. When supplied, its mean, components, and "
            "explained-variance ratios are used instead of fitting a PCA on the matched pair."
        ),
    )
    parser.add_argument(
        "--basis-label",
        default="CNN Filter DB reference basis",
        help="Human-readable label used when --pca-basis is supplied.",
    )
    parser.add_argument(
        "--selected-pair-file",
        default="",
        help="Optional selected_pair.json; defaults to <experiment-dir>/selected_pair.json.",
    )
    parser.add_argument("--imagenet-checkpoint", default="", help="Override the selected ImageNet checkpoint path.")
    parser.add_argument("--lejepa-checkpoint", default="", help="Override the selected LeJEPA checkpoint path.")
    parser.add_argument("--imagenet-accuracy", type=float, default=None)
    parser.add_argument("--lejepa-accuracy", type=float, default=None)
    parser.add_argument("--imagenet-epoch", type=int, default=None)
    parser.add_argument("--lejepa-epoch", type=int, default=None)
    parser.add_argument("--include-stem", action="store_true", default=True)
    parser.add_argument(
        "--no-include-stem",
        action="store_false",
        dest="include_stem",
        help="Exclude the CIFAR stem conv1 filter bank from filter analysis.",
    )
    return parser.parse_args()


def resolve_path(path_text: str, experiment_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = experiment_dir / path_text
    if candidate.exists():
        return candidate
    return path


def load_selected_pair(experiment_dir: Path, selected_pair_file: str = "") -> dict:
    selected_path = Path(selected_pair_file) if selected_pair_file else experiment_dir / "selected_pair.json"
    if selected_path.exists():
        payload = json.loads(selected_path.read_text())
        return payload["selected"]
    matches_path = experiment_dir / "matched_checkpoints.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"Missing {selected_path} and {matches_path}")
    df = pd.read_csv(matches_path).sort_values("rank")
    return df.iloc[0].to_dict()


def apply_selected_overrides(selected: dict, args: argparse.Namespace) -> dict:
    """Return a fresh selected-pair record with explicit CLI overrides applied."""
    selected = dict(selected)
    if args.imagenet_checkpoint:
        selected["imagenet_checkpoint_path"] = args.imagenet_checkpoint
    if args.lejepa_checkpoint:
        # The filter loader only needs the embedded backbone, so an SSL backbone
        # checkpoint and a frozen-linear checkpoint are both accepted here.
        selected["lejepa_linear_checkpoint_path"] = args.lejepa_checkpoint
        selected["lejepa_ssl_checkpoint_path"] = args.lejepa_checkpoint
    if args.imagenet_accuracy is not None:
        selected["imagenet_accuracy"] = float(args.imagenet_accuracy)
    if args.lejepa_accuracy is not None:
        selected["lejepa_accuracy"] = float(args.lejepa_accuracy)
    if args.imagenet_epoch is not None:
        selected["imagenet_epoch"] = int(args.imagenet_epoch)
    if args.lejepa_epoch is not None:
        selected["lejepa_ssl_epoch"] = int(args.lejepa_epoch)
    if "imagenet_accuracy" in selected and "lejepa_accuracy" in selected:
        selected["abs_accuracy_delta"] = abs(
            float(selected["imagenet_accuracy"]) - float(selected["lejepa_accuracy"])
        )
    return selected


def npz_scalar(payload: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"Reference PCA provenance key {key!r} must be scalar, got {value.shape}")
    return value.reshape(()).item()


def reference_scale_filters(x: np.ndarray, preprocessing_id: str) -> np.ndarray:
    """Apply the target preprocessing used while fitting the external basis."""
    if preprocessing_id == FULL_DB_PREPROCESSING_ID:
        values16 = np.asarray(x).reshape(-1, 9).astype(np.float16)
        denominator = np.abs(values16).max(axis=1)
        denominator = np.where(denominator == 0, np.float16(1.0), denominator)[:, None]
        return (values16 / denominator).astype(np.float32)
    return common.scale_filters(x).astype(np.float32, copy=False)


def load_reference_pca(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = ["mean", "components", "explained_variance", "explained_variance_ratio"]
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"Reference PCA is missing arrays: {missing}")
        numeric_keys = required + (["singular_values"] if "singular_values" in payload.files else [])
        result = {key: np.asarray(payload[key]).copy() for key in numeric_keys}
        provenance = {}
        for key in payload.files:
            if key in numeric_keys or key == "covariance":
                continue
            value = np.asarray(payload[key])
            provenance[key] = value.reshape(()).item() if value.ndim == 0 else value.tolist()
    mean = np.asarray(result["mean"], dtype=np.float64)
    components = np.asarray(result["components"], dtype=np.float64)
    weights = np.asarray(result["explained_variance_ratio"], dtype=np.float64)
    if mean.shape != (9,) or components.shape != (9, 9) or weights.shape != (9,):
        raise ValueError(
            "Reference PCA must contain mean (9,), components (9, 9), and "
            f"explained_variance_ratio (9,); got {mean.shape}, {components.shape}, {weights.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(components).all() or not np.isfinite(weights).all():
        raise ValueError("Reference PCA contains non-finite values")
    orthogonality_error = float(np.max(np.abs(components @ components.T - np.eye(9))))
    if orthogonality_error > 1e-4:
        raise ValueError(f"Reference PCA components are not orthonormal (max error {orthogonality_error:.3e})")
    if np.any(weights < 0) or not np.isclose(float(weights.sum()), 1.0, atol=1e-4):
        raise ValueError(f"Invalid explained-variance ratios (sum={weights.sum():.8f})")
    result["mean"] = mean.astype(np.float32)
    result["components"] = components.astype(np.float32)
    result["explained_variance_ratio"] = weights.astype(np.float32)

    # Full CNN Filter DB artifacts carry enough information to prove that the
    # basis was fitted on every released filter.  Validate that evidence here,
    # while keeping backwards compatibility with simpler reference NPZ files.
    preprocessing_id = str(provenance.get("preprocessing_id", "legacy_float32_maxabs"))
    full_keys = {
        "n_samples",
        "full_dataset_filter_count",
        "filter_index_start",
        "filter_index_stop_exclusive",
        "is_full_dataset_fit",
        "n_features",
        "preprocessing_id",
    }
    if full_keys.intersection(provenance) and not full_keys.issubset(provenance):
        missing = sorted(full_keys - set(provenance))
        raise ValueError(f"Reference PCA has incomplete full-dataset provenance: {missing}")
    if full_keys.issubset(provenance):
        expected = EXPECTED_FULL_DB_FILTERS
        if (
            int(provenance["n_samples"]) != expected
            or int(provenance["full_dataset_filter_count"]) != expected
            or int(provenance["filter_index_start"]) != 0
            or int(provenance["filter_index_stop_exclusive"]) != expected
            or not bool(provenance["is_full_dataset_fit"])
            or int(provenance["n_features"]) != 9
        ):
            raise ValueError("Reference PCA does not describe a complete 1,464,797,156-filter fit")
        if preprocessing_id != FULL_DB_PREPROCESSING_ID:
            raise ValueError(
                f"Unexpected full-DB preprocessing {preprocessing_id!r}; "
                f"expected {FULL_DB_PREPROCESSING_ID!r}"
            )
        provenance["consumer_full_dataset_validation_passed"] = True
    else:
        provenance["consumer_full_dataset_validation_passed"] = False
    provenance["consumer_orthogonality_max_abs_error"] = orthogonality_error
    provenance["consumer_explained_variance_ratio_sum"] = float(weights.sum())
    provenance["consumer_projection_preprocessing_id"] = preprocessing_id
    return result, provenance


def load_model_infos(selected: dict, experiment_dir: Path, include_stem: bool) -> tuple[dict[str, dict], list[dict]]:
    infos: dict[str, dict] = {}
    quality_rows: list[dict] = []
    for spec in MODEL_SPECS:
        path = resolve_path(str(selected[spec["path_key"]]), experiment_dir)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[extract] Loading {spec['display']}: {path}", flush=True)
        checkpoint = torch.load(path, map_location="cpu")
        state = common.load_backbone_state(checkpoint)
        x, slices, rows = common.extract_filter_bank(
            state,
            spec["model_id"],
            include_stem=include_stem,
        )
        for row in rows:
            row.update(
                {
                    "display": spec["display"],
                    "checkpoint_path": str(path),
                    "test_accuracy": float(selected[spec["acc_key"]]),
                }
            )
        quality_rows.extend(rows)
        infos[spec["model_id"]] = {
            "spec": spec,
            "checkpoint": checkpoint,
            "checkpoint_path": path,
            "X": x,
            "slices": slices,
            "display": spec["display"],
            "color": spec["color"],
            "test_accuracy": float(selected[spec["acc_key"]]),
        }
        print(f"[extract] {spec['display']}: {len(x):,} 3x3 filters", flush=True)
    return infos, quality_rows


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    with path.open("w") as handle:
        handle.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            handle.write(label + "," + ",".join(f"{float(value):.10f}" for value in row) + "\n")


def residual_histogram_rows(
    z_a: np.ndarray,
    z_b: np.ndarray,
    weights: np.ndarray,
    bins: int,
    prefix: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    hist_rows: list[dict] = []
    summary_rows: list[dict] = []
    prefix = dict(prefix or {})
    for component in range(z_a.shape[1]):
        a_values = z_a[:, component]
        b_values = z_b[:, component]
        edges = common.component_edges([a_values, b_values], bins)
        p_a = common.probability_histogram(a_values, edges)
        p_b = common.probability_histogram(b_values, edges)
        residual = p_b - p_a
        centers = (edges[:-1] + edges[1:]) / 2.0
        tv = 0.5 * float(np.abs(residual).sum())
        skl = common.symmetric_kl(p_a, p_b)
        max_pos = int(np.argmax(residual))
        max_neg = int(np.argmin(residual))
        summary_row = dict(prefix)
        summary_row.update(
            {
                "component": component,
                "component_weight": float(weights[component]),
                "mean_imagenet": float(np.mean(a_values)),
                "mean_lejepa": float(np.mean(b_values)),
                "mean_delta_lejepa_minus_imagenet": float(np.mean(b_values) - np.mean(a_values)),
                "std_imagenet": float(np.std(a_values)),
                "std_lejepa": float(np.std(b_values)),
                "std_delta_lejepa_minus_imagenet": float(np.std(b_values) - np.std(a_values)),
                "tv_distance": tv,
                "weighted_tv_distance": float(weights[component] * tv),
                "symmetric_kl": skl,
                "weighted_symmetric_kl": float(weights[component] * skl),
                "max_positive_bin_center": float(centers[max_pos]),
                "max_positive_residual": float(residual[max_pos]),
                "max_negative_bin_center": float(centers[max_neg]),
                "max_negative_residual": float(residual[max_neg]),
            }
        )
        summary_rows.append(summary_row)
        for idx in range(bins):
            row = dict(prefix)
            row.update(
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
            hist_rows.append(row)
    return hist_rows, summary_rows


def draw_component_bars(rows: Sequence[dict], path: Path) -> None:
    df = pd.DataFrame(rows).sort_values("component")
    width, height = 980, 560
    left, right, top, bottom = 88, 48, 82, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(15)
    small_font = common.get_font(12)
    draw.text((24, 24), "Matched Pair Drift Contributions by PCA Component", font=title_font, fill=(20, 20, 20))
    values = df["weighted_symmetric_kl"].to_numpy(dtype=float)
    max_value = max(float(values.max()) if len(values) else 0.0, 1e-12)
    for k in range(5):
        y = top + k / 4 * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(235, 235, 235))
        label = max_value * (1 - k / 4)
        draw.text((18, y - 8), f"{label:.4f}", font=small_font, fill=(80, 80, 80))
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(50, 50, 50), width=1)
    bar_gap = 18
    bar_w = (plot_w - 2 * 24 - 8 * bar_gap) / 9
    for _, row in df.iterrows():
        component = int(row.component)
        value = float(row.weighted_symmetric_kl)
        x0 = left + 24 + component * (bar_w + bar_gap)
        bar_h = value / max_value * (plot_h - 36)
        y0 = top + plot_h - 22 - bar_h
        color = common.heat_color(value / max_value)
        draw.rectangle([x0, y0, x0 + bar_w, top + plot_h - 22], fill=color)
        common.draw_centered(draw, (x0 + bar_w / 2, top + plot_h - 9), str(component), small_font)
    common.draw_centered(draw, (left + plot_w / 2, height - 28), "PCA component", label_font)
    draw.text((18, top - 30), "weighted symmetric KL", font=label_font, fill=(20, 20, 20))
    img.save(path)


def draw_confusion_matrices(path: Path, confusion_csv: Path) -> None:
    if not confusion_csv.exists():
        return
    df = pd.read_csv(confusion_csv)
    models = list(df["model_id"].unique())
    cell = 34
    panel = cell * 10
    gap = 52
    left, top = 128, 92
    width = left + len(models) * panel + max(len(models) - 1, 0) * gap + 36
    height = top + panel + 84
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(13)
    small_font = common.get_font(11)
    draw.text((24, 24), "Matched Pair CIFAR10 Confusion Matrices", font=title_font, fill=(20, 20, 20))
    max_count = 1
    matrices = {}
    for model_id in models:
        sub = df[df["model_id"] == model_id]
        matrix = sub[common.CIFAR10_CLASSES].to_numpy(dtype=float)
        matrices[model_id] = matrix
        max_count = max(max_count, int(matrix.max()))
    for m_idx, model_id in enumerate(models):
        x_base = left + m_idx * (panel + gap)
        matrix = matrices[model_id]
        common.draw_centered(draw, (x_base + panel / 2, top - 24), model_id, label_font)
        for r, class_name in enumerate(common.CIFAR10_CLASSES):
            if m_idx == 0:
                draw.text((14, top + r * cell + 10), class_name[:12], font=small_font, fill=(20, 20, 20))
            for c in range(10):
                value = matrix[r, c]
                color = common.heat_color(value / max_count)
                x0 = x_base + c * cell
                y0 = top + r * cell
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color, outline=(230, 230, 230))
        for c, class_name in enumerate(common.CIFAR10_CLASSES):
            common.draw_centered(draw, (x_base + c * cell + cell / 2, top + panel + 18), class_name[:3], small_font)
        draw.rectangle([x_base, top, x_base + panel, top + panel], outline=(40, 40, 40))
    img.save(path)


def draw_matching_scatter(path: Path, matches_csv: Path) -> None:
    if not matches_csv.exists():
        return
    df = pd.read_csv(matches_csv)
    if df.empty:
        return
    width, height = 760, 700
    left, right, top, bottom = 92, 42, 78, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(15)
    small_font = common.get_font(12)
    draw.text((24, 24), "Checkpoint Accuracy Matching", font=title_font, fill=(20, 20, 20))
    xs = df["imagenet_accuracy"].to_numpy(dtype=float)
    ys = df["lejepa_accuracy"].to_numpy(dtype=float)
    lo = float(min(xs.min(), ys.min()))
    hi = float(max(xs.max(), ys.max()))
    pad = max((hi - lo) * 0.08, 0.02)
    lo -= pad
    hi += pad

    def xmap(x: float) -> float:
        return left + (x - lo) / (hi - lo) * plot_w

    def ymap(y: float) -> float:
        return top + (hi - y) / (hi - lo) * plot_h

    for k in range(6):
        t = k / 5
        x = left + t * plot_w
        y = top + t * plot_h
        draw.line([(x, top), (x, top + plot_h)], fill=(235, 235, 235))
        draw.line([(left, y), (left + plot_w, y)], fill=(235, 235, 235))
        value = lo + t * (hi - lo)
        common.draw_centered(draw, (x, top + plot_h + 22), f"{value:.2f}", small_font)
        draw.text((24, y - 8), f"{hi - t * (hi - lo):.2f}", font=small_font, fill=(80, 80, 80))
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(40, 40, 40))
    draw.line([(xmap(lo), ymap(lo)), (xmap(hi), ymap(hi))], fill=(100, 100, 100), width=2)
    for row in df.itertuples():
        x = xmap(float(row.imagenet_accuracy))
        y = ymap(float(row.lejepa_accuracy))
        color = (0, 158, 115) if int(row.rank) == 1 else (170, 170, 170)
        radius = 6 if int(row.rank) == 1 else 3
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline="white")
    best = df.sort_values("rank").iloc[0]
    draw.text(
        (left + 12, top + 12),
        f"selected delta={best.abs_accuracy_delta:.4f}",
        font=label_font,
        fill=(20, 20, 20),
    )
    common.draw_centered(draw, (left + plot_w / 2, height - 26), "ImageNet FT test accuracy", label_font)
    draw.text((20, top - 32), "LeJEPA linear test accuracy", font=label_font, fill=(20, 20, 20))
    img.save(path)


def draw_c0_c1_density(
    model_infos: dict[str, dict],
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    bins = 180
    groups = {}
    for model_id, info in model_infos.items():
        groups[model_id] = common.sample_rows(info["Z"][:, :2], max_points, rng)
    all_z = np.vstack(list(groups.values()))
    x_range = (float(all_z[:, 0].min()), float(all_z[:, 0].max()))
    y_range = (float(all_z[:, 1].min()), float(all_z[:, 1].max()))
    hists = {}
    for model_id, z in groups.items():
        hist, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=bins, range=[x_range, y_range])
        hists[model_id] = hist.T.astype(np.float64)
    panel = 290
    gap = 32
    left, top = 44, 88
    width = left * 2 + 3 * panel + 2 * gap
    height = top + panel + 72
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(15, bold=True)
    small_font = common.get_font(12)
    draw.text((24, 24), "Matched Pair c0/c1 Filter Density", font=title_font, fill=(20, 20, 20))
    panel_specs = [
        ("imagenet_ft", "ImageNet FT"),
        ("lejepa_ssl_linear", "LeJEPA SSL + Linear"),
        ("residual", "LeJEPA - ImageNet"),
    ]
    residual = hists["lejepa_ssl_linear"] / max(hists["lejepa_ssl_linear"].sum(), 1.0) - hists["imagenet_ft"] / max(
        hists["imagenet_ft"].sum(),
        1.0,
    )
    max_abs_res = max(float(np.abs(residual).max()), 1e-12)
    for idx, (panel_id, label) in enumerate(panel_specs):
        x0 = left + idx * (panel + gap)
        y0 = top
        if panel_id == "residual":
            arr = np.zeros((bins, bins, 3), dtype=np.uint8)
            for r in range(bins):
                for c in range(bins):
                    arr[r, c] = common.diverging_color(float(residual[r, c]), max_abs_res)
        else:
            density = np.log1p(hists[panel_id])
            if density.max() > 0:
                density = density / density.max()
            arr = np.zeros((bins, bins, 3), dtype=np.uint8)
            for level in range(256):
                mask = (density >= level / 256) & (density < (level + 1) / 256)
                arr[mask] = common.density_color(level / 255)
        density_img = Image.fromarray(np.flipud(arr), mode="RGB").resize((panel, panel), Image.Resampling.BILINEAR)
        img.paste(density_img, (x0, y0))
        draw.rectangle([x0, y0, x0 + panel, y0 + panel], outline=(40, 40, 40), width=1)
        common.draw_centered(draw, (x0 + panel / 2, y0 - 22), label, label_font)
    draw.text(
        (left, height - 34),
        f"x: c0 [{x_range[0]:.2f}, {x_range[1]:.2f}], y: c1 [{y_range[0]:.2f}, {y_range[1]:.2f}]",
        font=small_font,
        fill=(80, 80, 80),
    )
    img.save(path)


def draw_residual_histograms(hist_rows: Sequence[dict], path: Path) -> None:
    df = pd.DataFrame(hist_rows)
    cols = 3
    rows = 3
    cell_w = 300
    cell_h = 170
    left, top = 70, 82
    width = left + cols * cell_w + 36
    height = top + rows * cell_h + 58
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(14, bold=True)
    small_font = common.get_font(11)
    draw.text((24, 24), "Global PCA Residual Histograms", font=title_font, fill=(20, 20, 20))
    max_abs = max(float(np.abs(df["residual_lejepa_minus_imagenet"]).max()), 1e-12)
    for component in range(9):
        sub = df[df["component"] == component].sort_values("bin_index")
        r = component // cols
        c = component % cols
        x0 = left + c * cell_w
        y0 = top + r * cell_h
        plot_w = cell_w - 34
        plot_h = cell_h - 42
        baseline = y0 + plot_h / 2 + 16
        draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(50, 50, 50))
        common.draw_centered(draw, (x0 + plot_w / 2, y0 - 16), f"c{component}", label_font)
        values = sub["residual_lejepa_minus_imagenet"].to_numpy(dtype=float)
        centers = sub["bin_center"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        for idx, value in enumerate(values):
            x = x0 + idx / max(len(values) - 1, 1) * plot_w
            h = abs(value) / max_abs * (plot_h / 2 - 8)
            color = (178, 24, 43) if value >= 0 else (33, 102, 172)
            if value >= 0:
                draw.line([(x, baseline), (x, baseline - h)], fill=color)
            else:
                draw.line([(x, baseline), (x, baseline + h)], fill=color)
        draw.line([(x0, baseline), (x0 + plot_w, baseline)], fill=(80, 80, 80))
        draw.text((x0, y0 + plot_h + 6), f"{centers.min():.2f}", font=small_font, fill=(80, 80, 80))
        draw.text((x0 + plot_w - 34, y0 + plot_h + 6), f"{centers.max():.2f}", font=small_font, fill=(80, 80, 80))
    draw.text(
        (left, height - 34),
        "Red: LeJEPA overrepresented. Blue: ImageNet overrepresented.",
        font=small_font,
        fill=(80, 80, 80),
    )
    img.save(path)


def draw_pair_drift_bar(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: Path,
    vmax: float | None = None,
) -> None:
    """Chart counterpart for a two-model symmetric drift heatmap."""
    if matrix.shape != (2, 2) or len(labels) != 2:
        raise ValueError(f"Pair drift chart expects a 2x2 matrix and two labels, got {matrix.shape}")
    value = float((matrix[0, 1] + matrix[1, 0]) / 2.0)
    maximum = max(float(vmax) if vmax is not None else value * 1.15, value, 1e-12)
    width, height = 940, 360
    left, right, top, bottom = 210, 70, 96, 80
    plot_w = width - left - right
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(15)
    value_font = common.get_font(15, bold=True)
    small_font = common.get_font(12)
    draw.text((24, 22), title, font=title_font, fill=(20, 20, 20))
    label = f"{labels[0]} vs {labels[1]}"
    draw.text((24, top + 48), label, font=label_font, fill=(30, 30, 30))
    bar_y0, bar_y1 = top + 24, top + 92
    draw.rectangle([left, bar_y0, left + plot_w, bar_y1], fill=(238, 238, 238))
    bar_width = value / maximum * plot_w
    draw.rectangle([left, bar_y0, left + bar_width, bar_y1], fill=(0, 120, 160))
    draw.text((left + min(bar_width + 12, plot_w - 65), bar_y0 + 22), f"{value:.6f}", font=value_font, fill=(20, 20, 20))
    for tick in range(6):
        x = left + tick / 5 * plot_w
        draw.line([(x, bar_y1 + 4), (x, bar_y1 + 10)], fill=(60, 60, 60))
        common.draw_centered(draw, (x, bar_y1 + 28), f"{maximum * tick / 5:.3f}", small_font)
    common.draw_centered(draw, (left + plot_w / 2, height - 30), "symmetric PCA-distribution drift D", label_font)
    image.save(path)


def draw_stage_heatmaps(stage_matrices: dict[str, np.ndarray], labels: Sequence[str], figure_dir: Path) -> None:
    max_value = max(float(matrix.max()) for matrix in stage_matrices.values()) if stage_matrices else 1.0
    for stage, matrix in stage_matrices.items():
        common.draw_heatmap(
            matrix,
            labels,
            f"Stage Drift: {stage}",
            figure_dir / f"drift_stage_{stage}_heatmap.png",
            vmin=0.0,
            vmax=max_value,
        )
        draw_pair_drift_bar(
            matrix,
            labels,
            f"Stage Drift Chart: {stage}",
            figure_dir / f"drift_stage_{stage}_chart.png",
            vmax=max_value,
        )
    if stage_matrices:
        stages = list(stage_matrices)
        common.draw_line_series(
            [
                {
                    "x": list(range(1, len(stages) + 1)),
                    "y": [float(stage_matrices[stage][0, 1]) for stage in stages],
                    "label": f"{labels[0]} vs {labels[1]}",
                    "color": (0, 120, 160),
                }
            ],
            "PCA-Distribution Drift Across RN18 Stages",
            "stage order (" + ", ".join(f"{index + 1}={stage}" for index, stage in enumerate(stages)) + ")",
            "drift D",
            figure_dir / "drift_stage_chart.png",
            y_min=0.0,
        )


def draw_training_figures(experiment_dir: Path, figure_dir: Path) -> None:
    imagenet_path = experiment_dir / "imagenet_finetune_metrics.csv"
    linear_path = experiment_dir / "lejepa_linear_metrics.csv"
    quick_path = experiment_dir / "lejepa_quick_probe_metrics.csv"
    ssl_path = experiment_dir / "lejepa_ssl_metrics.csv"
    series = []
    if imagenet_path.exists():
        df = pd.read_csv(imagenet_path)
        if not df.empty:
            series.append(
                {
                    "x": df["epoch"].tolist(),
                    "y": df["test_accuracy"].tolist(),
                    "label": "ImageNet FT",
                    "color": (0, 114, 178),
                }
            )
    if linear_path.exists():
        df = pd.read_csv(linear_path)
        if not df.empty:
            series.append(
                {
                    "x": df["ssl_epoch"].tolist(),
                    "y": df["final_test_accuracy"].tolist(),
                    "label": "LeJEPA fixed linear",
                    "color": (0, 158, 115),
                }
            )
    if quick_path.exists():
        df = pd.read_csv(quick_path)
        if not df.empty:
            series.append(
                {
                    "x": df["ssl_epoch"].tolist(),
                    "y": df["quick_probe_accuracy"].tolist(),
                    "label": "LeJEPA quick probe",
                    "color": (230, 159, 0),
                }
            )
    if series:
        common.draw_line_series(
            series,
            "CIFAR10 Accuracy Across Saved Checkpoints",
            "epoch / SSL epoch",
            "test accuracy",
            figure_dir / "accuracy_curves.png",
            y_min=0.0,
            y_max=1.0,
        )
    if ssl_path.exists():
        df = pd.read_csv(ssl_path)
        if not df.empty:
            common.draw_line_series(
                [
                    {
                        "x": df["epoch"].tolist(),
                        "y": df["ssl_loss"].tolist(),
                        "label": "SSL loss",
                        "color": (86, 180, 233),
                    }
                ],
                "LeJEPA SSL Loss",
                "SSL epoch",
                "loss",
                figure_dir / "ssl_loss_curve.png",
            )
    draw_matching_scatter(figure_dir / "checkpoint_accuracy_matching.png", experiment_dir / "matched_checkpoints.csv")
    draw_confusion_matrices(figure_dir / "matched_confusion_matrices.png", experiment_dir / "matched_confusion_matrices.csv")


def save_summary(
    path: Path,
    selected: dict,
    global_matrix: np.ndarray,
    group_ids: Sequence[str],
    stage_rows: list[dict],
    layer_rows: list[dict],
    quality_rows: list[dict],
    residual_summary: list[dict],
    explained_variance_ratio: np.ndarray,
    basis_description: str,
    basis_path: str,
    pca_figure_name: str,
    include_stem: bool,
    bins: int,
) -> None:
    def display(model_id: str) -> str:
        for spec in MODEL_SPECS:
            if spec["model_id"] == model_id:
                return spec["display"]
        return model_id

    pair_drift = global_matrix[group_ids.index(PAIR[0]), group_ids.index(PAIR[1])]
    layer_df = pd.DataFrame(layer_rows)
    pair_layers = layer_df[
        (layer_df["group_a"] == PAIR[0]) & (layer_df["group_b"] == PAIR[1])
    ].sort_values("drift_D", ascending=False)
    residual_df = pd.DataFrame(residual_summary).sort_values("weighted_symmetric_kl", ascending=False)
    qdf = pd.DataFrame(quality_rows)
    lines = []
    lines.append("# RN18 CIFAR10 Matched Filter Analysis")
    lines.append("")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Matched Checkpoints")
    lines.append("")
    lines.append(
        f"- ImageNet FT epoch `{int(selected['imagenet_epoch'])}` accuracy `{float(selected['imagenet_accuracy']):.4f}`"
    )
    lines.append(
        f"- LeJEPA SSL epoch `{int(selected['lejepa_ssl_epoch'])}` linear accuracy `{float(selected['lejepa_accuracy']):.4f}`"
    )
    lines.append(f"- Absolute accuracy gap: `{float(selected['abs_accuracy_delta']):.4f}`")
    lines.append("")
    lines.append("## PCA")
    lines.append("")
    lines.append(f"- Basis: {basis_description}")
    if basis_path:
        lines.append(f"- Basis artifact: `{basis_path}`")
    lines.append(
        f"- Filter scope: {'CIFAR stem plus residual blocks' if include_stem else '16 residual-block 3x3 convolutions; CIFAR stem excluded'}"
    )
    lines.append(
        f"- Histogram protocol: Experiment 3-compatible `{bins}` bins with component bounds recomputed "
        "from the compared groups in each global/stage/layer scope; only the PCA basis is replaced."
    )
    lines.append("")
    lines.append("Explained variance ratio:")
    lines.append("")
    lines.append("```text")
    lines.append(str([round(float(v), 6) for v in explained_variance_ratio]))
    lines.append("```")
    lines.append("")
    lines.append("## Filter Drift")
    lines.append("")
    lines.append(f"- Global matched-pair drift D: `{pair_drift:.6f}`")
    lines.append("")
    lines.append("| Stage | Drift D |")
    lines.append("|---|---:|")
    for row in stage_rows:
        if row["group_a"] == PAIR[0] and row["group_b"] == PAIR[1]:
            lines.append(f"| {row['stage']} | {float(row['drift_D']):.6f} |")
    lines.append("")
    lines.append("Top layer drifts:")
    lines.append("")
    lines.append("| Layer | Stage | Drift D |")
    lines.append("|---|---|---:|")
    for row in pair_layers.head(8).itertuples():
        lines.append(f"| `{row.layer_key}` | {row.stage} | {float(row.drift_D):.6f} |")
    lines.append("")
    lines.append("Top PCA component contributions:")
    lines.append("")
    lines.append("| Component | Weighted KL | Weighted TV | Mean Delta |")
    lines.append("|---:|---:|---:|---:|")
    for row in residual_df.head(5).itertuples():
        lines.append(
            f"| {int(row.component)} | {float(row.weighted_symmetric_kl):.6f} | "
            f"{float(row.weighted_tv_distance):.6f} | {float(row.mean_delta_lejepa_minus_imagenet):.6f} |"
        )
    lines.append("")
    lines.append("## Layer Quality Flags")
    lines.append("")
    lines.append("Flags are heuristic: `S > 0.01`, `H/TH > 0.95`, or `H < 0.50`.")
    lines.append("")
    lines.append("| Model | Layer | H | S | H/TH | Reason |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in qdf.itertuples():
        reasons = []
        if float(row.sparsity_S) > 0.01:
            reasons.append("sparse")
        if float(row.H_over_TH) > 0.95:
            reasons.append("near_random_threshold")
        if float(row.entropy_H) < 0.50:
            reasons.append("low_diversity")
        if reasons:
            lines.append(
                f"| {display(row.model_id)} | `{row.layer_key}` | {float(row.entropy_H):.4f} | "
                f"{float(row.sparsity_S):.4f} | {float(row.H_over_TH):.4f} | {', '.join(reasons)} |"
            )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in [
        "accuracy_curves.png",
        "checkpoint_accuracy_matching.png",
        "matched_confusion_matrices.png",
        pca_figure_name,
        "drift_global_heatmap.png",
        "drift_global_chart.png",
        "drift_stage_chart.png",
        "drift_layer_by_depth.png",
        "component_drift_contributions.png",
        "ridge_matched_pair.png",
        "c0_c1_density_matched.png",
        "global_component_residual_histograms.png",
        "entropy_by_depth.png",
        "sparsity_by_depth.png",
    ]:
        lines.append(f"- `figures/{fig}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "analysis"
    figure_dir = output_dir / "figures"
    common.ensure_dir(output_dir)
    common.ensure_dir(figure_dir)
    print(f"[start] experiment_dir={experiment_dir}", flush=True)
    print(f"[start] output_dir={output_dir}", flush=True)

    selected = apply_selected_overrides(
        load_selected_pair(experiment_dir, args.selected_pair_file),
        args,
    )
    (output_dir / "selected_pair_used.json").write_text(json.dumps(selected, indent=2) + "\n")
    model_infos, quality_rows = load_model_infos(selected, experiment_dir, include_stem=args.include_stem)
    pd.DataFrame(quality_rows).to_csv(output_dir / "local_layer_quality.csv", index=False)

    group_ids = [spec["model_id"] for spec in MODEL_SPECS]
    labels = {spec["model_id"]: spec["display"] for spec in MODEL_SPECS}
    colors = {spec["model_id"]: spec["color"] for spec in MODEL_SPECS}
    basis_path_text = ""
    if args.pca_basis:
        basis_path = Path(args.pca_basis).expanduser().resolve()
        basis_path_text = str(basis_path)
        print(f"[pca] Loading reference PCA basis: {basis_path}", flush=True)
        pca, basis_metadata = load_reference_pca(basis_path)
        preprocessing_id = str(
            basis_metadata.get("consumer_projection_preprocessing_id", "legacy_float32_maxabs")
        )
        for gid in group_ids:
            scaled = reference_scale_filters(model_infos[gid]["X"], preprocessing_id)
            model_infos[gid]["X_scaled"] = scaled
            model_infos[gid]["Z"] = common.project_pca(
                scaled,
                pca["mean"],
                pca["components"],
            )
        copied_basis = output_dir / "pca_basis_used.npz"
        if basis_path != copied_basis.resolve():
            shutil.copy2(basis_path, copied_basis)
        basis_sha256 = hashlib.sha256(basis_path.read_bytes()).hexdigest()
        published_mean_passed = basis_metadata.get("published_mean_validation_passed")
        if published_mean_passed is False:
            print(
                "[pca-warning] Full released dataset fit is internally valid, but its mean "
                "does not reproduce the paper's printed centering vector; recording this "
                "discrepancy in provenance.",
                flush=True,
            )
        (output_dir / "pca_basis_provenance.json").write_text(
            json.dumps(
                {
                    "basis_label": args.basis_label,
                    "source_path": str(basis_path),
                    "sha256": basis_sha256,
                    "projection_preprocessing": (
                        "cast each target kernel to float16, then divide by its float16 "
                        "maximum absolute coefficient"
                        if preprocessing_id == FULL_DB_PREPROCESSING_ID
                        else "each target kernel divided by its own maximum absolute coefficient"
                    ),
                    "projection_preprocessing_id": preprocessing_id,
                    "fit_on_matched_pair": False,
                    "artifact_metadata": basis_metadata,
                    "published_mean_discrepancy_warning": (
                        "The basis is the exact finite covariance eigensystem of all filters "
                        "in the checksum-verified released dataset, but its computed mean "
                        "differs from the paper's printed mean; see artifact metadata."
                        if published_mean_passed is False
                        else ""
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        basis_description = args.basis_label
        pca_figure_name = "pca_eigenfilters_cnn_filter_db_basis.png"
    else:
        print("[pca] Fitting matched RN18/CIFAR10 PCA basis", flush=True)
        all_x = np.vstack([model_infos[gid]["X"] for gid in group_ids]).astype(np.float32, copy=False)
        all_scaled = common.scale_filters(all_x)
        pca = common.fit_pca(all_scaled, n_components=9)
        np.savez_compressed(
            output_dir / "pca_matched_rn18_cifar10.npz",
            components=pca["components"],
            mean=pca["mean"],
            explained_variance=pca["explained_variance"],
            explained_variance_ratio=pca["explained_variance_ratio"],
            singular_values=pca["singular_values"],
        )
        cursor = 0
        for gid in group_ids:
            n = len(model_infos[gid]["X"])
            model_infos[gid]["X_scaled"] = all_scaled[cursor : cursor + n]
            model_infos[gid]["Z"] = pca["z"][cursor : cursor + n]
            cursor += n
        del all_x, all_scaled
        basis_description = "PCA fitted jointly on the two matched RN18 checkpoints"
        pca_figure_name = "pca_eigenfilters_matched_basis.png"
    weights = pca["explained_variance_ratio"].astype(np.float64)

    print("[drift] Computing global drift", flush=True)
    global_groups = {gid: model_infos[gid]["Z"] for gid in group_ids}
    global_matrix, global_contribs = common.compute_drift(global_groups, weights, args.bins)
    write_matrix_csv(output_dir / "drift_global_target_basis.csv", global_matrix, group_ids)
    pd.DataFrame(common.matrix_to_rows(global_matrix, group_ids)).to_csv(
        output_dir / "drift_global_pairs.csv",
        index=False,
    )
    pair_contribs = global_contribs[PAIR]
    pd.DataFrame(
        [
            {
                "group_a": PAIR[0],
                "group_b": PAIR[1],
                "component": component,
                "weighted_contribution": float(value),
                "component_weight": float(weights[component]),
            }
            for component, value in enumerate(pair_contribs)
        ]
    ).to_csv(output_dir / "drift_component_contributions.csv", index=False)

    hist_rows, residual_summary = residual_histogram_rows(
        model_infos[PAIR[0]]["Z"],
        model_infos[PAIR[1]]["Z"],
        weights,
        args.bins,
    )
    pd.DataFrame(hist_rows).to_csv(output_dir / "global_component_residual_histograms.csv", index=False)
    pd.DataFrame(residual_summary).to_csv(output_dir / "global_component_residual_summary.csv", index=False)

    print("[drift] Computing stage and layer drift", flush=True)
    stage_rows: list[dict] = []
    stage_matrices = {}
    available_stages = [stage for stage in STAGE_ORDER if any(s.stage == stage for s in model_infos[group_ids[0]]["slices"])]
    for stage in available_stages:
        groups = {gid: common.subset_by_stage(model_infos[gid], stage) for gid in group_ids}
        matrix, _ = common.compute_drift(groups, weights, args.bins)
        stage_matrices[stage] = matrix
        stage_rows.extend(common.matrix_to_rows(matrix, group_ids, prefix={"stage": stage}))
        write_matrix_csv(output_dir / f"drift_stage_{stage}.csv", matrix, group_ids)
    pd.DataFrame(stage_rows).to_csv(output_dir / "drift_stage_target_basis.csv", index=False)

    layer_rows: list[dict] = []
    num_layers = len(model_infos[group_ids[0]]["slices"])
    for layer_order in range(num_layers):
        groups = {}
        layer_info = None
        for gid in group_ids:
            z, layer_slice = common.subset_by_layer(model_infos[gid], layer_order)
            groups[gid] = z
            layer_info = layer_slice
        matrix, _ = common.compute_drift(groups, weights, args.bins)
        assert layer_info is not None
        layer_rows.extend(
            common.matrix_to_rows(
                matrix,
                group_ids,
                prefix={
                    "layer_order": layer_order,
                    "layer_key": layer_info.layer_key,
                    "stage": layer_info.stage,
                    "conv_depth_norm": layer_info.conv_depth_norm,
                },
            )
        )
    pd.DataFrame(layer_rows).to_csv(output_dir / "drift_layer_target_basis.csv", index=False)

    print("[plot] Writing figures", flush=True)
    draw_training_figures(experiment_dir, figure_dir)
    display_labels = [labels[gid] for gid in group_ids]
    common.draw_eigenfilters(
        pca["components"],
        weights,
        figure_dir / pca_figure_name,
        f"{basis_description} Eigenfilters",
    )
    common.draw_heatmap(
        global_matrix,
        display_labels,
        "Matched Pair Global Drift",
        figure_dir / "drift_global_heatmap.png",
        vmin=0.0,
        vmax=max(float(global_matrix.max()), 1e-12),
    )
    draw_pair_drift_bar(
        global_matrix,
        display_labels,
        "Matched Pair Global Drift Chart",
        figure_dir / "drift_global_chart.png",
    )
    draw_stage_heatmaps(stage_matrices, display_labels, figure_dir)
    pair_layer_df = pd.DataFrame(layer_rows)
    pair_layer_df = pair_layer_df[
        (pair_layer_df["group_a"] == PAIR[0]) & (pair_layer_df["group_b"] == PAIR[1])
    ].sort_values("layer_order")
    common.draw_line_series(
        [
            {
                "x": pair_layer_df["conv_depth_norm"].tolist(),
                "y": pair_layer_df["drift_D"].tolist(),
                "label": "ImageNet vs LeJEPA",
                "color": (0, 120, 160),
            }
        ],
        "Layer-Level Drift by Depth",
        "normalized convolution depth",
        "drift D",
        figure_dir / "drift_layer_by_depth.png",
        y_min=0.0,
    )
    draw_component_bars(residual_summary, figure_dir / "component_drift_contributions.png")
    common.draw_ridge_plot(
        {gid: model_infos[gid]["Z"] for gid in group_ids},
        labels,
        colors,
        args.bins,
        figure_dir / "ridge_matched_pair.png",
        "Matched Pair PCA Coefficient Ridges",
    )
    draw_c0_c1_density(model_infos, figure_dir / "c0_c1_density_matched.png", args.max_density_points, args.seed)
    draw_residual_histograms(hist_rows, figure_dir / "global_component_residual_histograms.png")
    qdf = pd.DataFrame(quality_rows)
    entropy_series = []
    sparsity_series = []
    for spec in MODEL_SPECS:
        sub = qdf[qdf["model_id"] == spec["model_id"]].sort_values("layer_order")
        entropy_series.append(
            {
                "x": sub["conv_depth_norm"].tolist(),
                "y": sub["entropy_H"].tolist(),
                "label": spec["display"],
                "color": spec["color"],
            }
        )
        sparsity_series.append(
            {
                "x": sub["conv_depth_norm"].tolist(),
                "y": sub["sparsity_S"].tolist(),
                "label": spec["display"],
                "color": spec["color"],
            }
        )
    common.draw_line_series(
        entropy_series,
        "Layer Entropy by Depth",
        "normalized convolution depth",
        "entropy H",
        figure_dir / "entropy_by_depth.png",
    )
    common.draw_line_series(
        sparsity_series,
        "Layer Sparsity by Depth",
        "normalized convolution depth",
        "sparsity S",
        figure_dir / "sparsity_by_depth.png",
        y_min=0.0,
    )
    save_summary(
        output_dir / "summary.md",
        selected,
        global_matrix,
        group_ids,
        stage_rows,
        layer_rows,
        quality_rows,
        residual_summary,
        weights,
        basis_description,
        basis_path_text,
        pca_figure_name,
        args.include_stem,
        args.bins,
    )
    print(f"[done] Wrote analysis to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
