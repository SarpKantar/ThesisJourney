#!/usr/bin/env python
"""Threshold-sensitivity analysis for ResNet18 3x3 filter quality metrics.

This script deliberately does not fit or load a global PCA basis.  Layer
entropy is computed once from the raw 3x3 filter bank and then held fixed while
decision cutoffs are swept.  Sparsity is recomputed over a grid of relative
near-zero thresholds.  An optional, separately named ``clean_entropy_H``
diagnostic recomputes entropy after removing filters classified as sparse.

Model inputs are explicit and repeatable, for example::

    python scripts/analyze_rn18_threshold_sensitivity.py \
      --model-spec 'e001::LeJEPA epoch 1::outputs/run/checkpoints/lejepa_ssl_epoch001.pth' \
      --model-spec 'e110::LeJEPA epoch 110::outputs/run/checkpoints/lejepa_ssl_epoch110.pth' \
      --output-dir outputs/threshold_sensitivity

The shorter ``MODEL_ID::CHECKPOINT_PATH`` form is also accepted.  A JSON model
spec with ``model_id``, ``display``, and ``checkpoint_path`` keys is useful when
labels or paths themselves contain ``::``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env.
    raise ModuleNotFoundError(
        "This script requires PyTorch. Use the `minitron` conda environment "
        "or the project's Slurm runtime."
    ) from exc

from PIL import Image, ImageDraw

import rn18_cifar10_common as common


DEFAULT_SPARSITY_EPSILONS = "0.001,0.0025,0.005,0.01,0.02,0.05,0.1"
DEFAULT_LOW_ENTROPY_CUTOFFS = "0.30,0.40,0.50,0.60,0.70,0.80,0.90"
DEFAULT_RANDOM_RATIO_CUTOFFS = "0.75,0.80,0.85,0.90,0.95,1.00,1.05"
LEGACY_SPARSITY_EPSILON = 0.01
LEGACY_LOW_ENTROPY_CUTOFF = 0.50
LEGACY_RANDOM_RATIO_CUTOFF = 0.95
ENTROPY_MAX = math.log10(9.0)
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PALETTE = [
    (0, 114, 178),
    (0, 158, 115),
    (230, 159, 0),
    (204, 121, 167),
    (86, 180, 233),
    (213, 94, 0),
    (240, 228, 66),
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display: str
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep sparsity and entropy decision thresholds for explicit RN18 checkpoints. "
            "No global PCA basis is used."
        )
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        required=True,
        metavar="MODEL_ID::DISPLAY::CHECKPOINT",
        help=(
            "Repeat for every checkpoint. DISPLAY may be omitted: MODEL_ID::CHECKPOINT. "
            "A JSON object with model_id/display/checkpoint_path is also accepted."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for CSV, PNG, and summary outputs.")
    parser.add_argument(
        "--sparsity-epsilons",
        default=DEFAULT_SPARSITY_EPSILONS,
        help=(
            "Comma/space-separated fractions of each layer's absolute peak. Must contain at "
            "least seven values and include the legacy 0.01 threshold."
        ),
    )
    parser.add_argument(
        "--low-entropy-cutoffs",
        default=DEFAULT_LOW_ENTROPY_CUTOFFS,
        help=(
            "Comma/space-separated H cutoffs for the H < cutoff decision. Must contain at "
            "least seven values and include the legacy 0.50 cutoff."
        ),
    )
    parser.add_argument(
        "--random-ratio-cutoffs",
        default=DEFAULT_RANDOM_RATIO_CUTOFFS,
        help=(
            "Comma/space-separated H/TH cutoffs for the H/TH > cutoff decision. Must contain "
            "at least six values and include the legacy 0.95 cutoff."
        ),
    )
    parser.add_argument(
        "--include-stem",
        action="store_true",
        help="Include the CIFAR-style conv1 stem. The default is residual-block convolutions only.",
    )
    parser.add_argument(
        "--compute-clean-entropy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also compute a separately labeled entropy after sparse-filter exclusion for each "
            "epsilon. Baseline entropy H always remains fixed."
        ),
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance for legacy-threshold regression checks.",
    )
    return parser.parse_args()


def parse_model_spec(text: str) -> ModelSpec:
    text = text.strip()
    if text.startswith("{"):
        payload = json.loads(text)
        model_id = str(payload["model_id"]).strip()
        display = str(payload.get("display", model_id)).strip()
        checkpoint_text = str(payload["checkpoint_path"]).strip()
    else:
        parts = text.split("::", 2)
        if len(parts) == 2:
            model_id, checkpoint_text = parts
            display = model_id
        elif len(parts) == 3:
            model_id, display, checkpoint_text = parts
        else:
            raise ValueError(
                "Invalid --model-spec. Use MODEL_ID::CHECKPOINT, "
                "MODEL_ID::DISPLAY::CHECKPOINT, or a JSON object."
            )
        model_id = model_id.strip()
        display = display.strip()
        checkpoint_text = checkpoint_text.strip()
    if not model_id or not MODEL_ID_RE.fullmatch(model_id):
        raise ValueError(
            f"Invalid model_id {model_id!r}; use letters, digits, dot, underscore, or hyphen."
        )
    if not display:
        raise ValueError(f"Empty display label for {model_id}")
    checkpoint = Path(checkpoint_text).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint for {model_id} does not exist: {checkpoint}")
    return ModelSpec(model_id=model_id, display=display, checkpoint_path=checkpoint.resolve())


def parse_grid(
    text: str,
    name: str,
    minimum_count: int,
    required_value: float,
) -> list[float]:
    tokens = [token for token in re.split(r"[\s,]+", text.strip()) if token]
    try:
        values = sorted(set(float(token) for token in tokens))
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-numeric value: {text!r}") from exc
    if len(values) < minimum_count:
        raise ValueError(f"{name} needs at least {minimum_count} distinct values; got {len(values)}")
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} values must be finite and non-negative")
    if not any(math.isclose(value, required_value, rel_tol=0.0, abs_tol=1e-12) for value in values):
        raise ValueError(f"{name} must include the legacy value {required_value:g}")
    return values


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        return torch.load(path, map_location="cpu")


def entropy_from_filters(x: np.ndarray) -> float:
    """Base-10 covariance-spectrum entropy, matching layer_quality mathematically."""
    if x.ndim != 2 or x.shape[1] != 9:
        raise ValueError(f"Expected [N, 9] filter matrix, got {x.shape}")
    if x.shape[0] < 2:
        return 0.0 if x.shape[0] == 1 else math.nan
    x64 = x.astype(np.float64, copy=False)
    centered = x64 - x64.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(x64.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if not np.isfinite(total) or total <= 0.0:
        return 0.0
    return common.entropy_base10(eigenvalues / total)


def checkpoint_metadata(checkpoint: dict) -> dict[str, object]:
    epoch = checkpoint.get("epoch", checkpoint.get("ssl_epoch", ""))
    return {
        "checkpoint_method": str(checkpoint.get("method", "")),
        "checkpoint_epoch": epoch,
        "checkpoint_architecture": str(checkpoint.get("architecture", "")),
        "checkpoint_dataset": str(checkpoint.get("dataset", "")),
    }


def threshold_label(value: float) -> str:
    return f"{value:.4g}"


def threshold_filename(prefix: str, value: float) -> str:
    """Return a stable, shell-friendly filename for one swept threshold."""
    token = f"{value:.8g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_{token}.png"


def analyze_models(
    specs: Sequence[ModelSpec],
    sparsity_epsilons: Sequence[float],
    low_entropy_cutoffs: Sequence[float],
    random_ratio_cutoffs: Sequence[float],
    include_stem: bool,
    compute_clean_entropy: bool,
    regression_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_rows: list[dict] = []
    sparsity_rows: list[dict] = []
    decision_rows: list[dict] = []
    regression_rows: list[dict] = []

    for spec in specs:
        print(f"[load] {spec.display}: {spec.checkpoint_path}", flush=True)
        checkpoint = load_checkpoint(spec.checkpoint_path)
        state = common.load_backbone_state(checkpoint)
        items = common.sorted_resnet18_filter_items(state, include_stem=include_stem)
        if not items:
            raise ValueError(f"No RN18 3x3 filters found in {spec.checkpoint_path}")
        metadata = checkpoint_metadata(checkpoint)
        for layer_order, (layer_key, tensor) in enumerate(items):
            raw = common.tensor_to_numpy(tensor)
            x = raw.reshape(-1, 9).astype(np.float32, copy=False)
            num_filters = int(x.shape[0])
            stage = common.layer_stage(layer_key)
            depth = layer_order / max(len(items) - 1, 1)
            quality = common.layer_quality(x)
            entropy_h = float(quality["entropy_H"])
            random_th = float(common.random_entropy_threshold(num_filters))
            h_over_th = entropy_h / random_th if random_th else math.nan
            base = {
                "model_id": spec.model_id,
                "display": spec.display,
                "checkpoint_path": str(spec.checkpoint_path),
                **metadata,
                "include_stem": bool(include_stem),
                "layer_key": layer_key,
                "stage": stage,
                "layer_order": layer_order,
                "conv_depth_norm": depth,
                "shape": "x".join(str(int(value)) for value in raw.shape),
                "num_filters": num_filters,
                "entropy_H_fixed": entropy_h,
                "random_threshold_TH": random_th,
                "H_over_TH_fixed": h_over_th,
                "abs_peak_weight": float(quality["abs_peak_weight"]),
                "mean_abs_weight": float(quality["mean_abs_weight"]),
                "std_weight": float(quality["std_weight"]),
            }

            legacy_sweep_row = None
            for epsilon in sparsity_epsilons:
                is_legacy = math.isclose(
                    epsilon,
                    LEGACY_SPARSITY_EPSILON,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                absolute_threshold = (
                    float(quality["sparsity_threshold"])
                    if is_legacy
                    else float(quality["abs_peak_weight"]) * float(epsilon)
                )
                sparse_mask = (np.abs(x) < absolute_threshold).all(axis=1)
                sparse_count = int(sparse_mask.sum())
                sparsity_s = sparse_count / max(num_filters, 1)
                clean_h = math.nan
                if compute_clean_entropy:
                    clean_h = entropy_from_filters(x[~sparse_mask])
                row = {
                    **base,
                    "epsilon_fraction": float(epsilon),
                    "epsilon_label": threshold_label(epsilon),
                    "absolute_weight_threshold": absolute_threshold,
                    "threshold_scale": "fraction_of_layer_abs_peak",
                    "sparse_filter_rule": "all_9_abs_weights_below_threshold",
                    "sparse_filter_count": sparse_count,
                    "non_sparse_filter_count": num_filters - sparse_count,
                    "sparsity_S": sparsity_s,
                    "has_any_sparse_filters": bool(sparse_count > 0),
                    "layer_sparsity_gt_0p01": bool(sparsity_s > 0.01),
                    "is_legacy_epsilon_0p01": is_legacy,
                    "clean_entropy_computed": bool(compute_clean_entropy),
                    "clean_entropy_H": clean_h,
                    "clean_entropy_delta_from_fixed_H": (
                        clean_h - entropy_h if np.isfinite(clean_h) else math.nan
                    ),
                }
                sparsity_rows.append(row)
                if is_legacy:
                    legacy_sweep_row = row

            if legacy_sweep_row is None:  # guarded by grid validation.
                raise AssertionError("Legacy epsilon row was not produced")
            legacy_sparsity = float(quality["sparsity_S"])
            observed_sparsity = float(legacy_sweep_row["sparsity_S"])
            sparsity_delta = abs(observed_sparsity - legacy_sparsity)
            threshold_delta = abs(
                float(legacy_sweep_row["absolute_weight_threshold"])
                - float(quality["sparsity_threshold"])
            )
            entropy_delta = abs(entropy_h - float(quality["entropy_H"]))
            baseline_row = {
                **base,
                "legacy_entropy_H_common": float(quality["entropy_H"]),
                "legacy_entropy_abs_delta": entropy_delta,
                "legacy_entropy_regression_pass": entropy_delta <= regression_tolerance,
                "legacy_sparsity_epsilon": LEGACY_SPARSITY_EPSILON,
                "legacy_sparsity_threshold_common": float(quality["sparsity_threshold"]),
                "sweep_sparsity_threshold_at_0p01": float(
                    legacy_sweep_row["absolute_weight_threshold"]
                ),
                "legacy_sparsity_threshold_abs_delta": threshold_delta,
                "legacy_sparsity_S_common": legacy_sparsity,
                "sweep_sparsity_S_at_0p01": observed_sparsity,
                "legacy_sparsity_abs_delta": sparsity_delta,
                "legacy_sparsity_regression_pass": (
                    sparsity_delta <= regression_tolerance
                    and threshold_delta <= regression_tolerance
                ),
                "legacy_low_entropy_cutoff": LEGACY_LOW_ENTROPY_CUTOFF,
                "legacy_low_entropy_flag": bool(entropy_h < LEGACY_LOW_ENTROPY_CUTOFF),
                "legacy_random_ratio_cutoff": LEGACY_RANDOM_RATIO_CUTOFF,
                "legacy_random_like_flag": bool(h_over_th > LEGACY_RANDOM_RATIO_CUTOFF),
            }
            baseline_rows.append(baseline_row)

            regression_rows.extend(
                [
                    {
                        **{key: base[key] for key in [
                            "model_id",
                            "display",
                            "checkpoint_path",
                            "layer_key",
                            "stage",
                            "layer_order",
                            "conv_depth_norm",
                        ]},
                        "check_name": "fixed_entropy_matches_common_layer_quality",
                        "expected": float(quality["entropy_H"]),
                        "observed": entropy_h,
                        "absolute_delta": entropy_delta,
                        "tolerance": regression_tolerance,
                        "passed": entropy_delta <= regression_tolerance,
                    },
                    {
                        **{key: base[key] for key in [
                            "model_id",
                            "display",
                            "checkpoint_path",
                            "layer_key",
                            "stage",
                            "layer_order",
                            "conv_depth_norm",
                        ]},
                        "check_name": "epsilon_0p01_threshold_matches_legacy",
                        "expected": float(quality["sparsity_threshold"]),
                        "observed": float(legacy_sweep_row["absolute_weight_threshold"]),
                        "absolute_delta": threshold_delta,
                        "tolerance": regression_tolerance,
                        "passed": threshold_delta <= regression_tolerance,
                    },
                    {
                        **{key: base[key] for key in [
                            "model_id",
                            "display",
                            "checkpoint_path",
                            "layer_key",
                            "stage",
                            "layer_order",
                            "conv_depth_norm",
                        ]},
                        "check_name": "epsilon_0p01_sparsity_matches_legacy",
                        "expected": legacy_sparsity,
                        "observed": observed_sparsity,
                        "absolute_delta": sparsity_delta,
                        "tolerance": regression_tolerance,
                        "passed": sparsity_delta <= regression_tolerance,
                    },
                ]
            )

            for cutoff in low_entropy_cutoffs:
                flagged = bool(entropy_h < cutoff)
                is_legacy = math.isclose(
                    cutoff,
                    LEGACY_LOW_ENTROPY_CUTOFF,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                decision_rows.append(
                    {
                        **base,
                        "decision_family": "low_entropy_H",
                        "decision_metric": "entropy_H_fixed",
                        "decision_comparator": "<",
                        "decision_cutoff": float(cutoff),
                        "decision_cutoff_label": threshold_label(cutoff),
                        "decision_value": entropy_h,
                        "flagged": flagged,
                        "is_legacy_decision_cutoff": is_legacy,
                    }
                )
                if is_legacy:
                    expected = bool(entropy_h < LEGACY_LOW_ENTROPY_CUTOFF)
                    regression_rows.append(
                        {
                            **{key: base[key] for key in [
                                "model_id",
                                "display",
                                "checkpoint_path",
                                "layer_key",
                                "stage",
                                "layer_order",
                                "conv_depth_norm",
                            ]},
                            "check_name": "H_0p50_low_entropy_decision_matches_legacy",
                            "expected": int(expected),
                            "observed": int(flagged),
                            "absolute_delta": abs(int(flagged) - int(expected)),
                            "tolerance": 0.0,
                            "passed": flagged == expected,
                        }
                    )

            for cutoff in random_ratio_cutoffs:
                flagged = bool(h_over_th > cutoff)
                is_legacy = math.isclose(
                    cutoff,
                    LEGACY_RANDOM_RATIO_CUTOFF,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                decision_rows.append(
                    {
                        **base,
                        "decision_family": "random_like_H_over_TH",
                        "decision_metric": "H_over_TH_fixed",
                        "decision_comparator": ">",
                        "decision_cutoff": float(cutoff),
                        "decision_cutoff_label": threshold_label(cutoff),
                        "decision_value": h_over_th,
                        "flagged": flagged,
                        "is_legacy_decision_cutoff": is_legacy,
                    }
                )
                if is_legacy:
                    expected = bool(h_over_th > LEGACY_RANDOM_RATIO_CUTOFF)
                    regression_rows.append(
                        {
                            **{key: base[key] for key in [
                                "model_id",
                                "display",
                                "checkpoint_path",
                                "layer_key",
                                "stage",
                                "layer_order",
                                "conv_depth_norm",
                            ]},
                            "check_name": "H_over_TH_0p95_decision_matches_legacy",
                            "expected": int(expected),
                            "observed": int(flagged),
                            "absolute_delta": abs(int(flagged) - int(expected)),
                            "tolerance": 0.0,
                            "passed": flagged == expected,
                        }
                    )
        print(f"[analyze] {spec.display}: {len(items)} layers", flush=True)

    return (
        pd.DataFrame(baseline_rows),
        pd.DataFrame(sparsity_rows),
        pd.DataFrame(decision_rows),
        pd.DataFrame(regression_rows),
    )


def build_threshold_summary(
    sparsity_df: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (model_id, display, epsilon), sub in sparsity_df.groupby(
        ["model_id", "display", "epsilon_fraction"], sort=False
    ):
        rows.append(
            {
                "model_id": model_id,
                "display": display,
                "threshold_family": "sparsity_epsilon",
                "threshold": float(epsilon),
                "criterion": "sparsity_S > 0",
                "num_layers": len(sub),
                "flagged_layer_count": int(sub["has_any_sparse_filters"].sum()),
                "flagged_layer_fraction": float(sub["has_any_sparse_filters"].mean()),
                "secondary_criterion": "sparsity_S > 0.01",
                "secondary_flagged_layer_count": int(sub["layer_sparsity_gt_0p01"].sum()),
                "mean_metric_value": float(sub["sparsity_S"].mean()),
                "maximum_metric_value": float(sub["sparsity_S"].max()),
                "is_legacy_threshold": bool(sub["is_legacy_epsilon_0p01"].iloc[0]),
            }
        )
    for (model_id, display, family, cutoff), sub in decision_df.groupby(
        ["model_id", "display", "decision_family", "decision_cutoff"], sort=False
    ):
        rows.append(
            {
                "model_id": model_id,
                "display": display,
                "threshold_family": family,
                "threshold": float(cutoff),
                "criterion": f"{sub['decision_metric'].iloc[0]} {sub['decision_comparator'].iloc[0]} cutoff",
                "num_layers": len(sub),
                "flagged_layer_count": int(sub["flagged"].sum()),
                "flagged_layer_fraction": float(sub["flagged"].mean()),
                "secondary_criterion": "",
                "secondary_flagged_layer_count": math.nan,
                "mean_metric_value": float(sub["decision_value"].mean()),
                "maximum_metric_value": float(sub["decision_value"].max()),
                "is_legacy_threshold": bool(sub["is_legacy_decision_cutoff"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def draw_fixed_entropy_by_depth(baseline_df: pd.DataFrame, path: Path) -> None:
    series = []
    model_ids = list(baseline_df["model_id"].drop_duplicates())
    for index, model_id in enumerate(model_ids):
        sub = baseline_df[baseline_df["model_id"] == model_id].sort_values("layer_order")
        series.append(
            {
                "x": sub["conv_depth_norm"].tolist(),
                "y": sub["entropy_H_fixed"].tolist(),
                "label": f"{sub['display'].iloc[0]}: H",
                "color": PALETTE[index % len(PALETTE)],
            }
        )
    first = baseline_df[baseline_df["model_id"] == model_ids[0]].sort_values("layer_order")
    series.append(
        {
            "x": first["conv_depth_norm"].tolist(),
            "y": first["random_threshold_TH"].tolist(),
            "label": "Random-like reference TH(n)",
            "color": (100, 100, 100),
        }
    )
    common.draw_line_series(
        series,
        "Fixed Layer Entropy by Residual-Block Depth",
        "normalized convolution depth",
        "base-10 entropy H",
        path,
        y_min=0.0,
        y_max=max(1.0, float(baseline_df["random_threshold_TH"].max()) * 1.03),
    )


def draw_threshold_heatmap(
    df: pd.DataFrame,
    path: Path,
    title: str,
    subtitle: str,
    row_field: str,
    value_field: str,
    row_prefix: str,
    binary: bool = False,
    vmax: float | None = None,
) -> None:
    model_ids = list(df["model_id"].drop_duplicates())
    row_values = sorted(float(value) for value in df[row_field].drop_duplicates())
    first_model = df[df["model_id"] == model_ids[0]]
    layer_order = (
        first_model[["layer_order", "layer_key"]]
        .drop_duplicates()
        .sort_values("layer_order")
    )
    layer_keys = layer_order["layer_key"].tolist()
    num_layers = len(layer_keys)
    cell_w, cell_h = 56, 30
    left, top, right, bottom = 225, 126, 40, 190
    block_gap = 54
    block_h = 34 + len(row_values) * cell_h
    width = left + num_layers * cell_w + right
    height = top + len(model_ids) * block_h + max(len(model_ids) - 1, 0) * block_gap + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = common.get_font(24, bold=True)
    label_font = common.get_font(13, bold=True)
    small_font = common.get_font(10)
    value_font = common.get_font(9)
    draw.text((24, 22), title, font=title_font, fill=(20, 20, 20))
    draw.text((24, 58), subtitle, font=small_font, fill=(75, 75, 75))
    if vmax is None and not binary:
        finite = pd.to_numeric(df[value_field], errors="coerce")
        vmax = max(float(finite.max()), 1e-12)
    for model_index, model_id in enumerate(model_ids):
        model_df = df[df["model_id"] == model_id]
        y_block = top + model_index * (block_h + block_gap)
        display = str(model_df["display"].iloc[0])
        draw.text((24, y_block), display, font=label_font, fill=PALETTE[model_index % len(PALETTE)])
        grid_top = y_block + 30
        for row_index, threshold in enumerate(row_values):
            y0 = grid_top + row_index * cell_h
            label = f"{row_prefix}{threshold_label(threshold)}"
            draw.text((28, y0 + 8), label, font=small_font, fill=(35, 35, 35))
            threshold_df = model_df[np.isclose(model_df[row_field].astype(float), threshold)].set_index(
                "layer_key"
            )
            for column, layer_key in enumerate(layer_keys):
                x0 = left + column * cell_w
                if layer_key not in threshold_df.index:
                    color = (225, 225, 225)
                    text_value = "NA"
                else:
                    raw_value = threshold_df.loc[layer_key, value_field]
                    if isinstance(raw_value, pd.Series):
                        raw_value = raw_value.iloc[0]
                    value = float(raw_value)
                    if not np.isfinite(value):
                        color = (225, 225, 225)
                        text_value = "NA"
                    elif binary:
                        color = (178, 24, 43) if bool(value) else (225, 238, 247)
                        text_value = "yes" if bool(value) else "·"
                    else:
                        color = common.heat_color(value / max(float(vmax), 1e-12))
                        if value == 0.0:
                            text_value = "0"
                        elif abs(value) < 0.01:
                            text_value = f"{value:.1g}"
                        else:
                            text_value = f"{value:.2f}"
                draw.rectangle(
                    [x0, y0, x0 + cell_w, y0 + cell_h],
                    fill=color,
                    outline=(238, 238, 238),
                )
                text_color = (255, 255, 255) if common.luminance(color) < 105 else (30, 30, 30)
                common.draw_centered(
                    draw,
                    (x0 + cell_w / 2, y0 + cell_h / 2),
                    text_value,
                    value_font,
                    text_color,
                )
        draw.rectangle(
            [left, grid_top, left + num_layers * cell_w, grid_top + len(row_values) * cell_h],
            outline=(60, 60, 60),
            width=1,
        )

    final_grid_bottom = top + (len(model_ids) - 1) * (block_h + block_gap) + 30 + len(row_values) * cell_h
    for column, layer_key in enumerate(layer_keys):
        label = layer_key.replace(".weight", "")
        text_w, text_h = common.text_size(draw, label, small_font)
        text_image = Image.new("RGBA", (text_w + 6, text_h + 6), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text((3, 3), label, font=small_font, fill=(30, 30, 30))
        text_image = text_image.rotate(90, expand=True)
        x0 = left + column * cell_w
        image.paste(
            text_image,
            (int(x0 + cell_w / 2 - text_image.width / 2), final_grid_bottom + 12),
            text_image,
        )
    legend_y = height - 34
    if binary:
        draw.rectangle([left, legend_y - 8, left + 24, legend_y + 8], fill=(178, 24, 43))
        draw.text((left + 32, legend_y - 8), "criterion true", font=small_font, fill=(30, 30, 30))
        draw.rectangle([left + 160, legend_y - 8, left + 184, legend_y + 8], fill=(225, 238, 247))
        draw.text((left + 192, legend_y - 8), "criterion false", font=small_font, fill=(30, 30, 30))
    else:
        legend_w = 260
        for index in range(legend_w):
            value = index / max(legend_w - 1, 1)
            draw.line(
                [(left + index, legend_y - 8), (left + index, legend_y + 8)],
                fill=common.heat_color(value),
            )
        draw.text((left, legend_y + 13), "0", font=small_font, fill=(30, 30, 30))
        draw.text(
            (left + legend_w - 40, legend_y + 13),
            f"{float(vmax):.3g}",
            font=small_font,
            fill=(30, 30, 30),
        )
    image.save(path)


def draw_metric_lines_for_thresholds(
    df: pd.DataFrame,
    output_dir: Path,
    threshold_field: str,
    metric_field: str,
    threshold_prefix: str,
    title_prefix: str,
    ylabel: str,
    *,
    add_cutoff_line: bool = False,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Write one ImageNet-vs-LeJEPA depth chart per swept threshold.

    Decision heatmaps encode booleans, but their companion charts intentionally
    plot the fixed continuous statistic plus its cutoff.  This makes both the
    comparison and every threshold crossing visible instead of reducing a line
    chart to a sequence of zeros and ones.
    """
    common.ensure_dir(output_dir)
    model_ids = list(df["model_id"].drop_duplicates())
    thresholds = sorted(float(value) for value in df[threshold_field].drop_duplicates())
    for threshold in thresholds:
        threshold_df = df[np.isclose(df[threshold_field].astype(float), threshold)]
        series = []
        for model_index, model_id in enumerate(model_ids):
            model_df = threshold_df[threshold_df["model_id"] == model_id].sort_values("layer_order")
            if model_df.empty:
                continue
            series.append(
                {
                    "x": model_df["conv_depth_norm"].astype(float).tolist(),
                    "y": model_df[metric_field].astype(float).tolist(),
                    "label": str(model_df["display"].iloc[0]),
                    "color": PALETTE[model_index % len(PALETTE)],
                }
            )
        if add_cutoff_line and not threshold_df.empty:
            ordered = threshold_df.sort_values("layer_order")
            x_values = ordered["conv_depth_norm"].astype(float)
            series.append(
                {
                    "x": [float(x_values.min()), float(x_values.max())],
                    "y": [threshold, threshold],
                    "label": f"decision cutoff = {threshold_label(threshold)}",
                    "color": (95, 95, 95),
                }
            )
        common.draw_line_series(
            series,
            f"{title_prefix}: {threshold_prefix}{threshold_label(threshold)}",
            "normalized convolution depth",
            ylabel,
            output_dir / threshold_filename(threshold_prefix.rstrip("="), threshold),
            y_min=y_min,
            y_max=y_max,
        )


def write_summary(
    path: Path,
    specs: Sequence[ModelSpec],
    baseline_df: pd.DataFrame,
    sparsity_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    sparsity_epsilons: Sequence[float],
    low_entropy_cutoffs: Sequence[float],
    random_ratio_cutoffs: Sequence[float],
    include_stem: bool,
    compute_clean_entropy: bool,
) -> None:
    lines = [
        "# RN18 Filter-Quality Threshold Sensitivity",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope and Definitions",
        "",
        f"- Stem included: `{include_stem}`. The default and this run's paper-aligned residual scope are recorded explicitly.",
        "- Baseline entropy `H` is computed once per raw layer from the normalized covariance eigenvalues using base-10 Shannon entropy.",
        "- Baseline `H` is mathematically fixed across every cutoff; entropy cutoff sweeps change only the decision flag.",
        "- `TH(n)` is the published random-layer reference and `H/TH` is also fixed.",
        "- Sparsity epsilon is multiplied by that layer's maximum absolute raw weight.",
        "- A filter is sparse only when all nine absolute coefficients are below the resulting threshold.",
        f"- Clean entropy after sparse-filter exclusion computed: `{compute_clean_entropy}`. It is a separate sensitivity diagnostic, not a replacement for fixed `H`.",
        "- No global PCA basis is used by any metric in this report.",
        "",
        "Threshold grids:",
        "",
        f"- Sparsity epsilons: `{[float(value) for value in sparsity_epsilons]}`",
        f"- Low-entropy decisions `H < cutoff`: `{[float(value) for value in low_entropy_cutoffs]}`",
        f"- Random-like decisions `H/TH > cutoff`: `{[float(value) for value in random_ratio_cutoffs]}`",
        "",
        "## Models",
        "",
        "| ID | Display | Checkpoint | Layers |",
        "|---|---|---|---:|",
    ]
    for spec in specs:
        count = int((baseline_df["model_id"] == spec.model_id).sum())
        lines.append(f"| `{spec.model_id}` | {spec.display} | `{spec.checkpoint_path}` | {count} |")
    lines.extend(
        [
            "",
            "## Legacy Regression Checks",
            "",
        ]
    )
    passed = int(regression_df["passed"].sum())
    total = len(regression_df)
    lines.append(f"- Passed: `{passed}/{total}`")
    failures = regression_df[~regression_df["passed"]]
    if failures.empty:
        lines.append("- Legacy entropy, epsilon `0.01` sparsity, `H < 0.50`, and `H/TH > 0.95` behavior was preserved.")
    else:
        lines.append(f"- **Warning:** `{len(failures)}` legacy regression checks failed; inspect `regression_checks.csv`.")
    lines.extend(["", "## Fixed Entropy Extremes", ""])
    lines.extend(["| Model | Minimum-H layer | H min | Maximum-H layer | H max |", "|---|---|---:|---|---:|"])
    for model_id, sub in baseline_df.groupby("model_id", sort=False):
        minimum = sub.loc[sub["entropy_H_fixed"].idxmin()]
        maximum = sub.loc[sub["entropy_H_fixed"].idxmax()]
        lines.append(
            f"| {minimum['display']} | `{minimum['layer_key']}` | {float(minimum['entropy_H_fixed']):.4f} | "
            f"`{maximum['layer_key']}` | {float(maximum['entropy_H_fixed']):.4f} |"
        )

    lines.extend(["", "## Sparsity Sensitivity", ""])
    lines.extend(
        [
            "| Model | Epsilon | Layers with S>0 | Layers with S>0.01 | Mean S | Max S |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    sparse_summary = summary_df[summary_df["threshold_family"] == "sparsity_epsilon"]
    for row in sparse_summary.itertuples():
        lines.append(
            f"| {row.display} | {float(row.threshold):.4g} | {int(row.flagged_layer_count)}/{int(row.num_layers)} | "
            f"{int(row.secondary_flagged_layer_count)}/{int(row.num_layers)} | "
            f"{float(row.mean_metric_value):.4f} | {float(row.maximum_metric_value):.4f} |"
        )

    lines.extend(["", "## Entropy Decision Sensitivity", ""])
    lines.extend(
        [
            "| Model | Decision | Cutoff | Flagged layers |",
            "|---|---|---:|---:|",
        ]
    )
    entropy_summary = summary_df[summary_df["threshold_family"] != "sparsity_epsilon"]
    for row in entropy_summary.itertuples():
        decision = "H < cutoff" if row.threshold_family == "low_entropy_H" else "H/TH > cutoff"
        lines.append(
            f"| {row.display} | {decision} | {float(row.threshold):.4g} | "
            f"{int(row.flagged_layer_count)}/{int(row.num_layers)} |"
        )

    if compute_clean_entropy:
        finite_clean = sparsity_df[np.isfinite(sparsity_df["clean_entropy_H"])]
        if not finite_clean.empty:
            largest = finite_clean.iloc[
                finite_clean["clean_entropy_delta_from_fixed_H"].abs().argmax()
            ]
            lines.extend(
                [
                    "",
                    "## Clean-Entropy Diagnostic",
                    "",
                    "This diagnostic removes sparse filters separately at each epsilon and then recomputes entropy. It must not be described as a threshold-dependent version of baseline `H`.",
                    "",
                    f"Largest absolute clean-H change: `{largest['display']}` / `{largest['layer_key']}` at epsilon "
                    f"`{float(largest['epsilon_fraction']):.4g}`: delta `{float(largest['clean_entropy_delta_from_fixed_H']):.6f}`.",
                ]
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `layer_quality_baseline.csv`: one row per model/layer with fixed H, TH, H/TH, raw-scale context, and legacy regression fields.",
            "- `sparsity_sensitivity.csv`: tidy model/layer/epsilon values, actual absolute thresholds, sparse counts, S, and optional clean entropy.",
            "- `entropy_decision_sensitivity.csv`: tidy model/layer/cutoff decision flags with fixed underlying H and H/TH values.",
            "- `threshold_model_summary.csv`: model-level counts and aggregate values for every threshold.",
            "- `regression_checks.csv`: explicit observed/expected legacy checks.",
            "- `figures/fixed_entropy_by_depth.png`: fixed H and published TH(n) reference.",
            "- `figures/sparsity_threshold_by_depth_heatmap.png`: threshold-by-layer sparsity values.",
            "- `figures/sparsity_threshold_by_depth_heatmap/`: one ImageNet-versus-LeJEPA depth line chart per epsilon.",
            "- `figures/low_entropy_decision_heatmap.png`: H-cutoff decision sensitivity.",
            "- `figures/low_entropy_decision_heatmap/`: one fixed-H comparison chart, with its cutoff line, per H cutoff.",
            "- `figures/random_like_decision_heatmap.png`: H/TH-cutoff decision sensitivity.",
            "- `figures/random_like_decision_heatmap/`: one fixed-H/TH comparison chart, with its cutoff line, per ratio cutoff.",
        ]
    )
    if compute_clean_entropy:
        lines.extend(
            [
                "- `figures/clean_entropy_after_sparse_exclusion_heatmap.png`: separately labeled post-exclusion clean H.",
                "- `figures/clean_entropy_after_sparse_exclusion_heatmap/`: one ImageNet-versus-LeJEPA clean-H chart per epsilon.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Relative-to-peak sparsity is scale-normalized within a layer but remains sensitive to a single extreme coefficient.",
            "- Threshold sweeps reveal decision stability; they do not turn entropy or sparsity into direct measures of model accuracy.",
            "- Results are descriptive for the supplied checkpoints and do not establish seed-level reproducibility.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    specs = [parse_model_spec(text) for text in args.model_spec]
    model_ids = [spec.model_id for spec in specs]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Every --model-spec must have a unique model_id")
    if args.regression_tolerance < 0.0:
        raise ValueError("--regression-tolerance must be non-negative")
    sparsity_epsilons = parse_grid(
        args.sparsity_epsilons,
        "sparsity epsilons",
        minimum_count=7,
        required_value=LEGACY_SPARSITY_EPSILON,
    )
    low_entropy_cutoffs = parse_grid(
        args.low_entropy_cutoffs,
        "low-entropy cutoffs",
        minimum_count=7,
        required_value=LEGACY_LOW_ENTROPY_CUTOFF,
    )
    random_ratio_cutoffs = parse_grid(
        args.random_ratio_cutoffs,
        "random-ratio cutoffs",
        minimum_count=6,
        required_value=LEGACY_RANDOM_RATIO_CUTOFF,
    )

    output_dir = Path(args.output_dir).expanduser()
    figure_dir = output_dir / "figures"
    common.ensure_dir(output_dir)
    common.ensure_dir(figure_dir)
    print(f"[start] output_dir={output_dir.resolve()}", flush=True)
    print(f"[start] include_stem={args.include_stem}", flush=True)

    baseline_df, sparsity_df, decision_df, regression_df = analyze_models(
        specs,
        sparsity_epsilons,
        low_entropy_cutoffs,
        random_ratio_cutoffs,
        include_stem=args.include_stem,
        compute_clean_entropy=args.compute_clean_entropy,
        regression_tolerance=args.regression_tolerance,
    )
    summary_df = build_threshold_summary(sparsity_df, decision_df)

    baseline_df.to_csv(output_dir / "layer_quality_baseline.csv", index=False)
    sparsity_df.to_csv(output_dir / "sparsity_sensitivity.csv", index=False)
    decision_df.to_csv(output_dir / "entropy_decision_sensitivity.csv", index=False)
    summary_df.to_csv(output_dir / "threshold_model_summary.csv", index=False)
    regression_df.to_csv(output_dir / "regression_checks.csv", index=False)

    config = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": [
            {
                "model_id": spec.model_id,
                "display": spec.display,
                "checkpoint_path": str(spec.checkpoint_path),
            }
            for spec in specs
        ],
        "include_stem": bool(args.include_stem),
        "compute_clean_entropy": bool(args.compute_clean_entropy),
        "sparsity_epsilons": sparsity_epsilons,
        "low_entropy_cutoffs": low_entropy_cutoffs,
        "random_ratio_cutoffs": random_ratio_cutoffs,
        "legacy_thresholds": {
            "sparsity_epsilon": LEGACY_SPARSITY_EPSILON,
            "low_entropy_H": LEGACY_LOW_ENTROPY_CUTOFF,
            "random_like_H_over_TH": LEGACY_RANDOM_RATIO_CUTOFF,
        },
        "regression_tolerance": args.regression_tolerance,
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")

    print("[plot] writing threshold sensitivity figures", flush=True)
    draw_fixed_entropy_by_depth(baseline_df, figure_dir / "fixed_entropy_by_depth.png")
    draw_threshold_heatmap(
        sparsity_df,
        figure_dir / "sparsity_threshold_by_depth_heatmap.png",
        "Sparsity Sensitivity Across RN18 Depth",
        "Rows sweep epsilon in |w| < epsilon * layer peak; cells show the fraction of all-nine-near-zero kernels.",
        row_field="epsilon_fraction",
        value_field="sparsity_S",
        row_prefix="epsilon=",
        binary=False,
        vmax=max(float(sparsity_df["sparsity_S"].max()), 1e-12),
    )
    draw_metric_lines_for_thresholds(
        sparsity_df,
        figure_dir / "sparsity_threshold_by_depth_heatmap",
        "epsilon_fraction",
        "sparsity_S",
        "epsilon=",
        "Sparsity Across Depth",
        "sparse-filter fraction S",
        y_min=0.0,
    )
    low_df = decision_df[decision_df["decision_family"] == "low_entropy_H"]
    draw_threshold_heatmap(
        low_df,
        figure_dir / "low_entropy_decision_heatmap.png",
        "Low-Entropy Decision Sensitivity",
        "Baseline H is fixed; only the decision H < cutoff changes across rows.",
        row_field="decision_cutoff",
        value_field="flagged",
        row_prefix="H<",
        binary=True,
    )
    draw_metric_lines_for_thresholds(
        low_df,
        figure_dir / "low_entropy_decision_heatmap",
        "decision_cutoff",
        "decision_value",
        "H=",
        "Fixed Layer Entropy and Low-Entropy Cutoff",
        "base-10 entropy H",
        add_cutoff_line=True,
        y_min=0.0,
        y_max=max(1.0, float(low_df["decision_cutoff"].max()) * 1.03),
    )
    random_df = decision_df[decision_df["decision_family"] == "random_like_H_over_TH"]
    draw_threshold_heatmap(
        random_df,
        figure_dir / "random_like_decision_heatmap.png",
        "Random-Like Entropy Decision Sensitivity",
        "Baseline H/TH is fixed; only the decision H/TH > cutoff changes across rows.",
        row_field="decision_cutoff",
        value_field="flagged",
        row_prefix="H/TH>",
        binary=True,
    )
    draw_metric_lines_for_thresholds(
        random_df,
        figure_dir / "random_like_decision_heatmap",
        "decision_cutoff",
        "decision_value",
        "H_over_TH=",
        "Fixed Entropy Ratio and Random-Like Cutoff",
        "entropy ratio H/TH",
        add_cutoff_line=True,
        y_min=0.0,
        y_max=max(1.1, float(random_df["decision_cutoff"].max()) * 1.03),
    )
    if args.compute_clean_entropy:
        draw_threshold_heatmap(
            sparsity_df,
            figure_dir / "clean_entropy_after_sparse_exclusion_heatmap.png",
            "Clean Entropy After Sparse-Filter Exclusion",
            "Separate diagnostic: remove epsilon-sparse kernels, then recompute H. Baseline H remains fixed elsewhere.",
            row_field="epsilon_fraction",
            value_field="clean_entropy_H",
            row_prefix="epsilon=",
            binary=False,
            vmax=ENTROPY_MAX,
        )
        draw_metric_lines_for_thresholds(
            sparsity_df,
            figure_dir / "clean_entropy_after_sparse_exclusion_heatmap",
            "epsilon_fraction",
            "clean_entropy_H",
            "epsilon=",
            "Clean Entropy After Sparse-Filter Exclusion",
            "base-10 clean entropy H",
            y_min=0.0,
            y_max=ENTROPY_MAX,
        )

    write_summary(
        output_dir / "summary.md",
        specs,
        baseline_df,
        sparsity_df,
        decision_df,
        summary_df,
        regression_df,
        sparsity_epsilons,
        low_entropy_cutoffs,
        random_ratio_cutoffs,
        include_stem=args.include_stem,
        compute_clean_entropy=args.compute_clean_entropy,
    )

    failures = regression_df[~regression_df["passed"]]
    if not failures.empty:
        print(
            f"[error] {len(failures)} legacy regression checks failed; see regression_checks.csv",
            flush=True,
        )
        return 2
    print(
        f"[done] {len(specs)} models, {len(baseline_df)} model/layer rows, "
        f"{len(regression_df)} legacy checks passed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
