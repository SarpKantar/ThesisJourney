#!/usr/bin/env python
"""Compute the prespecified layer-matrix weight diagnostics for Experiment 5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import DEFAULT_CONFIG, atomic_json_dump, config_sha256, load_config, sha256_file


RESIDUAL_3X3 = re.compile(r"^layer[1-4]\.\d+\.conv[12]\.weight$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def bn_key_for_weight(name: str, source: str) -> str | None:
    if source == "backbone":
        if name == "conv1.weight":
            return "bn1"
        match = re.match(r"^(layer[1-4]\.\d+)\.conv([12])\.weight$", name)
        if match:
            return f"{match.group(1)}.bn{match.group(2)}"
        match = re.match(r"^(layer[2-4]\.0)\.downsample\.0\.weight$", name)
        if match:
            return f"{match.group(1)}.downsample.1"
    if source == "projector":
        match = re.match(r"^net\.(0|3)\.weight$", name)
        if match:
            return f"net.{int(match.group(1)) + 1}"
    return None


def stage_for(name: str, source: str) -> str:
    if source == "projector":
        return "projector"
    if name.startswith("layer"):
        return name.split(".", 1)[0]
    if name == "conv1.weight":
        return "stem"
    return "backbone_other"


def matrix_kind(name: str, tensor: torch.Tensor, source: str) -> str:
    if source == "projector":
        return "projector_linear"
    if name == "conv1.weight":
        return "stem_3x3"
    if RESIDUAL_3X3.match(name):
        return "residual_3x3"
    if re.match(r"^layer[2-4]\.0\.downsample\.0\.weight$", name):
        return "shortcut_1x1"
    return "other_matrix"


def iter_matrices(state: Mapping[str, torch.Tensor], source: str):
    for name, tensor in state.items():
        if not name.endswith(".weight") and name != "conv1.weight":
            continue
        if source == "backbone" and tensor.ndim == 4:
            yield name, tensor.detach().float().reshape(tensor.shape[0], -1)
        elif source == "projector" and tensor.ndim == 2:
            yield name, tensor.detach().float().reshape(tensor.shape[0], -1)


def folded_matrix(
    name: str,
    matrix: torch.Tensor,
    state: Mapping[str, torch.Tensor],
    source: str,
) -> torch.Tensor | None:
    prefix = bn_key_for_weight(name, source)
    if prefix is None:
        return None
    required = [f"{prefix}.weight", f"{prefix}.running_var"]
    if not all(key in state for key in required):
        return None
    gamma = state[f"{prefix}.weight"].detach().float()
    running_var = state[f"{prefix}.running_var"].detach().float()
    scale = gamma / torch.sqrt(running_var + 1e-5)
    if len(scale) != matrix.shape[0]:
        raise ValueError(f"BN fold shape mismatch for {source}.{name}")
    return matrix * scale[:, None]


def spectral_metrics(matrix: torch.Tensor) -> tuple[dict[str, float], np.ndarray]:
    values = torch.linalg.svdvals(matrix.double()).cpu().numpy()
    energy = np.square(values)
    total = float(energy.sum())
    rank_limit = min(matrix.shape)
    if total <= 0:
        q = np.zeros_like(energy)
        effective_rank = 0.0
        stable_rank = 0.0
    else:
        q = energy / total
        positive = q[q > 0]
        effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
        stable_rank = float(total / max(float(energy.max()), np.finfo(np.float64).tiny))
    row_norms = torch.linalg.vector_norm(matrix.double(), dim=1).cpu().numpy()
    gram = matrix.double() @ matrix.double().T if matrix.shape[0] <= matrix.shape[1] else matrix.double().T @ matrix.double()
    gram_trace = float(torch.trace(gram).cpu())
    identity_scaled = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device) * (
        gram_trace / max(gram.shape[0], 1)
    )
    return (
        {
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "rank_limit": int(rank_limit),
            "weight_energy_effective_rank": effective_rank,
            "normalized_weight_energy_effective_rank": effective_rank / max(rank_limit, 1),
            "stable_rank": stable_rank,
            "normalized_stable_rank": stable_rank / max(rank_limit, 1),
            "frobenius_norm": float(matrix.double().norm().cpu()),
            "spectral_norm": float(values.max()) if len(values) else 0.0,
            "mean_row_norm": float(row_norms.mean()) if len(row_norms) else 0.0,
            "std_row_norm": float(row_norms.std()) if len(row_norms) else 0.0,
            "scale_normalized_gram_anisotropy": float((gram - identity_scaled).norm().cpu())
            / max(gram_trace, np.finfo(np.float64).tiny),
        },
        values,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA analysis was requested but is not available.")
    analysis_device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != config["experiment"]:
        raise ValueError(f"Not an Experiment 5 checkpoint: {checkpoint_path}")
    if checkpoint.get("config_sha256") != config_sha256(config):
        raise ValueError("Checkpoint protocol config does not match the analysis config.")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent.parent / "diagnostics" / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for source, state_key in [("backbone", "backbone_state"), ("projector", "projector_state")]:
        state = checkpoint[state_key]
        for name, raw in iter_matrices(state, source):
            representations = {"raw": raw}
            folded = folded_matrix(name, raw, state, source)
            if folded is not None:
                representations["bn_folded"] = folded
            row_norm = raw.norm(dim=1, keepdim=True)
            representations["row_direction"] = raw / row_norm.clamp_min(torch.finfo(raw.dtype).eps)
            for representation, matrix in representations.items():
                metrics, singular_values = spectral_metrics(matrix.to(analysis_device))
                base = {
                    "experiment": checkpoint["experiment"],
                    "arm": checkpoint["arm"],
                    "seed": checkpoint["seed"],
                    "update": checkpoint["update"],
                    "equivalent_epoch": checkpoint["equivalent_epoch"],
                    "source": source,
                    "stage": stage_for(name, source),
                    "matrix_kind": matrix_kind(name, raw, source),
                    "parameter": name,
                    "representation": representation,
                    "centered": False,
                }
                metric_rows.append({**base, **metrics})
                energy = np.square(singular_values)
                normalized_energy = energy / energy.sum() if float(energy.sum()) > 0 else np.zeros_like(energy)
                for index, (singular_value, energy_share) in enumerate(zip(singular_values, normalized_energy), start=1):
                    spectrum_rows.append(
                        {
                            **base,
                            "singular_index": index,
                            "singular_value": float(singular_value),
                            "normalized_weight_energy": float(energy_share),
                        }
                    )

    primary_rows = [
        row
        for row in metric_rows
        if row["source"] == "backbone"
        and row["matrix_kind"] == "residual_3x3"
        and row["representation"] == config["primary_endpoint"]["parameter_representation"]
    ]
    if len(primary_rows) != 16:
        raise RuntimeError(f"Primary endpoint expected 16 residual 3x3 matrices, found {len(primary_rows)}")
    primary_value = float(
        np.mean([row["normalized_weight_energy_effective_rank"] for row in primary_rows])
    )
    stage_rows = []
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        selected = [row for row in primary_rows if row["stage"] == stage]
        stage_rows.append(
            {
                "arm": checkpoint["arm"],
                "seed": checkpoint["seed"],
                "update": checkpoint["update"],
                "stage": stage,
                "layer_count": len(selected),
                "mean_normalized_weight_energy_effective_rank": float(
                    np.mean([row["normalized_weight_energy_effective_rank"] for row in selected])
                ),
            }
        )
    write_csv(output_dir / "weight_matrix_metrics.csv", metric_rows)
    write_csv(output_dir / "weight_matrix_spectra.csv", spectrum_rows)
    write_csv(output_dir / "primary_stage_summary.csv", stage_rows)
    summary = {
        "experiment": checkpoint["experiment"],
        "arm": checkpoint["arm"],
        "seed": checkpoint["seed"],
        "update": checkpoint["update"],
        "equivalent_epoch": checkpoint["equivalent_epoch"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": config_sha256(config),
        "primary_endpoint_definition": config["primary_endpoint"],
        "primary_endpoint_value": primary_value,
        "primary_endpoint_is_scheduled_here": int(checkpoint["update"])
        == int(config["primary_endpoint"]["update"]),
        "matrix_metric_rows": len(metric_rows),
        "spectrum_rows": len(spectrum_rows),
        "notes": [
            "All spectra are uncentered.",
            "BN folding uses gamma/sqrt(running_var+1e-5) for a directly following BN.",
            "Row-direction spectra are scale controls; row norms remain in weight_matrix_metrics.csv.",
            "Stem, shortcuts, and projector matrices are secondary diagnostics and excluded from the primary equal-layer mean.",
        ],
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
