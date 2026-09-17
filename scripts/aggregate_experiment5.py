#!/usr/bin/env python
"""Aggregate seed-level Experiment 5 A/B/C primary effects without filter-level pseudoreplication."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import DEFAULT_CONFIG, atomic_json_dump, config_sha256, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-root", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/discovery")
    parser.add_argument("--output-dir", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/aggregate_discovery")
    parser.add_argument("--seeds", default="13,17,23,31,43")
    return parser.parse_args()


def exact_two_sided_sign_flip_pvalue(effects: list[float]) -> float:
    observed = abs(float(np.mean(effects)))
    values = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(effects)):
        values.append(abs(float(np.mean(np.asarray(effects) * np.asarray(signs)))))
    return sum(value >= observed - 1e-15 for value in values) / len(values)


def paired_summary(effects: list[float]) -> dict[str, Any]:
    n = len(effects)
    mean = statistics.mean(effects)
    standard_deviation = statistics.stdev(effects) if n > 1 else float("nan")
    standard_error = standard_deviation / math.sqrt(n) if n > 1 else float("nan")
    # Exact t critical values for df 1..9 avoid introducing an analysis dependency.
    t975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    critical = t975.get(n - 1, 1.96)
    return {
        "n_independent_paired_seeds": n,
        "mean_paired_effect": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "t_interval_95": [mean - critical * standard_error, mean + critical * standard_error]
        if n > 1
        else [float("nan"), float("nan")],
        "exact_two_sided_sign_flip_pvalue": exact_two_sided_sign_flip_pvalue(effects),
        "all_effects_same_nonzero_sign": all(value > 0 for value in effects)
        or all(value < 0 for value in effects),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    update = int(config["primary_endpoint"]["update"])
    values: dict[int, dict[str, float]] = {}
    initial_hashes: dict[int, dict[str, tuple[str, str]]] = {}
    for seed in seeds:
        values[seed] = {}
        initial_hashes[seed] = {}
        for arm in ["A", "B", "C"]:
            run_dir = input_root / f"arm_{arm}" / f"seed_{seed}"
            manifest_path = run_dir / "manifest.json"
            summary_path = run_dir / "diagnostics" / f"update_{update:06d}" / "summary.json"
            if not manifest_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"Missing manifest/primary diagnostic for {arm}, seed {seed}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if manifest["config_sha256"] != config_sha256(config):
                raise ValueError(f"Config mismatch for {arm}, seed {seed}")
            if not summary["primary_endpoint_is_scheduled_here"]:
                raise ValueError(f"Diagnostic is not at the primary update for {arm}, seed {seed}")
            initial_hashes[seed][arm] = (
                manifest["initial_backbone_state_sha256"],
                manifest["initial_projector_state_sha256"],
            )
            values[seed][arm] = float(summary["primary_endpoint_value"])
        if len(set(initial_hashes[seed].values())) != 1:
            raise RuntimeError(f"A/B/C did not share identical initialization for seed {seed}")

    rows = []
    for seed in seeds:
        rows.append(
            {
                "seed": seed,
                "arm_A": values[seed]["A"],
                "arm_B": values[seed]["B"],
                "arm_C": values[seed]["C"],
                "B_minus_C": values[seed]["B"] - values[seed]["C"],
                "B_minus_A": values[seed]["B"] - values[seed]["A"],
                "paired_initialization_verified": True,
            }
        )
    with (output_dir / "primary_endpoint_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": config["experiment"],
        "stage": "discovery",
        "training_run_is_experimental_unit": True,
        "primary_endpoint": config["primary_endpoint"],
        "primary_contrast_B_minus_C": paired_summary([row["B_minus_C"] for row in rows]),
        "secondary_contrast_B_minus_A": paired_summary([row["B_minus_A"] for row in rows]),
        "seeds": seeds,
        "caution": (
            "Discovery intervals quantify seed variation but do not turn selected endpoints or candidates into confirmation. "
            "With five paired runs the minimum exact two-sided sign-flip p-value is 0.0625."
        ),
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
