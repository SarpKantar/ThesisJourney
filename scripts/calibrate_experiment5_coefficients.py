#!/usr/bin/env python
"""Calibrate arm C's coefficient against arm B regularizer-gradient scale."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import (
    DEFAULT_CONFIG,
    assert_pinned_lejepa,
    atomic_json_dump,
    autocast_context,
    build_model_pair,
    build_sigreg,
    config_sha256,
    load_config,
    resolve_device,
    seed_streams,
    sigreg_over_views,
    vicreg_variance_covariance,
)
from experiment5.data import load_split, make_base_cifar10, make_training_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split-file", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/protocol/cifar10_split_indices.npz")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/pilot/coefficient_ledger.json")
    parser.add_argument("--batches-per-seed", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--allow-dirty-lejepa", action="store_true")
    return parser.parse_args()


def group_norm(grads: Sequence[torch.Tensor | None]) -> float:
    squares = [gradient.detach().float().square().sum() for gradient in grads if gradient is not None]
    return float(torch.sqrt(torch.stack(squares).sum()).cpu()) if squares else 0.0


def gradient_norms(loss: torch.Tensor, backbone, projector, retain_graph: bool) -> dict[str, float]:
    backbone_parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    projector_parameters = [parameter for parameter in projector.parameters() if parameter.requires_grad]
    parameters = backbone_parameters + projector_parameters
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    backbone_gradients = gradients[: len(backbone_parameters)]
    projector_gradients = gradients[len(backbone_parameters) :]
    return {
        "backbone": group_norm(backbone_gradients),
        "projector": group_norm(projector_gradients),
        "joint": group_norm(gradients),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    assert_pinned_lejepa(config, allow_dirty=args.allow_dirty_lejepa)
    device = resolve_device(args.device)
    split = load_split(args.split_file)
    dataset = make_base_cifar10(args.data_dir, train=True, download=False)
    records: list[dict[str, Any]] = []

    for seed in config["seeds"]["pilot"]:
        streams = seed_streams(int(seed), config["seeds"]["stream_names"])
        backbone, projector = build_model_pair(config, streams["initialization"], device)
        sigreg = build_sigreg(config, device, streams["sigreg_directions"])
        loader = make_training_loader(
            dataset,
            split["train_indices"],
            config,
            streams,
            start_update=0,
            num_workers=args.num_workers,
        )
        backbone.train()
        projector.train()
        for batch_index, (views, _labels, _indices) in enumerate(loader):
            if batch_index >= args.batches_per_seed:
                break
            views = views.to(device, non_blocking=True)
            batch_size, num_views = views.shape[:2]
            with autocast_context(device):
                features = backbone(views.flatten(0, 1))
                projected = projector(features)
            projections = projected.float().view(batch_size, num_views, -1).transpose(0, 1)
            sig_value = sigreg_over_views(sigreg, projections)
            vc_value, vc_components = vicreg_variance_covariance(
                projections,
                variance_target=float(config["vicreg_comparator"]["variance_target"]),
                variance_epsilon=float(config["vicreg_comparator"]["variance_epsilon"]),
                variance_weight=float(config["vicreg_comparator"]["variance_weight"]),
                covariance_weight=float(config["vicreg_comparator"]["covariance_weight"]),
            )
            sig_norms = gradient_norms(sig_value, backbone, projector, retain_graph=True)
            vc_norms = gradient_norms(vc_value, backbone, projector, retain_graph=False)
            records.append(
                {
                    "seed": int(seed),
                    "batch": batch_index + 1,
                    "sigreg_value": float(sig_value.detach().cpu()),
                    "vc_value": float(vc_value.detach().cpu()),
                    "vc_variance": float(vc_components["variance"].detach().cpu()),
                    "vc_covariance": float(vc_components["covariance"].detach().cpu()),
                    "sigreg_gradient_norms": sig_norms,
                    "vc_gradient_norms": vc_norms,
                }
            )
            del views, features, projected, projections, sig_value, vc_value

    sig_coefficient = float(config["training"]["sigreg_coefficient"])
    median_sig = statistics.median(record["sigreg_gradient_norms"]["joint"] for record in records)
    median_vc = statistics.median(record["vc_gradient_norms"]["joint"] for record in records)
    if not np.isfinite(median_sig) or not np.isfinite(median_vc) or median_vc <= 0:
        raise RuntimeError(f"Invalid calibration medians: SIGReg={median_sig}, VC={median_vc}")
    alpha0 = sig_coefficient * median_sig / median_vc
    ledger = {
        "experiment": config["experiment"],
        "purpose": "gradient-scale convention only; not evidence of equivalent action",
        "calibration_stage": "initialization/first deterministic training batches; no parameter updates",
        "pilot_seeds": list(map(int, config["seeds"]["pilot"])),
        "batches_per_seed": args.batches_per_seed,
        "config_sha256": config_sha256(config),
        "sigreg_reference_coefficient": sig_coefficient,
        "median_unscaled_joint_sigreg_gradient_norm": median_sig,
        "median_unscaled_joint_vc_gradient_norm": median_vc,
        "alpha0": alpha0,
        "vc_candidate_coefficients": [alpha0 / 3.0, alpha0, 3.0 * alpha0],
        "sigreg_sensitivity_coefficients": [sig_coefficient / 3.0, sig_coefficient, 0.06],
        "group_summary": {
            group: {
                "median_sigreg": statistics.median(
                    record["sigreg_gradient_norms"][group] for record in records
                ),
                "median_vc": statistics.median(record["vc_gradient_norms"][group] for record in records),
            }
            for group in ["backbone", "projector", "joint"]
        },
        "records": records,
    }
    atomic_json_dump(args.output, ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
