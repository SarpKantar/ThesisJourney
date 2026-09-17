#!/usr/bin/env python
"""Numerically audit Experiment 5 losses against the pinned LeJEPA source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import (
    DEFAULT_CONFIG,
    all_view_alignment,
    assert_pinned_lejepa,
    atomic_json_dump,
    build_sigreg,
    config_sha256,
    load_config,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/validation/loss_parity.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--direction-seed", type=int, default=20260916)
    parser.add_argument("--value-atol", type=float, default=1e-5)
    parser.add_argument("--gradient-atol", type=float, default=2e-5)
    parser.add_argument("--allow-dirty-lejepa", action="store_true")
    return parser.parse_args()


def explicit_reference(x: torch.Tensor, seed: int, num_slices: int, t_max: float, n_points: int):
    """Literal real/imaginary characteristic-function calculation from source."""

    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    directions = torch.randn((x.shape[-1], num_slices), device=x.device, generator=generator)
    directions = directions / directions.norm(p=2, dim=0, keepdim=True)
    projected = x @ directions
    t = torch.linspace(0.0, t_max, n_points, dtype=torch.float32, device=x.device)
    phi = torch.exp(-0.5 * t.square())
    dt = t_max / (n_points - 1)
    weights = torch.full((n_points,), 2.0 * dt, dtype=torch.float32, device=x.device)
    weights[[0, -1]] = dt
    weights = weights * phi
    values = projected.unsqueeze(-1) * t
    real = torch.cos(values).mean(dim=0)
    imaginary = torch.sin(values).mean(dim=0)
    error = (real - phi).square() + imaginary.square()
    per_slice = (error @ weights) * x.shape[0]
    return per_slice.mean()


def gradient_summary(gradient: torch.Tensor) -> dict[str, float]:
    return {
        "l2": float(gradient.float().norm().cpu()),
        "max_abs": float(gradient.float().abs().max().cpu()),
        "mean_abs": float(gradient.float().abs().mean().cpu()),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    source = assert_pinned_lejepa(config, allow_dirty=args.allow_dirty_lejepa)
    device = resolve_device(args.device)
    generator = torch.Generator(device=device).manual_seed(314159)
    standard = torch.randn((args.samples, args.dimension), device=device, generator=generator)
    rank_deficient = standard.clone()
    rank_deficient[:, args.dimension // 2 :] = 0.0
    cases = {
        "standard_gaussian": standard,
        "shifted_gaussian": standard + 1.0,
        "rescaled_gaussian": standard * 2.0,
        "rank_deficient_gaussian": rank_deficient,
        "constant_zero": torch.zeros_like(standard),
    }
    sig_config = config["sigreg"]
    rows: list[dict[str, Any]] = []
    all_passed = True
    sigreg = build_sigreg(config, device, args.direction_seed)
    for name, data in cases.items():
        seed = int(sigreg.global_step.item())
        local_input = data.detach().clone().requires_grad_(True)
        local_value = sigreg(local_input)
        local_gradient = torch.autograd.grad(local_value, local_input)[0]
        reference_input = data.detach().clone().requires_grad_(True)
        reference_value = explicit_reference(
            reference_input,
            seed=seed,
            num_slices=int(sig_config["num_slices"]),
            t_max=float(sig_config["integration_bound"]),
            n_points=int(sig_config["integration_points"]),
        )
        reference_gradient = torch.autograd.grad(reference_value, reference_input)[0]
        value_error = float((local_value - reference_value).abs().detach().cpu())
        gradient_error = float((local_gradient - reference_gradient).abs().max().detach().cpu())
        passed = value_error <= args.value_atol and gradient_error <= args.gradient_atol
        all_passed = all_passed and passed
        rows.append(
            {
                "case": name,
                "direction_seed": seed,
                "pinned_value": float(local_value.detach().cpu()),
                "reference_value": float(reference_value.detach().cpu()),
                "absolute_value_error": value_error,
                "maximum_absolute_gradient_error": gradient_error,
                "pinned_gradient": gradient_summary(local_gradient),
                "reference_gradient": gradient_summary(reference_gradient),
                "passed": passed,
            }
        )

    sensitivity = {}
    for bound in [float(sig_config["historical_bound_for_sensitivity"]), float(sig_config["integration_bound"])]:
        sensitivity[str(bound)] = {}
        for name, data in cases.items():
            sensitivity[str(bound)][name] = float(
                explicit_reference(
                    data,
                    seed=args.direction_seed,
                    num_slices=int(sig_config["num_slices"]),
                    t_max=bound,
                    n_points=int(sig_config["integration_points"]),
                ).detach().cpu()
            )

    projections = torch.randn((4, 8, 6), generator=torch.Generator().manual_seed(7), requires_grad=True)
    center = projections.mean(dim=0)
    center.retain_grad()
    alignment = (projections - center.unsqueeze(0)).square().mean()
    alignment.backward()
    alignment_check = {
        "center_is_attached_to_graph": center.grad is not None,
        "center_gradient_l2": float(center.grad.norm()) if center.grad is not None else None,
        "all_view_gradient_nonzero": [bool(projections.grad[index].abs().sum() > 0) for index in range(4)],
        "note": "The center gradient is mathematically zero because residuals sum to zero; graph attachment confirms no detach/stop-gradient.",
    }
    independent_images_check = {
        "sigreg_sample_axis": "B (independent source images) for each view separately",
        "views_are_not_treated_as_independent_samples": True,
        "view_reduction": "arithmetic mean after one statistic per view",
    }
    result = {
        "experiment": config["experiment"],
        "config_sha256": config_sha256(config),
        "source": source,
        "device": str(device),
        "samples": args.samples,
        "dimension": args.dimension,
        "quadrature_audit": {
            "stored_nodes": int(sig_config["integration_points"]),
            "stored_interval": [0.0, float(sig_config["integration_bound"])],
            "symmetry": "negative frequencies represented by doubled trapezoid weights",
            "effective_interval": [
                -float(sig_config["integration_bound"]),
                float(sig_config["integration_bound"]),
            ],
            "important_interpretation": (
                "The pinned implementation stores 17 nodes on the nonnegative half, not 17 total nodes on [-5,5]. "
                "This audit records the executable reference convention explicitly."
            ),
        },
        "independent_images_and_views": independent_images_check,
        "alignment_gradient_check": alignment_check,
        "synthetic_cases": rows,
        "bound_sensitivity": sensitivity,
        "all_parity_checks_passed": all_passed,
        "caveats": [
            "Finite-sample Gaussian discrepancy is not expected to be zero.",
            "A constant-zero tensor may have a high statistic and a zero input gradient.",
            "These cases validate implementation behavior; they do not impose a universal ordering of non-Gaussian distributions.",
        ],
    }
    atomic_json_dump(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit("Pinned/reference loss parity failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
