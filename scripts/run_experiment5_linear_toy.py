#!/usr/bin/env python
"""Run one prespecified analytic-control cell for Experiment 5A."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    seed_streams,
    set_global_seed,
    soft_orthogonality,
    vicreg_variance_covariance,
)


METHODS = ["sigreg", "vc", "so"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--condition-number", required=True, type=int, choices=[1, 10, 100])
    parser.add_argument("--seed", required=True, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/linear_toy")
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-dirty-lejepa", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


class TwoLayerLinear(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.first = nn.Linear(dimension, dimension, bias=False)
        self.second = nn.Linear(dimension, dimension, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value))

    def composed(self) -> torch.Tensor:
        return self.second.weight @ self.first.weight


def mixing_matrix(dimension: int, condition_number: float, generator: torch.Generator) -> torch.Tensor:
    left, _ = torch.linalg.qr(torch.randn((dimension, dimension), generator=generator))
    right, _ = torch.linalg.qr(torch.randn((dimension, dimension), generator=generator))
    singular_values = torch.logspace(0.0, math.log10(condition_number), dimension)
    return left @ torch.diag(singular_values) @ right.T


def make_data(
    count: int,
    dimension: int,
    condition_number: float,
    seed: int,
    device: torch.device,
    mix: torch.Tensor | None = None,
):
    generator = torch.Generator().manual_seed(seed)
    if mix is None:
        mix = mixing_matrix(dimension, condition_number, generator)
    else:
        mix = mix.detach().cpu()
    latent = torch.randn((count, dimension), generator=generator)
    shared = 0.95 * latent
    noise_scale = math.sqrt(1.0 - 0.95**2)
    view1 = (shared + noise_scale * torch.randn((count, dimension), generator=generator)) @ mix.T
    view2 = (shared + noise_scale * torch.randn((count, dimension), generator=generator)) @ mix.T
    return torch.stack([view1, view2], dim=0).to(device), mix.to(device)


def covariance_metrics(outputs: torch.Tensor) -> dict[str, float | list[float]]:
    centered = outputs - outputs.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0).flip(0)
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    return {
        "covariance_frobenius_error": float((covariance - identity).norm().cpu()),
        "mean_coordinate_variance": float(torch.diag(covariance).mean().cpu()),
        "covariance_eigenvalues_descending": eigenvalues.cpu().tolist(),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    source = assert_pinned_lejepa(config, allow_dirty=args.allow_dirty_lejepa)
    device = resolve_device(args.device)
    dimension = 16
    train_count = 10_000
    heldout_count = 2_000
    updates = min(args.updates, 3) if args.smoke_test else args.updates
    streams = seed_streams(args.seed, config["seeds"]["stream_names"])
    set_global_seed(streams["initialization"])
    train_views, mix = make_data(
        train_count, dimension, args.condition_number, streams["data_order"] % (2**31), device
    )
    heldout_views, _ = make_data(
        heldout_count,
        dimension,
        args.condition_number,
        (streams["data_order"] + 1) % (2**31),
        device,
        mix=mix,
    )
    model = TwoLayerLinear(dimension).to(device)
    sigreg = build_sigreg(config, device, streams["sigreg_directions"]) if args.method == "sigreg" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    batch_generator = torch.Generator(device="cpu").manual_seed(streams["data_order"] % (2**63 - 1))
    rows = []
    for update in range(1, updates + 1):
        indices = torch.randint(0, train_count, (args.batch_size,), generator=batch_generator).to(device)
        batch = train_views[:, indices]
        outputs = torch.stack([model(batch[view]) for view in range(2)], dim=0)
        alignment = all_view_alignment(outputs)
        if args.method == "sigreg":
            regularizer = torch.stack([sigreg(outputs[view].float()) for view in range(2)]).mean()
            coefficient = 0.02
        elif args.method == "vc":
            regularizer, _ = vicreg_variance_covariance(outputs)
            coefficient = 0.02
        else:
            regularizer = 0.5 * (
                soft_orthogonality(model.first.weight) + soft_orthogonality(model.second.weight)
            )
            coefficient = 0.02
        loss = 0.98 * alignment + coefficient * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update in {1, 10, 100, 1000, updates}:
            rows.append(
                {
                    "update": update,
                    "loss": float(loss.detach().cpu()),
                    "alignment": float(alignment.detach().cpu()),
                    "regularizer": float(regularizer.detach().cpu()),
                }
            )

    with torch.no_grad():
        heldout_outputs = torch.stack([model(heldout_views[view]) for view in range(2)], dim=0)
        combined = heldout_outputs.transpose(0, 1).reshape(-1, dimension)
        composed = model.composed()
        singular_values = torch.linalg.svdvals(composed.double())
        effective_map = composed @ mix
        effective_singular_values = torch.linalg.svdvals(effective_map.double())
    output_dir = (
        Path(args.output_root)
        / f"condition_{args.condition_number:03d}"
        / args.method
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": config["experiment"],
        "control": "two_layer_linear_correlated_gaussian_views",
        "scientific_result": not args.smoke_test,
        "method": args.method,
        "condition_number": args.condition_number,
        "seed": args.seed,
        "dimension": dimension,
        "training_pairs": train_count,
        "heldout_pairs": heldout_count,
        "updates": updates,
        "config_sha256": config_sha256(config),
        "lejepa_source": source,
        "output_metrics": covariance_metrics(combined),
        "composed_map_singular_values": singular_values.cpu().tolist(),
        "composed_map_times_mixing_singular_values": effective_singular_values.cpu().tolist(),
        "training_metrics": rows,
        "scope_warning": "This analytic control tests whitening assumptions; it is not evidence of image representation quality.",
    }
    atomic_json_dump(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
