#!/usr/bin/env python
"""Train one fixed-update, matched-objective Experiment 5 arm."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import (
    ARMS,
    DEFAULT_CONFIG,
    append_jsonl,
    assert_pinned_lejepa,
    atomic_json_dump,
    autocast_context,
    build_model_pair,
    build_sigreg,
    capture_rng_state,
    config_sha256,
    cpu_state_dict,
    git_metadata,
    learning_rate_for_update,
    load_config,
    objective,
    optimizer_parameter_groups,
    optimizer_to_device,
    package_versions,
    resolve_device,
    restore_rng_state,
    seed_streams,
    set_global_seed,
    set_optimizer_learning_rate,
    sha256_file,
    state_dict_sha256,
)
from experiment5.data import (
    DeterministicFakeCIFAR10,
    load_split,
    make_base_cifar10,
    make_training_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split-file", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/protocol/cifar10_split_indices.npz")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output-root", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/discovery")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--coefficient", type=float, default=None, help="Required calibrated coefficient for arm C.")
    parser.add_argument(
        "--coefficient-ledger",
        default="",
        help="Pilot ledger that authorized --coefficient; its config hash and candidate grid are verified.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", default="", help="Exact checkpoint to resume; no best-checkpoint selection is supported.")
    parser.add_argument("--allow-dirty-lejepa", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny fake-data code-path check, never a result.")
    return parser.parse_args()


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["dataset"].update({"train_size": 20, "validation_size": 10})
    result["augmentation"]["views"] = 2
    result["model"]["projector_dims"] = [512, 32, 32, 8]
    result["training"].update(
        {
            "batch_size": 10,
            "complete_batches_per_epoch": 2,
            "total_updates": 2,
            "equivalent_epochs": 1,
            "warmup_updates": 1,
            "metrics_every_updates": 1,
            "checkpoint_updates": [0, 1, 2],
            "full_diagnostic_updates": [0, 2],
        }
    )
    result["sigreg"].update({"num_slices": 8, "integration_points": 5})
    return result


def torch_save_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_payload(
    *,
    update: int,
    arm: str,
    seed: int,
    coefficient: float,
    backbone: torch.nn.Module,
    projector: torch.nn.Module,
    sigreg: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    streams: Mapping[str, int],
    split_hash: str | None,
    smoke_test: bool,
) -> dict[str, Any]:
    return {
        "experiment": config["experiment"],
        "protocol_version": config["protocol_version"],
        "arm": arm,
        "arm_name": ARMS[arm],
        "seed": int(seed),
        "update": int(update),
        "equivalent_epoch": float(update) / float(config["training"]["complete_batches_per_epoch"]),
        "coefficient": float(coefficient),
        "backbone_state": cpu_state_dict(backbone),
        "projector_state": cpu_state_dict(projector),
        "sigreg_state": None if sigreg is None else cpu_state_dict(sigreg),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "streams": dict(streams),
        "config": copy.deepcopy(dict(config)),
        "config_sha256": config_sha256(config),
        "split_file_sha256": split_hash,
        "smoke_test": bool(smoke_test),
        "created_at_unix": time.time(),
    }


def save_checkpoint(output_dir: Path, update: int, **kwargs: Any) -> Path:
    path = output_dir / "checkpoints" / f"update_{update:06d}.pth"
    torch_save_atomic(path, checkpoint_payload(update=update, **kwargs))
    return path


def gradients_are_finite(modules: list[torch.nn.Module]) -> bool:
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                return False
    return True


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.output_root) / f"arm_{args.arm}" / f"seed_{args.seed}"


def main() -> int:
    args = parse_args()
    base_config = load_config(args.config)
    config = smoke_config(base_config) if args.smoke_test else base_config
    arm = args.arm.upper()
    if arm == "C" and args.coefficient is None and config["training"]["vc_coefficient"] is None:
        raise ValueError("Arm C requires --coefficient from the pilot calibration ledger.")
    coefficient = (
        float(config["training"]["sigreg_coefficient"])
        if arm == "B" and args.coefficient is None
        else float(args.coefficient or config["training"]["vc_coefficient"] or 0.0)
    )
    coefficient_ledger = None
    coefficient_ledger_hash = None
    if args.coefficient_ledger:
        ledger_path = Path(args.coefficient_ledger)
        coefficient_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if coefficient_ledger.get("config_sha256") != config_sha256(config):
            raise ValueError("Coefficient ledger was calibrated under a different protocol config.")
        candidate_key = "sigreg_sensitivity_coefficients" if arm == "B" else "vc_candidate_coefficients"
        candidates = [float(value) for value in coefficient_ledger.get(candidate_key, [])]
        if arm in {"B", "C"} and not any(math.isclose(coefficient, value, rel_tol=1e-12) for value in candidates):
            raise ValueError(f"Coefficient {coefficient} is not in the ledger's {candidate_key}.")
        coefficient_ledger_hash = sha256_file(ledger_path)
    output_dir = output_dir_for(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"{manifest_path} already exists. Refusing to mix runs; pass --resume with an exact checkpoint."
        )
    device = resolve_device(args.device)
    lejepa_metadata = assert_pinned_lejepa(config, allow_dirty=args.allow_dirty_lejepa)
    stream_names = list(config["seeds"]["stream_names"])
    streams = seed_streams(args.seed, stream_names)
    set_global_seed(streams["initialization"])

    if args.smoke_test:
        base_dataset = DeterministicFakeCIFAR10(size=int(config["dataset"]["train_size"]), seed=args.seed)
        train_indices = np.arange(len(base_dataset), dtype=np.int64)
        split_hash = None
    else:
        split_path = Path(args.split_file)
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing split artifact {split_path}. Run scripts/prepare_experiment5.py first."
            )
        split = load_split(split_path)
        train_indices = split["train_indices"]
        base_dataset = make_base_cifar10(args.data_dir, train=True, download=False)
        split_hash = sha256_file(split_path)

    backbone, projector = build_model_pair(config, streams["initialization"], device)
    initial_backbone_sha = state_dict_sha256(cpu_state_dict(backbone))
    initial_projector_sha = state_dict_sha256(cpu_state_dict(projector))
    sigreg = build_sigreg(config, device, streams["sigreg_directions"]) if arm == "B" else None
    groups, group_names = optimizer_parameter_groups(
        [backbone, projector], float(config["training"]["matrix_weight_decay"])
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(training["learning_rate"]),
        betas=tuple(map(float, training["betas"])),
        eps=float(training["epsilon"]),
    )
    start_update = 0
    if args.resume:
        checkpoint_path = Path(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key, expected in [("arm", arm), ("seed", args.seed), ("config_sha256", config_sha256(config))]:
            if checkpoint.get(key) != expected:
                raise ValueError(f"Resume checkpoint {key}={checkpoint.get(key)!r}, expected {expected!r}")
        if checkpoint.get("split_file_sha256") != split_hash:
            raise ValueError("Resume checkpoint was created with a different split artifact.")
        backbone.load_state_dict(checkpoint["backbone_state"], strict=True)
        projector.load_state_dict(checkpoint["projector_state"], strict=True)
        if sigreg is not None:
            sigreg.load_state_dict(checkpoint["sigreg_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to_device(optimizer, device)
        restore_rng_state(checkpoint["rng_state"])
        start_update = int(checkpoint["update"])
        print(f"[resume] {checkpoint_path} at completed update {start_update}", flush=True)

    loader = make_training_loader(
        base_dataset,
        train_indices,
        config,
        streams,
        start_update=start_update,
        num_workers=0 if args.smoke_test else args.num_workers,
    )
    manifest = {
        "experiment": config["experiment"],
        "protocol_version": config["protocol_version"],
        "scientific_result": not args.smoke_test,
        "smoke_test": args.smoke_test,
        "arm": arm,
        "arm_name": ARMS[arm],
        "seed": args.seed,
        "coefficient": coefficient,
        "coefficient_ledger": None if not args.coefficient_ledger else str(Path(args.coefficient_ledger).resolve()),
        "coefficient_ledger_sha256": coefficient_ledger_hash,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": config_sha256(config),
        "base_protocol_config_sha256": config_sha256(base_config),
        "split_file": None if args.smoke_test else str(Path(args.split_file).resolve()),
        "split_file_sha256": split_hash,
        "rng_streams": streams,
        "initial_backbone_state_sha256": initial_backbone_sha,
        "initial_projector_state_sha256": initial_projector_sha,
        "optimizer_parameter_groups": group_names,
        "lejepa_source": lejepa_metadata,
        "repository_source": git_metadata(Path(__file__).resolve().parents[1]),
        "packages": package_versions(),
        "device": str(device),
        "resume_checkpoint": args.resume or None,
        "command": sys.argv,
        "config": config,
    }
    atomic_json_dump(manifest_path, manifest)
    shutil.copy2(args.config, output_dir / "source_config.json")

    checkpoint_kwargs = dict(
        arm=arm,
        seed=args.seed,
        coefficient=coefficient,
        backbone=backbone,
        projector=projector,
        sigreg=sigreg,
        optimizer=optimizer,
        config=config,
        streams=streams,
        split_hash=split_hash,
        smoke_test=args.smoke_test,
    )
    checkpoint_updates = {int(value) for value in training["checkpoint_updates"]}
    if start_update == 0:
        save_checkpoint(output_dir, 0, **checkpoint_kwargs)

    collapse_streak = 0
    completed_update = start_update
    wall_start = time.perf_counter()
    try:
        for batch_offset, (views, _labels, source_indices) in enumerate(loader, start=1):
            update = start_update + batch_offset
            if update > int(training["total_updates"]):
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            update_start = time.perf_counter()
            views = views.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate_for_update(update, config)
            set_optimizer_learning_rate(optimizer, lr)
            batch_size, num_views = int(views.shape[0]), int(views.shape[1])
            with autocast_context(device, enabled=True):
                features = backbone(views.flatten(0, 1))
                projected = projector(features)
            projections = projected.float().view(batch_size, num_views, -1).transpose(0, 1)
            loss, components = objective(
                arm, projections, config, sigreg=sigreg, coefficient=coefficient
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at update {update}: {loss.detach().cpu().item()}")
            loss.backward()
            if not gradients_are_finite([backbone, projector]):
                raise FloatingPointError(f"Nonfinite gradient at update {update}")
            optimizer.step()
            completed_update = update
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - update_start

            coordinate_variance = projections.detach().reshape(-1, projections.shape[-1]).var(dim=0, unbiased=True)
            mean_coordinate_variance = float(coordinate_variance.mean().cpu())
            collapse_streak = collapse_streak + 1 if mean_coordinate_variance < 1e-4 else 0
            should_log = (
                update <= 10
                or update % int(training["metrics_every_updates"]) == 0
                or update in checkpoint_updates
            )
            if should_log:
                row = {
                    "update": update,
                    "equivalent_epoch": update / float(training["complete_batches_per_epoch"]),
                    "arm": arm,
                    "seed": args.seed,
                    "loss": float(loss.detach().cpu()),
                    "alignment": float(components["alignment"].detach().cpu()),
                    "regularizer": float(components["regularizer"].detach().cpu()),
                    "sigreg": float(components["sigreg"].detach().cpu()),
                    "vc_variance": float(components["vc_variance"].detach().cpu()),
                    "vc_covariance": float(components["vc_covariance"].detach().cpu()),
                    "regularizer_coefficient": coefficient,
                    "learning_rate": lr,
                    "mean_coordinate_variance": mean_coordinate_variance,
                    "collapse_flag_three_checks": collapse_streak >= 3,
                    "batch_unique_source_images": int(torch.unique(source_indices).numel()),
                    "image_views": batch_size * num_views,
                    "milliseconds_per_update": elapsed * 1000.0,
                    "images_per_second": batch_size / max(elapsed, 1e-12),
                    "peak_device_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                    ),
                }
                append_jsonl(metrics_path, row)
                print(
                    f"[{arm} seed={args.seed}] update {update:06d}/{training['total_updates']} "
                    f"loss={row['loss']:.5f} align={row['alignment']:.5f} "
                    f"reg={row['regularizer']:.5f} lr={lr:.3e}",
                    flush=True,
                )
            if update in checkpoint_updates:
                path = save_checkpoint(output_dir, update, **checkpoint_kwargs)
                print(f"[checkpoint] {path}", flush=True)
    except (FloatingPointError, RuntimeError) as exc:
        atomic_json_dump(
            output_dir / "failure.json",
            {
                "classification": "nonfinite_arithmetic" if isinstance(exc, FloatingPointError) else "runtime_failure",
                "completed_update": completed_update,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_at_unix": time.time(),
            },
        )
        raise

    final_update = int(training["total_updates"])
    if completed_update != final_update:
        raise RuntimeError(f"Training ended at update {completed_update}, expected {final_update}")
    atomic_json_dump(
        output_dir / "completed.json",
        {
            "completed_update": completed_update,
            "final_checkpoint": str(output_dir / "checkpoints" / f"update_{completed_update:06d}.pth"),
            "total_wall_seconds": time.perf_counter() - wall_start,
            "scientific_result": not args.smoke_test,
        },
    )
    print(f"[done] completed {completed_update} updates in {time.perf_counter() - wall_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
