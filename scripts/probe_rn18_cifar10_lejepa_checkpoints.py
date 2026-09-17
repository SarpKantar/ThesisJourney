#!/usr/bin/env python
"""Train stronger frozen-backbone linear probes for saved RN18/CIFAR10 LeJEPA checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("Use the `minitron` conda environment or the provided Slurm job.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rn18_cifar10_common import ensure_dir
from train_rn18_cifar10_experiment import (
    append_csv_row,
    build_lejepa_linear_inference_model,
    configure_cuda_backend,
    evaluate_matched_pair,
    load_existing_imagenet_rows,
    make_datasets,
    make_loader,
    match_checkpoints,
    resolve_device,
    set_seed,
    train_all_linear_heads,
)


def parse_epochs(text: str) -> set[int] | None:
    if not text.strip():
        return None
    out = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc linear probe sweep for true-LeJEPA RN18/CIFAR10 checkpoints.")
    parser.add_argument("--source-experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--fake-train-samples", type=int, default=64)
    parser.add_argument("--fake-test-samples", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--imagenet-source-dir", default="outputs/rn18_cifar10_experiment_cudnn")
    parser.add_argument(
        "--checkpoint-epochs",
        default="80,90,100,110,120,130,140,150,180,190,220,250",
        help="Comma-separated SSL epochs to probe. Empty means all checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--linear-epochs", type=int, default=100)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    parser.add_argument("--linear-weight-decay", type=float, default=1e-6)
    parser.add_argument("--linear-optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument(
        "--linear-head-type",
        choices=["linear", "layernorm_linear", "batchnorm_linear"],
        default="layernorm_linear",
    )
    return parser.parse_args()


def make_probe_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=args.data_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fake_data=args.fake_data,
        fake_train_samples=args.fake_train_samples,
        fake_test_samples=args.fake_test_samples,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        ssl_views=2,
        ssl_crop_min=0.45,
        ssl_crop_max=1.0,
        ssl_jitter_brightness=0.35,
        ssl_jitter_contrast=0.35,
        ssl_jitter_saturation=0.35,
        ssl_jitter_hue=0.10,
        ssl_grayscale_p=0.20,
        ssl_blur_p=0.25,
        ssl_solarize_p=0.0,
        linear_epochs=args.linear_epochs,
        linear_lr=args.linear_lr,
        linear_weight_decay=args.linear_weight_decay,
        linear_optimizer=args.linear_optimizer,
        linear_head_type=args.linear_head_type,
        momentum=0.9,
        weight_decay=5e-4,
        amp=args.amp,
    )


def checkpoint_epoch(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit("epoch", 1)[1])


def write_probe_summary(output_dir: Path, args: argparse.Namespace, selected: dict[str, Any], linear_rows: list[dict[str, Any]]) -> None:
    best = max(linear_rows, key=lambda row: float(row["final_test_accuracy"]))
    lines = []
    lines.append("# RN18/CIFAR10 LeJEPA Linear Probe Sweep")
    lines.append("")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- Source experiment: `{args.source_experiment_dir}`")
    lines.append(f"- Head: `{args.linear_head_type}`")
    lines.append(f"- Optimizer: `{args.linear_optimizer}`")
    lines.append(f"- LR: `{args.linear_lr}`")
    lines.append(f"- Weight decay: `{args.linear_weight_decay}`")
    lines.append(f"- Linear epochs: `{args.linear_epochs}`")
    lines.append("")
    lines.append(f"- Best LeJEPA final accuracy: `{float(best['final_test_accuracy']):.4f}` at SSL epoch `{int(best['ssl_epoch'])}`")
    lines.append(
        "- Selected closest pair: "
        f"ImageNet epoch `{selected['imagenet_epoch']}` acc `{float(selected['imagenet_accuracy']):.4f}` vs "
        f"LeJEPA SSL epoch `{selected['lejepa_ssl_epoch']}` acc `{float(selected['lejepa_accuracy']):.4f}` "
        f"(delta `{float(selected['abs_accuracy_delta']):.4f}`)."
    )
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    source_dir = Path(args.source_experiment_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = source_dir / "checkpoints"
    linear_dir = output_dir / "linear_heads"
    ensure_dir(output_dir)
    ensure_dir(linear_dir)

    device = resolve_device(args.device)
    configure_cuda_backend(args, device)
    print(f"[start] source={source_dir}", flush=True)
    print(f"[start] output={output_dir}", flush=True)
    print(f"[start] device={device}", flush=True)
    print(
        f"[probe] head={args.linear_head_type} opt={args.linear_optimizer} "
        f"lr={args.linear_lr} wd={args.linear_weight_decay} epochs={args.linear_epochs}",
        flush=True,
    )

    probe_args = make_probe_namespace(args)
    train_dataset, _, test_dataset = make_datasets(probe_args)
    train_loader = make_loader(train_dataset, probe_args, shuffle=True)
    test_loader = make_loader(test_dataset, probe_args, shuffle=False)

    imagenet_rows = load_existing_imagenet_rows(Path(args.imagenet_source_dir), output_dir / "imagenet_finetune_metrics.csv")
    shutil.copyfile(source_dir / "lejepa_ssl_metrics.csv", output_dir / "lejepa_ssl_metrics.csv")
    if (source_dir / "lejepa_quick_probe_metrics.csv").exists():
        shutil.copyfile(source_dir / "lejepa_quick_probe_metrics.csv", output_dir / "lejepa_quick_probe_metrics.csv")

    keep_epochs = parse_epochs(args.checkpoint_epochs)
    ssl_checkpoints = sorted(checkpoint_dir.glob("lejepa_ssl_epoch*.pth"), key=checkpoint_epoch)
    if keep_epochs is not None:
        ssl_checkpoints = [path for path in ssl_checkpoints if checkpoint_epoch(path) in keep_epochs]
    if not ssl_checkpoints:
        raise RuntimeError(f"No SSL checkpoints selected from {checkpoint_dir}")

    metrics_path = output_dir / "lejepa_linear_metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    linear_rows = train_all_linear_heads(probe_args, ssl_checkpoints, train_loader, test_loader, device, linear_dir, metrics_path)
    selected = match_checkpoints(imagenet_rows, linear_rows, output_dir)
    evaluate_matched_pair(selected, test_loader, device, output_dir)

    config = vars(args).copy()
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["selected_checkpoints"] = [str(path) for path in ssl_checkpoints]
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_probe_summary(output_dir, args, selected, linear_rows)
    print(
        "[done] best LeJEPA acc="
        f"{max(float(row['final_test_accuracy']) for row in linear_rows):.4f}; "
        f"selected delta={float(selected['abs_accuracy_delta']):.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
