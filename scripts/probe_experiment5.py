#!/usr/bin/env python
"""Run the fixed frozen-backbone 1%/10%/100% Experiment 5 linear probes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    build_backbone,
    config_sha256,
    derive_seed,
    load_config,
    resolve_device,
    sha256_file,
)
from experiment5.data import load_split, make_base_cifar10, make_eval_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split-file", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/protocol/cifar10_split_indices.npz")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--extract-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def extract(backbone, dataset, batch_size: int, num_workers: int, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features, labels, indices = [], [], []
    backbone.eval()
    for images, batch_labels, source_indices in loader:
        features.append(backbone(images.to(device, non_blocking=True)).float().cpu())
        labels.append(batch_labels.long().cpu())
        indices.append(source_indices.long().cpu())
    return torch.cat(features), torch.cat(labels), torch.cat(indices)


@torch.inference_mode()
def evaluate(head, features: torch.Tensor, labels: torch.Tensor, device: torch.device, batch_size: int):
    head.eval()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    correct = 0
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for batch_features, batch_labels in loader:
        logits = head(batch_features.to(device))
        target = batch_labels.to(device)
        total_loss += float(criterion(logits, target).cpu())
        correct += int((logits.argmax(dim=1) == target).sum().cpu())
        total += len(target)
    return {"loss": total_loss / total, "accuracy": correct / total}


def train_head(
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    test_features,
    test_labels,
    config,
    seed: int,
    device: torch.device,
):
    probe = config["probe"]
    generator = torch.Generator().manual_seed(seed % (2**63 - 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed % (2**63 - 1))
        head = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(probe["learning_rate"]), weight_decay=float(probe["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(probe["epochs"]))
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=int(probe["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    history = []
    for epoch in range(1, int(probe["epochs"]) + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_labels)
            total_correct += int((logits.argmax(dim=1) == batch_labels).sum().detach().cpu())
            total += len(batch_labels)
        validation = evaluate(
            head, validation_features, validation_labels, device, int(probe["batch_size"])
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total,
                "train_accuracy": total_correct / total,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step()
    final_validation = evaluate(head, validation_features, validation_labels, device, int(probe["batch_size"]))
    final_test = evaluate(head, test_features, test_labels, device, int(probe["batch_size"]))
    return history, final_validation, final_test


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != config["experiment"] or checkpoint.get("config_sha256") != config_sha256(config):
        raise ValueError("Checkpoint and probe protocol do not match.")
    split = load_split(args.split_file)
    train_base = make_base_cifar10(args.data_dir, train=True, download=False)
    test_base = make_base_cifar10(args.data_dir, train=False, download=False)
    train_dataset = make_eval_subset(train_base, split["train_indices"])
    validation_dataset = make_eval_subset(train_base, split["validation_indices"])
    test_dataset = make_eval_subset(test_base, np.arange(len(test_base), dtype=np.int64))
    backbone = build_backbone().to(device)
    backbone.load_state_dict(checkpoint["backbone_state"], strict=True)
    train_features, train_labels, train_ids = extract(
        backbone, train_dataset, args.extract_batch_size, args.num_workers, device
    )
    validation_features, validation_labels, _ = extract(
        backbone, validation_dataset, args.extract_batch_size, args.num_workers, device
    )
    test_features, test_labels, _ = extract(
        backbone, test_dataset, args.extract_batch_size, args.num_workers, device
    )
    id_to_position = {int(source_id): position for position, source_id in enumerate(train_ids.tolist())}
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent.parent / "probes" / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_history = []
    results = []
    for draw_index, _draw_seed in enumerate(config["probe"]["label_draw_seeds"]):
        for fraction in config["probe"]["label_fractions"]:
            fraction_key = f"{float(fraction):.2f}".replace(".", "p")
            source_ids = split[f"probe_draw{draw_index}_{fraction_key}"]
            positions = torch.tensor([id_to_position[int(source_id)] for source_id in source_ids], dtype=torch.long)
            head_seed = derive_seed(
                int(checkpoint["seed"]), f"linear_probe:draw={draw_index}:fraction={float(fraction)}"
            )
            history, validation, test = train_head(
                train_features[positions],
                train_labels[positions],
                validation_features,
                validation_labels,
                test_features,
                test_labels,
                config,
                head_seed,
                device,
            )
            for row in history:
                all_history.append(
                    {
                        "arm": checkpoint["arm"],
                        "pretraining_seed": checkpoint["seed"],
                        "update": checkpoint["update"],
                        "label_draw": draw_index,
                        "label_fraction": float(fraction),
                        **row,
                    }
                )
            results.append(
                {
                    "arm": checkpoint["arm"],
                    "pretraining_seed": checkpoint["seed"],
                    "update": checkpoint["update"],
                    "label_draw": draw_index,
                    "label_fraction": float(fraction),
                    "labeled_images": len(source_ids),
                    "head_seed": head_seed,
                    "final_validation_loss": validation["loss"],
                    "final_validation_accuracy": validation["accuracy"],
                    "final_test_loss": test["loss"],
                    "final_test_accuracy": test["accuracy"],
                    "test_used_for_selection": False,
                }
            )
    write_csv(output_dir / "probe_history.csv", all_history)
    write_csv(output_dir / "probe_results.csv", results)
    primary = [row for row in results if row["label_fraction"] == float(config["probe"]["primary_label_fraction"])]
    summary = {
        "experiment": checkpoint["experiment"],
        "arm": checkpoint["arm"],
        "pretraining_seed": checkpoint["seed"],
        "update": checkpoint["update"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_file_sha256": sha256_file(args.split_file),
        "protocol": config["probe"],
        "primary_10_percent_mean_validation_accuracy": float(
            np.mean([row["final_validation_accuracy"] for row in primary])
        ),
        "primary_10_percent_mean_test_accuracy": float(
            np.mean([row["final_test_accuracy"] for row in primary])
        ),
        "test_use": "Reported only after the fixed final epoch; never used for selection or coefficient tuning.",
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
