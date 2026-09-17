#!/usr/bin/env python
"""Create and verify the immutable Experiment 5 CIFAR10 split artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import DEFAULT_CONFIG, atomic_json_dump, config_sha256, load_config, sha256_file
from experiment5.data import create_split_artifact, make_base_cifar10, validate_split_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output-dir", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/protocol")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "cifar10_split_indices.npz"
    manifest_path = output_dir / "split_manifest.json"

    dataset = make_base_cifar10(args.data_dir, train=True, download=args.download)
    labels = dataset.targets
    if split_path.is_file():
        with np.load(split_path, allow_pickle=False) as loaded:
            artifact = {key: loaded[key] for key in loaded.files}
        if not args.verify_only:
            expected = create_split_artifact(labels, config)
            if set(expected) != set(artifact) or any(
                not np.array_equal(expected[key], artifact[key]) for key in expected
            ):
                raise RuntimeError(
                    "Existing split artifact differs from the configured deterministic split; refusing to overwrite it."
                )
            print(f"[split] Reusing identical immutable artifact: {split_path}")
    elif args.verify_only:
        raise FileNotFoundError(split_path)
    else:
        artifact = create_split_artifact(labels, config)
        temporary = split_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **artifact)
        temporary.replace(split_path)

    validation = validate_split_artifact(artifact, labels, config)
    manifest = {
        "experiment": config["experiment"],
        "protocol_version": config["protocol_version"],
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": config_sha256(config),
        "split_path": str(split_path.resolve()),
        "split_file_sha256": sha256_file(split_path),
        "split_seed": int(config["dataset"]["split_seed"]),
        "official_cifar10_train_size": len(dataset),
        **validation,
    }
    if manifest_path.exists() and args.verify_only:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("split_file_sha256") != manifest["split_file_sha256"]:
            raise RuntimeError("Split manifest hash does not match the split artifact.")
    atomic_json_dump(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
