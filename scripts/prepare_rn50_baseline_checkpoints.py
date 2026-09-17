#!/usr/bin/env python
"""Prepare RN50 baseline checkpoints without running filter analysis.

The saved files follow the local checkpoint wrapper expected by the analysis
scripts: a dictionary with a ``backbone_state`` entry and lightweight metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from torchvision.models import resnet50


HF_REPO_ID = "timm/resnet50.a1_in1k"
RANDOM_OUT = "rn50_random_init_timm_default_backbone.pth"
IMAGENET_OUT = "rn50_imagenet_a1_in1k_pretrained_backbone.pth"
MANIFEST_OUT = "rn50_baseline_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RN50 baseline checkpoint wrappers.")
    parser.add_argument("--model-dir", default="models", help="Directory where checkpoints are written.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for random RN50 initialization.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing baseline checkpoint files.",
    )
    return parser.parse_args()


def strip_classifier(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in state_dict.items()
        if not key.startswith("fc.")
    }


def save_checkpoint(path: Path, checkpoint: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        print(f"[skip] {path} already exists; use --force to overwrite.", flush=True)
        return
    torch.save(checkpoint, path)
    print(f"[write] {path}", flush=True)


def create_random_baseline(model_dir: Path, seed: int, force: bool) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = resnet50(weights=None, zero_init_residual=True)
    backbone_state = strip_classifier(model.state_dict())
    checkpoint = {
        "backbone": "rn50_random_init_timm_default",
        "dataset": "none",
        "backbone_state": backbone_state,
        "probe_state": None,
        "best_bacc": None,
        "epochs": 0,
        "source": "torchvision.models.resnet50(weights=None, zero_init_residual=True)",
        "initialization": {
            "conv": "Kaiming normal, fan_out, relu",
            "residual_last_bn": "zero_init_residual=True, matching timm zero_init_last=True intent",
            "bn_gamma_beta": "PyTorch defaults: gamma=1, beta=0 except residual branch final BN gamma=0",
            "seed": seed,
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_checkpoint(model_dir / RANDOM_OUT, checkpoint, force)
    return {
        "file": RANDOM_OUT,
        "backbone": checkpoint["backbone"],
        "status": "created" if force or not (model_dir / RANDOM_OUT).exists() else "exists_or_created",
        "state_tensors": len(backbone_state),
        "note": "Classifier fc.* removed to match local backbone_state format.",
    }


def load_hf_state_dict(repo_id: str) -> tuple[dict[str, torch.Tensor], str]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        return load_safetensors(path, device="cpu"), path
    except Exception as safetensor_error:
        print(f"[warn] safetensors load failed: {safetensor_error}", flush=True)
        path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint type from {path}: {type(state)}")
        return state, path


def create_imagenet_baseline(model_dir: Path, force: bool) -> dict[str, Any]:
    state, downloaded_path = load_hf_state_dict(HF_REPO_ID)
    backbone_state = strip_classifier(state)
    checkpoint = {
        "backbone": "rn50_imagenet_a1_in1k_pretrained",
        "dataset": "imagenet1k",
        "backbone_state": backbone_state,
        "probe_state": None,
        "best_bacc": None,
        "epochs": 0,
        "source": HF_REPO_ID,
        "source_file": downloaded_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_checkpoint(model_dir / IMAGENET_OUT, checkpoint, force)
    return {
        "file": IMAGENET_OUT,
        "backbone": checkpoint["backbone"],
        "status": "created" if force or not (model_dir / IMAGENET_OUT).exists() else "exists_or_created",
        "state_tensors": len(backbone_state),
        "source": HF_REPO_ID,
        "source_file": downloaded_path,
        "note": "Classifier fc.* removed to match local backbone_state format.",
    }


def validate_backbone_state(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("backbone_state")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a backbone_state dictionary")
    conv2_3x3 = [
        key
        for key, value in state.items()
        if key.startswith("layer")
        and key.endswith(".conv2.weight")
        and hasattr(value, "shape")
        and tuple(value.shape[-2:]) == (3, 3)
    ]
    return {
        "file": path.name,
        "backbone": checkpoint.get("backbone"),
        "state_tensors": len(state),
        "has_fc": "fc.weight" in state or "fc.bias" in state,
        "rn50_bottleneck_3x3_conv2_tensors": len(conv2_3x3),
        "conv1_shape": list(state["conv1.weight"].shape),
        "first_conv2_shape": list(state["layer1.0.conv2.weight"].shape),
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] Model dir: {model_dir}", flush=True)
    print("[baseline] Creating random-initialized RN50 wrapper", flush=True)
    random_info = create_random_baseline(model_dir, args.seed, args.force)

    print(f"[baseline] Downloading ImageNet-pretrained RN50 from {HF_REPO_ID}", flush=True)
    imagenet_info = create_imagenet_baseline(model_dir, args.force)

    validations = [
        validate_backbone_state(model_dir / RANDOM_OUT),
        validate_backbone_state(model_dir / IMAGENET_OUT),
    ]
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baselines": [random_info, imagenet_info],
        "validation": validations,
        "lejepa_note": (
            "No separate LeJEPA pre-SSL initialization or pre-target checkpoint was identified "
            "from the provided information. Existing models/rn50_lejepa_ssl_backbone.pth is "
            "the available LeJEPA SSL backbone checkpoint."
        ),
    }
    manifest_path = model_dir / MANIFEST_OUT
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[write] {manifest_path}", flush=True)
    print("[done] Baseline checkpoint preparation complete; no analysis scripts were run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
