#!/usr/bin/env python
"""Run the two existing-checkpoint falsification controls from the Experiment 5 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment5.core import DEFAULT_CONFIG, atomic_json_dump, build_backbone, load_config, resolve_device, sha256_file
from experiment5.data import load_split, make_base_cifar10, make_eval_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split-file", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/protocol/cifar10_split_indices.npz")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--shuffle-layer", default="layer4.1.conv2.weight")
    parser.add_argument("--permutation-block", default="layer4.1")
    parser.add_argument("--images", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--output", default="outputs/(5thEXP)rn18_cifar10_sigreg_structure/existing_checkpoint_falsifications/result.json")
    return parser.parse_args()


def load_backbone_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("backbone_state")
    if isinstance(state, dict):
        return {key: value.detach().cpu().clone() for key, value in state.items()}
    state = checkpoint.get("model_state")
    if isinstance(state, dict):
        return {
            key.removeprefix("module."): value.detach().cpu().clone()
            for key, value in state.items()
            if not key.removeprefix("module.").startswith("fc.")
        }
    raise ValueError("Checkpoint contains neither backbone_state nor model_state.")


def kernel_multiset_sha256(weight: torch.Tensor) -> str:
    kernels = weight.detach().cpu().float().reshape(-1, weight.shape[-2] * weight.shape[-1]).numpy()
    order = np.lexsort(tuple(kernels[:, column] for column in reversed(range(kernels.shape[1]))))
    sorted_kernels = np.ascontiguousarray(kernels[order])
    digest = hashlib.sha256()
    digest.update(np.asarray(sorted_kernels.shape, dtype=np.int64).tobytes())
    digest.update(sorted_kernels.tobytes())
    return digest.hexdigest()


def shuffle_channel_pair_kernels(
    state: Mapping[str, torch.Tensor], layer: str, seed: int
) -> dict[str, torch.Tensor]:
    if layer not in state:
        raise KeyError(layer)
    output = {key: value.clone() for key, value in state.items()}
    weight = state[layer]
    if weight.ndim != 4 or tuple(weight.shape[-2:]) != (3, 3):
        raise ValueError(f"{layer} is not a 3x3 convolution tensor.")
    generator = torch.Generator().manual_seed(seed)
    flat = weight.reshape(-1, 3, 3)
    permutation = torch.randperm(len(flat), generator=generator)
    output[layer] = flat[permutation].reshape_as(weight).clone()
    return output


def coordinated_intermediate_channel_permutation(
    state: Mapping[str, torch.Tensor], block: str, seed: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    required = [
        f"{block}.conv1.weight",
        f"{block}.bn1.weight",
        f"{block}.bn1.bias",
        f"{block}.bn1.running_mean",
        f"{block}.bn1.running_var",
        f"{block}.conv2.weight",
    ]
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"Missing block tensors: {missing}")
    output = {key: value.clone() for key, value in state.items()}
    channels = state[f"{block}.conv1.weight"].shape[0]
    if state[f"{block}.conv2.weight"].shape[1] != channels:
        raise ValueError("Block conv1 output and conv2 input channel counts do not match.")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(channels, generator=generator)
    output[f"{block}.conv1.weight"] = state[f"{block}.conv1.weight"][permutation].clone()
    for suffix in ["weight", "bias", "running_mean", "running_var"]:
        key = f"{block}.bn1.{suffix}"
        output[key] = state[key][permutation].clone()
    output[f"{block}.conv2.weight"] = state[f"{block}.conv2.weight"][:, permutation].clone()
    return output, permutation


def build_model(state: Mapping[str, torch.Tensor], device: torch.device):
    model = build_backbone()
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("fc.")]
    if missing or unexpected:
        raise ValueError(f"Backbone state mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_features(model, loader, device):
    features = []
    indices = []
    for images, _labels, source_indices in loader:
        features.append(model(images.to(device, non_blocking=True)).float().cpu())
        indices.append(source_indices.cpu())
    return torch.cat(features), torch.cat(indices)


def linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double() - first.double().mean(dim=0, keepdim=True)
    second = second.double() - second.double().mean(dim=0, keepdim=True)
    cross = first.T @ second
    first_gram = first.T @ first
    second_gram = second.T @ second
    return float(cross.square().sum() / torch.sqrt(first_gram.square().sum() * second_gram.square().sum()))


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = candidate - reference
    cosine = F.cosine_similarity(reference, candidate, dim=1)
    return {
        "maximum_absolute_feature_difference": float(delta.abs().max()),
        "relative_feature_l2_difference": float(delta.norm() / reference.norm().clamp_min(1e-12)),
        "mean_per_image_cosine": float(cosine.mean()),
        "linear_cka": linear_cka(reference, candidate),
        "reference_mean_coordinate_variance": float(reference.var(dim=0, unbiased=True).mean()),
        "candidate_mean_coordinate_variance": float(candidate.var(dim=0, unbiased=True).mean()),
    }


def singular_values(weight: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(weight.double().reshape(weight.shape[0], -1))


def main() -> int:
    args = parse_args()
    load_config(args.config)  # Validate the Experiment 5 protocol used for image IDs.
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference_state = load_backbone_state(checkpoint)
    shuffled_state = shuffle_channel_pair_kernels(reference_state, args.shuffle_layer, args.seed)
    permuted_state, permutation = coordinated_intermediate_channel_permutation(
        reference_state, args.permutation_block, args.seed + 1
    )

    before_hash = kernel_multiset_sha256(reference_state[args.shuffle_layer])
    shuffled_hash = kernel_multiset_sha256(shuffled_state[args.shuffle_layer])
    if before_hash != shuffled_hash:
        raise RuntimeError("Kernel shuffle did not preserve the exact marginal kernel multiset.")
    for layer in [f"{args.permutation_block}.conv1.weight", f"{args.permutation_block}.conv2.weight"]:
        if kernel_multiset_sha256(reference_state[layer]) != kernel_multiset_sha256(permuted_state[layer]):
            raise RuntimeError(f"Coordinated permutation changed the marginal kernel multiset for {layer}")

    split = load_split(args.split_file)
    image_ids = split["diagnostic_indices"][: args.images]
    dataset = make_base_cifar10(args.data_dir, train=True, download=False)
    eval_dataset = make_eval_subset(dataset, image_ids)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    reference_features, reference_ids = extract_features(build_model(reference_state, device), loader, device)
    shuffled_features, shuffled_ids = extract_features(build_model(shuffled_state, device), loader, device)
    permuted_features, permuted_ids = extract_features(build_model(permuted_state, device), loader, device)
    if not torch.equal(reference_ids, shuffled_ids) or not torch.equal(reference_ids, permuted_ids):
        raise RuntimeError("Feature comparisons used different image orders.")

    spectrum_differences = {}
    for layer in [f"{args.permutation_block}.conv1.weight", f"{args.permutation_block}.conv2.weight"]:
        reference_spectrum = singular_values(reference_state[layer])
        permuted_spectrum = singular_values(permuted_state[layer])
        spectrum_differences[layer] = float((reference_spectrum - permuted_spectrum).abs().max())
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "image_count": len(image_ids),
        "image_ids_sha256": hashlib.sha256(np.ascontiguousarray(image_ids).tobytes()).hexdigest(),
        "kernel_shuffle": {
            "layer": args.shuffle_layer,
            "kernel_multiset_sha256_before": before_hash,
            "kernel_multiset_sha256_after": shuffled_hash,
            "marginal_descriptors_preserved_exactly": before_hash == shuffled_hash,
            "feature_effect": comparison(reference_features, shuffled_features),
            "interpretation": (
                "The intervention preserves the layer's complete 3x3 channel-pair kernel population but changes connectivity. "
                "Behavioral damage would show that marginal descriptors do not uniquely characterize function."
            ),
        },
        "coordinated_permutation": {
            "block": args.permutation_block,
            "permuted_channels": int(len(permutation)),
            "marginal_kernel_populations_preserved_exactly": True,
            "maximum_singular_value_differences": spectrum_differences,
            "feature_invariance": comparison(reference_features, permuted_features),
            "numerical_invariance_tolerance": 1e-4,
            "outputs_numerically_invariant": float((reference_features - permuted_features).abs().max()) < 1e-4,
            "interpretation": (
                "Conv1 output rows, BN1 channels, and Conv2 input columns are permuted together inside one residual block. "
                "This is a valid function-preserving control, not an arbitrary ResNet channel shuffle."
            ),
        },
    }
    if not result["coordinated_permutation"]["outputs_numerically_invariant"]:
        raise RuntimeError("The coordinated channel permutation failed numerical output invariance.")
    atomic_json_dump(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
