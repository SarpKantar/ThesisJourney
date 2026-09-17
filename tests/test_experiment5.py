from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment5.core import (  # noqa: E402
    DEFAULT_CONFIG,
    all_view_alignment,
    build_backbone,
    derive_seed,
    learning_rate_for_update,
    load_config,
    objective,
    optimizer_parameter_groups,
    seed_streams,
    soft_orthogonality,
    vicreg_variance_covariance,
)
from experiment5.data import (  # noqa: E402
    DeterministicFakeCIFAR10,
    DeterministicMultiViewDataset,
    FixedUpdateBatchSampler,
    make_ssl_transform,
    stratified_split,
)
from analyze_experiment5_weights import matrix_kind, spectral_metrics  # noqa: E402
from diagnose_experiment5_representations import lidar_metrics, representation_metrics  # noqa: E402
from check_experiment5_weight_falsifications import (  # noqa: E402
    coordinated_intermediate_channel_permutation,
    kernel_multiset_sha256,
    shuffle_channel_pair_kernels,
)


def test_main_config_is_internally_consistent():
    config = load_config(DEFAULT_CONFIG)
    assert config["training"]["total_updates"] == 120 * 175
    assert config["training"]["batch_size"] == 256
    assert config["augmentation"]["views"] == 4
    assert config["model"]["projector_dims"] == [512, 2048, 2048, 64]


def test_rng_streams_are_stable_and_distinct():
    names = ["initialization", "data_order", "augmentation", "sigreg_directions"]
    first = seed_streams(13, names)
    second = seed_streams(13, names)
    assert first == second
    assert len(set(first.values())) == len(names)
    assert first != seed_streams(17, names)
    assert derive_seed(13, "augmentation") == first["augmentation"]


def test_stratified_split_is_exact_and_disjoint():
    labels = np.repeat(np.arange(10), 100)
    train, validation = stratified_split(labels, validation_size=200, seed=20260916)
    assert len(train) == 800
    assert len(validation) == 200
    assert not np.intersect1d(train, validation).size
    assert np.bincount(labels[validation]).tolist() == [20] * 10


def test_fixed_update_sampler_replays_and_resumes_exactly():
    kwargs = dict(source_indices=np.arange(23), batch_size=5, total_updates=9, data_order_seed=44)
    full = list(FixedUpdateBatchSampler(start_update=0, **kwargs))
    repeated = list(FixedUpdateBatchSampler(start_update=0, **kwargs))
    resumed = list(FixedUpdateBatchSampler(start_update=4, **kwargs))
    assert full == repeated
    assert resumed == full[4:]
    assert len(full) == 9
    assert all(len(batch) == 5 for batch in full)
    assert all(len({source for source, _update in batch}) == 5 for batch in full)
    assert [batch[0][1] for batch in full] == list(range(1, 10))


def test_augmentations_are_keyed_and_do_not_depend_on_call_order():
    config = load_config(DEFAULT_CONFIG)
    base = DeterministicFakeCIFAR10(4, seed=1)
    dataset = DeterministicMultiViewDataset(base, make_ssl_transform(config), views=4, augmentation_seed=99)
    first = dataset[(2, 17)][0]
    _ = dataset[(0, 3)]
    repeated = dataset[(2, 17)][0]
    another_update = dataset[(2, 18)][0]
    assert torch.equal(first, repeated)
    assert not torch.equal(first, another_update)


def test_alignment_matches_explicit_definition_and_keeps_center_attached():
    values = torch.randn(4, 7, 5, requires_grad=True)
    center = values.mean(dim=0)
    expected = ((values - center[None]) ** 2).sum() / (4 * 7 * 5)
    actual = all_view_alignment(values)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert all(bool(values.grad[view].abs().sum() > 0) for view in range(4))


def test_vicreg_uses_unbiased_covariance_and_expected_scaling():
    values = torch.tensor(
        [
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, -1.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )
    total, parts = vicreg_variance_covariance(values)
    manual_variance = []
    manual_covariance = []
    for view in values:
        centered = view - view.mean(0, keepdim=True)
        variance = centered.square().sum(0) / (len(view) - 1)
        manual_variance.append(torch.relu(1 - torch.sqrt(variance + 1e-4)).mean())
        covariance = centered.T @ centered / (len(view) - 1)
        manual_covariance.append((covariance[0, 1].square() + covariance[1, 0].square()) / 2)
    expected_variance = torch.stack(manual_variance).mean()
    expected_covariance = torch.stack(manual_covariance).mean()
    assert torch.allclose(parts["variance"], expected_variance)
    assert torch.allclose(parts["covariance"], expected_covariance)
    assert torch.allclose(total, 25 * expected_variance + expected_covariance)


def test_arm_objectives_have_fixed_alignment_coefficient():
    config = load_config(DEFAULT_CONFIG)
    values = torch.randn(4, 8, 6)
    loss_a, components_a = objective("A", values, config)
    assert torch.allclose(loss_a, 0.98 * all_view_alignment(values))
    loss_c, components_c = objective("C", values, config, coefficient=0.125)
    assert torch.allclose(
        loss_c, 0.98 * components_c["alignment"] + 0.125 * components_c["regularizer"]
    )
    with pytest.raises(ValueError, match="coefficient is unset"):
        objective("C", values, config)


def test_soft_orthogonality_uses_smaller_gram():
    wide_identity = torch.cat([torch.eye(3), torch.zeros(3, 2)], dim=1)
    tall_identity = torch.cat([torch.eye(3), torch.zeros(2, 3)], dim=0)
    assert float(soft_orthogonality(wide_identity)) == pytest.approx(0.0)
    assert float(soft_orthogonality(tall_identity)) == pytest.approx(0.0)


def test_optimizer_decay_groups_only_linear_and_convolution_matrices():
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=True),
        torch.nn.BatchNorm1d(3),
        torch.nn.Linear(3, 2, bias=False),
    )
    groups, names = optimizer_parameter_groups([module], matrix_weight_decay=5e-4)
    assert groups[0]["weight_decay"] == pytest.approx(5e-4)
    assert groups[1]["weight_decay"] == 0.0
    assert names["matrix_decay"] == ["backbone.0.weight", "backbone.2.weight"]
    assert set(names["bias_norm_no_decay"]) == {
        "backbone.0.bias",
        "backbone.1.weight",
        "backbone.1.bias",
    }


def test_schedule_hits_warmup_peak_and_final_floor():
    config = load_config(DEFAULT_CONFIG)
    assert learning_rate_for_update(1, config) == pytest.approx(0.002 / 175)
    assert learning_rate_for_update(175, config) == pytest.approx(0.002)
    assert learning_rate_for_update(21000, config) == pytest.approx(0.000002)
    with pytest.raises(ValueError):
        learning_rate_for_update(0, config)


def test_weight_and_representation_diagnostics_on_small_matrices():
    metrics, spectrum = spectral_metrics(torch.eye(4))
    assert metrics["normalized_weight_energy_effective_rank"] == pytest.approx(1.0)
    assert metrics["normalized_stable_rank"] == pytest.approx(1.0)
    assert len(spectrum) == 4
    assert matrix_kind("layer2.0.downsample.0.weight", torch.empty(4, 4), "backbone") == "shortcut_1x1"

    features = torch.randn(24, 6)
    representation, covariance_spectrum = representation_metrics(features, torch.device("cpu"))
    assert representation["samples"] == 24
    assert representation["dimension"] == 6
    assert len(covariance_spectrum) == 6

    views = torch.randn(4, 24, 6)
    lidar, lidar_spectrum = lidar_metrics(views, delta=1e-6, epsilon=1e-12, analysis_device=torch.device("cpu"))
    assert 1.0 <= lidar["lidar"] <= 6.0
    assert lidar["surrogate_classes"] == 24
    assert lidar["views_per_class"] == 4
    assert len(lidar_spectrum) == 6


def test_falsification_edits_preserve_marginals_and_coordinated_permutation_preserves_function():
    model = build_backbone().eval()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    layer = "layer4.1.conv2.weight"
    shuffled = shuffle_channel_pair_kernels(state, layer, seed=5)
    assert kernel_multiset_sha256(state[layer]) == kernel_multiset_sha256(shuffled[layer])
    assert not torch.equal(state[layer], shuffled[layer])

    permuted, _ = coordinated_intermediate_channel_permutation(state, "layer4.1", seed=7)
    permuted_model = build_backbone().eval()
    permuted_model.load_state_dict(permuted, strict=True)
    inputs = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        reference_output = model(inputs)
        permuted_output = permuted_model(inputs)
    assert torch.allclose(reference_output, permuted_output, atol=1e-6, rtol=1e-5)
