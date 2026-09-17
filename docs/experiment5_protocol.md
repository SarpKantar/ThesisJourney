# Experiment 5: SIGReg and actionable local weight structure

This directory-level protocol turns `THESIS_EXPERIMENT_5_RESEARCH_PLAN.md` into an executable, gated experiment. It does not modify or reuse the invalid EMA/predictor lineage. The existing Experiment 3/4 scripts and outputs remain historical artifacts.

## What is implemented

- Experiment 5A loss audit against the local LeJEPA checkout pinned at commit `c293d291ca87cd4fddee9d3fffe4e914c7272052`, including synthetic Gaussian, shifted, rescaled, rank-deficient, and constant inputs; value/gradient parity; and the historical bound-3 sensitivity check.
- Existing-checkpoint kernel-population shuffle and valid function-preserving residual-block channel-permutation controls.
- The 16-dimensional, two-layer linear control for condition numbers 1, 10, and 100; SIGReg, VICReg variance/covariance, and dimension-normalized soft orthogonality; five seeds; 5,000 updates.
- Experiment 5B arms A/B/C with identical per-seed initialization, fixed update count, fixed schedule, explicit AdamW decay groups, deterministic split/data/augmentation/SIGReg streams, exact resume state, and all planned checkpoints.
- Pilot coefficient calibration and equal-budget coefficient-neighborhood jobs for both B and C.
- Uncentered layer-matrix spectra for raw, BN-folded, and row-direction parameter views. The primary endpoint is frozen in the config as the equal-layer mean normalized weight-energy effective rank over the 16 residual 3x3 matrices at update 5,250 (epoch 30), with B-C as the primary contrast.
- Fixed-image stage, backbone, projector, final-z, covariance-rank, RankMe, LiDAR, multi-view, and independent Gaussian-discrepancy diagnostics.
- Frozen-backbone 1%, 10%, and 100% linear probes with three nested label draws and the fixed 50-epoch AdamW protocol.
- Seed-level paired aggregation with initialization-hash verification, 95% paired t intervals, and the exact two-sided sign-flip result.

The Experiment 5C learned local candidate and Experiment 5D replacement branches are intentionally gated. The research plan requires the local statistic, coefficients, layer scope, and endpoints to be chosen from completed discovery evidence and then frozen. Creating a nominal “new penalty” before A/B/C finishes would violate that rule. Soft orthogonality is already implemented as the initial parameter-only baseline and in the analytic control.

## Frozen files

- `configs/experiment5/main.json`: scientific protocol and checkpoint schedule.
- `scripts/experiment5/core.py`: objectives, model, optimizer grouping, RNG streams, and schedule.
- `scripts/experiment5/data.py`: immutable split logic, deterministic augmentations, and update-indexed sampling.
- `tests/test_experiment5.py`: unit checks for the controlled definitions.

Every run writes its full config, config hash, source commits/status, package versions, split hash, RNG seeds, initialization-state hashes, optimizer groups, command line, and checkpoints. A smoke run is marked `scientific_result=false` and cannot be confused with a real run.

## Execution order

Run from the repository root. The first submission downloads/verifies CIFAR10 and publishes the split indices and hashes.

```bash
sbatch jobs/prepare_experiment5.sbatch
sbatch jobs/validate_experiment5.sbatch
sbatch jobs/check_experiment5_weight_falsifications.sbatch
sbatch jobs/run_experiment5_linear_toy.sbatch
sbatch jobs/calibrate_experiment5.sbatch
```

Inspect `outputs/(5thEXP)rn18_cifar10_sigreg_structure/pilot/coefficient_ledger.json`. The ledger reports backbone, projector, and joint gradient norms and defines `alpha0`, `alpha0/3`, and `3*alpha0`. Gradient matching is only a scale convention.

The full two-seed coefficient sensitivity grid is available but expensive:

```bash
sbatch jobs/run_experiment5_pilot_sensitivity.sbatch
```

The screening array defaults to the first three discovery seeds (nine A/B/C runs):

```bash
sbatch jobs/run_experiment5_abc.sbatch
```

After confirming that the protocol behaves as intended, extend only the remaining two seeds:

```bash
sbatch --array=9-14 jobs/run_experiment5_abc.sbatch
```

Full activation diagnostics are scheduled at epochs 0, 1, 5, 10, 30, 60, and 120. The default array covers the nine screening runs; extend it after completing seeds 31 and 43:

```bash
sbatch jobs/diagnose_experiment5.sbatch
sbatch --array=63-104 jobs/diagnose_experiment5.sbatch
```

The final-checkpoint linear probes follow the same screen-then-extend convention:

```bash
sbatch jobs/probe_experiment5.sbatch
sbatch --array=9-14 jobs/probe_experiment5.sbatch
```

Once all epoch-30 weight summaries exist, aggregate only independent training seeds:

```bash
python scripts/aggregate_experiment5.py
```

## Local smoke check

This uses deterministic fake images solely to exercise the code path:

```bash
python scripts/train_experiment5.py \
  --arm B \
  --seed 13 \
  --device cpu \
  --num-workers 0 \
  --smoke-test \
  --output-dir /tmp/experiment5-smoke
```

Do not report fake-data or pilot runs as scientific outcomes.

## Important audited convention

The pinned LeJEPA implementation stores 17 quadrature nodes on the nonnegative interval `[0, bound]` and represents negative frequencies through symmetric trapezoid weights. Its effective interval is `[-bound, bound]`, but it is not literally 17 stored nodes spread across that full interval. `validate_experiment5_losses.py` records this convention and must pass before the main run.

## Resume and failure policy

Resume only from an exact saved checkpoint via `--resume path/to/update_NNNNNN.pth`. The model, optimizer, BN buffers, SIGReg direction counter, Python/NumPy/Torch RNG states, and completed update are restored. Data order and augmentations are pure functions of the stored stream seeds and update index.

Finite collapsed runs continue and remain in the result set. A final-z mean coordinate variance below `1e-4` for three logged checks is a diagnostic flag, not an exclusion rule. Nonfinite loss or gradients stop immediately and write `failure.json`; no arm-specific clipping or recovery is added.
