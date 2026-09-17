# CNN Filter DB Experiment Progress and Agent Handoff

Last updated: 2026-08-07

## Project Goal

This project studies whether different training methods leave measurable patterns
in convolutional kernel distributions. The main emphasis is learned `3x3`
filters, following `papers/CNN_Filter_DB(Paper).pdf`, rather than treating model
accuracy as the only result.

Three main experiments have been completed:

| Experiment | Architecture / data | Main comparison | Canonical output |
|---|---|---|---|
| 1 | ResNet50 / MLL23 | Scratch FT vs ImageNet FT vs LeJEPA SSL | `outputs/(1stEXP)rn50_filter_analysis/` |
| 2 | ResNet50 / MLL23 | Signed LeJEPA SSL minus ImageNet FT residuals | `outputs/(2ndEXP)rn50_lejepa_imagenet_residuals/` |
| 3 | ResNet18 / CIFAR10 | ImageNet1K FT vs true LeJEPA SSL + frozen probe | `outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/` |

The current scientific objective is not to prove that LeJEPA has higher
classification accuracy. It is to compare kernels from two reasonably close
performing models and identify repeatable distributional patterns associated
with their training routes.

## Shared Analysis Method

- A convolution tensor is separated into one `3x3` kernel for every
  input-channel/output-channel pair.
- Each kernel is usually divided by its own maximum absolute coefficient before
  shape analysis. This removes magnitude and preserves spatial shape.
- PCA is fitted on a shared filter population so the compared models use the
  same nine-dimensional basis.
- Histogram differences are measured with explained-variance-weighted symmetric
  KL drift. Lower drift means more similar filter distributions.
- Layer entropy and sparsity provide heuristic local quality information. They
  are descriptive and are not direct measures of model quality.
- Signed residual, total variation, and mean coefficient shift localize how two
  otherwise similar distributions differ by PCA component and network depth.
- Experiment 3 additionally compares raw weights, BatchNorm-folded effective
  weights, per-kernel normalized shapes, spatial frequency, center energy, and
  roughness.

Do not compare channels kernel-by-kernel across independently trained models.
Channels can be permuted without changing function, so distribution-level
comparisons are safer. Results currently use one seed and cannot yet establish
that a pattern is universally method-specific.

## Experiment 1: RN50 Three-Model Filter Analysis

### Setup

Three local ResNet50 backbones trained/evaluated on MLL23 were compared:

| Model | Training route | Stored best balanced accuracy |
|---|---|---:|
| Scratch FT | Random initialization, full target fine-tuning | 0.5679 |
| ImageNet FT | ImageNet-pretrained, full target fine-tuning | 0.7244 |
| LeJEPA SSL | SSL backbone with logistic-regression target head | 0.8377 |

Each RN50 has 16 bottleneck `conv2` tensors with `3x3` kernels, totaling
1,257,472 channel-pair filters per model. The `7x7` stem and all `1x1`
convolutions were excluded.

### Main Results

| Pair | Global drift D |
|---|---:|
| Scratch FT vs ImageNet FT | 0.482610 |
| Scratch FT vs LeJEPA SSL | 0.314835 |
| ImageNet FT vs LeJEPA SSL | 0.045309 |

ImageNet FT and LeJEPA SSL were very close in global `3x3` filter space, while
the scratch model was much farther from both. This ranking was stable in the
bootstrap analysis. The scratch filters also had high entropy and zero measured
sparsity, making many layers look comparatively random-like. LeJEPA showed more
sparsity in later layers.

This experiment established the PCA, drift, entropy, sparsity, ridge, scatter,
stage, and layer analysis pipeline. Its main limitation is the large performance
gap between the checkpoints, especially ImageNet FT at 0.7244 and LeJEPA at
0.8377. That gap motivated Experiment 3.

### Key Files

- Main script: `scripts/analyze_rn50_filters.py`
- Slurm job: `jobs/analyze_rn50_filters.sbatch`
- Checkpoints: `models/rn50_rand_ft_backbone.pth`,
  `models/rn50_imagenet_ft_backbone.pth`, and
  `models/rn50_lejepa_ssl_backbone.pth`
- Summary: `outputs/(1stEXP)rn50_filter_analysis/summary.md`
- Detailed report:
  `outputs/(1stEXP)rn50_filter_analysis/report_first_experiments.md`
- Completed job: `1330456` on `palamut1`

## Experiment 2: RN50 LeJEPA-ImageNet Signed Residuals

### Purpose

Experiment 1 showed that RN50 ImageNet FT and LeJEPA SSL were globally close.
Experiment 2 reused the same normalized filters and PCA basis, then analyzed the
signed residual defined as:

```text
LeJEPA SSL - ImageNet FT
```

Only the RN50 bottleneck `3x3` filters were included.

### Main Results

- Global pair drift remained `0.045309`, exactly matching Experiment 1.
- PCA component `c0` was the largest residual after explained-variance
  weighting: global weighted TV `0.025807`, TV `0.066379`, mean shift `+0.033900`.
- The residual was localized by depth rather than spread uniformly.
- Early and late layers generally shifted positively along `c0`; several
  layer2/layer3 blocks shifted negatively.
- ImageNet was much sparser in the first bottleneck, while LeJEPA was much
  sparser in late layer4 blocks.

The interpretation is that LeJEPA did not learn a completely separate RN50
filter universe. It produced a structured, depth-dependent residual on top of a
broadly ImageNet-like filter distribution.

### Key Files

- Main script: `scripts/analyze_lejepa_imagenet_residuals.py`
- Slurm job: `jobs/analyze_lejepa_imagenet_residuals.sbatch`
- Summary: `outputs/(2ndEXP)rn50_lejepa_imagenet_residuals/summary.md`
- Completed job: `1341201` on `palamut1`

## Experiment 3: RN18/CIFAR10 Matched-Accuracy Comparison

### Motivation

The third experiment changed both architecture and dataset and deliberately
saved many checkpoints. The intended comparison is:

1. ImageNet1K-pretrained ResNet18, fully fine-tuned on labeled CIFAR10.
2. Randomly initialized ResNet18, pretrained with LeJEPA on unlabeled CIFAR10,
   then evaluated with a separately trained frozen-backbone linear head.

The test-set accuracy is used to select a close pair, but kernel distributions
are the primary result.

### Important Implementation History

- The first fake-data run was only a smoke test for code execution. It is not a
  scientific CIFAR10 result.
- The first real-data SSL implementation used an EMA target encoder, predictor,
  stop-gradient, and cosine target-prediction objective. After the supplied
  LeJEPA paper and `lejepa/` repository were inspected, this was recognized as
  not LeJEPA. Its SSL checkpoints must not be used or described as LeJEPA.
- The supervised ImageNet-pretrained fine-tuning checkpoints from that run were
  valid and were reused by the corrected runs.
- Corrected LeJEPA uses view-center prediction/invariance plus SIGReg with the
  local Epps-Pulley and slicing implementations. It has no EMA target encoder,
  target projector, predictor MLP, stop-gradient, or prototypes.

The corrected implementation is documented in
`docs/rn18_cifar10_lejepa_protocol.md` and implemented in
`scripts/train_rn18_cifar10_experiment.py`.

### Corrected LeJEPA Protocol

- Dataset: real CIFAR10, 50,000 training and 10,000 test images.
- Backbone: torchvision ResNet18 with a CIFAR-style `3x3`, stride-1 stem.
- SSL labels: never used for encoder/projector gradients.
- Loss: `(1 - lambda) * prediction_loss + lambda * SIGReg_loss`.
- Prediction loss: pulls each image's view embeddings toward their view center.
- For ResNet, all views are global views, following the supplied paper's
  non-ViT rule.
- SIGReg: 1,024 slices, `t_max=3.0`, 17 integration points.
- Evaluation: freeze the SSL backbone and train a separate classification head.
- Reported LeJEPA accuracy means backbone plus the separately trained head on
  the CIFAR10 test set. It is not an accuracy produced by SSL alone.
- Quick probes are monitoring tools. Final comparisons use fixed-epoch heads.

### Configurations Tried

| Run | Main changes | Best final test accuracy | Outcome |
|---|---|---:|---|
| True LeJEPA baseline | 250 SSL epochs, `V=8`, `lambda=0.05`, projector 256, moderate augmentation, SGD linear probe for 20 epochs | 0.8605 at SSL epoch 110 | Plateaued; later SSL epochs did not improve accuracy |
| Post-hoc probe | Baseline SSL checkpoints, LayerNorm+Linear, AdamW, 100 head epochs | 0.8580 at SSL epoch 190 | Longer/better head did not fix the representation gap |
| Strong p16 | 120 SSL epochs, `V=4`, `lambda=0.02`, projector 16, hidden 2048, strong augmentation, SSL LR `2e-3`, 50-epoch AdamW probe | 0.8873 at SSL epoch 120 | Large improvement |
| Strong p64, canonical | Same as p16 but projector 64 | 0.8913 at SSL epoch 110 | Best completed run |
| Moderate V8 p64 | `V=8`, `lambda=0.02`, projector 64 | No result | Job `1429988` failed during CUDA initialization on `kolyoz13` |
| Strong lambda 0.05 p64 | `V=4`, `lambda=0.05`, projector 64 | No result | Job `1429989` failed during CUDA initialization on `kolyoz13` |

The failed CUDA jobs are untested configurations, not negative scientific
results. Sweep details are in `docs/rn18_cifar10_lejepa_sweep_20260729.md`.

### Accuracy Interpretation

- Best ImageNet FT accuracy: `0.9595` at epoch 19.
- Lowest saved ImageNet FT accuracy: `0.9125` at epoch 1.
- Best corrected LeJEPA final linear-probe accuracy: `0.8913` at SSL epoch 110.
- Advisor-selected comparison: ImageNet FT epoch 1 at `0.9125` versus LeJEPA
  SSL epoch 110 plus its frozen probe at `0.8913`.
- Absolute test-accuracy gap for this pair: `0.0212`.

The strong p64 run stabilized around 89% from roughly epochs 90-120. More head
epochs did not help the weaker baseline, so the earlier 86% ceiling was not just
linear-probe undertraining. The remaining gap is plausibly related to the very
different pretraining data budgets: ImageNet pretraining used about 1.2 million
images before CIFAR10 fine-tuning, while LeJEPA saw only the 50,000 CIFAR10
training images. This is a hypothesis, not a proved conclusion.

### Canonical Kernel Comparison

Use this pair for the current RN18 analysis:

- ImageNet FT checkpoint: epoch 1, accuracy `0.9125`.
- LeJEPA SSL backbone: epoch 110 from the strong p64 run.
- LeJEPA frozen linear head: epoch-110 head, final accuracy `0.8913`.

The dedicated output is:

```text
outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/
  analysis_imagenet_epoch001_vs_lejepa_epoch110/
```

The direct kernel-distribution report found that raw scale is heavily confounded
by BatchNorm: LeJEPA raw pooled standard deviation was `7.02x` ImageNet, but the
ratio became `0.33x` after BatchNorm folding. After per-kernel normalization,
global shape TV was `0.0531`. LeJEPA also showed higher high-frequency ratio and
roughness, with lower center energy. These observations are descriptive and
need replication across seeds.

### Stem Exclusion and Current Drift Result

The experiment's ResNet18 uses a CIFAR-style `3x3` stem instead of torchvision's
stock ImageNet `7x7` stem. This adaptation is shared by both compared models,
but it is a special input layer and previously dominated the depth graph.

For the paper-aligned `3x3` PCA/drift analysis, the stem is now excluded. The
current analysis contains the 16 residual-block `3x3` convolutions only, totaling
1,220,608 channel-pair kernels per model.

Current no-stem PCA/drift results:

| Scope | Drift D |
|---|---:|
| Global | 0.128806 |
| layer1 | 0.079488 |
| layer2 | 0.015189 |
| layer3 | 0.017935 |
| layer4 | 0.185562 |

The largest layer drift is `layer4.1.conv2.weight` at `0.779819`. The current
depth graph is:

```text
outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/
  analysis_imagenet_epoch001_vs_lejepa_epoch110/pca_shape_analysis/
  figures/drift_layer_by_depth.png
```

Important: the parent direct kernel-distribution report was generated before
the stem exclusion and still includes the CIFAR stem in its tables and global
descriptor pools. Treat `pca_shape_analysis/summary.md` as the authoritative
stem-excluded drift report. Do not cite the stem ranking from the parent report
as a residual-block result.

### Key Files

- Training/common code: `scripts/train_rn18_cifar10_experiment.py` and
  `scripts/rn18_cifar10_common.py`
- Corrected cuDNN job: `jobs/run_rn18_cifar10_true_lejepa_cudnn.sbatch`
- Probe script: `scripts/probe_rn18_cifar10_lejepa_checkpoints.py`
- Direct distribution analysis:
  `scripts/analyze_rn18_cifar10_kernel_distributions.py`
- PCA/drift analysis: `scripts/analyze_rn18_cifar10_filters.py`
- Canonical training summary:
  `outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/summary.md`
- Canonical pair report:
  `outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/analysis_imagenet_epoch001_vs_lejepa_epoch110/summary.md`
- Canonical no-stem drift summary:
  `outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/analysis_imagenet_epoch001_vs_lejepa_epoch110/pca_shape_analysis/summary.md`

## Infrastructure Findings

- CIFAR10 data was downloaded to `data/cifar10`; scientific jobs use real data,
  not `FakeData`.
- cuDNN failed on `palamut6` with `CUDNN_STATUS_NOT_INITIALIZED`. CUDA worked
  there only when cuDNN was disabled.
- cuDNN was successfully probed and used on several `kolyoz-cuda` H100 nodes,
  including `kolyoz22`, `kolyoz24`, and `kolyoz33`.
- `kolyoz13` produced CUDA initialization failures for jobs `1429988` and
  `1429989`. Avoid treating those failures as model/configuration failures.
- The working Slurm resource pattern is one GPU on `kolyoz-cuda`, 16 CPUs, one
  task, one node. See `ServerInformation.txt` for commands and history.

## Current Reproducibility Caveats

1. Several output directories were renamed with experiment prefixes after jobs
   completed. Generated JSON/Markdown files may still contain their old paths,
   such as `outputs/rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong`.
2. The RN18 ImageNet epoch checkpoints were originally stored under
   `outputs/rn18_cifar10_experiment_cudnn/`. That directory is not present in
   the current repository tree. Existing analyses and metrics remain available,
   but rerunning the exact matched-pair analysis requires restoring or
   regenerating `imagenet_ft_epoch001.pth`.
3. Experiment 3 currently uses one seed. Repeat both training routes across
   seeds before presenting a kernel pattern as method-specific.
4. Accuracy matching reduces one confound but does not equalize pretraining
   dataset size, label exposure, optimization, or total compute.
5. The CIFAR-style stem differs from stock ImageNet ResNet18. Keep it excluded
   when the question is specifically about comparable residual-block `3x3`
   filters.

## Recommended Next Steps

1. Preserve or restore the RN18 ImageNet epoch-1 checkpoint inside a canonical,
   stable output directory and update stale paths.
2. Repeat the canonical RN18 comparison with multiple random seeds.
3. Run bin-count sensitivity and bootstrap confidence intervals for the RN18
   drift and descriptor differences.
4. Inspect representative kernels from the highest-residual PCA bins, especially
   `layer4.1.conv2`, early layer1 blocks, and their dominant `c0` residuals.
5. If resources allow, test equal-data or larger-unlabeled-data LeJEPA
   pretraining before concluding that the remaining accuracy gap is intrinsic.
6. CIFAR100 can be a useful follow-up for transfer and class-granularity tests,
   but changing to CIFAR100 does not by itself equalize ImageNet's pretraining
   data advantage.

## Read These First

For a new agent or chat, read in this order:

1. This file.
2. `docs/rn18_cifar10_lejepa_protocol.md` for the corrected LeJEPA definition.
3. `outputs/(1stEXP)rn50_filter_analysis/report_first_experiments.md` for the
   original filter-analysis method and RN50 findings.
4. `outputs/(2ndEXP)rn50_lejepa_imagenet_residuals/summary.md` for the signed
   residual interpretation.
5. The canonical Experiment 3 training, pair-comparison, and no-stem PCA
   summaries listed above.
6. `ServerInformation.txt` before submitting GPU jobs on TRUBA.
