# Implementation Plan: CNN Filter DB Analysis for Three RN50 Backbones

This document explains how to adapt the CNN Filter DB evaluation method to the
three local ResNet-50 backbone checkpoints in `models/`:

- `rn50_rand_ft_backbone.pth`: RN50 trained from scratch on the target dataset.
- `rn50_imagenet_ft_backbone.pth`: ImageNet-pretrained RN50 fine-tuned on the target dataset.
- `rn50_lejepa_ssl_backbone.pth`: LeJepa SSL RN50 backbone with a target-dataset logistic-regression head.

The target dataset stored in the checkpoints is `mll23` for all three models, so
the intended comparison isolates the training/pretraining recipe while keeping
architecture and target dataset fixed.

## 1. What the Paper Measures

The paper does not analyze activations or input images. It analyzes the learned
weights of trained CNNs, especially regular `3x3` convolution kernels. Each
`3x3` kernel is flattened into a 9-dimensional vector. The paper then asks:

1. Do filters trained under different conditions occupy different regions of the
   same 9D filter space?
2. Are some layers degenerated, meaning sparse, low-diversity, or still close to
   random initialization?
3. Do distribution shifts in filter space correlate with metadata such as task,
   visual category, architecture family, pretraining dataset, training dataset,
   or convolution depth?

The main pipeline is:

1. Extract only regular `3x3` convolution filters from trained CNNs.
2. Flatten every filter to shape `(9,)`.
3. Normalize each filter by its own maximum absolute weight:

   ```text
   f_scaled = f / max(abs(f))
   ```

   If the denominator is zero, leave the filter unchanged.

4. Fit a full-rank PCA with `n_components=9`.
5. Represent every filter by its 9 PCA coefficients.
6. Compare groups of filters by histogramming each PCA coefficient and computing
   a weighted symmetric KL divergence.

The paper's drift score between two filter groups `P` and `Q` is:

```text
D(P, Q) = sum_i explained_variance_ratio[i] * KL_sym(P_i, Q_i)
KL_sym(P_i, Q_i) = KL(P_i || Q_i) + KL(Q_i || P_i)
```

where `P_i` and `Q_i` are the 1D coefficient histograms for principal component
`i`. The notebook uses 70 bins and a shared histogram range across all groups
being compared.

The second major measurement is layer degeneration:

- `sparsity S`: the fraction of filters in a layer whose weights are all near
  zero. The notebook implements the threshold as `abs(layer).max() / 100`.
- `entropy H`: Shannon entropy of the layer's SVD/PCA explained-variance ratio.
  Low entropy means filters in that layer are structurally redundant. Very high
  entropy, near the paper's random-initialization threshold, suggests filters may
  still be close to random.

The paper's random-layer threshold is:

```text
TH(n) = 1.26 / (1 + exp(-0.89 * (log2(n) - 2.30))) - 0.31
```

For the RN50 `3x3` layers here, `n` is large, so `TH(n)` is approximately `0.95`.
Layers with `H` near or above this value should be inspected as possibly
random-like. Layers with low `H` and/or high `S` should be inspected as possible
low-diversity or sparse degeneration.

The paper also uses qualitative visual phenotypes in the first PCA dimensions:

- `sun`: roughly Gaussian-like coefficient distributions.
- `spikes`: local hot spots, usually low-diversity degeneration.
- `symbols`: multimodal, off-center, sparse, or otherwise non-normal structure.
- `point`: most coefficients concentrated near zero, usually sparsity degeneration.

These phenotypes should be used as visual diagnostics, not as standalone proof.

## 2. What `main-partial.ipynb` Shows

The partial notebook is the same analysis style, but it was run on a small prefix
of the huge CNN Filter DB dataset.

Observed run details:

- Full database size: `1,464,797,156` filters.
- Processed fraction: `1.0%`.
- Processed filters: `14,647,971`.
- Metadata after filtering to the processed prefix: `257` layer rows.
- PCA output shape: `(14647971, 9)`.
- The first global PCA component explains about `0.41` of variance; the first
  three components explain about `0.70` cumulatively.
- The subset PCA auto-selected `natural` because it was the most frequent visual
  category in that partial metadata slice.
- The partial ridge plot only had four visual categories: `art`, `map`,
  `natural`, and `plants`.

Important caveats from the partial notebook:

1. The partial analysis uses the first 1 percent of the HDF5 filter table, not a
   stratified random sample. If the HDF5 rows are ordered by model/category, the
   result is biased toward whichever models appear early.
2. Small metadata groups can create misleading spikes. In the visible partial
   results, `plants` is very spiky and far from the other groups, but it is a
   tiny group in the subset.
3. The task KL matrix only compares `Classification` and `GAN-Generator`, so it
   should not be interpreted as a robust task-level result.
4. The subset PCA cell calls `scale(subset_raw)`, but in the visible notebook
   source the active scaling function is named `scale_chunk`; `scale` appears in
   a commented cell. That means the successful run probably depended on hidden
   notebook state. The RN50 implementation should be scripted and self-contained.
5. The notebook is useful as a method prototype, but the RN50 experiment should
   not reuse the partial subset results as scientific baselines without a
   balanced reference selection.

## 3. Local Checkpoint Inventory

The three local checkpoints are PyTorch archives with a top-level
`backbone_state` dictionary.

| File | Backbone metadata | Dataset | Best balanced accuracy | Epochs |
|---|---|---:|---:|---:|
| `rn50_rand_ft_backbone.pth` | `rn50_rand_finetuned` | `mll23` | `0.5679` | `100` |
| `rn50_imagenet_ft_backbone.pth` | `rn50_imagenet_finetuned` | `mll23` | `0.7244` | `100` |
| `rn50_lejepa_ssl_backbone.pth` | `rn50_lejepa_ssl` | `mll23` | `0.8377` | not stored |

Each checkpoint contains:

- `53` convolution tensors total.
- `1` `7x7` stem convolution: `conv1.weight`.
- `36` `1x1` convolutions.
- `16` `3x3` bottleneck convolutions, all named like `layerX.Y.conv2.weight`.

For paper-compatible analysis, use only those `16` `3x3` bottleneck tensors.
The stage layout is:

| Stage | Number of `3x3` conv layers | Tensor shapes |
|---|---:|---|
| `layer1` | 3 | `(64, 64, 3, 3)` |
| `layer2` | 4 | `(128, 128, 3, 3)` |
| `layer3` | 6 | `(256, 256, 3, 3)` |
| `layer4` | 3 | `(512, 512, 3, 3)` |

Each RN50 has `1,257,472` `3x3` filters:

```text
3 * 64 * 64 + 4 * 128 * 128 + 6 * 256 * 256 + 3 * 512 * 512 = 1,257,472
```

This is a good experimental setup because architecture and target dataset are
held constant. The main independent variable is the training recipe.

## 4. Main Experimental Questions

### A. Which training recipe produces the most similar filter distribution?

Compare the three RN50s directly:

- scratch target training vs ImageNet fine-tuning.
- scratch target training vs LeJepa SSL plus logistic regression.
- ImageNet fine-tuning vs LeJepa SSL plus logistic regression.

This is the cleanest comparison because all three models are RN50 backbones and
all target the same dataset. A global pairwise KL drift matrix should be the
first result.

### B. Where in the network do the training recipes differ?

Compute drift at multiple granularities:

- full backbone, all `3x3` filters pooled.
- stage-level groups: `layer1`, `layer2`, `layer3`, `layer4`.
- layer-level groups: each of the 16 `conv2.weight` tensors.
- normalized depth bins, if we want the same style as CNN Filter DB.

This answers whether pretraining mainly changes early visual filters, middle
representation filters, or the final residual stage. For fine-tuning advice,
stage-level and layer-level drift are more useful than one global score.

### C. Do higher-performing models have healthier filters?

Compute degeneration metrics for every `3x3` conv layer:

- entropy `H`.
- sparsity `S`.
- optional `H / TH(n)` ratio, where `TH(n)` is the paper's random threshold.

Then compare these profiles against balanced accuracy:

```text
scratch FT:    bacc 0.5679
ImageNet FT:   bacc 0.7244
LeJepa SSL:    bacc 0.8377
```

The goal is not simply "more sparse is bad" or "higher entropy is good". The
paper's interpretation is more nuanced:

- high sparsity can mean unused filters.
- very low entropy can mean redundant filters.
- very high entropy near the random threshold can mean undertrained/random-like
  filters.
- ideal layers tend to have high-but-not-random entropy and low sparsity.

### D. Are these three RN50s normal relative to CNN Filter DB?

Use the original database as context, but do it carefully.

The direct 3-way RN50 comparison is primary. CNN Filter DB comparison is
secondary and should answer:

1. Are our target-trained RN50 filters close to ordinary ImageNet RN50 filters?
2. Are they closer to the paper's `natural` classification models or to another
   visual category, if the target dataset category is known?
3. Do any layers look like the paper's degenerated outliers?
4. Is the scratch-trained target RN50 unusually random-like or sparse compared
   with healthy pretrained RN50s?

Recommended reference groups:

- Same architecture/family first: `resnet50`, `timm resnet50`, and close RN50
  variants from CNN Filter DB.
- Same task next: classification models.
- Same visual category only if the `mll23` visual category is known and enough
  reference models exist.
- Full database only for broad context, not as the first baseline.

Avoid comparing our three RN50s primarily against arbitrary architectures. The
paper found that architecture family can be very stable internally, so same-family
comparison is the fairest reference.

### E. How much of the difference is target adaptation?

The current files contain final backbones only. To isolate target fine-tuning
from pretraining, ask for or produce extra checkpoints if possible:

- ImageNet RN50 before target fine-tuning.
- LeJepa SSL RN50 before target logistic regression or fine-tuning.
- scratch RN50 at initialization and several intermediate epochs.

With these, compute:

```text
D(pretrained, fine_tuned)
D(initialized, epoch_k)
D(epoch_k, final)
```

This would reproduce the paper's "formation during training" idea and make the
interpretation much stronger.

## 5. Proper Implementation Procedure

### Step 1: Extract RN50 `3x3` filters

Create a self-contained extraction script instead of relying on notebook state.

For each checkpoint:

1. Load with Torch.
2. Read `checkpoint["backbone_state"]`.
3. Select tensors with shape `(out_channels, in_channels, 3, 3)`.
4. For RN50, these should be exactly the 16 `layer*.conv2.weight` tensors.
5. Flatten each tensor to `(out_channels * in_channels, 9)`.
6. Build metadata per filter:

   ```text
   model_name
   training_recipe
   checkpoint_path
   target_dataset
   best_bacc
   layer_key
   stage
   layer_order_0_to_15
   conv_depth_norm = layer_order / 15
   filter_count_in_layer
   ```

7. Save arrays and metadata to reproducible files, for example:

   ```text
   outputs/rn50_filter_analysis/filters.npy
   outputs/rn50_filter_analysis/filter_meta.parquet
   outputs/rn50_filter_analysis/layer_meta.csv
   ```

Do not mix `1x1` and `7x7` kernels into the paper-compatible analysis. They are
different vector spaces. If needed, analyze them separately as an appendix.

### Step 2: Normalize filters

For PCA and KL drift, normalize each individual `3x3` filter by its own absolute
peak weight:

```python
den = np.abs(X).max(axis=1)
den = np.where(den == 0, 1, den)[:, None]
X_scaled = X / den
```

Use raw, unnormalized layer weights for the degeneration metrics unless a
sensitivity analysis explicitly says otherwise. This matches the notebook's
`layer_quality_worker`.

### Step 3: Fit PCA bases

Use at least two PCA bases.

1. `target_rn50_basis`

   Fit PCA on the union of the three local RN50s. This is the best basis for the
   main 3-way target experiment because all compared groups contribute equally to
   the coordinate system.

2. `reference_basis`

   Fit PCA on a curated CNN Filter DB reference set, preferably same-family RN50
   and ImageNet/classification models. This lets us say whether our filters are
   unusual relative to the published database.

Optionally also fit a `full_db_basis` using the full CNN Filter DB, but it is
computationally heavier and can be dominated by the database's class imbalance.

Do not compute KL drift from PCA bases fitted separately per model. Per-model PCA
is useful for qualitative eigenfilter inspection, but drift requires a shared
coordinate system.

### Step 4: Compute KL drift

For every comparison:

1. Transform filters into the chosen shared PCA basis.
2. Build histograms for each PCA coefficient.
3. Use the same histogram range and same bin count for all groups in that
   comparison.
4. Use 70 bins for paper compatibility.
5. Replace zero-probability bins with a small epsilon before normalization.
6. Compute weighted symmetric KL using PCA explained-variance ratios.

Report:

- global `3x3` model-to-model drift matrix.
- stage-specific drift matrices.
- layer-specific drift curves.
- component-level drift contributions.
- drift to reference groups from CNN Filter DB.

### Step 5: Balance the comparisons

A naive global filter pool is dominated by late RN50 layers because `layer4`
contains many more filters than `layer1`.

Report both:

1. filter-count-weighted drift, where every filter has equal weight.
2. layer-balanced or stage-balanced drift, where every layer/stage contributes
   equally.

For robustness, add bootstrap confidence intervals:

- sample the same number of filters from each group.
- repeat with multiple seeds.
- compute mean and confidence interval of drift.
- for reference comparisons, sample by model/layer when possible to avoid one
  large model dominating the group.

### Step 6: Compute layer quality

For every selected `3x3` layer:

1. Use the raw layer matrix `W.reshape(-1, 9)`.
2. Center it column-wise.
3. Run SVD or PCA on the layer.
4. Compute explained-variance ratio.
5. Compute entropy `H`.
6. Compute sparsity `S` using the notebook threshold `abs(W).max() / 100`.
7. Compute the paper's random threshold `TH(n)`.

Save one row per layer:

```text
model_name, training_recipe, layer_key, stage, conv_depth_norm,
num_filters, entropy_H, random_threshold_TH, H_over_TH, sparsity_S
```

Plot:

- `H` vs normalized depth.
- `S` vs normalized depth.
- `H_over_TH` vs normalized depth.
- stage summaries.

### Step 7: Visualize distributions

Minimum figure set:

1. PCA eigenfilters and explained variance for `target_rn50_basis`.
2. Ridge plots: three models by nine PCA components.
3. Ridge plots split by stage.
4. Global pairwise drift heatmap for the three models.
5. Stage-specific drift heatmaps.
6. `c0` vs `c1` density scatter plots for each model and each stage.
7. Entropy/sparsity depth profiles.
8. Reference-distance table or heatmap against selected CNN Filter DB groups.

The ridge and scatter plots are important because KL scores alone can hide
whether a difference is a smooth distribution shift, a spike, a point mass, or a
multimodal pattern.

## 6. Recommended Experiment Matrix

| ID | Experiment | Groups | Basis | Output | Main question |
|---|---|---|---|---|---|
| E1 | Direct model drift | 3 RN50s | target RN50 | `3x3` KL heatmap | Which recipe is closest/farthest? |
| E2 | Stage drift | model x `layer1..4` | target RN50 | stage heatmaps | Where do recipes differ? |
| E3 | Layer drift | model x 16 conv layers | target RN50 | depth curves | Are changes early, middle, or late? |
| E4 | Degeneration | every `3x3` layer | none/PCA per layer | `H`, `S`, `TH` plots | Which layers look sparse, redundant, or random-like? |
| E5 | Component analysis | 3 RN50s | target RN50 | per-PC drift bars | Which eigenfilter directions explain the difference? |
| E6 | Reference RN50 comparison | local RN50s plus DB RN50s | reference RN50 | distance/rank table | Are local models normal for RN50s? |
| E7 | Reference category comparison | local RN50s plus DB task/category groups | curated DB | drift heatmap | Are local models close to a visual/task category? |
| E8 | Sensitivity | repeated sampled subsets | same as E1-E7 | CIs/tables | Are conclusions stable? |

## 7. Preliminary Local-Only Sanity Check

A quick in-memory dry run on the three local RN50s, using a PCA basis fitted only
on their combined `3x3` filters, produced:

```text
PCA explained variance ratio:
[0.3888, 0.1208, 0.1153, 0.0857, 0.0668, 0.0596, 0.0595, 0.0530, 0.0505]

Global drift, filter-count-weighted:
scratch FT    vs ImageNet FT:       0.474190
scratch FT    vs LeJepa SSL:        0.308608
ImageNet FT   vs LeJepa SSL:        0.045268
```

Stage-specific drift from the same dry run:

| Stage | scratch vs ImageNet | scratch vs LeJepa | ImageNet vs LeJepa |
|---|---:|---:|---:|
| `layer1` | `0.377312` | `0.326717` | `0.088883` |
| `layer2` | `0.185356` | `0.146801` | `0.043907` |
| `layer3` | `0.301690` | `0.361373` | `0.037874` |
| `layer4` | `0.621027` | `0.313948` | `0.080742` |

Layer quality summary from the same dry run:

| Model | Mean H | Min H | Max H | Mean S | Max S |
|---|---:|---:|---:|---:|---:|
| scratch FT | `0.9346` | `0.8991` | `0.9501` | `0.000000` | `0.000000` |
| ImageNet FT | `0.7210` | `0.4011` | `0.9111` | `0.032397` | `0.502441` |
| LeJepa SSL | `0.7258` | `0.5708` | `0.8609` | `0.063215` | `0.264069` |

Interpret this only as a sanity check, not as final evidence. It has no
bootstrap confidence intervals, no reference basis, and no layer-balanced
correction. Still, it suggests useful priorities:

1. ImageNet FT and LeJepa SSL are very close in global filter structure.
2. The scratch-trained RN50 is much farther from both pretrained backbones.
3. The scratch model has high entropy and zero sparsity, so inspect whether it is
   closer to random-like filters.
4. The pretrained models show nonzero sparsity in specific layers, so inspect
   whether those are true degenerated filters or harmless learned pruning.
5. `layer4` and `layer1` deserve special attention because they show the largest
   recipe-dependent drift.

## 8. How to Use CNN Filter DB Properly as a Reference

Use the full database to contextualize the local RN50s, not to replace the direct
controlled comparison.

Recommended reference protocol:

1. Load `dataset.h5` from:

   ```text
   /arf/scratch/skantar/CNN-Filter-DB/data/dataset.h5
   ```

2. Select reference models with enough metadata support:

   - same-family RN50 ImageNet classification models.
   - same-task classification models.
   - same visual category as `mll23`, if known and adequately represented.
   - robust ImageNet models if robustness/degeneration is relevant.

3. Avoid tiny groups unless they are explicitly marked as low-confidence.
4. Fit a reference PCA basis on a balanced reference set.
5. Transform both reference filters and local RN50 filters into that basis.
6. Compute drift with equal sampling per model/layer.
7. Report nearest reference groups and outlier distances.

This answers questions like:

- Is scratch target training closer to random-like CNN Filter DB phenotypes?
- Are ImageNet FT and LeJepa SSL both inside the normal RN50/ImageNet region?
- Does `mll23` produce a distinct filter distribution, or does it mostly reuse
  common RN50 filter structure?
- Are any local layers more sparse or less diverse than typical RN50 layers?

## 9. Interpretation Rules

Use these rules when writing conclusions:

1. Because all three models share architecture and target dataset, direct drift
   differences are most plausibly due to training recipe.
2. The LeJepa checkpoint may represent SSL backbone quality plus a trained
   logistic-regression head, not necessarily target fine-tuning of the backbone.
   The filter analysis only sees the backbone.
3. A low global drift does not mean two models solve the task the same way. It
   means their `3x3` filter distributions are similar in this PCA/histogram
   representation.
4. High sparsity is not automatically bad. Check layer position, accuracy, and
   whether the pattern is also seen in strong reference models.
5. High entropy near `TH(n)` can indicate random-like filters, especially if it
   appears consistently across depth and is paired with low performance.
6. Database category comparisons are sensitive to group size and database bias.
   Prefer same-family references and bootstrapped estimates.
7. Do not over-interpret the partial notebook's small groups. Its value is the
   implementation pattern, not the final reference distribution.

## 10. Minimum Deliverable

The first complete implementation should produce:

```text
outputs/rn50_filter_analysis/
  local_filter_meta.parquet
  local_layer_quality.csv
  pca_target_rn50.npz
  drift_global_target_basis.csv
  drift_stage_target_basis.csv
  drift_layer_target_basis.csv
  figures/
    pca_eigenfilters_target_basis.png
    ridge_global_three_models.png
    ridge_by_stage.png
    drift_global_heatmap.png
    drift_stage_heatmaps.png
    entropy_by_depth.png
    sparsity_by_depth.png
    scatter_c0_c1_by_model.png
```

The second complete implementation should add:

```text
outputs/rn50_filter_analysis/
  reference_filter_meta.parquet
  pca_reference_rn50.npz
  drift_to_reference_groups.csv
  bootstrap_confidence_intervals.csv
  figures/
    reference_distance_heatmap.png
    reference_nearest_neighbors.png
    local_vs_reference_quality.png
```

If time is limited, prioritize `E1`, `E2`, and `E4`. Those directly answer the
professor's question for the three RN50 checkpoints.
