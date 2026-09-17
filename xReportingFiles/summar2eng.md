# Fourth Experiments and Reanalysis of the Third Experiment with Global PCA

This document explains in detail the technical audit of the fourth experiments conducted in the CNN Filter DB project, the analyses that were rerun, the results obtained, and the scientific interpretation of those results. Its scope is not limited to reporting the final numbers. It also explains why each checkpoint was used, how the global PCA basis was produced, exactly where the thresholds enter the calculations, which results are independent of the PCA basis, and which interpretations should remain cautious because of the single-seed/single-run limitation.

No new model training was performed at this stage. The figures from the third experiment and the analyses from the fourth experiment were regenerated from existing checkpoints. The sole exception is the previously completed deterministic baseline replay used to recover the selected ImageNet epoch-1 checkpoint that had been lost; the final analysis jobs covered by this document did not perform any further training.

## 1. Brief conclusion

The overall result of the technical audit is positive:

- The threshold-sensitivity analysis uses the correct two checkpoints, the correct 16 residual-block `3×3` layers, and seven different thresholds for each metric family.
- The entropy and sparsity calculations do not depend on the global PCA basis. These two measurements are computed locally from the raw filters in each layer.
- All `160/160` regression checks that reproduce the former default thresholds passed.
- The global PCA basis was calculated from all `1.464.797.156` filters in the released CNN Filter DB file; it was not fitted to a sample or only to the two ResNet18 models.
- The PCA covariance calculation is not approximate or mini-batch PCA. The exact nine-dimensional sample covariance was obtained by mathematically exact merging of chunk statistics.
- The PCA components are orthonormal, the explained-variance ratios sum to `1.0`, and the covariance is consistent with its eigenvalue/eigenvector decomposition.
- The maximum absolute difference between the centering vector printed in the paper supplement and the calculated vector is `0.00105831`. This small difference was not hidden; it is recorded as a warning in the artifact and consumer provenance records. After the user accepted a difference of this magnitude, it was treated as a documented limitation rather than a blocker.
- The global drift between ImageNet epoch 1 and LeJEPA epoch 110 is `0.128806` in the two-model local PCA basis and `0.133022` in the new full-data global basis. The absolute change is small (`+0.004216`, approximately `3.27%`). The main result—namely that the difference is concentrated especially in the final residual block and in some early layer1 layers—is preserved after changing the basis.
- The PCA-independent raw-weight, BN-folded-weight, normalized-filter-shape, and kernel-descriptor tables are numerically identical to the old third-experiment results. This additionally confirms that the regenerated ImageNet epoch-1 checkpoint is the correct checkpoint for the analysis.
- The complete figure package for the third experiment was regenerated in a separate directory with the new global PCA basis; the heatmaps were retained and corresponding chart versions were added.
- The four threshold heatmaps from the fourth experiment were retained, and a subdirectory containing seven ImageNet–LeJEPA line charts was created for each one.
- The LeJEPA-only training-dynamics section was completed on exactly 13 checkpoints at epochs `1, 10, 20, ..., 120`. Most of the global filter-distribution change occurs by epochs 20–30, while probe accuracy continues to improve at later epochs.
- All required checks for the LeJEPA dynamics passed, including SHA256 verification of the 13 checkpoints, independent reconstruction of the fixed histogram edges, symmetry of the `13x13` matrix, component-contribution sums, and readability of all 25 figures. The only `False` check is the `warning` row that keeps the accepted paper-mean difference visible.

The strongest overall interpretation is therefore as follows: the results are not merely a product of a PCA coordinate system fitted to the two compared models. When switching to the global, model-independent CNN Filter DB basis, the principal depth pattern and the dominant regions of difference are preserved. In contrast, highly detailed interpretations tied to component number and sign are more sensitive to basis choice and should not be presented as universal semantic directions.

## 2. Audited experimental scope

The fourth experiment is divided into three parts:

1. Sparsity/entropy threshold sensitivity for ImageNet epoch 1 and LeJEPA epoch 110.
2. Reanalysis of the same matched pair in the full CNN Filter DB global PCA basis instead of the former PCA fitted only to those two models.
3. Filter change throughout training using only the LeJEPA SSL backbone checkpoints at epochs `1, 10, 20, ..., 120`.

In addition, at the user's request, figure generation for the third experiment was rerun separately with the new global PCA basis without retraining either model. The selected comparison from the third experiment remains unchanged:

| Model | Checkpoint | Evaluation accuracy |
|---|---:|---:|
| ImageNet1K-pretrained, CIFAR10-fine-tuned RN18 | epoch 1 | `0.9125` |
| LeJEPA CIFAR10 SSL RN18 + frozen linear probe | SSL epoch 110 | `0.8913` |
| Absolute difference |  | `0.0212` |

These accuracies are not the target variable of the filter analysis. They were used only for checkpoint matching so that two models with a very large performance gap would not be compared.

## 3. Job history and the meaning of the canceled jobs

Not every Slurm job shown as canceled or `FAILED` represents a scientific failure. The job history should be interpreted as follows:

| Job | Status | Scientific meaning |
|---|---|---|
| `1460251` | canceled | This was the ImageNet replay started with the wrong `batch_size=256`; it was stopped after epoch 16 and explicitly separated as invalid under `*_batch256_invalid`. |
| `1460256` | `FAILED` | It processed all `1.464.797.156` filters. It exited without writing the artifact only because an excessively strict final-check tolerance of `5e-5` against the mean printed in the paper did not pass. The complete sufficient-statistics checkpoint remained intact. |
| `1460257` | completed | It regenerated 20/20 ImageNet checkpoints with the correct `batch_size=128`. The train/test loss, train/test accuracy, and learning-rate values at every epoch match the preserved third-experiment CSV exactly; only the file-path and wall-time fields are expected to differ. |
| `1460263` | completed | Initial threshold analysis; all `160/160` regression checks passed. |
| `1460264` | canceled | It never started because its `afterok:1460256` dependency was not satisfied, so it did not produce corrupted results. |
| `1470891` | completed | Fourth-experiment global-PCA ImageNet–LeJEPA pair analysis and PCA-independent controls. |
| `1470892` | completed | Figure/statistics-only rerun of the third experiment with the new global PCA basis. |
| `1470893` | completed | Updated threshold analysis, including the threshold-heatmap subdirectories and line-chart versions. |
| `1470905` | canceled | Duplicate threshold-figure job canceled before it started during queue reorganization; it produced no scientific output. |
| `1470908` | canceled | A duplicate job carrying the final line-chart scale adjustment was stopped after 12 seconds; the exact same job body was completed locally and all outputs were revalidated. |
| `1470909` | completed | Thirteen-checkpoint training-dynamics analysis using only the existing LeJEPA checkpoints; no new training was performed. |

The files from the incorrect batch-size run have been retained, but no final analysis uses them. The sole accepted ImageNet source path is:

```text
outputs/(4thEXP)rn18_cifar10_filter_dynamics/
  regenerated_imagenet_baseline/checkpoints/imagenet_ft_epoch001.pth
```

## 4. Audit of the global CNN Filter DB PCA basis

### 4.1. Why a global basis?

The former PCA basis in the third experiment was fitted to the union of the compared ImageNet and LeJEPA RN18 filters. That approach is internally valid because it places the two models in the same coordinate system, but the coordinate system depends on the models being compared. The new basis is fitted to the released CNN Filter DB population, which contains filters from many architectures, tasks, and datasets. Consequently:

- the two compared RN18 models do not determine the PCA axes;
- every LeJEPA checkpoint is projected onto the same fixed axes;
- training-time drift is easier to interpret because the basis is not refitted or rotated between checkpoints;
- it is possible to test whether the result from the old two-model basis persists in an external basis.

The paper reports 647 models, 21.436 convolution layers, and exactly `1.464.797.156` `3×3` filters. The `/filters` array in the local HDF5 file likewise has the form:

```text
(1464797156, 3, 3), dtype=float64
```

The sample range recorded in the artifact is `[0, 1464797156)`. Therefore, no filter range was skipped and no subset sampling was used.

### 4.2. Source-data provenance

The artifact records the following source information:

- DOI: `10.5281/zenodo.6371680`
- MD5 of the downloaded compressed source: `11d0ca3f9c3f7b6e0f73d389db35105a`
- HDF5 file size: `105.488.156.386` bytes
- Full filter count: `1.464.797.156`
- PCA artifact SHA256: `1706ba1b78db4115763bff5e34e02e21c393c6fb739ad20bedf4f6c30d06b135`

The model/layer filter ranges in the HDF5 metadata are contiguous from beginning to end; no gaps or overlaps were detected.

### 4.3. Preprocessing consistent with the paper and original repository

Equation 9 of the paper defines division of every `3×3` filter by its own largest absolute coefficient for filter-shape analysis:

\[
\widetilde f =
\begin{cases}
f/\max_j |f_j|, & \max_j |f_j|>0 \\
f, & \max_j |f_j|=0.
\end{cases}
\]

The current `main.ipynb` code path in the original repository first reshapes the raw HDF5 data into nine columns and casts it to `float16`, then applies per-filter maximum-absolute-value normalization. The recomputation order is therefore:

1. row-major transformation `[N,3,3] -> [N,9]`;
2. conversion to `float16`;
3. division of every row by its own `float16` maximum absolute coefficient;
4. accumulation of the chunk mean and centered cross-products in `float64`;
5. calculation of the sample covariance using `n-1`;
6. decomposition of the symmetric `9×9` covariance using `numpy.linalg.eigh`.

The main paper does not state that `float16` is mandatory; this is an implementation detail of the currently published repository notebook. It would therefore be inaccurate to call the artifact we use “the authors' unpublished bit-level original PCA file.” The accurate description is:

> A global PCA basis recomputed from every filter in the released CNN Filter DB data file, following the paper's maximum-absolute-value normalization and the dtype order in the current original repository.

The same `float16 -> float16 max-abs -> centering -> projection` sequence is applied when projecting RN18 target filters onto this basis. The float32 target-normalization mismatch present in the initial draft was detected during the audit and corrected before the final runs.

### 4.4. Why the streaming calculation is not approximate

Instead of holding 1.46 billion rows in a single RAM matrix, the sample count, mean, and centered sum-of-products matrix were calculated for each chunk. The sufficient statistics of two chunks were merged using:

\[
\delta=\mu_B-\mu_A,
\qquad
\mu=\mu_A+\delta\frac{n_B}{n_A+n_B},
\]

\[
M_2=M_{2,A}+M_{2,B}+\delta\delta^\top\frac{n_A n_B}{n_A+n_B}.
\]

The final sample covariance was calculated as:

\[
C=\frac{M_2}{n-1}.
\]

This method is algebraically identical to centering all rows at once and calculating `X^T X/(n-1)`. Neither `IncrementalPCA`, randomized SVD, nor a low-rank approximation was used. Because there are only nine features, a full eigendecomposition of the final `9×9` covariance was performed directly.

In a controlled small-data test, uninterrupted calculation and checkpoint/resume calculation produced bit-identical results. In independent sklearn PCA comparisons, the explained-variance ratios and component axes agreed to numerical precision. Chunking is therefore only a memory-management technique, not an approximation to the result.

### 4.5. Numerical validations

The explained-variance ratios of the new basis are:

```text
[0.39815034, 0.18326029, 0.11703953,
 0.09852558, 0.06576389, 0.05521558,
 0.03007510, 0.02823360, 0.02373609]
```

Audit results:

- EVR sum: `0.9999999999999999`
- Maximum component-orthonormality error: `4.718×10^-16`
- The covariance is symmetric
- The saved covariance is consistent with reconstruction by `V^T diag(lambda) V`
- All mean, covariance, eigenvalue, EVR, and component values are finite
- `n_samples == full_dataset_filter_count == 1.464.797.156`

### 4.6. Difference from the paper's centering vector

The paper supplement gives the following SVD/PCA centering vector for the full dataset:

```text
[-0.04262863, -0.04113670, -0.04461834,
 -0.04071190, -0.03574134, -0.04268694,
 -0.04350573, -0.04138637, -0.04486743]
```

The vector calculated from the released data using the current notebook preprocessing is:

```text
[-0.04223531, -0.04043981, -0.04412311,
 -0.04007022, -0.03468303, -0.04198876,
 -0.04310292, -0.04076126, -0.04435161]
```

| Position | Paper | Calculated | Absolute difference |
|---:|---:|---:|---:|
| 0 | -0.04262863 | -0.04223531 | 0.00039332 |
| 1 | -0.04113670 | -0.04043981 | 0.00069689 |
| 2 | -0.04461834 | -0.04412311 | 0.00049523 |
| 3 | -0.04071190 | -0.04007022 | 0.00064168 |
| 4 | -0.03574134 | -0.03468303 | 0.00105831 |
| 5 | -0.04268694 | -0.04198876 | 0.00069818 |
| 6 | -0.04350573 | -0.04310292 | 0.00040281 |
| 7 | -0.04138637 | -0.04076126 | 0.00062511 |
| 8 | -0.04486743 | -0.04435161 | 0.00051582 |

The maximum absolute difference is `0.00105831`. The `5×10^-5` tolerance in the first job was much stricter than this difference, so the Slurm job was marked `FAILED` only at the final safeguard. This does not mean that the data scan stopped early; the full-scan checkpoint completed with `next_index=1.464.797.156`.

The available evidence does not establish the exact cause of this difference. The paper calculation may have used a different internal snapshot, dtype, or small undocumented implementation detail. The difference is small on the unit scale of the normalized coefficients and does not compromise the internal consistency of the covariance. After the user accepted a difference of this magnitude, the final policy is as follows: the difference is not hidden, it remains stored as `published_mean_validation_passed=False`, and consumer analyses emit a prominent warning; however, the artifact is used because the full-data and numerical validations pass.

### 4.7. PCA sign ambiguity

If `v` is a PCA eigenvector, `-v` is an equally valid axis. The paper supplement likewise states that component flips are not a characteristic difference. For reproducible files, a deterministic sign convention was chosen so that the loading with the largest absolute value in each component is positive.

This choice:

- does not change the explained variance;
- does not change the symmetric-KL/TV drift magnitude because it is applied to both models simultaneously;
- may mirror a density graph along the corresponding axis;
- makes the sign of the signed `LeJEPA - ImageNet` mean difference dependent on the convention.

Component signs should therefore not be interpreted as physical “positive/negative feature” directions.

### 4.8. Additional sanity check of the directions and the limit of the comparison

The paper/repository does not publish a separate, machine-readable `9x9` PCA artifact that permits element-by-element comparison with the nine directions fitted by the authors. It is therefore not possible to claim that this result is “bit-identical to the authors' file.” The direct numerical anchor from the paper is the centering vector compared above, whose maximum difference is `0.00105831`. In contrast, several complementary pieces of evidence support the correctness of the direction calculation: the complete source range and preprocessing provenance, independent reconstruction of the full covariance, sklearn/SVD agreement on controlled subsets, orthonormality/EVR checks, and direction visualizations consistent with the published eigenfilter family.

As an additional sanity check, the new global directions were matched to the two-model PCA directions from the third experiment by absolute cosine similarity:

| Global component | Closest old component | Absolute cosine |
|---:|---:|---:|
| 0 | 0 | 0.9989 |
| 1 | 1 | 0.9992 |
| 2 | 3 | 0.9731 |
| 3 | 2 | 0.9712 |
| 4 | 4 | 0.9979 |
| 5 | 5 | 0.9991 |
| 6 | 6 | 0.9991 |
| 7 | 7 | 0.9990 |
| 8 | 8 | 0.9992 |

Seven directions are nearly coincident; the principal difference is that old directions 2 and 3 exchange order in the global basis and rotate slightly within the same two-dimensional subspace. The cumulative EVR of the first two components is `0.581737` in the old basis and `0.581411` in the global basis; for the first four, the corresponding values are `0.794582` and `0.796976`. This similarity does not prove the global calculation by itself, because the old basis is also specific to the RN18 pair; it is nevertheless a strong cross-check against major implementation errors such as an incorrect reshape, the wrong centering axis, or a completely different normalization.

Given the user's instruction that a small difference is acceptable, the calculation was not reopened to pursue another uncertain dtype combination. The current artifact was accepted with the small paper-mean difference explicitly marked, and every final consumer was connected to this single fully checksummed artifact.

## 5. Why was the ImageNet baseline regenerated for 20 epochs, and where is it used?

### 5.1. The missing item was the weight file, not the metric

The selected ImageNet epoch-1 accuracy, the old metric CSVs, and the old analysis tables from the third experiment had been preserved. However, the original `imagenet_ft_epoch001.pth` weight file needed to regenerate the figures was no longer present after directory renaming/cleanup. The objective was therefore not to design a new baseline, but to recover the same previously selected model state.

### 5.2. Why not only one epoch?

The original supervised fine-tuning protocol used 20 epochs and `CosineAnnealingLR(T_max=20)`. Running only through epoch 1 could have produced a weight file, but it would not have fully replayed the original run with the same total training horizon and scheduler configuration. Replaying all 20 epochs provided the following strong validation:

- seed `13` was preserved;
- the correct batch size of `128` was used;
- the same ImageNet1K_V1 RN18 initialization and CIFAR stem were used;
- all train/test loss, train/test accuracy, and learning-rate values over 20 epochs matched the preserved third-experiment CSV exactly;
- the best accuracy was again `0.9595` at epoch 19, epoch-20 accuracy was again `0.9591`, and epoch-1 accuracy was again `0.9125`.

Agreement across this complete trajectory is stronger evidence of reproducibility than agreement of a single accuracy value.

### 5.3. Which checkpoint is actually used?

Only the regenerated ImageNet epoch-1 weights are used in the current final analyses. They are used in:

1. Fourth experiment, Part 1: comparison with LeJEPA epoch 110 in the threshold-sensitivity analysis.
2. Fourth experiment, Part 2: ImageNet–LeJEPA pair analysis in the full CNN Filter DB PCA basis.
3. Regeneration of the third experiment's figures/statistics with the global PCA basis.
4. PCA-independent raw, BN-folded, normalized-shape, and descriptor control analyses.

The epoch 2–20 checkpoints are not used in the current main comparison. They are retained to validate the replay trajectory, preserve archive completeness, and reduce the need to retrain if another epoch is required in the future. The LeJEPA-only training-dynamics section does not use an ImageNet checkpoint at all.

The short answer is:

> The actual need was to recover the epoch-1 weights. Replaying all twenty epochs provided a regression check proving that the original run was genuinely reproduced with the same total training horizon and scheduler. The current comparisons use only epoch 1.

## 6. How are the sparsity and entropy thresholds used in the calculations?

### 6.1. There is no PCA dependency

Every residual `3×3` convolution tensor is reshaped from `[c_out,c_in,3,3]` into:

\[
X_L\in\mathbb{R}^{n_L\times 9},\qquad n_L=c_{out}c_{in}.
\]

Entropy is computed from the local layer covariance, and sparsity is computed from raw-weight magnitudes. The global CNN Filter DB PCA components are not inputs to either metric.

### 6.2. Entropy H

The filters in a layer are centered around their own nine-dimensional mean. Variance ratios are calculated from the SVD singular values, or equivalently from the covariance eigenvalues:

\[
\lambda_i=\frac{\sigma_i^2}{n_L-1},\qquad
p_i=\frac{\lambda_i}{\sum_j\lambda_j}.
\]

The base-10 Shannon entropy used in the paper is:

\[
H(L)=-\sum_{i=0}^{8}p_i\log_{10}p_i.
\]

When the nine directions have equal variance, the theoretical maximum is `log10(9)≈0.95424`. A low `H` means that layer-filter variance is concentrated in a small number of directions; a high `H` means it is distributed more evenly.

The baseline `H` is calculated only once for each layer. The `0.30,...,0.90` cutoff sweep does not change the SVD or the value of `H`; it changes only the decision:

\[
\text{low-entropy}(L;c)=\mathbf{1}[H(L)<c].
\]

As the cutoff rises, the number of marked layers increases monotonically.

### 6.3. Randomness reference TH(n)

The reference fitted by the paper to its random-normal-filter experiments is:

\[
TH(n)=\frac{1.26}{1+\exp[-0.89(\log_2 n-2.30)]}-0.31.
\]

The paper's direct random rule is `H>TH(n)`, or equivalently `H/TH>1`. The analysis first calculates the fixed ratio:

\[
R(L)=H(L)/TH(n_L),
\]

and then changes only the `R>cutoff` decision for cutoffs `0.75,0.80,0.85,0.90,0.95,1.00,1.05`. `1.00` is the paper's actual boundary, while `0.95` is the more permissive “random-like” heuristic used in the previous experiment.

The combined degeneration-ablation criterion in the paper supplement is:

\[
(H\ge TH-0.02)\ \lor\ [(H<0.5)\land(S\ge0.14)].
\]

Our former `H/TH>0.95` flag is not identical to this criterion; it is a separate sensitivity heuristic. Including `1.00` in the new sweep also makes the paper's direct random boundary visible.

### 6.4. Sparsity epsilon

First, the peak magnitude among all raw coefficients in the relevant layer is found:

\[
M_L=\max_{f\in L,j}|f_j|.
\]

For each epsilon, the actual absolute near-zero threshold is:

\[
\tau_L(\epsilon)=\epsilon M_L.
\]

A `3×3` filter is considered sparse only if all nine coefficients are below this threshold:

\[
\text{sparse}(f;\epsilon)=
\mathbf{1}[\forall j, |f_j|<\tau_L(\epsilon)].
\]

The layer sparsity ratio is calculated as:

\[
S(L;\epsilon)=\frac{\#\text{sparse filters}}{n_L}.
\]

The epsilons used are:

```text
0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10
```

`epsilon=0.01` does not mean that the absolute weight threshold is `0.01`; it means one percent of the layer's peak absolute weight. For example, if `M_L=0.8`, the actual threshold is `0.008`.

Two different uses of one percent must not be confused:

- `epsilon=0.01`: the ratio of the coefficient near-zero threshold to the layer peak;
- `S>0.01`: more than one percent of the filters in the layer are sparse.

As epsilon increases, the near-zero range expands, so the sparse-filter set cannot shrink. The results likewise show `S(epsilon)` to be monotonically increasing or constant for every model/layer.

### 6.5. Clean entropy

Baseline `H` is calculated without removing sparse filters and remains fixed throughout the cutoff sweep. The separately named clean-entropy diagnostic instead performs the following for each epsilon:

1. identifies the epsilon-sparse filters;
2. removes those filters from the layer samples;
3. recomputes covariance entropy from the remaining filters.

\[
H_{clean}(L;\epsilon)=H(\{f\in L:\neg sparse(f;\epsilon)\}).
\]

Thus, baseline entropy is not epsilon-dependent; the separate post-exclusion `H_clean` measurement is. `H_clean` need not be monotonic.

### 6.6. What the thresholds are not

These thresholds are not model-training hyperparameters: they do not modify the weights, perform pruning, change the PCA basis, or by themselves prove accuracy or representation quality. Their purpose is to show how sensitive the “sparse/low-entropy/random-like” decision is to the selected numerical boundary.

## 7. Part 1: threshold-sensitivity results

The analysis excludes the stem and examines 16 residual-block `3×3` layers per model. The baseline summary is:

| Model | Mean H | Min H | Max H | Mean H/TH | epsilon=0.01 mean S | epsilon=0.01 max S |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet FT epoch 1 | 0.731123 | 0.209724 | 0.872600 | 0.769668 | 0.007925 | 0.125244 |
| LeJEPA SSL epoch 110 | 0.804338 | 0.688594 | 0.888862 | 0.846737 | 0.000000 | 0.000000 |

For both models, the minimum entropy occurs in `layer4.1.conv2.weight`. However, `H=0.209724` is a very pronounced low-diversity condition for ImageNet, whereas the same LeJEPA layer has `H=0.688594`. ImageNet's maximum entropy occurs in `layer2.0.conv1`, while LeJEPA's maximum occurs in `layer2.0.conv2`.

### 7.1. Sparsity epsilon sweep

| Epsilon | ImageNet mean S | ImageNet max S | ImageNet layers with S>0 | LeJEPA mean S | LeJEPA max S | LeJEPA layers with S>0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0010 | 0.007812 | 0.125000 | 1 | 0.000000 | 0.000000 | 0 |
| 0.0025 | 0.007813 | 0.125000 | 2 | 0.000000 | 0.000000 | 0 |
| 0.0050 | 0.007814 | 0.125000 | 2 | 0.000000 | 0.000000 | 0 |
| 0.0100 | 0.007925 | 0.125244 | 4 | 0.000000 | 0.000000 | 0 |
| 0.0200 | 0.011669 | 0.133057 | 15 | 0.000002 | 0.000015 | 3 |
| 0.0500 | 0.106536 | 0.477062 | 16 | 0.001589 | 0.007080 | 15 |
| 0.1000 | 0.504424 | 0.901447 | 16 | 0.080413 | 0.229980 | 16 |

The most important pattern is that ImageNet shows approximately `12.5%` sparse filters in a particular layer even at small epsilons, whereas LeJEPA measures zero in all residual layers for `epsilon<=0.01`. At very large epsilon values, sparsity rises in both models because the definition begins to classify increasingly many small but nonzero filters as “sparse.” Nevertheless, even with a very broad boundary such as `epsilon=0.10`, the ImageNet mean is `0.5044`, compared with `0.0804` for LeJEPA.

This shows that the sparsity difference at the selected checkpoints is not specific only to the exact `1%` threshold. However, `epsilon=0.05`, and especially `0.10`, are now very broad near-zero definitions; these rows should be read as the upper end of threshold sensitivity, not as the proportion of genuinely “prunable zero filters.”

### 7.2. Low-entropy cutoff sweep

| H cutoff | ImageNet marked /16 | LeJEPA marked /16 |
|---:|---:|---:|
| 0.30 | 1 | 0 |
| 0.40 | 1 | 0 |
| 0.50 | 1 | 0 |
| 0.60 | 2 | 0 |
| 0.70 | 4 | 1 |
| 0.80 | 9 | 6 |
| 0.90 | 16 | 16 |

ImageNet's low-entropy result for `layer4.1.conv2` is stable throughout the `0.30–0.50` range. LeJEPA has no layer with `H<0.50`. When the cutoff is raised to `0.8–0.9`, it approaches the theoretical maximum of `0.95424`, and the “low entropy” label loses its selectivity; nearly every layer is marked. The purpose of the sweep is therefore not to declare one correct cutoff, but to show where the decision is stable and where it becomes arbitrary.

### 7.3. H/TH random-like cutoff sweep

| H/TH cutoff | ImageNet marked /16 | LeJEPA marked /16 |
|---:|---:|---:|
| 0.75 | 12 | 15 |
| 0.80 | 10 | 12 |
| 0.85 | 6 | 10 |
| 0.90 | 1 | 2 |
| 0.95 | 0 | 0 |
| 1.00 | 0 | 0 |
| 1.05 | 0 | 0 |

At the paper's direct `H>TH` boundary (`H/TH>1`), neither model has any layer marked as random. The result is also zero at the former, more permissive `0.95` boundary. However, when the cutoff is lowered to values such as `0.85`, LeJEPA approaches the random reference in more layers. This does not mean that “LeJEPA layers are random”; none crosses the paper boundary. A more accurate interpretation is that LeJEPA's local variance spectrum is generally more even and closer to the random reference than ImageNet's, while remaining below that reference.

### 7.4. Clean entropy

At small epsilons, removing sparse filters changes the baseline entropy almost not at all. Under the broadest `epsilon=0.10` condition, the mean clean-H difference is approximately `-0.0322` for ImageNet and `-0.0030` for LeJEPA. The largest absolute single-layer change is approximately `0.1046` for ImageNet and `0.0114` for LeJEPA. This shows that ImageNet's entropy profile is more affected by filters removed under the broad sparsity definition. However, this is not part of the baseline `H` sweep; it is a separate post-exclusion diagnostic.

### 7.5. Part 1 figures

The four heatmaps were retained. A subdirectory containing seven individual charts was created for each heatmap:

```text
figures/sparsity_threshold_by_depth_heatmap/                 # 7 epsilon charts
figures/low_entropy_decision_heatmap/                        # 7 H-cutoff charts
figures/random_like_decision_heatmap/                        # 7 H/TH-cutoff charts
figures/clean_entropy_after_sparse_exclusion_heatmap/        # 7 epsilon charts
```

Rather than plotting only a `0/1` line, the chart counterparts of the decision heatmaps more informatively show the fixed continuous `H` or `H/TH` curves for ImageNet and LeJEPA together with the relevant horizontal cutoff line.

## 8. Part 2: ImageNet–LeJEPA pair result in the global PCA basis

The PCA analysis excludes the stem and uses `1.220.608` residual-block `3×3` filters per model. The overall result in the new global basis is:

```text
Global drift D = 0.1330218026
```

Stage results:

| Stage | Global-basis drift D |
|---|---:|
| layer1 | 0.070725 |
| layer2 | 0.011124 |
| layer3 | 0.017624 |
| layer4 | 0.193849 |

The highest layer drifts are:

| Layer | Stage | Drift D |
|---|---|---:|
| `layer4.1.conv2.weight` | layer4 | 0.779966 |
| `layer1.0.conv1.weight` | layer1 | 0.285191 |
| `layer1.0.conv2.weight` | layer1 | 0.235985 |
| `layer1.1.conv1.weight` | layer1 | 0.181766 |
| `layer4.1.conv1.weight` | layer4 | 0.125888 |
| `layer1.1.conv2.weight` | layer1 | 0.106079 |
| `layer2.0.conv1.weight` | layer2 | 0.096252 |
| `layer2.0.conv2.weight` | layer2 | 0.071522 |

The top five components in the global weighted symmetric-KL contribution are `c0`, `c2`, `c5`, `c7`, and `c6`. The `c0` contribution of `0.102808` accounts for a large part of the total difference. However, the component orderings in the old two-model PCA and the new global PCA do not have identical semantics; for example, the old `c3` direction may be closest to another component number in the new basis. Interpretations based only on a number, such as “c2 is biologically/spatially this,” should therefore be avoided.

### 8.1. Comparison with the old two-model PCA

The global drift in the old dedicated no-stem analysis was `0.128806`. The new value is `0.133022`:

| Basis | Global D |
|---|---:|
| PCA fitted only to the two matched RN18 models | 0.128806 |
| Full CNN Filter DB global PCA | 0.133022 |
| Absolute difference | +0.004216 |
| Relative difference | approximately +3.27% |

Stage comparison:

| Stage | Old local PCA | New global PCA | Change |
|---|---:|---:|---:|
| layer1 | 0.079488 | 0.070725 | -0.008763 |
| layer2 | 0.015189 | 0.011124 | -0.004065 |
| layer3 | 0.017935 | 0.017624 | -0.000311 |
| layer4 | 0.185562 | 0.193849 | +0.008287 |

The main absolute ordering is preserved: layer4 is the most different, layer1 is second, and the middle layer2/layer3 stages are closer. In particular, the `layer4.1.conv2` drift is `0.779819` in the old basis and `0.779966` in the new basis, so it is almost unchanged. The strong difference in this layer does not appear to be an artifact of the PCA basis.

Individual magnitudes within early layer1 change more. This is expected: when the component directions and their EVR weights change, histogram drift is redistributed quantitatively. The most robust conclusion is therefore the pattern of a secondary cluster of differences in the early blocks and a very strong difference in the final block, rather than the exact value of a single layer1 sublayer.

### 8.2. PCA-independent controls

All raw-weight, BN-folded-weight, per-kernel normalized-shape, kernel-descriptor, BN-scale, and spatial-energy tables are exactly identical to the old third experiment. The maximum numerical difference across the six CSV families is `0.0`. The main results of these controls remain unchanged:

- LeJEPA's raw-weight scale is much larger than ImageNet's, but the ordering reverses after BatchNorm folding.
- Raw-weight magnitude alone therefore cannot be interpreted as a functional difference.
- After magnitude is removed, the global normalized-shape TV difference is approximately `0.0531`.
- At the selected checkpoint, LeJEPA has a higher high-frequency ratio and roughness and a lower center-energy ratio.
- Normalized-shape/descriptor differences are concentrated in the stem and final residual block; the middle blocks are mostly more similar.

It is expected that these numbers do not change with the new PCA basis because these analyses do not use PCA coefficients. The exact agreement is also strong regression evidence that the regenerated ImageNet epoch-1 checkpoint fully recovers the weight behavior from the old analysis.

### 8.3. Heatmap and chart versions

The PCA drift heatmaps were retained, and `drift_global_chart.png`, `drift_stage_<stage>_chart.png` for every stage, and `drift_stage_chart.png`, which displays all stages together, were added. In the kernel-distribution section:

- `layerwise_weight_distance_chart.png` alongside `layerwise_weight_distance_heatmap.png`;
- `layerwise_descriptor_distance_chart.png` alongside `layerwise_descriptor_distance_heatmap.png`;
- a separate line-chart subdirectory for every global/stage scope alongside `spatial_energy_maps.png`

were produced.

## 9. Figure-only rerun of the third experiment with global PCA

The new output is located under:

```text
outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/
  analysis_global_cnn_filter_db_basis_epoch001_vs_epoch110/
```

This job did not call a training script; it loaded only the two existing checkpoints and produced analyses/figures. The output contains 63 files and 39 PNGs. All 18 CSVs comparable with Fourth Experiment Part 2 have the same shapes, columns, and numerical values, with a maximum numerical difference of `0.0`. There is therefore no computational divergence between the pair result in the fourth experiment and the rerun under the third-experiment directory.

The old outputs were not overwritten. This is important because the two-model local-PCA result and the new global-PCA result remain available for side-by-side audit.

## 10. Part 3: LeJEPA training dynamics

### 10.1. Scope and calculation protocol

No ImageNet model is used in this section. The objective is not to compare two models, but to track how the filter distribution of a single LeJEPA SSL backbone changes throughout training in the same external coordinate system. The analysis uses exactly the following checkpoints:

```text
1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120
```

Every checkpoint contains the same 16 residual `3x3` convolution layers, excluding the stem, and a total of `1.220.608` filters. The layer names, tensor shapes, and filter ranges are identical across all 13 checkpoints. The analysis did not update any checkpoint, produce intermediate checkpoints, interpolate missing epochs, or train a model. The job reads only the existing LeJEPA weights in the third-experiment directory.

Consistent with the global artifact, every target filter is first cast to `float16`, divided by its own `float16` maximum absolute coefficient, centered using the global mean, and projected onto the fixed nine PCA directions. PCA is not refitted at each epoch. This is a critical design choice: a PCA axis that rotates from epoch to epoch would mix genuine distribution movement with movement of the coordinate system.

Drift uses 70 histogram bins per component. Bin boundaries are not selected again for every checkpoint pair. First, the combined minimum and maximum of the relevant PCA component across all 13 checkpoints are found, a small outer padding is added, and the resulting single boundary set is reused for every global, stage, and layer scope. Thus, for example, `D(e1,e20)` and `D(e80,e100)` are computed over the same numerical partitions. This is why the saved `fixed_pca_histogram_edges.csv` table contains `9 x 71 = 639` rows.

The total value between two epochs is:

\[
D(a,b)=\sum_{k=0}^{8} w_k
\left[KL(P_{a,k}\|P_{b,k})+KL(P_{b,k}\|P_{a,k})\right],
\]

where `w_k` is the explained-variance ratio of the full CNN Filter DB basis, and `P_{a,k}` is the component-`k` histogram for epoch `a`. This version uses the sum of KL in both directions; it is not a percentage, an accuracy difference, or a bounded distance. It is a comparative distribution-drift measure when the same protocol is retained.

### 10.2. Job and output validation

The LeJEPA dynamics completed as Slurm job `1470909`: status `COMPLETED`, exit code `0:0`, duration `00:01:17`. The final directory is:

```text
outputs/(4thEXP)rn18_cifar10_filter_dynamics/lejepa_training_dynamics/
```

It contains 15 CSVs, 25 PNGs, two provenance JSON files, and one summary file. The main table sizes are:

| Output | Rows | Expected structure |
|---|---:|---|
| `global_pairwise_epoch_drift.csv` | 169 | `13 x 13` epoch pairs |
| `global_pairwise_component_contributions.csv` | 1.521 | `13 x 13 x 9` components |
| `global_drift_trajectories.csv` | 13 | one row per checkpoint |
| `stage_drift_trajectories.csv` | 52 | `13 x 4` stages |
| `layer_drift_trajectories.csv` | 208 | `13 x 16` layers |
| `global_component_drift_contributions.csv` | 351 | `13 x 3 x 9` reference types/components |
| `layer_quality_by_epoch.csv` | 208 | `13 x 16` local-quality diagnostics |
| `distribution_summary_by_epoch.csv` | 3.549 | raw, BN-folded, shape, and descriptor summaries |
| `spatial_energy_by_epoch.csv` | 1.755 | global/stage `3x3` energy maps |
| `pca_component_statistics.csv` | 117 | `13 x 9` component summaries |
| `quality_aggregate_by_epoch.csv` | 39 | `13 x 3` entropy/H-TH/sparsity summaries |
| `milestone_scalar_histograms.csv` | 4.550 | fixed-bin histograms for five milestones |
| `dynamics_training_context.csv` | 13 | SSL-loss and probe context |
| `fixed_pca_histogram_edges.csv` | 639 | `9 x 71` fixed boundaries |
| `validation_checks.csv` | 9 | eight required passes and one warning |

The audit did not rely only on the script's own `validation_checks.csv`. The following checks were also repeated independently:

- The SHA256 value of all 13 source checkpoints was recalculated and matched the provenance.
- Each checkpoint was verified to exclude the stem, contain 16 residual layers, and contain `1.220.608` filters.
- The symmetry error of the `13x13` drift matrix is exactly `0`, and its maximum diagonal value is exactly `0`.
- For every epoch pair, the sum of the nine component contributions reconstructs the global `D` value with at most `4.996x10^-16` error.
- The fixed 71 boundaries are strictly increasing for every component, cover the union projection range, and leave no coefficient outside the histogram.
- When all 13 checkpoints were reprojected and their histograms independently reconstructed, the saved pairwise matrix was reproduced with at most `9.78x10^-17` difference.
- The maximum histogram probability-sum error is `2.22x10^-16`; the maximum milestone-histogram sum error is `3.77x10^-15`.
- All 25 PNG files were opened, and representative heatmap/chart/trajectory outputs were also inspected visually.
- There are no `NaN` or infinite values in the core numerical tables. Some empty `stage/layer_order` fields in the descriptor table occur only because those metadata fields are structurally inapplicable to the global scope.

All eight required validation checks passed. The only `False` row is `published_mean_matches_supplement`, whose severity is explicitly `warning`. The accepted maximum paper-mean difference of `0.00105830942` remains visible in the validation CSV, both provenance JSON files, and the section summary. This row is therefore not a hidden dynamics-computation error.

### 10.3. Global drift trajectory

The complete global results are:

| Epoch | D from epoch 1 | D from previous saved checkpoint | D to epoch 120 |
|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 0.419884 |
| 10 | 0.172998 | 0.172998 | 0.053262 |
| 20 | 0.341305 | 0.027179 | 0.005730 |
| 30 | 0.417520 | 0.003989 | 0.000948 |
| 40 | 0.441594 | 0.000520 | 0.000678 |
| 50 | 0.442967 | 0.000182 | 0.000560 |
| 60 | 0.437054 | 0.000203 | 0.000340 |
| 70 | 0.431374 | 0.000180 | 0.000227 |
| 80 | 0.426087 | 0.000124 | 0.000123 |
| 90 | 0.423185 | 0.000102 | 0.000096 |
| 100 | 0.420848 | 0.000112 | 0.000042 |
| 110 | 0.420222 | 0.000042 | 0.000007 |
| 120 | 0.419884 | 0.000007 | 0.000000 |

The total global drift between epoch 1 and epoch 120 is `0.419884`. The largest change between consecutive saved checkpoints is `0.172998` over the epoch 1->10 interval; it falls to `0.027179` over 10->20 and `0.003989` over 20->30. At epoch 30, the remaining distribution distance to the final checkpoint is only `0.000948`. Most of the global histogram organization observed in the fixed global PCA coordinates therefore occurs during the first 20–30 epochs.

The first interval spans nine epochs, from 1 to 10, while subsequent saved intervals mostly span ten epochs. The consecutive values should therefore not be read as a precise “rate per epoch.” Nevertheless, the fact that the first two intervals are one to two orders of magnitude larger than the later values cannot be explained only by this one-epoch difference in interval length.

`D(e1,e)` is not monotonic. The distance from epoch 1 reaches its maximum of `0.442967` at epoch 50 and then falls slightly to `0.419884` at the final checkpoint. This describes an “overshoot/relaxation” pattern in which the distribution moves away from the initial state and later partially approaches the same initial distribution again. Because symmetric KL is not a metric satisfying the triangle inequality, this return should not be interpreted as a linear path length or as individual filters moving backward.

It is also inappropriate to directly compare the magnitude of the `0.419884` value with the ImageNet–LeJEPA value of `D=0.133022` from Part 2. Even though the PCA basis is the same, Part 2 establishes its bin bounds from the two-model pair, whereas the dynamics analysis fixes them from the union range of 13 checkpoints. The strong result here is the time pattern within a single dynamics protocol, not the ratio of the two absolute numbers.

### 10.4. Movement at stage and layer level

The epoch 1->120 stage results are:

| Stage | Filter count | Share of global pool | D(e1,e120) |
|---|---:|---:|---:|
| layer1 | 16.384 | 1.34% | 0.529162 |
| layer2 | 57.344 | 4.70% | 0.236437 |
| layer3 | 229.376 | 18.79% | 0.359590 |
| layer4 | 917.504 | 75.17% | 0.466528 |

The largest stage drift is in `layer1`, followed by `layer4`. However, because layer4 filters account for approximately three quarters of the global pool, the global `D` is naturally more sensitive to deep-layer distributions. The global value is not a simple filter-weighted mean of the stage values; KL is computed after the histograms are pooled, so the operation is nonlinear. Nevertheless, the filter-count imbalance explains why the global graph is insufficient by itself.

The time patterns also differ by stage. At epoch 10, layer4 shows strong early movement of `0.190914` and layer3 shows `0.163763`, whereas layer1 is only `0.063439`. Layer4 reaches approximately `0.497` around epoch 40 and then relaxes to `0.466528`. In contrast, layer1 continues to change for longer: `0.063439` at epoch 10, `0.313164` at epoch 30, `0.472242` at epoch 50, and `0.529162` at the final checkpoint. Thus, the statement that “the global distribution stabilizes around epoch 30” does not mean that all layers freeze simultaneously; continued movement in early layers containing relatively few filters can be suppressed in the global pool.

The largest layer drifts at the final endpoint are:

| Layer | Stage | D(e1,e120) |
|---|---|---:|
| `layer1.0.conv2.weight` | layer1 | 1.128378 |
| `layer1.1.conv2.weight` | layer1 | 0.941002 |
| `layer1.0.conv1.weight` | layer1 | 0.847634 |
| `layer4.1.conv2.weight` | layer4 | 0.776601 |
| `layer1.1.conv1.weight` | layer1 | 0.762926 |
| `layer3.0.conv1.weight` | layer3 | 0.606682 |
| `layer3.1.conv2.weight` | layer3 | 0.600993 |
| `layer2.0.conv1.weight` | layer2 | 0.587541 |

`layer1.0.conv2` has the largest final change and relaxes to its final value of `1.128` from a higher point of approximately `1.167` around epoch 100. `layer4.1.conv2` likewise rises to approximately `0.837` at epoch 50 before falling to `0.777`. These layer-level overshoot examples are consistent with the slight return in the global curve.

There is no contradiction between `layer4.1.conv2` being the strongest ImageNet–LeJEPA difference in Part 2 and `layer1.0.conv2` moving the most during LeJEPA's own training. The first measures the endpoint difference between two different training protocols; the second measures time change within one protocol relative to epoch 1.

### 10.5. Which global PCA components carry the change?

The component contributions to the epoch 1->120 global drift are:

| Component | Weighted symmetric KL | Share of total D |
|---:|---:|---:|
| c0 | 0.309096 | 73.615% |
| c1 | 0.050059 | 11.922% |
| c8 | 0.020019 | 4.768% |
| c7 | 0.013642 | 3.249% |
| c6 | 0.013483 | 3.211% |
| c5 | 0.005855 | 1.395% |
| c4 | 0.003629 | 0.864% |
| c2 | 0.003389 | 0.807% |
| c3 | 0.000711 | 0.169% |

The first two components carry `85.54%` of the total drift. The sign-canonicalized `c0` visually has similar signs across all `3x3` coefficients and is a DC/mean-like direction, while `c1` is primarily a spatial-contrast direction. However, PCA directions are statistical axes and their signs are conventional; this analysis alone is insufficient to assign them learned semantic-feature labels.

The component-coefficient scales also show variance becoming concentrated in the leading directions. The standard deviation of `c0` rises from `0.5811` at epoch 1 to `0.9448` at the final checkpoint, while `c1` rises from `0.5613` to `0.7728`. In contrast, the standard deviations of `c6`, `c7`, and `c8` change by approximately `0.5367->0.3367`, `0.5377->0.3302`, and `0.5377->0.2768`, respectively. Training therefore does not merely increase overall scale; it redistributes normalized filter-shape variance toward particular global directions.

### 10.6. Changes in entropy and sparsity throughout training

Local layer entropy is independent of PCA. The equal-weight mean across layers is:

| Epoch | Mean H | Filter-weighted H | Mean H/TH | Mean S, epsilon=0.01 |
|---:|---:|---:|---:|---:|
| 1 | 0.953554 | 0.953275 | 1.003817 | 0.000000 |
| 10 | 0.898432 | 0.862468 | 0.945792 | 0.000000 |
| 20 | 0.848626 | 0.793140 | 0.893362 | 0.000000 |
| 30 | 0.824790 | 0.771432 | 0.868270 | 0.000000 |
| 50 | 0.807711 | 0.767450 | 0.850289 | 0.000000 |
| 80 | 0.803970 | 0.772406 | 0.846349 | 0.000000 |
| 110 | 0.804338 | 0.774045 | 0.846737 | 0.000000 |
| 120 | 0.804359 | 0.774083 | 0.846759 | 0.000000 |

Because the theoretical maximum over nine directions is `log10(9)=0.95424`, the epoch-1 mean is very close to the maximum. At epoch 1, the `H/TH` ratios of all 16 layers lie between `1.0031–1.0043` and slightly exceed the paper's direct random reference. By epoch 10, the range has fallen to `0.8750–0.9920`; no layer now satisfies `H/TH>1`. This shows that the local second-order filter statistic of the epoch-1 checkpoint is very close to the random reference, and that variance becomes concentrated in a smaller number of structural directions as training progresses. Because epoch 1 is a checkpoint after one epoch of training, not a random-initialization file, this should not be overgeneralized into the claim that its filters are “completely random.”

Most of the entropy decline again occurs within the first 30 epochs: `0.953554->0.824790`. The layer mean then plateaus around `0.804`. At the final checkpoint, the lowest entropy is `H=0.688687` in `layer4.1.conv2`, and the highest is `H=0.888859` in `layer2.0.conv2`. The closeness of the epoch-120 numbers to the epoch-110 Part 1 values is consistent with the expected late plateau.

Under the legacy `epsilon=0.01` definition, sparsity is exactly zero across all 13 checkpoints and all 16 residual layers. This does not mean that there are no small weight coefficients. For a filter to count as sparse, all nine of its coefficients must be smaller than one percent of the peak of its own layer. The result says only that this strong filter-level near-zero condition never occurs along the LeJEPA training trajectory.

### 10.7. Raw weights, BatchNorm, and shape descriptors

PCA drift was not interpreted in isolation. The PCA-independent global endpoint summaries are:

| Metric | Epoch 1 | Epoch 120 | Important point for interpretation |
|---|---:|---:|---|
| Mean raw kernel L2 | 0.07149 | 0.39805 | The raw convolution norm grows strongly. |
| Mean BN-folded kernel L2 | 0.04543 | 0.07111 | The increase on a scale closer to functional behavior is much smaller. |
| Mean absolute BN scale | 0.65330 | 0.17622 | The BN scale partially compensates for raw-weight growth. |
| High-frequency ratio | 0.87512 | 0.70721 | The very high initial frequency content decreases. |
| Spatial roughness | 2.59997 | 1.56371 | Normalized filter shapes become smoother. |
| Orientation anisotropy | 0.23420 | 0.35878 | Directional inequality increases. |
| Center-energy ratio | 0.11078 | 0.11104 | It is nearly constant in the global mean. |

Raw kernel L2 grows by approximately `5.57` times, whereas BN-folded L2 grows by only approximately `1.57` times. This divergence is a direct example of Conv–BatchNorm scale freedom. An interpretation such as “the weight norm grew, so the model learned stronger filters” is therefore incomplete; the result must be read together with the BN parameters.

The decline in high-frequency content and roughness, together with the rise in orientation anisotropy, suggests that an initially more noise-like and directionally balanced filter population becomes organized into smoother but more directional shapes through training. The unchanged mean center energy shows that not all structural change can be explained by moving energy to the center coefficient. These remain population summaries, not evidence of individual-filter semantics or causal performance contributions.

### 10.8. Reading the result together with SSL loss and probe accuracy

The checkpoint-aligned context is:

| Epoch | SSL loss | Quick probe | Fixed-probe final accuracy |
|---:|---:|---:|---:|
| 1 | 0.537856 | 0.4662 | 0.5151 |
| 10 | 0.254591 | 0.7522 | 0.7764 |
| 20 | 0.223640 | 0.8042 | 0.8208 |
| 30 | 0.206163 | 0.8271 | 0.8408 |
| 60 | 0.173338 | 0.8639 | 0.8746 |
| 90 | 0.155432 | 0.8803 | 0.8905 |
| 110 | 0.150305 | 0.8849 | 0.8913 |
| 120 | 0.148598 | 0.8847 | 0.8905 |

Between epochs 30 and 120, the global histogram's distance to the final checkpoint is only `0.000948`, while quick-probe accuracy rises from `0.8271->0.8847` and fixed-probe final accuracy rises from `0.8408->0.8905`. In other words, representation quality improves by approximately five percentage points after the coarse global normalized-filter distribution has stabilized early.

This does not show that filter change is unimportant for performance. Histogram analysis does not measure channel correspondence, the inputs on which particular filters activate, channel composition, the subsequent nonlinear function, or small but task-important parameter movements. The more accurate inference is that late-stage accuracy gains do not require large-scale reorganization of the global `3x3` filter-coefficient histogram alone.

At epoch 110, quick probe reaches `0.8849` and fixed-probe final reaches `0.8913`, their peaks among the saved checkpoints; the epoch-120 values are slightly lower at `0.8847` and `0.8905`, respectively. This is consistent with selecting epoch 110 for the third experiment. However, the difference is small and comes from a single run; it is not evidence of a sharp optimum.

The fact that loss continues to fall while the probe plateaus and filter drift becomes small much earlier shows that the three curves measure different objects. Their early co-movement does not imply that a particular PCA component causes accuracy.

### 10.9. Main interpretation of Part 3

At the beginning of training, LeJEPA's residual `3x3` filter population shows entropy near the theoretical maximum and a local covariance structure close to the random reference. A large reorganization occurs in the fixed global PCA coordinates during the first 20–30 epochs: entropy falls, variance becomes concentrated in the leading global components, high-frequency content/roughness declines, and directional anisotropy rises. The global distribution then changes very little, while probe accuracy and SSL loss continue to improve.

This movement is not simultaneous and homogeneous across the network. Layer4, which contains many filters, strongly affects the global curve and organizes early; layer1, which has few filters, continues to change for longer and has the largest stage drift at the final endpoint. Global, stage, and layer graphs must therefore be reported together.

The most defensible conclusion is:

> During LeJEPA training, the coarse global organization of the normalized `3x3` filter-shape distribution emerges early; later representation gains continue without large global histogram movement. This does not mean that late training is unimportant, but rather that late gains may occur in channel, layer, activation, or optimization structures finer than the global filter-population statistic.

## 11. Limits of scientific interpretation

### 11.1. Single seed

The ImageNet and LeJEPA comparison is based on one seed and one selected checkpoint pair. Although the depth pattern is clear and stable across the two PCA bases, multiple independent seeds are required to conclude that “the LeJEPA method always produces this filter signature.”

### 11.2. Channel permutation

Channels can be permuted in independently trained networks. The analysis therefore does not semantically match two kernels with the same index; it performs distribution-level comparison. This choice is appropriate, but a distribution difference is not evidence of functional causality.

### 11.3. Histograms and bin count

Drift is derived from 70-bin coefficient histograms and symmetric KL. The same protocol was retained for comparability with the third experiment. Histogram-based values can be sensitive to the bin count and edge protocol. Using one shared fixed edge set for all epochs in the training-dynamics analysis prevents that sensitivity from distorting the time comparison.

### 11.4. PCA basis and component semantics

The global basis makes the comparison model-independent, but it does not mean that every component has a universal semantic label. The strongest interpretations concern total drift, stage/layer localization, and patterns repeated across different representation controls. Component sign and number should be interpreted more cautiously.

### 11.5. Entropy and sparsity are not quality scores

Low entropy can indicate redundancy, high entropy can indicate a random-like variance spectrum, and sparsity can indicate the proportion of near-zero filters. None of them alone measures model accuracy or representation quality. In particular, LeJEPA's higher mean `H` does not directly mean “better”; it says only that variance at the selected checkpoint is distributed more evenly across the nine local directions.

## 12. Overall assessment

After the audit, the main methodological structure of the fourth experiments was found to be valid. The two important remaining implementation issues—using the same dtype/normalization order for target filters as the global artifact and completing the LeJEPA dynamics script—were corrected before the final jobs. Chart counterparts were added for the heatmaps, and the old heatmaps were retained.

Part 1 shows that the sparsity difference does not depend only on one epsilon choice, that the final low-entropy ImageNet layer remains stable across a broad cutoff interval, and that neither model has a layer crossing the paper's actual random boundary.

Part 2 shows that the ImageNet–LeJEPA global drift and depth pattern are not caused only by the PCA basis fitted to the two models. Although global D changes by approximately `3.27%`, the dominant final-layer difference and the overall stage ordering are preserved. The bit-identical PCA-independent control tables strengthen confidence in the checkpoint replay and the analysis pipeline.

Part 3 shows that the largest reorganization of LeJEPA's global normalized-filter distribution occurs during the first 20–30 epochs. Although the epoch 1->120 global drift is `0.419884`, epoch 30 is only `0.000948` from the final checkpoint. In contrast, fixed-probe accuracy rises from `0.8408` at epoch 30 to `0.8905` at epoch 120. The stage/layer separation shows that layer1, which contains few filters, continues to change after the global plateau; the entropy and descriptor controls show that initially high-entropy shapes close to the random reference organize into a more directional and lower-frequency population. This relationship is descriptive and does not claim causality.

The most accurate wording of the result is:

> In the selected, similarly performing RN18 checkpoint pair, LeJEPA does not produce a completely separate filter universe from ImageNet. The differences are globally limited but structured by depth; the strongest separation appears in the final residual block, with secondary separation in the early residual blocks. At the selected checkpoint, LeJEPA also has fewer near-zero filters and higher local variance-spectrum entropy. These patterns persist in the global, model-independent CNN Filter DB PCA basis, but a method-level claim requires validation across multiple seeds.

## 13. Main output paths

```text
outputs/(4thEXP)rn18_cifar10_filter_dynamics/
  original_cnn_filter_db_pca/
    cnn_filter_db_full_pca.npz
  regenerated_imagenet_baseline/
  part1_threshold_sensitivity/
  part2_original_basis_pair/
  lejepa_training_dynamics/

outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/
  analysis_global_cnn_filter_db_basis_epoch001_vs_epoch110/
```

## 14. Main local sources used in the audit

- `papers/CNN_Filter_DB(Paper).pdf`
- `main.ipynb` in both the original Git history and the current commit
- `(3)EXPERIMENT_PROGRESS.md`
- `scripts/compute_cnn_filter_db_pca.py`
- `scripts/analyze_rn18_threshold_sensitivity.py`
- `scripts/analyze_rn18_cifar10_filters.py`
- `scripts/analyze_rn18_cifar10_kernel_distributions.py`
- `scripts/analyze_rn18_lejepa_training_dynamics.py`
- `scripts/rn18_cifar10_common.py`
- old and regenerated ImageNet metric CSVs
- the old two-model-PCA and kernel-distribution results from the third experiment
