**Testing whether SIGReg induces actionable local weight structure**

Research assessment and proposed Experiment 5 for Sarp Kantar · 16 September 2026

This memo reviews the supplied `EXPERIMENTS_HANDOFF.mf` and primary literature. The handoff is the source for all statements about completed experiments; the underlying code, checkpoints, and raw outputs were not supplied or independently executed. All new configurations, numerical decision margins, and interventions below are proposals, not completed results. Literature coverage is targeted rather than a systematic review with exhaustive coverage.

**The recommendation is to replace training-route comparisons with a controlled regularizer experiment, followed by replacement interventions.** Keep CNN Filter DB measurements as secondary descriptors. Make the connection between weights, representations, and downstream behavior the central object of study.

The hypothesis has three separable parts:

| Claim | Evidence required | What would weaken it |
|---|---|---|
| H1: SIGReg causes a reproducible weight signature | Change only the regularizer; observe a prespecified weight statistic changing across independent training seeds and surviving relevant parameterization controls | Differences are smaller than seed variation, confined to arbitrary scale, or absent outside the projector |
| H2: That signature is actionable | Directly promoting it changes representation quality or collapse behavior; removing or perturbing it has a selective effect beyond matched controls | Matching the statistic does not improve the representation, or unrelated perturbations explain the effect |
| H3: It supports a useful local substitute | Train from random initialization with alignment plus a parameter-only penalty; retain performance and stability at lower measured cost, including a held-out setting | It works only after SIGReg pretraining, requires data-derived online statistics, or costs as much as an efficient SIGReg baseline |

Detectability does not imply locality, and locality does not imply causal sufficiency. A regularizer acts through gradients on model parameters, so some parameter change is unsurprising. The scientific question is whether a compact, transferable description of that change captures a useful part of the regularizer's action.

**What the existing experiments establish**

| Experiment | Finding to retain | Main limitation for the thesis hypothesis | Consequence for the next study |
|---|---|---|---|
| 1: RN50/MLL23 | Broad filter distributions of the supplied ImageNet and LeJEPA checkpoints were relatively close under the chosen measurement | Training provenance, data, evaluation, and independent seeds are unavailable or unequal | Use as exploratory evidence; do not attribute differences to SIGReg |
| 2: Signed residuals | The remaining difference has depth structure and basis-dependent directional information | Reanalysis of Experiment 1; not an independent replication; leading-PC importance depends on weighting | Retain residual plots, but preregister metrics before new runs |
| 3: RN18/CIFAR10 | Strong final-block differences and the raw-versus-BN-folded scale reversal | Random-init SSL versus supervised ImageNet transfer changes many factors; test-adaptive pairing compares epoch 110 to epoch 1 | Compare same-init SSL arms at the same optimizer steps |
| 4: Robustness and dynamics | The broad depth pattern survives the external PCA pipeline; coarse filter shape organizes early | Same selected pair and single SSL trajectory; external-basis comparison also changes numerical preprocessing | Reuse the measurement infrastructure and investigate structure hidden by the pooled histogram |

The handoff already identifies many of these limitations correctly. Four experiment families do not constitute four independent tests of the hypothesis: Experiment 2 reuses Experiment 1, and much of Experiment 4 reuses Experiment 3.

Three observations should drive the design:

1. **Raw weight scale reverses after BN folding.** The pooled LeJEPA/ImageNet standard-deviation ratio changes from 7.02 to 0.33. This rules out interpreting the raw ratio as a direct functional-strength signature. Folding is an inference-mode diagnostic based on stored running statistics; it neither identifies all equivalent parameterizations nor provides a strictly data-independent training penalty.
2. **Histogram convergence precedes downstream convergence.** From epoch 30 to 120, the reported final-histogram distance starts at only 0.000948, yet probe accuracy rises from 84.08% to 89.05%, a 4.97 percentage-point gain. This is evidence of measurement insensitivity to some continued learning, not evidence that weights stop mattering.
3. **Pooling heavily favors the last stage.** Layer4 contains 75.17% of the analyzed RN18 filters. Global stability can conceal changes elsewhere. Report equal-layer summaries alongside filter-pooled summaries.

Accuracy matching is not the repair for the causal problem. Accuracy is itself an outcome of initialization, training, and regularization; selecting on it can distort comparisons. Fixed-step comparisons should be primary. Validation-based performance matching can be a secondary descriptive analysis. Because CIFAR10 test accuracy already influenced previous choices, a new validation split cannot make that test set historically untouched. Treat CIFAR10 as development and reserve another benchmark for confirmation.

The approximately 3.27% change under the external PCA pipeline is a useful robustness check. It does not establish causal specificity, and it changes both basis and float16/float32 preprocessing. Further efforts to make this PCA reconstruction bit-identical should not take priority over controlled training.

The handoff does not document a measured collapse of the canonical corrected SIGReg run. Its two CUDA-initialization failures are infrastructure failures, not collapse events. If instability is part of the thesis motivation, define it and reproduce it under valid training conditions. Also separate low unlabeled-data requirements, few labeled examples at evaluation, and small-domain pretraining outperforming transfer: these are different forms of data efficiency.

**Why the proposed local replacement needs a narrower theoretical claim**

For a centered linear map, let z = Wx and Cov(x) = Σx. Then

\[
\operatorname{Cov}(z)=W\Sigma_xW^\top.
\]

An orthogonality constraint WWᵀ = I gives output whitening when the relevant input covariance is identity. It does not generally whiten anisotropic inputs. The same nonzero W cannot produce covariance I for both Σx = I and Σx = 2I. This is a counterexample to universal equivalence between a fixed weight-only whitening constraint and a distribution-dependent output constraint; it is not a proof that useful weight-only regularization is impossible.

Likewise, normally distributed weight coefficients do not imply normally distributed activations. Even exact output whitening does not imply Gaussianity. Nonlinearities, biases, normalization, data support, and compositions of layers all matter.

“Local” also needs an explicit definition. A separable penalty on each individual 3×3 kernel cannot directly measure relationships between filters. A layer-level penalty can measure those relationships. An adjacent-block penalty can address composition. Prefer the smallest scope that actually works rather than requiring the strongest filter-level claim from the outset.

Use this operational thesis statement:

> Under controlled data, architecture, and optimization, SIGReg induces reproducible structure in selected layer parameters. A fixed parameter-only penalty promoting that structure can reproduce part of SIGReg's benefit on representation quality and training stability, within a specified set of regimes.

“Data-independent” should mean that evaluating the replacement penalty and its gradient needs no input examples, feature batches, or running activation statistics. The alignment objective still uses data. If the penalty or its coefficients are discovered from SIGReg-trained models, call it a data-informed design with data-independent evaluation. BatchNorm in the backbone also remains data-dependent; this does not invalidate a parameter-only penalty, but it limits claims about the whole training system.

**Literature that materially changes the experimental design**

The following are primary sources. Recent preprints supply hypotheses and comparison methods, not settled guarantees. The implications in the final column are recommendations for this project.

| Source | Relevant result or scope | Design implication |
|---|---|---|
| [CNN Filter DB — Gavrikov & Keuper, 2022](https://arxiv.org/abs/2203.15331) | Studies populations of spatial 3×3 kernels across trained CNNs | Retain it as a morphological baseline; add channel and operator structure |
| [LeJEPA — Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544) | Combines prediction/alignment with Gaussian distribution matching and reports scalable, stable training | Reproduce the exact regularizer and qualify general claims of instability or universal data efficiency |
| [VICReg — Bardes et al., 2022](https://arxiv.org/abs/2105.04906) | Separates invariance, a variance floor, and covariance decorrelation | Hold the alignment term fixed when comparing the two regularization mechanisms |
| [WERank — Saheb Pasand et al., 2024](https://arxiv.org/abs/2402.09586) | Proposes weight regularization against rank degradation, including graph SSL and controlled models | “Weight-only anti-collapse regularization” is already prior art; include a weight-regularization baseline |
| [Preventing Dimensional Collapse via Orthogonality Regularization — He et al., 2024](https://arxiv.org/abs/2411.00392) | Adds SO/SRIP penalties to SSL encoders; examines weight and feature spectra in CNNs and Transformers | This is the closest practical baseline; adding OR to SSL does not establish that OR replaces SIGReg |
| [Orthogonal Convolutional Neural Networks — Wang et al., 2020](https://arxiv.org/abs/1911.12207) | Distinguishes kernel orthogonality from orthogonality of the convolution operator | Do not treat flattened-filter orthogonality as full convolutional isometry |
| [The Singular Values of Convolutional Layers — Sedghi et al., 2019](https://arxiv.org/abs/1805.10408) | Characterizes convolution spectra under its operator assumptions | Add frequency-domain channel-mixing diagnostics; respect boundary and stride assumptions |
| [Understanding Dimensional Collapse — Jing et al., 2022](https://arxiv.org/abs/2110.09348) | Analyzes collapse dynamics and the roles of augmentation and overparameterization | Include initialization, dense early measurements, and augmentation controls |
| [Exploring the Gap between Collapsed & Whitened Features — He & Ozay, 2022](https://proceedings.mlr.press/v162/he22c.html) | Relates spectral decay, projection, and downstream generalization | Do not assume maximally flat spectra are always the desired target |
| [Variance-Covariance Regularization Enforces Pairwise Independence — Mialon et al., 2024 version](https://arxiv.org/abs/2209.14905) | Analyzes how projector-output constraints can affect its input representation under stated conditions | Measure backbone and projector separately; neither assume isolation nor identical geometry |
| [Guillotine Regularization — Bordes et al.](https://arxiv.org/abs/2206.13378) | Studies why removing final training layers can improve transfer | Performance at the backbone and geometry at the regularized output can differ |
| [RankMe — Garrido et al., 2023](https://arxiv.org/abs/2210.02885) | Uses effective rank as an unsupervised representation-quality indicator | Use as one diagnostic with the published singular-value convention |
| [LiDAR — Thilak et al., 2024](https://arxiv.org/abs/2312.04000) | Separates between-instance and within-augmentation variation | Helps distinguish useful diversity from augmentation noise |
| [LDReg — Huang et al., 2024](https://arxiv.org/abs/2401.10474) | Studies local dimensional collapse despite global dimensionality | Optional neighborhood diagnostic; its “local” refers to data neighborhoods, not local weights |
| [CKA — Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html) | Provides a representation-comparison statistic | Useful descriptive bridge across seeds, not proof of causal equivalence |
| [KerJEPA — Zimmermann et al., 2025](https://arxiv.org/abs/2512.19605) | Generalizes JEPA distribution matching using kernel discrepancies and examines projection tradeoffs | Distinguish Gaussian-target effects from finite-slicing or estimator effects |
| [VISReg — Wu et al., 2026](https://arxiv.org/abs/2606.02572) | Separates scale and shape regularization and analyzes weak SIGReg gradients near collapse | Add as a stronger comparison in the later stability study; verify the claimed regime in your implementation |
| [Rectified LpJEPA — Kuang et al., 2026](https://arxiv.org/abs/2602.01456) | Uses alternative target distributions to control representation sparsity | An optional target-distribution intervention can test whether a weight signature is Gaussian-specific |
| [Gaussian Embeddings / JEPA-SCORE — Balestriero et al., 2025](https://arxiv.org/abs/2510.05949) | Relates learned density to model Jacobians under the paper's assumptions | Provides motivation for sparse Jacobian diagnostics; these remain data-dependent measurements |
| [When Does LeJEPA Learn a World Model? — Klindt et al., 2026](https://arxiv.org/abs/2605.26379) | Gives identifiability results under specified latent and transition assumptions | Use a controlled latent-variable toy study; do not infer local filter identifiability from an output-level theorem |
| [SPHERE-JEPA — Nicollier et al., 2026](https://arxiv.org/abs/2605.26900) | Investigates alternative optimal geometry under manifold-based assumptions | Gaussianity should be treated as a specified target, not a universal quality score |

The first implementation priorities are the OR baseline, separate weight/feature spectra, LiDAR plus output-distribution diagnostics, and paired objective training. The newer distribution-matching alternatives are useful specificity controls after the core experiment works.

**Experiment 5A: validate the mechanism and measurement before a large run**

Use the corrected LeJEPA lineage exclusively. Exclude the invalid EMA/predictor implementation, fake-data smoke run, and invalid batch-256 supervised replay. Audit the local loss against a pinned version of the [authors' implementation](https://github.com/galilai-group/lejepa); matching the method name is insufficient.

Check the number of independent images versus views, per-view averaging, normalized projection directions, real and imaginary characteristic-function terms, integration bounds/weights, sample-count factors, all loss reductions, and whether gradients reach both sides of the view-center objective. Do not standardize embeddings inside a diagnostic meant to detect loss of scale. Compute statistical losses in float32 even if backbone training uses bfloat16.

The paper's printed example and the historical local run use different integration settings. Preserve the historical run as historical. For the proposed main experiment, specify 1,024 slices and 17 equally spaced points over [-5,5], with Gaussian weighting and the sample-count/reduction convention of the verified reference. Run a short sensitivity check against the historical bound of 3. A coefficient of 0.02 is not comparable across implementations if reductions differ. The exact source commit and numerical parity result belong in the experiment manifest.

Validate the loss on synthetic embeddings: standard Gaussian, shifted Gaussian, rescaled Gaussian, rank-deficient Gaussian, and a constant tensor. Compare values and gradients to the reference; a constant zero embedding may be stationary even though it has a high loss. A Gaussian finite-sample discrepancy need not equal zero. These are scientific validation checks, not claims that a loss must order all non-Gaussian distributions identically.

Add a small analytic-control experiment: a two-layer linear network in 16 dimensions, isotropic latent variables, and correlated Gaussian views. Observe them through invertible mixing matrices with condition numbers 1, 10, and 100. Compare alignment + SIGReg, alignment + VC terms, and alignment + SO. Use 10,000 training pairs, 2,000 held-out pairs, five seeds, and a fixed 5,000-update budget. Measure output covariance and composed-map singular values. Its purpose is to expose the assumptions under which local orthogonality approximates output whitening; it cannot establish image representation quality.

**Experiment 5B: a matched-objective discovery study**

Define a common model z = gφ(fθ(x)), backbone representation y = fθ(x), and identical all-global-view center alignment:

\[
\bar z_i=\frac1V\sum_v z_i^{(v)},\qquad
L_{\mathrm{align}}=\frac1{BVd}\sum_{i,v}\|z_i^{(v)}-\bar z_i\|_2^2.
\]

There is no stop-gradient on the center. Keep the alignment coefficient at 0.98 for all controlled arms so the SIGReg arm agrees with the historical 0.02 tradeoff convention once its reductions are verified.

| Arm | Training objective | Purpose |
|---|---|---|
| A | 0.98 Lalign | Negative control for the role of anti-collapse regularization |
| B | 0.98 Lalign + 0.02 RSIG | Main reference |
| C | 0.98 Lalign + α RVC | Moment-based comparator with identical architecture and alignment |

Call C “LeJEPA alignment + VICReg variance/covariance terms,” not a reproduction of the entire original VICReg recipe. Do not add its invariance loss a second time.

For C, compute each view's sample covariance with denominator B−1, a mean variance hinge max(0, 1−sqrt(var+10⁻⁴)), and squared off-diagonal covariance summed and divided by d. Define RVC as the mean over views of 25 times the variance term plus the covariance term. This explicitly fixes the variance/covariance ratio; α controls its scale relative to the shared alignment term. If poor behavior traces to that ratio, add a separately labeled ratio ablation rather than silently changing the main arm.

B versus C does not isolate higher-order Gaussianity: the variance hinge permits variances above its floor and does not set the mean to zero. If the mechanism claim becomes specifically about higher-order distribution matching, add a fourth arm M with the same alignment and a penalty on ||mean(z)||²/d + ||Cov(z)−I||F²/d, averaged over views. Calibrate its coefficient with the same pilot budget and measure the achieved moments. Differences between B and M are more informative about distributional effects beyond first and second moments, although differing optimization dynamics still matter. This adds five runs to the five-seed discovery design; it is an optional specificity experiment, not silently included in the fifteen-run budget.

Use two pilot seeds, 901 and 902, only for numerical validation and coefficient calibration. Choose α0 so that the median regularizer-gradient norm over a fixed small set of early training batches is similar to B, reporting backbone and projector norms separately. Test {α0/3, α0, 3α0}. For B, evaluate the analogous coefficient neighborhood {0.02/3, 0.02, 0.06} while holding alignment at 0.98; retain 0.02 as the controlled reference and report the alternatives as sensitivity runs. Equal numerical coefficients or equal scalar losses do not equate regularization strength. Gradient calibration is a scale convention, not evidence of equivalent action. Give compared methods equal pilot budgets.

| Setting | Proposed value |
|---|---|
| Dataset | CIFAR10 for development |
| Split | Fixed stratified 45,000 training / 5,000 validation from the official training split; split seed 20260916; publish indices and hashes |
| Test use | Historical test is acknowledged as development-exposed; no checkpoint or coefficient selection on it |
| Backbone | Same CIFAR ResNet18 in every arm, random initialization, 3×3 stride-1 stem, no max-pool |
| Initialization pairing | Within a seed, copy the identical backbone and projector state into A/B/C |
| Projector | 512 → 2048 → 2048 → 64, BN/GELU in hidden blocks, linear output; log exact biases and BN settings |
| Input/views | 32×32 inputs, four independently augmented global views per image |
| Augmentation | Historical strong recipe: crop scale [0.08,1], jitter (0.8,0.8,0.8,0.2) at p=.8, grayscale .2, blur .5, solarize .2, horizontal flip .5; freeze exact kernel/sigma/threshold/order in code |
| Optimizer | AdamW, lr .002, betas (.9,.999), epsilon 10⁻⁸; weight decay 5×10⁻⁴ on Conv/Linear matrices, zero on biases and normalization affine parameters |
| Duration | 21,000 optimizer updates: 120 full-data equivalent epochs with 175 complete batches per epoch |
| Schedule | 175-update linear warmup, then cosine to lr 2×10⁻⁶; indexed by optimizer update |
| Batch | 256 distinct source images; four views gives 1,024 image views per update |
| Precision | Backbone bfloat16 autocast; covariance, characteristic-function statistics, and diagnostic spectra float32/float64 as appropriate |
| Discovery seeds | 13, 17, 23, 31, 43 |
| Confirmation seeds | 101, 103, 107, 109, 113, used after candidate and endpoints are frozen |
| Main checkpoint | Fixed final update, not best observed validation or test checkpoint |
| Probe | Frozen backbone in eval mode; one fixed protocol shared across all methods |

The new weight-decay grouping is a proposed explicit convention, not a claim that it reproduces unspecified historical grouping. If exact reproduction is required, resolve that convention before running and keep it identical across arms.

Use separate deterministic RNG streams for initialization, data order, augmentation, and SIGReg directions. Adding random slice draws must not change future image views in one arm only. Same-seed pairing reduces nuisance variation, but only independently trained seeds supply replication.

Save weights, optimizer, scheduler, scaler if used, BN buffers, and RNG states at initialization; updates 1, 10, 50; and equivalent epochs 1, 2, 5, 10, 20, 30, 60, 90, 120. Store light training metrics frequently; reserve full activation diagnostics for epochs 0, 1, 5, 10, 30, 60, 120.

Do not stop a finite but collapsed run and silently remove it. Record its trajectory and failure classification. Stop immediately for nonfinite arithmetic and record the numerical failure separately. Avoid adding clipping or recovery logic to only one arm.

**Measure weight structure at three spatial scales**

| Scale | Measurement | What it can reveal |
|---|---|---|
| Individual channel-pair 3×3 kernels | Existing PCA distributions, spatial roughness, frequency energy, norms | Marginal spatial morphology |
| Entire output filters | Aℓ = reshape(Wℓ) in R^(Cout × Cin k²); uncentered singular values, normalized Gram structure, filter coherence | Relationships and rank across input/output channels |
| Convolution/block function | Sampled frequency-response matrices and sparse block-Jacobian measurements | Channel mixing, composition, and effects of nonlinearities |

The covariance entropy of millions of rows in R⁹ is not the rank of a layer whose output space has hundreds of channels. Its maximum dimension is nine. This distinction is central to the thesis.

For an uncentered weight matrix A, compute qj = σj² / Σkσk², weight-energy effective rank exp(−Σj qj log qj), normalized by min(m,n), and stable rank ||A||F² / ||A||2². Report the complete normalized spectrum at selected layers, not only scalar summaries. Centering A across its rows changes the question and should be a separately labeled analysis.

Use three parallel parameter views: raw matrices; inference BN-folded matrices with the exact corresponding affine scale; and normalized row/filter directions with norms reported separately. Include BN gamma/beta and running-statistic summaries, projector matrices, and 1×1 shortcuts. Keep the stem separate for compatibility with prior plots, but do not omit it from the functional account.

Predeclare the equal-layer mean normalized weight-energy effective rank at epoch 30 as the primary descriptive weight endpoint. Test B versus C; B versus A is a secondary contrast because collapse alone may explain their separation. The historical last-block finding makes layer4.1.conv2 a justified secondary location, not a guaranteed universal locus. Analyze stage summaries and complete spectra as prespecified secondary outcomes. A null scalar effect limits this endpoint; it does not rule out all weight signatures.

For frequency analysis, transform the spatial kernels and inspect the resulting Cout × Cin channel-mixing matrix at a fixed small frequency grid. Label this a frequency-response diagnostic for the actual padded/strided network. Exact singular-value identities from circular, stride-one convolution do not automatically transfer to zero padding, stride changes, pooling, and residual additions. If exact operator norms are required, use implicit operator and adjoint products with the actual implementation.

Retain the checksum-pinned external PCA; never refit it per arm or checkpoint. Keep normalization precision identical. Use fixed component-wise histogram edges derived from discovery runs and frozen for confirmation, including explicit underflow/overflow accounting. Report 50/70/100-bin sensitivity once, and prefer a fixed-direction sliced Wasserstein distance in the original normalized 9D kernel space as a complementary bin-free, joint-distribution-sensitive measure. Do not interpret either metric as a quality score. Do not discard lower-variance PCs merely because the original weighting suppresses them.

**Connect those measurements to representations**

Extract stage outputs, GAP backbone features y, both projector hidden outputs, and final z. For CNN stage outputs use per-image global average pooling as the primary feature definition; spatial samples are a separate diagnostic and are not independent training replicates.

Use 2,048 fixed validation images with one deterministic evaluation view, plus four fixed augmented views of the same images for view-based diagnostics. Reuse image IDs, views, and random projection directions across methods. Use a disjoint 2,048-image validation calibration subset if a diagnostic estimates whitening or standardization statistics. Do not fit local-regularizer coefficients on these validation images.

Measure:

- Feature covariance spectra, total variance, coordinate standard deviations, and covariance-energy effective rank. Label this convention separately from RankMe, which uses normalized singular values rather than their squares.
- LiDAR from the same multi-view image groups, to separate variation across images from nuisance variation across views.
- Distribution discrepancy of final z against N(0,I), evaluated with independent directions and a finer frequency grid than training. Compare to matched-size Gaussian simulations to contextualize finite-sample error. A low finite-projection statistic is evidence of closeness under that measurement, not proof of multivariate Gaussianity.
- Mean, scale, covariance anisotropy, and shape separately. Optionally compute a post-whitening shape diagnostic using calibration statistics estimated on different images. Never report this as evidence that raw outputs have correct scale or covariance.
- Paired-view alignment, final-backbone frozen-probe accuracy, and sparse CKA comparisons across seeds. High rank can encode nuisance noise; high CKA does not imply equal task behavior.
- At selected checkpoints, active-unit fractions and pre/post-normalization behavior. Intermediate ReLU outputs cannot themselves be standard Gaussians; their failure of a Gaussianity test is not evidence of failed training.

For the main performance endpoint, use a plain linear classifier on y, trained for 50 epochs with AdamW lr .001, weight decay 10⁻⁶, batch 256, and cosine decay under a fixed common protocol. Cache deterministic features when practical. Log convergence on validation; decide any common extension in pilots and freeze it. The historical LayerNorm+Linear evaluation is a separately reported continuity check, since LayerNorm is a nonlinear feature transformation even with a linear classification layer.

Evaluate shared, nested 1%, 10%, and 100% labeled subsets of the 45,000-image training pool. Select classifier hyperparameters on validation with equal budgets, never on test. Label fractions and unlabeled pretraining fractions are separate experimental axes. For downstream few-label comparisons, pair the label subsets across methods and report variation across a small prespecified set of label draws; keep pretraining seed as the main inferential unit.

**Two inexpensive falsification checks on existing checkpoints**

First, shuffle complete channel-pair 3×3 kernels within a layer without compensating subsequent channels. This preserves that layer's raw marginal kernel population exactly but changes how kernels are connected. Recompute the original descriptors and assess feature/probe behavior. If descriptors are unchanged while behavior deteriorates, the descriptors do not uniquely characterize function. This does not show those descriptors are useless as an inductive bias.

Second, apply a valid, coordinated channel permutation with matching BN, downstream, and residual-path transformations. Confirm that outputs are numerically unchanged. A statistic intended to be permutation-invariant should remain unchanged. Do not treat an arbitrary uncoordinated ResNet permutation as a function-preserving control. Likewise, do not claim generic scale invariance for every weight-spectrum statistic.

**Experiment 5C: ask whether a local penalty points in the same useful direction**

At fixed checkpoints, isolate the SIGReg gradient, rather than using the total alignment-plus-SIGReg gradient:

\[
g_{\ell,b}^{\mathrm{SIG}}=\nabla_{W_\ell}R_{\mathrm{SIG}}(Z_b),\qquad
\bar g_\ell^{\mathrm{SIG}}=\frac1K\sum_{b=1}^K g_{\ell,b}^{\mathrm{SIG}}.
\]

Use eight independently sampled training batches at epochs 1, 5, 30, and 120. Restore model buffers and RNG state between measurements so the diagnostic does not alter subsequent training. Either preserve training-mode BN consistently or explicitly declare a separate eval-mode analysis. Keep those two questions separate.

For each proposed parameter-only penalty Rlocal, compare its layer gradient to the averaged SIGReg gradient: cosine, relative norm, and predicted first-order change in held-out SIGReg loss. Estimate gradient variability across batches and across fresh slicing directions separately. A useful approximation need not match stochastic batch noise. Report near-zero norms explicitly; cosine is not informative there.

Also compare against the alignment gradient and the VICReg-style gradient. This identifies whether the candidate approximates a distinctive SIGReg direction or merely a broad anti-collapse tendency. At an optimum SIGReg gradients may be tiny, which makes early checkpoints more informative.

This is a proposed diagnostic, not a sufficient causal test. AdamW momentum and preconditioning mean raw-gradient alignment does not guarantee matching trajectories. If desired, additionally compare one-step parameter changes from cloned optimizer states.

A small nonnegative combination of gradients of scalar weight statistics can be fitted on discovery runs, giving an actual scalar potential Rlocal = Σj aj φj(θ). Freeze aj before confirmation, and validate on new seeds and new data regimes. An arbitrary learned vector field need not integrate to a scalar regularizer; avoid calling one a loss without establishing that property. Disclose the data used to design the coefficients.

**Choose a baseline before inventing a new penalty**

Start with Soft Orthogonality on the smaller Gram matrix. For A in R^(m×n), let G = AAᵀ when m≤n and G = AᵀA otherwise; r = min(m,n). A dimension-normalized version is ||G−I_r||F²/r. This normalization is a deliberate variant; reproduce the published baseline's sum and coefficient convention separately when comparing against it. SO and SRIP are already described in the [OR paper](https://arxiv.org/html/2411.00392v1).

Do not constrain an impossible identity on the larger side of a rectangular matrix. Do not assume all layers should become isotropic. Do not confuse a finite-dimensional patch-matrix constraint with full convolution-operator orthogonality.

A candidate informed by the discovery study is a soft band on a small number of differentiable layer statistics:

\[
R_{\mathrm{band}}=\sum_{\ell,j}a_{\ell j}\left[(l_{\ell j}-\phi_j(W_\ell))_+^2+(\phi_j(W_\ell)-u_{\ell j})_+^2\right].
\]

Possible φ are normalized Gram anisotropy, a smooth participation-ratio proxy, and parameter energy. Choose only statistics with a reproducible effect and acceptable parameterization behavior. Set bounds from training-only discovery evidence; freeze them before new-seed/dataset tests. This allows preservation of a measured spectral range without assuming that the best target is always an identity Gram matrix. It is a candidate family, not a demonstrated new algorithm or an established SIGReg surrogate.

A scale-normalized Gram term alone can be insensitive to uniform weight shrinkage. Track absolute scale and normalization gains; include a clearly disclosed parameter-only scale term if needed. A weight-only guarantee can still be defeated by zero normalization gains, biases, nonlinear gating, a collapsing projector, or incompatible layer composition. Standard SO itself has zero gradient at A=0. Therefore neither this penalty nor a full-rank weight matrix proves recovery from arbitrary collapse.

For a cheap estimator, a squared Gram penalty can use E_v ||(G−I)v||² with random vectors satisfying E[vvᵀ]=I. This uses matrix-vector products without materializing the full Gram. Fix probe count and refresh policy; compare estimator variance and wall time. A sampled penalty is data-independent but still computational work.

**Experiment 5D: replacement and timing interventions**

Start with the B checkpoints at epoch 30 because the historical morphology stabilizes near that point. Clone the complete training state and continue to the same final update:

| Branch | Remaining objective | Interpretation |
|---|---|---|
| B→B | Continue SIGReg | Paired reference; reuse the continued parent run |
| B→A | Remove SIGReg | Tests whether continued explicit regularization is needed in this state |
| B→Local | Replace SIGReg with the fixed local penalty | Tests maintenance of a learned state |
| B→VC, optional | Replace with moment regularization | Tests whether late higher-order distribution matching is needed |

Retain alignment coefficient, schedule, optimizer state, data order, and augmentation streams. Do not reset the optimizer only in the replacement branch. Repeat at epoch 5 if the epoch-30 swap is promising or if timing is the question. An optional remove-then-reintroduce branch can test recovery; it is not required for the initial screen.

Warm-start success is evidence for a continuation strategy, not for learning from scratch. The decisive additional arms are:

| Arm | Start/objective | Question |
|---|---|---|
| D | Random initialization + alignment + local penalty | Can it replace SIGReg throughout training? |
| E | Random initialization + alignment + SIGReg + local penalty | Is it only complementary? |
| F | Random initialization + alignment + SIGReg with 256 slices | Does a simpler reduction in SIGReg cost achieve the same practical gain? |
| G, stronger study | Alignment + published SO/SRIP baseline if D is a different candidate | Does the learned candidate outperform established weight regularization? |

If D uses SO, G is already represented and need not duplicate it. Scope ablations should compare backbone-only, projector-only, and backbone-plus-projector penalties with disclosed coefficient normalization. A backbone-only failure caused by an unconstrained projector is informative but is not a decisive rejection of all local regularization.

Add a projector-capacity study only after the baseline experiment: compare the shared 512→2048→2048→64 projector with a 512→64 linear projector under both B and C, and then D if warranted. Keep dimensions fixed. If signatures relocate with projector capacity, investigate that mediation. Changing projection dimension and depth simultaneously would confound this test.

A targeted weight intervention can test the proposed mechanism more directly: move a selected singular-value statistic toward or away from its SIGReg range while preserving singular vectors, and compare with equal-relative-norm random perturbations in the same layers. Test multiple small perturbation sizes. Verify which statistics changed, and assess frozen features both before and after a common BN recalibration diagnostic. A single destructive edit proves little. Directional response, matched controls, and rescue provide stronger evidence, but do not establish a complete mediation theorem.

**Statistical decisions and success criteria**

The training run is the experimental unit. Filters, images, layers, checkpoints, and forked descendants are nested measurements; none creates additional independent pretrained models. Analyze fork outcomes as paired contrasts within parent seed.

Use seed-level paired effects and 95% intervals, show every seed, and separate exploratory feature selection from confirmation. Five seeds are a practical start with limited precision; use pilot variance to decide whether the prespecified scientific margin needs eight to ten or more confirmation seeds. With five paired runs an exact two-sided sign-flip test cannot attain p<.05 even if every difference has the same sign (minimum 2/32=.0625); do not select a convenient testing convention afterward.

For the single primary weight endpoint, report B−C and its paired uncertainty. Correct exploratory per-layer testing for multiplicity or present it as descriptive. A method classifier trained on weight summaries must hold out entire seeds and, later, datasets; splitting filters from one model into train/test is leakage. With only five seeds, such a classifier is exploratory and cannot be the sole signature test.

For the replacement stage, predeclare a practical performance margin, for example one percentage point in final frozen-backbone accuracy on the primary label regime. Evidence for noninferiority requires the lower confidence bound for D−B to exceed −1 percentage point; a nonsignificant difference is not equivalence. Assess few-label and transfer performance separately, not only the full-label score.

For an exact initial declaration, use the 10% labeled probe (4,500 labeled training images) as the primary replacement-performance endpoint, with 1% and 100% as secondary endpoints. Require no observed complete-collapse or numerical-failure excess while reporting the uncertainty in those rates. For a practical efficiency target, choose before timing either at least a 10% end-to-end time reduction at noninferior accuracy, or a stated memory reduction at comparable time; these are proposed engineering margins, not universal scientific thresholds. A useful structural result can remain publishable even if the efficiency target fails, but the claim must change accordingly.

Track complete collapse using output variance together with chance-level discrimination and nonfinite failures separately. One proposed diagnostic flag is final-z mean coordinate variance below 10⁻⁴ over three consecutive checks; a flag triggers inspection, not automatic deletion. Report dimensional collapse using continuous normalized effective rank and spectra; a threshold such as rank/d<.1 is a prespecified screen, not a universal pathology criterion. Evaluate backbone and projector separately and report low performance without collapse as its own failure mode.

Do not claim a low collapse probability from zero failures in five runs. Under independent identically distributed runs, zero failures out of five still allows an approximately 45% failure probability at a one-sided 95% binomial upper bound. A publication claim about robustness needs many inexpensive stress runs or a more limited formulation.

Define success at three levels:

| Outcome | Defensible claim |
|---|---|
| B−C weight signature replicates, but no intervention works | Reproducible regularizer-associated parameter structure |
| B→Local works, D fails | Local constraint can help maintain a SIGReg-learned state |
| E works, D fails | Complementary regularization, not replacement |
| D is noninferior and stable on the development setting | Empirical replacement within that setting |
| D also transfers with frozen design choices and lowers measured cost | Evidence for a useful transferable parameter-only substitute |
| D matches performance but not SIGReg-like representation diagnostics | An alternative successful mechanism, not established mechanistic equivalence |
| No consistent marginal signature, but operator/activation differences exist | The hypothesis must move to a larger structural scale |
| No useful weight signature survives controls | The tested signature family fails; this does not exclude all possible parameter descriptions |

**Measure computational advantage instead of assuming it**

For V views, B images, embedding dimension d, S slices, and T quadrature points, a direct SIGReg implementation is roughly O(VBS(d+T)); a covariance regularizer is roughly O(VBd²). Constants, fusion, communication, and memory layout matter. A layer Gram penalty costs roughly O(rmn), summed across selected matrices, and can exceed the embedding-space cost. In particular, d=64 is a small covariance problem while the RN18 layer matrices are large. These are implementation-level complexity estimates, not measured speedups; kernel discrepancies and slicing tradeoffs are discussed in [KerJEPA](https://arxiv.org/abs/2512.19605).

Measure end-to-end training milliseconds per update, images per second, peak device memory, regularizer forward/backward time, distributed communication if used, and time to a preregistered validation target. Use identical hardware, precision, batch, views, and synchronization. Warm up, then time a fixed block of at least 200 updates; report variation across blocks. Exclude offline scientific analysis consistently from training-time comparisons and report its cost separately.

Compare B, F, C, and D with reasonably efficient implementations. If SIGReg accounts for fraction f of total update time, even eliminating it entirely gives a maximum speedup of 1/(1−f), before adding replacement overhead. If f=.05, that ceiling is only about 1.053×. In that case, simplicity or robustness may be the stronger engineering argument.

**A compute-aware order of execution**

1. Existing-checkpoint probes, kernel-shuffle/permutation checks, loss parity, and the linear toy experiment. These resolve measurement and implementation issues cheaply.
2. A/B/C on three paired seeds for screening: nine full runs. Extend to five discovery seeds if the protocol behaves as intended: fifteen full runs total. This is discovery, not independent confirmation of a selected candidate.
3. From the five B parents, add B→A and B→Local at epoch 30. Continuing the original B run already supplies B→B. The ten extra 90-epoch branches cost 7.5 full 120-epoch equivalents.
4. Add D, E, and F for five seeds after selecting a candidate: fifteen additional full runs. Run additional baselines or scope ablations only to answer a concrete remaining question.
5. Freeze the candidate and endpoints, then compare it with fresh B/C/F controls on the confirmation seeds. Do not describe reusing discovery seeds as held-out confirmation.

Do not estimate GPU hours without measuring your implementation on the available hardware. The table above gives run-equivalent budgets; storage can be reduced by saving full optimizer states only where forks or exact resumes are required.

For a stronger study, first add an unlabeled-data fraction of 10% (4,500 unique images) at the same 21,000-update budget. This isolates unique-data quantity while permitting repeated exposure. Then run an equal-epoch sensitivity analysis because equal updates and equal passes answer different questions. Keep labeled probe budgets separately fixed. A 1% unlabeled setting is an additional stress test, not needed for the initial causal comparison.

Next reserve a confirmation dataset, such as CIFAR100 if it has not already influenced this method's design. Fix its train/validation/test split and candidate selection rules before seeing the new results. Replicate the reference, moment comparator, selected local candidate, and economical SIGReg baseline at full and reduced unlabeled-data budgets. A domain-shift dataset is more informative than another CIFAR result if the claim centers on low-data domain transfer.

Only then extend to a Transformer. A small ViT with an appropriate patch size for its input resolution is sufficient. Analyze Q/K/V projections separately, attention output projections, MLP matrices, normalization gains, and projector matrices. Full-rank Q/K weights do not guarantee diverse attention, because attention depends on data-conditioned products and softmax. Keep architecture-specific training choices equal across regularizers; changing backbone and dataset together without within-setting controls prevents attribution.

**Outputs needed for an auditable result**

Keep a stable experiment directory with a manifest, source commit, package/environment versions, split hashes, full configurations, seed/RNG policy, coefficient-pilot ledger, checkpoint registry, and PCA artifact checksum. Use tables keyed by dataset, fraction, architecture, arm, seed, update, layer, parameter representation, and metric. Preserve all outcomes, including failures and excluded implementation checks with their reasons.

The final scientific figures should show: paired B−C weight effects by stage; weight and activation spectra on the same timeline; backbone-versus-projector localization; local/SIGReg gradient agreement; continuation-branch outcomes; and accuracy/stability versus measured training cost. Avoid using a pooled histogram as the main causal figure.

The immediate recommendation is **A/B/C matched training plus stage/projector probes, followed by the epoch-30 remove/replace branches and a from-scratch local arm**. This makes the thesis falsifiable while preserving the useful measurement infrastructure already built. A defensible contribution would be identifying when embedding regularization can be approximated by parameter-local constraints, where that approximation fails, and what practical benefit survives those limitations.
