# CNN Filter DB Experiment Guide

This file explains how to run this repo on the remote machine and what each notebook experiment is trying to measure.

## 1. Environment

Use the conda environment already created for this repo:

```bash
cd /arf/scratch/skantar/CNN-Filter-DB
conda activate cnn-filter-db
```

If you need to recreate it from scratch:

```bash
conda env create -f environment.yml
conda activate cnn-filter-db
python -m ipykernel install --user --name cnn-filter-db --display-name "Python (cnn-filter-db)"
```

The notebook kernel should be:

```text
Python (cnn-filter-db)
```

## 2. Dataset

The dataset has already been downloaded and extracted by the Slurm job:

```text
/arf/scratch/skantar/CNN-Filter-DB/data/dataset.h5
```

Current verified size:

```text
105.49 GB decimal
98.24 GiB
```

The file contains:

```text
filters: shape (1464797156, 3, 3), dtype float64
meta:    21436 rows x 128 columns
```

The compressed Zenodo file was `dataset.h5.xz`, about `47.0 GB`, and it expanded to about `98.2 GiB`.

To re-run the download job if needed:

```bash
sbatch jobs/download_dataset.sbatch
```

Logs are written to:

```text
logs/cnn-filter-db-data-<job_id>.out
logs/cnn-filter-db-data-<job_id>.err
```

## 3. Quick Dataset Check

Before running the full notebook, run the lightweight check:

```bash
conda activate cnn-filter-db
python scripts/check_dataset.py --sample-size 100000
```

This verifies that:

1. `dataset.h5` opens successfully.
2. The `filters` and `meta` keys are present.
3. Metadata can be loaded with pandas.
4. A small PCA experiment works on a filter sample.

Expected final line:

```text
status: ok
```

This is the safe first test because it does not load the full 98 GiB filter array into memory.

## 4. Running Jupyter Remotely

Start Jupyter from a compute allocation, not the login node.

Example:

```bash
srun --pty -p palamut-cuda --cpus-per-task=16 --mem=250G --time=12:00:00 bash
cd /arf/scratch/skantar/CNN-Filter-DB
conda activate cnn-filter-db
jupyter lab --no-browser
```

Then open the tokenized Jupyter link from your local browser.

The notebook path is:

```text
main.ipynb
```

The dataset path cell is already set to:

```python
dataset_path = "/arf/scratch/skantar/CNN-Filter-DB/data/dataset.h5"
```

## 5. Memory Warning

The full filter matrix is very large:

```text
filters as float64: about 98.2 GiB
dX as float16:      about 24.6 GiB
dX_scaled:          about 24.6 GiB
```

The notebook cell below loads the full filter dataset:

```python
with h5py.File(dataset_path, "r") as f:
    dX = f["filters"][...].reshape(-1, 9).astype(np.float16)
```

During loading and conversion, memory usage can temporarily be much larger than the final `float16` array. Use a compute node with a large memory allocation.

## 6. Notebook Experiments

Run the notebook in this order.

### Experiment 1: Load Data

Cells:

1. Imports.
2. `dataset_path`.
3. Load `filters` into `dX`.
4. Load `meta` into `df_meta`.

What it does:

The dataset contains many trained `3x3` convolution kernels. The notebook reshapes every filter into a vector of length `9`, so the full dataset becomes:

```text
number_of_filters x 9
```

The metadata table maps groups of filter IDs back to model information such as task, visual category, architecture, and convolution depth.

Why it matters:

Almost every later experiment asks: "Do filters trained under different conditions occupy different regions of this 9-dimensional filter space?"

### Experiment 2: Scale Filters

Cell:

```python
dX_scaled = scale(dX)
```

What it does:

Each filter is divided by its largest absolute weight. This removes raw magnitude and focuses the analysis on filter shape.

Why it matters:

Two filters may have the same pattern but different scale. Scaling makes the experiments compare pattern geometry rather than absolute weight size.

### Experiment 3: Global PCA

Cells:

```python
pca = PCA(n_components=9)
dX_n = pca.fit_transform(dX_scaled)
```

What it does:

PCA finds the main axes of variation across all filters. Since a `3x3` filter has `9` values, there are at most `9` principal components.

Outputs:

1. `pca.mean_`: the average filter.
2. Cumulative explained variance plot.
3. PCA covariance plot.
4. `dX_n`: every filter represented in PCA coordinates.

Why it matters:

This gives a common coordinate system for comparing all filters. Later ridge plots, KL matrices, and scatter plots are based on `dX_n`.

Important notebook caveat:

The current PCA cell deletes `dX_scaled`:

```python
del dX_scaled
```

But the subset PCA section later uses `dX_scaled` again. If you want to run the subset PCA section, either run it before deleting `dX_scaled`, remove/comment out `del dX_scaled`, or rerun the scaling cell before the subset section.

### Experiment 4: Subset PCA

Default subset:

```python
df_meta[df_meta["Visual Category"] == "fractals"]
```

What it does:

This repeats PCA for only one subset of the dataset. The default example uses filters from models trained on fractal/formula-like data.

Outputs:

1. Subset eigenimages.
2. Subset cumulative explained variance.

Why it matters:

Global PCA tells us the dominant filter patterns across everything. Subset PCA asks whether a specific data category has its own dominant filter structure.

How to change the subset:

Examples:

```python
df_meta[df_meta["Visual Category"] == "medical xray"]
df_meta[df_meta["Task"] == "Classification"]
df_meta[df_meta["Task"] == "Segmentation"]
```

### Experiment 5: Ridge Plots

Cell group:

```python
datatype_distributions = df_meta.groupby("Visual Category").filter_ids.apply(...)
ridge_plot(...)
```

What it does:

For each visual category, it plots the distribution of PCA coefficients.

Why it matters:

These plots show whether different training data types produce visibly different filter distributions. If two categories have similar ridges, their trained filters are distributed similarly in PCA space.

### Experiment 6: KL Matrices

Cell groups:

1. KL helper functions.
2. KL by `Task`.
3. KL by `Visual Category`.
4. KL by normalized convolution depth.

What it does:

KL divergence compares probability distributions. The notebook builds histograms of PCA coefficients and compares groups pairwise.

The symmetric KL score is high when two groups have different filter distributions and low when they are similar.

Why it matters:

The KL matrices quantify distribution shift. Instead of only looking at plots, you get a numeric distance between categories such as:

```text
Classification vs Segmentation
natural vs medical xray
early layers vs late layers
```

### Experiment 7: KL Boxplots

Cell:

```python
for grouping in ["Task", "Visual Category"]:
    ...
plt.boxplot(...)
```

What it does:

It summarizes all pairwise KL distances for a grouping into a boxplot.

Why it matters:

This gives a compact view of which grouping creates stronger distribution shifts overall. For example, it can help compare whether `Task` or `Visual Category` separates filter distributions more strongly.

### Experiment 8: PCA Scatter Plot

Cell:

```python
scatter(dX_n[:,0], dX_n[:,1], ...)
```

What it does:

It plots the density of filters along the first two principal components.

Why it matters:

This is a global map of the learned filter space. Dense regions represent common filter patterns; sparse regions represent unusual filters.

### Experiment 9: Filter Quality / Degeneration

Cells:

```python
layer_quality_worker(...)
gen_layer_weights(...)
pool.map(...)
```

What it does:

For every layer/group of filters, it computes:

1. `variance_entropy`: how spread out the layer's filter variation is across SVD directions.
2. `sparsity`: the share of filters that are nearly zero after a small threshold.

Why it matters:

Degenerated filters are filters that carry little useful structure. A layer with many near-zero or low-variation filters may be less robust or less useful for fine-tuning.

This section can be expensive because it iterates over all metadata filter groups and uses multiprocessing.

## 7. Third Experiment: ResNet18 on CIFAR10

This experiment moves from RN50/MLL23 to ResNet18/CIFAR10 and compares:

1. ImageNet1k-pretrained ResNet18 fine-tuned on CIFAR10.
2. LeJEPA SSL-pretrained ResNet18 with a frozen backbone and fixed-epoch linear head.

The important design change is that the comparison uses performance-matched
checkpoints. The LeJEPA SSL stage saves checkpoints throughout SSL pretraining,
then every saved SSL checkpoint gets the same fixed-length linear-head training.
The selected pair is the ImageNet fine-tuning checkpoint and LeJEPA linear-head
checkpoint with the smallest CIFAR10 test-accuracy gap.

Current LeJEPA implementation note:

The earlier RN18/CIFAR10 SSL path was only a non-label EMA/predictor proxy and
must not be treated as LeJEPA. The corrected script now follows the supplied
LeJEPA paper/repository: multi-view prediction/invariance loss plus SIGReg,
implemented with the original local `lejepa/lejepa/univariate/epps_pulley.py`
and `lejepa/lejepa/multivariate/slicing.py` source files. For ResNet18, Algorithm 2
in the paper says to set `global_views = all_views`, so the CIFAR10 adaptation
uses same-resolution stochastic views rather than ViT-style local token crops.
There is no target encoder, EMA, predictor, stop-gradient, prototype layer, or
label use in the SSL backbone training.

The ImageNet-pretrained fine-tuned checkpoints already exist in
`outputs/rn18_cifar10_experiment_cudnn/checkpoints`, so true-LeJEPA runs should
reuse them instead of retraining Method A.

Run the full GPU experiment:

```bash
cd /arf/home/skantar/CNN-Filter-DB
sbatch jobs/run_rn18_cifar10_experiment.sbatch
```

Run the corrected cuDNN true-LeJEPA experiment:

```bash
cd /arf/home/skantar/CNN-Filter-DB
sbatch jobs/run_rn18_cifar10_true_lejepa_cudnn.sbatch
```

By default, CIFAR10 is cached under `data/cifar10` in this checkout because the
dataset is small. Set `RN18_CIFAR10_DATA_DIR=/path/to/cache` if you want it
elsewhere. The first full run may spend extra time downloading CIFAR10 and the
torchvision ImageNet-pretrained ResNet18 weights.

On the current `palamut-cuda` runtime, cuDNN initialization failed on a simple
ResNet18 convolution even though CUDA itself worked. The Slurm wrapper therefore
passes `--disable-cudnn`, which keeps CUDA enabled but avoids that runtime error.
The corrected true-LeJEPA job uses `kolyoz-cuda`, where the cuDNN probe passed,
and intentionally keeps cuDNN enabled.

Useful overrides:

```bash
RN18_CIFAR10_OUTPUT_DIR=outputs/rn18_cifar10_experiment_long \
RN18_CIFAR10_IMAGENET_EPOCHS=30 \
RN18_CIFAR10_SSL_EPOCHS=80 \
RN18_CIFAR10_LINEAR_EPOCHS=30 \
RN18_CIFAR10_SSL_SAVE_EVERY=5 \
sbatch jobs/run_rn18_cifar10_experiment.sbatch
```

Useful overrides for the corrected job:

```bash
RN18_CIFAR10_LEJEPA_OUTPUT_DIR=outputs/rn18_cifar10_true_lejepa_cudnn_lam02 \
RN18_CIFAR10_SSL_EPOCHS=800 \
RN18_CIFAR10_SSL_VIEWS=8 \
RN18_CIFAR10_LEJEPA_LAMBDA=0.02 \
RN18_CIFAR10_SSL_SAVE_EVERY=25 \
sbatch jobs/run_rn18_cifar10_true_lejepa_cudnn.sbatch
```

Smoke-test the code path without downloading CIFAR10:

```bash
/arf/home/skantar/anaconda3/envs/minitron/bin/python \
  scripts/train_rn18_cifar10_experiment.py \
  --output-dir outputs/rn18_cifar10_smoke \
  --smoke-test

/arf/home/skantar/anaconda3/envs/minitron/bin/python \
  scripts/analyze_rn18_cifar10_filters.py \
  --experiment-dir outputs/rn18_cifar10_smoke \
  --output-dir outputs/rn18_cifar10_smoke/analysis \
  --bins 20 \
  --max-density-points 20000
```

Main output files:

```text
outputs/rn18_cifar10_experiment/selected_pair.json
outputs/rn18_cifar10_experiment/matched_checkpoints.csv
outputs/rn18_cifar10_experiment/imagenet_finetune_metrics.csv
outputs/rn18_cifar10_experiment/lejepa_ssl_metrics.csv
outputs/rn18_cifar10_experiment/lejepa_linear_metrics.csv
outputs/rn18_cifar10_experiment/analysis/summary.md
outputs/rn18_cifar10_experiment/analysis/figures/
```

The most useful figures for the accuracy-matched comparison are:

```text
analysis/figures/accuracy_curves.png
analysis/figures/checkpoint_accuracy_matching.png
analysis/figures/drift_global_heatmap.png
analysis/figures/drift_layer_by_depth.png
analysis/figures/component_drift_contributions.png
analysis/figures/ridge_matched_pair.png
analysis/figures/c0_c1_density_matched.png
analysis/figures/global_component_residual_histograms.png
analysis/figures/entropy_by_depth.png
analysis/figures/sparsity_by_depth.png
```

## 8. Recommended Run Order

For a first serious run:

1. Run `scripts/check_dataset.py`.
2. Start Jupyter on a compute node with enough memory.
3. Run notebook cells through metadata loading.
4. Run scaling.
5. Run global PCA.
6. Inspect explained variance and covariance plots.
7. Run one subset PCA, such as `fractals`.
8. Run one ridge plot by `Visual Category`.
9. Run one KL matrix by `Task`.
10. Run the scatter plot.
11. Only then run the filter-quality section.

This order lets you catch problems early before spending time on the heaviest analysis.

## 9. Useful Monitoring Commands

Check jobs:

```bash
squeue -u $USER
```

Check completed job status:

```bash
sacct -j <job_id> --format=JobID,State,Elapsed,Timelimit,CPUTime,ExitCode,NodeList
```

Watch Slurm logs:

```bash
tail -f logs/cnn-filter-db-data-<job_id>.out
tail -f logs/cnn-filter-db-data-<job_id>.err
```

Check memory while a notebook is running:

```bash
free -h
```

Check dataset size:

```bash
du -sh data/dataset.h5
```

## 10. Interpreting Results

Use this mental model:

1. PCA creates a coordinate system for all `3x3` filters.
2. Ridge plots show how groups occupy that coordinate system.
3. KL matrices measure how different those group distributions are.
4. Scatter plots show the overall density of common and rare filters.
5. Quality metrics look for layers with low variation or near-zero filters.

Together, these experiments investigate whether learned CNN filters change systematically with training data, task, architecture depth, and model quality.
