# ThesisJourney / CNN Filter DB experiments

This repository extends [CNN Filter DB (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Gavrikov_CNN_Filter_DB_An_Empirical_Investigation_of_Trained_Convolutional_Filters_CVPR_2022_paper.html) with five local experiment families comparing learned `3x3` convolutional-filter structure across ImageNet transfer, supervised training, and LeJEPA/SIGReg training.

This README is the execution runbook for a new researcher or LLM. Run commands from the repository root, preserve the declared seeds and configs, and do not treat smoke tests, fake-data runs, or coefficient pilots as scientific results.

## Start here: Experiment 5

Experiment 5 is the current priority. Its frozen protocol is [configs/experiment5/main.json](configs/experiment5/main.json), its implementation notes are [docs/experiment5_protocol.md](docs/experiment5_protocol.md), and the full research rationale is [THESIS_EXPERIMENT_5_RESEARCH_PLAN.md](THESIS_EXPERIMENT_5_RESEARCH_PLAN.md).

For a new machine, this is the shortest safe path:

```bash
git clone --recurse-submodules https://github.com/SarpKantar/ThesisJourney.git
cd ThesisJourney
git submodule update --init --recursive

conda env create -f environment.yml
conda activate cnn-filter-db
mkdir -p logs

python -m pytest -q tests/test_experiment5.py
python scripts/validate_experiment5_losses.py --device cpu
python scripts/prepare_experiment5.py --data-dir data/cifar10 --download

# Code-path check only; never report this as an experiment result.
python scripts/train_experiment5.py \
  --arm B \
  --seed 13 \
  --device cpu \
  --num-workers 0 \
  --smoke-test \
  --output-dir /tmp/experiment5-smoke
```

The real run needs a CUDA GPU and is normally submitted through Slurm. First calibrate the comparator coefficient, then run the A/B/C screening array:

```bash
sbatch --partition=YOUR_GPU_PARTITION jobs/calibrate_experiment5.sbatch
sbatch --partition=YOUR_GPU_PARTITION jobs/run_experiment5_abc.sbatch
```

`run_experiment5_abc.sbatch` defaults to array tasks `0-8`: arms A/B/C for discovery seeds 13, 17, and 23. After those runs pass inspection, extend to seeds 31 and 43:

```bash
sbatch --partition=YOUR_GPU_PARTITION --array=9-14 jobs/run_experiment5_abc.sbatch
```

Do not submit the literal placeholder `YOUR_GPU_PARTITION`. Replace it with the local GPU partition, or omit `--partition` when the cluster's default partition accepts `--gres=gpu:1`. On TRUBA, the known working value is `kolyoz-cuda`.

After training, run diagnostics, final probes, and aggregation:

```bash
# The default diagnostics array covers the first nine A/B/C runs.
sbatch --partition=YOUR_GPU_PARTITION jobs/diagnose_experiment5.sbatch
sbatch --partition=YOUR_GPU_PARTITION jobs/probe_experiment5.sbatch

# Add the two remaining discovery seeds after tasks 9-14 finish.
sbatch --partition=YOUR_GPU_PARTITION --array=63-104 jobs/diagnose_experiment5.sbatch
sbatch --partition=YOUR_GPU_PARTITION --array=9-14 jobs/probe_experiment5.sbatch

EXP5_SEEDS=13,17,23,31,43 \
  sbatch --partition=YOUR_GPU_PARTITION jobs/aggregate_experiment5.sbatch
```

The complete Experiment 5 order is:

1. `prepare_experiment5.sbatch`: download/verify CIFAR10 and create the immutable split artifact.
2. `validate_experiment5.sbatch`: check LeJEPA loss parity and run all unit tests.
3. `run_experiment5_linear_toy.sbatch`: execute the five-seed analytic control.
4. `calibrate_experiment5.sbatch`: write `pilot/coefficient_ledger.json`.
5. Optionally run `run_experiment5_pilot_sensitivity.sbatch`; it is expensive and remains a pilot.
6. `run_experiment5_abc.sbatch`: run equal-initialization arms A, B, and C.
7. `diagnose_experiment5.sbatch`: calculate weight and representation diagnostics.
8. `probe_experiment5.sbatch`: train the fixed 1%, 10%, and 100% linear probes.
9. `aggregate_experiment5.sbatch`: aggregate independent seed-level contrasts.

`check_experiment5_weight_falsifications.sbatch` is also part of the protocol, but it requires Experiment 3's epoch-110 LeJEPA checkpoint. Run it after Experiment 3, or set `EXP5_EXISTING_CHECKPOINT` to an equivalent checkpoint.

Experiment 5A/5B and their analyses are implemented. The learned local candidate (5C) and replacement/timing intervention (5D) are intentionally gated: their statistic, coefficient, and layer scope must be selected from completed A/B/C discovery evidence and frozen before those branches are implemented. A new LLM must not invent 5C/5D defaults prematurely.

## Portable environment setup

### Clone correctly

LeJEPA is a Git submodule pinned to commit `c293d291ca87cd4fddee9d3fffe4e914c7272052`. Experiment 3 and Experiment 5 load source files from `lejepa/lejepa`, so a clone without the submodule is incomplete.

```bash
git clone --recurse-submodules https://github.com/SarpKantar/ThesisJourney.git
cd ThesisJourney
git submodule update --init --recursive
git -C lejepa rev-parse HEAD
git -C lejepa status --short
```

The first command printed by `rev-parse` must be the pinned commit above, and the submodule should be clean. Experiment 5 refuses a dirty or mismatched LeJEPA checkout unless `--allow-dirty-lejepa` is deliberately supplied and recorded.

### Create the Python environment

The experiment code uses Python 3.10 syntax and was exercised with PyTorch 2.5.1 / torchvision 0.20.1. The old upstream Python-3.8-only environment is not sufficient for the numbered experiments.

```bash
conda env create -f environment.yml
conda activate cnn-filter-db
python -V
python -m pip check
```

If the cluster requires a site-specific CUDA build, create the Python 3.10 environment, install the matching PyTorch 2.5.1 / torchvision 0.20.1 wheels using the cluster's CUDA instructions, and then run `pip install -r requirements.txt`. Do not silently change the PyTorch/torchvision version pair between arms of one experiment.

Basic source checks:

```bash
python -m compileall -q scripts tests
python -m pytest -q tests/test_experiment5.py
```

### Slurm on another cluster

All launchers resolve the repository from `SLURM_SUBMIT_DIR` or their own file location. They no longer depend on an `/arf/home/...` checkout. Submit from the repository root.

The launchers activate the `cnn-filter-db` Conda environment through `jobs/_common.sh`. If Conda is not visible inside a batch shell, export its activation script before submission:

```bash
export CNN_FILTER_DB_CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
```

Useful overrides are:

```bash
export EXP5_CONDA_ENV=cnn-filter-db
export RN18_CIFAR10_CONDA_ENV=cnn-filter-db
export RN50_FILTER_CONDA_ENV=cnn-filter-db
export CNN_FILTER_DB_CONDA_ENV=cnn-filter-db
```

If the environment is already inherited by compute nodes, set `CNN_FILTER_DB_SKIP_CONDA=1`. Review each launcher's `#SBATCH` CPU, memory, GPU, and time requests against the destination cluster. Partition directives are intentionally supplied at submission time rather than tied to one cluster.

## Experiment map and dependencies

| Experiment | Main action | Required inputs | Canonical launcher(s) |
|---|---|---|---|
| 1 | Analyze three RN50 filter populations | Three versioned RN50 checkpoints in `models/` | `analyze_rn50_filters.sbatch` |
| 2 | Signed LeJEPA-minus-ImageNet RN50 residual analysis | Experiment 1 checkpoints and PCA output | `analyze_lejepa_imagenet_residuals.sbatch` |
| 3 | Train canonical RN18/CIFAR10 ImageNet-transfer and LeJEPA routes | CIFAR10, pinned LeJEPA | `regenerate_rn18_imagenet_baseline.sbatch`, then `run_rn18_cifar10_true_lejepa_cudnn.sbatch` |
| 4 | Threshold robustness, full-database PCA pair analysis, and training dynamics | Experiment 3 checkpoints plus the approximately 99 GiB CNN Filter DB HDF5 file | `download_dataset.sbatch`, `compute_cnn_filter_db_pca.sbatch`, and three RN18 analysis jobs |
| 5 | Controlled SIGReg structure study | CIFAR10, pinned LeJEPA, frozen config | Experiment 5 launchers listed above |

Generated data, logs, checkpoints, and `outputs/` are intentionally Git-ignored. Copy those separately to durable storage when moving an in-progress run. Git alone carries code, configs, documentation, the pinned LeJEPA source reference, and the three irreplaceable Experiment 1 inputs.

## Experiment 1: RN50 three-route filter analysis

The required checkpoints are versioned in `models/`:

```text
models/rn50_rand_ft_backbone.pth
models/rn50_imagenet_ft_backbone.pth
models/rn50_lejepa_ssl_backbone.pth
```

Run locally on a sufficiently large CPU node:

```bash
python scripts/analyze_rn50_filters.py \
  --model-dir models \
  --output-dir 'outputs/(1stEXP)rn50_filter_analysis' \
  --bins 70 \
  --bootstrap-reps 30 \
  --bootstrap-sample-size 100000 \
  --seed 13
```

Or submit:

```bash
sbatch jobs/analyze_rn50_filters.sbatch
```

The three checkpoint SHA-256 values are:

```text
925c6b2304ad0f2ba0b2d2ae46890878f3d0b04d83abc4834212019a8d5fe121  rn50_rand_ft_backbone.pth
7efc2e41ca36c85a59e385d0a842d51cd35e1c0411828e3fae65ea44de7355f8  rn50_imagenet_ft_backbone.pth
c411236e72997a5f285ec86ab340428619ed86570fecac1923d126e042cdb633  rn50_lejepa_ssl_backbone.pth
```

## Experiment 2: RN50 signed residuals

Experiment 2 is analysis-only and depends on Experiment 1's saved PCA artifact.

```bash
python scripts/analyze_lejepa_imagenet_residuals.py \
  --model-dir models \
  --source-analysis-dir 'outputs/(1stEXP)rn50_filter_analysis' \
  --output-dir 'outputs/(2ndEXP)rn50_lejepa_imagenet_residuals' \
  --bins 70 \
  --seed 13
```

Or submit after Experiment 1 succeeds:

```bash
sbatch jobs/analyze_lejepa_imagenet_residuals.sbatch
```

## Experiment 3: canonical RN18/CIFAR10 comparison

Do not use the historical EMA/predictor lineage as LeJEPA. The canonical route uses view-center alignment plus the pinned SIGReg implementation.

First regenerate the 20-epoch ImageNet-initialized supervised baseline:

```bash
sbatch --partition=YOUR_GPU_PARTITION jobs/regenerate_rn18_imagenet_baseline.sbatch
```

Then train the canonical 120-epoch, four-view, `lambda=0.02`, 64-dimensional LeJEPA route and its 50-epoch frozen probes:

```bash
sbatch --partition=YOUR_GPU_PARTITION jobs/run_rn18_cifar10_true_lejepa_cudnn.sbatch
```

The second launcher now defaults to the canonical `p64 strong` configuration and writes:

```text
outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/
```

It also produces PCA-independent kernel controls and the stem-excluded selected-pair PCA analysis under `analysis_selected_pair/`. CIFAR10 is downloaded automatically into `data/cifar10` if absent.

`jobs/run_rn18_cifar10_experiment*.sbatch` and `jobs/probe_rn18_cifar10_lejepa_linear_cudnn.sbatch` remain for historical reproduction and diagnostics. They are not the canonical final Experiment 3 command.

## Experiment 4: full-database PCA and robustness

Experiment 4 reuses Experiment 3's 13 LeJEPA checkpoints (epochs 1, 10, ..., 120) and the regenerated ImageNet epoch-1 checkpoint. It also requires the released CNN Filter DB `dataset.h5`, which is approximately 99 GiB after download/decompression. Use scratch storage with comfortably more free space than the final file.

Download and verify the database:

```bash
export CNN_FILTER_DB_DATA_DIR=/path/on/large/scratch/cnn-filter-db
sbatch jobs/download_dataset.sbatch
export CNN_FILTER_DB_DATASET="$CNN_FILTER_DB_DATA_DIR/dataset.h5"
```

For a direct local check:

```bash
python scripts/check_dataset.py --dataset "$CNN_FILTER_DB_DATASET" --sample-size 100000
```

Compute the resumable full-database PCA. The released data's recomputed mean differs from the vector printed in the paper supplement; the completed local protocol records and explicitly accepts that discrepancy after its internal checks pass:

```bash
export CNN_FILTER_DB_PCA_ALLOW_MEAN_MISMATCH=1
sbatch jobs/compute_cnn_filter_db_pca.sbatch
```

Then run the three Experiment 4 analyses:

```bash
sbatch --partition=YOUR_GPU_PARTITION jobs/analyze_rn18_threshold_sensitivity.sbatch
sbatch --partition=YOUR_GPU_PARTITION jobs/analyze_rn18_original_db_basis_pair.sbatch
sbatch --partition=YOUR_GPU_PARTITION jobs/analyze_rn18_lejepa_training_dynamics.sbatch
```

The PCA job checkpoints sufficient statistics periodically and can resume. Do not delete its `.checkpoint.npz` while the full pass is incomplete.

## Experiment 5 details and failure policy

Arms in `train_experiment5.py` are:

- A: alignment only.
- B: alignment plus SIGReg.
- C: alignment plus the calibrated VICReg-style variance/covariance comparator.

The primary endpoint is the equal-layer mean normalized weight-energy effective rank over the 16 residual `3x3` matrices at update 5,250 (epoch 30), with B-C as the frozen contrast. The raw config file SHA-256 is:

```text
f8a8e2ae9dcbc40d8be0bd0c32e751568ff3f01eb1a7a71ce9333c97da66eaa2  configs/experiment5/main.json
```

Resume only from an exact saved checkpoint:

```bash
python scripts/train_experiment5.py \
  --arm B \
  --seed 13 \
  --device cuda \
  --resume 'outputs/(5thEXP)rn18_cifar10_sigreg_structure/discovery/arm_B/seed_13/checkpoints/update_005250.pth'
```

The checkpoint restores model/optimizer state, BatchNorm buffers, random-number states, data position, and SIGReg direction state. Finite collapsed runs remain in the result set. Nonfinite loss or gradients stop the run and write `failure.json`; do not introduce arm-specific recovery or clipping.

Monitor with:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,Timelimit,ExitCode,NodeList
tail -f logs/JOB_LOG.out
```

## What to inspect before claiming success

A new LLM should verify all of the following:

- The root repository and LeJEPA submodule commits are recorded in each run manifest.
- `scientific_result` is true; fake-data and smoke runs are excluded.
- A/B/C runs for one seed have identical initialization hashes.
- The coefficient ledger exists before any C run.
- All expected checkpoints exist, especially updates 5,250 and 21,000.
- Weight diagnostics, representation diagnostics, and probe results exist for every included seed.
- Aggregation treats independently trained seeds—not filters, images, layers, or forked descendants—as the inferential unit.
- Test-set results were not used to invent a new 5C/5D penalty and then presented as confirmation.

For the scientific history, exact completed results, caveats, and path lineage of Experiments 1-4, read [(First4Experiments)EXPERIMENTS_HANDOFF.mf](<(First4Experiments)EXPERIMENTS_HANDOFF.mf>). When prose and implementation differ, use this priority: frozen config, source script, launcher, run manifest, then narrative report.

## Original CNN Filter DB dataset

Dataset version 1.0.0 is available from [Zenodo](https://doi.org/10.5281/zenodo.6371680). The HDF5 file contains:

- `/filters`: `(N, 3, 3)` filters; reshape a filter to nine coefficients where needed.
- `/meta`: a pandas table containing model, architecture, task, dataset, depth, and filter-ID metadata.

Set `CNN_FILTER_DB_DATASET` to keep the database outside the Git checkout:

```bash
export CNN_FILTER_DB_DATASET=/absolute/path/to/dataset.h5
python scripts/check_dataset.py
```

The original upstream notebook remains useful for paper-era exploration, but the numbered experiment scripts and this README are the authoritative execution path for this thesis repository.

## Citation and license

If using CNN Filter DB, cite:

```bibtex
@InProceedings{Gavrikov_2022_CVPR,
  author    = {Gavrikov, Paul and Keuper, Janis},
  title     = {CNN Filter DB: An Empirical Investigation of Trained Convolutional Filters},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  pages     = {19066--19076}
}

@dataset{cnnfilterdb2022,
  author    = {Paul Gavrikov and Janis Keuper},
  title     = {CNN-Filter-DB},
  month     = jun,
  year      = 2022,
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.6371680}
}
```

The upstream project is licensed under [CC BY-SA 4.0](LICENSE).
