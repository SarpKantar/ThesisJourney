#!/usr/bin/env bash
#SBATCH --job-name=rn18-exp3-global-pca-figs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G

# Figure/statistics regeneration only. This job never trains either model.
set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
EXP3_DIR="${RN18_EXP3_SOURCE_DIR:-outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong}"
EXP4_ROOT="${RN18_EXP4_OUTPUT_ROOT:-outputs/(4thEXP)rn18_cifar10_filter_dynamics}"
BASIS_PATH="${CNN_FILTER_DB_PCA_BASIS:-${EXP4_ROOT}/original_cnn_filter_db_pca/cnn_filter_db_full_pca.npz}"
IMAGENET_CHECKPOINT="${RN18_EXP3_IMAGENET_CHECKPOINT:-${EXP4_ROOT}/regenerated_imagenet_baseline/checkpoints/imagenet_ft_epoch001.pth}"
LEJEPA_CHECKPOINT="${RN18_EXP3_LEJEPA_CHECKPOINT:-${EXP3_DIR}/checkpoints/lejepa_ssl_epoch110.pth}"
OUTPUT_DIR="${RN18_EXP3_GLOBAL_PCA_OUTPUT:-${EXP3_DIR}/analysis_global_cnn_filter_db_basis_epoch001_vs_epoch110}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

for required in "${BASIS_PATH}" "${IMAGENET_CHECKPOINT}" "${LEJEPA_CHECKPOINT}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

echo "[$(date)] Figure-only Experiment 3 rerun; no training"
echo "[$(date)] Full-DB PCA: ${BASIS_PATH}"
echo "[$(date)] ImageNet checkpoint: ${IMAGENET_CHECKPOINT}"
echo "[$(date)] LeJEPA checkpoint: ${LEJEPA_CHECKPOINT}"
echo "[$(date)] Output: ${OUTPUT_DIR}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python scripts/analyze_rn18_cifar10_filters.py \
  --experiment-dir "${EXP3_DIR}" \
  --output-dir "${OUTPUT_DIR}/pca_shape_analysis" \
  --pca-basis "${BASIS_PATH}" \
  --basis-label "PCA recomputed from all 1,464,797,156 released CNN Filter DB kernels" \
  --imagenet-checkpoint "${IMAGENET_CHECKPOINT}" \
  --lejepa-checkpoint "${LEJEPA_CHECKPOINT}" \
  --imagenet-accuracy 0.9125 \
  --lejepa-accuracy 0.8913 \
  --imagenet-epoch 1 \
  --lejepa-epoch 110 \
  --no-include-stem \
  --bins 70 \
  --max-density-points 400000

# These controls are PCA-independent, but regenerating them provides the full
# Experiment 3 figure suite against the byte-reproducible epoch-1 checkpoint.
python scripts/analyze_rn18_cifar10_kernel_distributions.py \
  --experiment-dir "${EXP3_DIR}" \
  --output-dir "${OUTPUT_DIR}/kernel_distributions" \
  --imagenet-checkpoint "${IMAGENET_CHECKPOINT}" \
  --lejepa-checkpoint "${LEJEPA_CHECKPOINT}" \
  --imagenet-accuracy 0.9125 \
  --lejepa-accuracy 0.8913 \
  --imagenet-epoch 1 \
  --lejepa-epoch 110 \
  --skip-selection-validation \
  --bins 120 \
  --max-stat-samples 250000

for required in \
  "${OUTPUT_DIR}/pca_shape_analysis/summary.md" \
  "${OUTPUT_DIR}/pca_shape_analysis/pca_basis_provenance.json" \
  "${OUTPUT_DIR}/pca_shape_analysis/figures/drift_global_heatmap.png" \
  "${OUTPUT_DIR}/pca_shape_analysis/figures/drift_global_chart.png" \
  "${OUTPUT_DIR}/pca_shape_analysis/figures/drift_stage_chart.png" \
  "${OUTPUT_DIR}/kernel_distributions/summary.md" \
  "${OUTPUT_DIR}/kernel_distributions/figures/layerwise_weight_distance_heatmap.png" \
  "${OUTPUT_DIR}/kernel_distributions/figures/layerwise_weight_distance_chart.png" \
  "${OUTPUT_DIR}/kernel_distributions/figures/layerwise_descriptor_distance_heatmap.png" \
  "${OUTPUT_DIR}/kernel_distributions/figures/layerwise_descriptor_distance_chart.png"; do
  test -s "${required}"
done

echo "[$(date)] Verified complete figure-only Experiment 3 rerun"
