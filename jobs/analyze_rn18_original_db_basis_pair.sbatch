#!/usr/bin/env bash
#SBATCH --job-name=rn18-exp4-paper-pca-pair
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
EXP3_DIR="${RN18_EXP4_SOURCE_DIR:-outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong}"
EXP4_ROOT="${RN18_EXP4_OUTPUT_ROOT:-outputs/(4thEXP)rn18_cifar10_filter_dynamics}"
BASIS_PATH="${CNN_FILTER_DB_PCA_BASIS:-${EXP4_ROOT}/original_cnn_filter_db_pca/cnn_filter_db_full_pca.npz}"
IMAGENET_CHECKPOINT="${RN18_EXP4_IMAGENET_CHECKPOINT:-${EXP4_ROOT}/regenerated_imagenet_baseline/checkpoints/imagenet_ft_epoch001.pth}"
LEJEPA_CHECKPOINT="${RN18_EXP4_LEJEPA_CHECKPOINT:-${EXP3_DIR}/checkpoints/lejepa_ssl_epoch110.pth}"
PAIR_OUTPUT="${RN18_EXP4_PAIR_OUTPUT:-${EXP4_ROOT}/part2_original_basis_pair}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

cd "${REPO_DIR}"
mkdir -p "${PAIR_OUTPUT}"

for required in "${BASIS_PATH}" "${IMAGENET_CHECKPOINT}" "${LEJEPA_CHECKPOINT}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Source experiment: ${EXP3_DIR}"
echo "[$(date)] Full-DB PCA: ${BASIS_PATH}"
echo "[$(date)] Regenerated ImageNet epoch 1: ${IMAGENET_CHECKPOINT}"
echo "[$(date)] LeJEPA epoch 110: ${LEJEPA_CHECKPOINT}"
echo "[$(date)] Output: ${PAIR_OUTPUT}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python scripts/analyze_rn18_cifar10_filters.py \
  --experiment-dir "${EXP3_DIR}" \
  --output-dir "${PAIR_OUTPUT}/pca_shape_analysis" \
  --pca-basis "${BASIS_PATH}" \
  --basis-label "PCA recomputed from all 1,464,797,156 CNN Filter DB kernels" \
  --imagenet-checkpoint "${IMAGENET_CHECKPOINT}" \
  --lejepa-checkpoint "${LEJEPA_CHECKPOINT}" \
  --imagenet-accuracy 0.9125 \
  --lejepa-accuracy 0.8913 \
  --imagenet-epoch 1 \
  --lejepa-epoch 110 \
  --no-include-stem \
  --bins 70 \
  --max-density-points 400000

# These controls are PCA-independent, but rerunning them against the regenerated
# epoch-1 baseline gives Part 2 a complete figure suite and tests regeneration.
python scripts/analyze_rn18_cifar10_kernel_distributions.py \
  --experiment-dir "${EXP3_DIR}" \
  --output-dir "${PAIR_OUTPUT}/kernel_distributions" \
  --imagenet-checkpoint "${IMAGENET_CHECKPOINT}" \
  --lejepa-checkpoint "${LEJEPA_CHECKPOINT}" \
  --imagenet-accuracy 0.9125 \
  --lejepa-accuracy 0.8913 \
  --imagenet-epoch 1 \
  --lejepa-epoch 110 \
  --skip-selection-validation \
  --bins 120 \
  --max-stat-samples 250000

test -s "${PAIR_OUTPUT}/pca_shape_analysis/summary.md"
test -s "${PAIR_OUTPUT}/pca_shape_analysis/pca_basis_provenance.json"
test -s "${PAIR_OUTPUT}/pca_shape_analysis/figures/drift_global_heatmap.png"
test -s "${PAIR_OUTPUT}/pca_shape_analysis/figures/drift_global_chart.png"
test -s "${PAIR_OUTPUT}/pca_shape_analysis/figures/drift_stage_chart.png"
test -s "${PAIR_OUTPUT}/pca_shape_analysis/figures/drift_layer_by_depth.png"
test -s "${PAIR_OUTPUT}/kernel_distributions/summary.md"
test -s "${PAIR_OUTPUT}/kernel_distributions/figures/global_kernel_descriptor_distributions.png"
test -s "${PAIR_OUTPUT}/kernel_distributions/figures/layerwise_weight_distance_heatmap.png"
test -s "${PAIR_OUTPUT}/kernel_distributions/figures/layerwise_weight_distance_chart.png"
test -s "${PAIR_OUTPUT}/kernel_distributions/figures/layerwise_descriptor_distance_heatmap.png"
test -s "${PAIR_OUTPUT}/kernel_distributions/figures/layerwise_descriptor_distance_chart.png"
test -s "${PAIR_OUTPUT}/kernel_distributions/figures/spatial_energy_maps/spatial_energy_global_chart.png"

echo "[$(date)] Verified pair analysis outputs under ${PAIR_OUTPUT}"
echo "[$(date)] Done"
