#!/usr/bin/env bash
#SBATCH --job-name=rn18-exp4-thresholds
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=24G

set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
EXP3_DIR="${RN18_EXP4_SOURCE_DIR:-outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong}"
EXP4_ROOT="${RN18_EXP4_OUTPUT_ROOT:-outputs/(4thEXP)rn18_cifar10_filter_dynamics}"
IMAGENET_CHECKPOINT="${RN18_EXP4_IMAGENET_CHECKPOINT:-${EXP4_ROOT}/regenerated_imagenet_baseline/checkpoints/imagenet_ft_epoch001.pth}"
LEJEPA_CHECKPOINT="${RN18_EXP4_LEJEPA_CHECKPOINT:-${EXP3_DIR}/checkpoints/lejepa_ssl_epoch110.pth}"
OUTPUT_DIR="${RN18_EXP4_THRESHOLD_OUTPUT:-${EXP4_ROOT}/part1_threshold_sensitivity}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"

for required in "${IMAGENET_CHECKPOINT}" "${LEJEPA_CHECKPOINT}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] ImageNet epoch 1: ${IMAGENET_CHECKPOINT}"
echo "[$(date)] LeJEPA epoch 110: ${LEJEPA_CHECKPOINT}"
echo "[$(date)] Scope: 16 residual-block 3x3 convolutions; stem excluded"
echo "[$(date)] Output: ${OUTPUT_DIR}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python scripts/analyze_rn18_threshold_sensitivity.py \
  --model-spec "imagenet_ft_epoch001::ImageNet FT epoch 1 (regenerated)::${IMAGENET_CHECKPOINT}" \
  --model-spec "lejepa_ssl_epoch110::LeJEPA SSL epoch 110::${LEJEPA_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --sparsity-epsilons "0.001,0.0025,0.005,0.01,0.02,0.05,0.1" \
  --low-entropy-cutoffs "0.30,0.40,0.50,0.60,0.70,0.80,0.90" \
  --random-ratio-cutoffs "0.75,0.80,0.85,0.90,0.95,1.00,1.05" \
  --compute-clean-entropy

test -s "${OUTPUT_DIR}/summary.md"
test -s "${OUTPUT_DIR}/regression_checks.csv"
test -s "${OUTPUT_DIR}/figures/sparsity_threshold_by_depth_heatmap.png"
test -s "${OUTPUT_DIR}/figures/low_entropy_decision_heatmap.png"
test "$(find "${OUTPUT_DIR}/figures/sparsity_threshold_by_depth_heatmap" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 7
test "$(find "${OUTPUT_DIR}/figures/low_entropy_decision_heatmap" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 7
test "$(find "${OUTPUT_DIR}/figures/random_like_decision_heatmap" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 7
test "$(find "${OUTPUT_DIR}/figures/clean_entropy_after_sparse_exclusion_heatmap" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 7

python - "${OUTPUT_DIR}/regression_checks.csv" <<'PY'
import sys
import pandas as pd

checks = pd.read_csv(sys.argv[1])
if not bool(checks["passed"].all()):
    raise SystemExit("At least one legacy-threshold regression check failed")
print(f"legacy regression checks passed: {len(checks)}/{len(checks)}")
PY

echo "[$(date)] Verified threshold-sensitivity outputs under ${OUTPUT_DIR}"
echo "[$(date)] Done"
