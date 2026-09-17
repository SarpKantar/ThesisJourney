#!/usr/bin/env bash
#SBATCH --job-name=rn18-exp4-lejepa-dynamics
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=48G

set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
EXP3_DIR="${RN18_EXP4_SOURCE_DIR:-outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong}"
EXP4_ROOT="${RN18_EXP4_OUTPUT_ROOT:-outputs/(4thEXP)rn18_cifar10_filter_dynamics}"
BASIS_PATH="${CNN_FILTER_DB_PCA_BASIS:-${EXP4_ROOT}/original_cnn_filter_db_pca/cnn_filter_db_full_pca.npz}"
OUTPUT_DIR="${RN18_EXP4_DYNAMICS_OUTPUT:-${EXP4_ROOT}/lejepa_training_dynamics}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

if [[ ! -s "${BASIS_PATH}" ]]; then
  echo "Missing full CNN Filter DB PCA artifact: ${BASIS_PATH}" >&2
  exit 1
fi
for epoch in 001 010 020 030 040 050 060 070 080 090 100 110 120; do
  checkpoint="${EXP3_DIR}/checkpoints/lejepa_ssl_epoch${epoch}.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing canonical LeJEPA SSL checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Source checkpoints: ${EXP3_DIR}/checkpoints"
echo "[$(date)] Full-DB PCA: ${BASIS_PATH}"
echo "[$(date)] Scope: epochs 1,10,...,120; 16 residual-block 3x3 convolutions; stem excluded"
echo "[$(date)] This is analysis only; it does not train or mutate checkpoints."
echo "[$(date)] Output: ${OUTPUT_DIR}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python scripts/analyze_rn18_lejepa_training_dynamics.py \
  --experiment-dir "${EXP3_DIR}" \
  --pca-basis "${BASIS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-epochs "1,10,20,30,40,50,60,70,80,90,100,110,120" \
  --milestone-epochs "1,30,60,90,120" \
  --bins 70 \
  --max-density-points 250000 \
  --descriptor-max-samples 200000 \
  --seed 13

test -s "${OUTPUT_DIR}/summary.md"
test -s "${OUTPUT_DIR}/analysis_provenance.json"
test -s "${OUTPUT_DIR}/validation_checks.csv"
test -s "${OUTPUT_DIR}/global_pairwise_epoch_drift.csv"
test -s "${OUTPUT_DIR}/figures/global_pairwise_epoch_drift_heatmap.png"
test -s "${OUTPUT_DIR}/figures/global_pairwise_epoch_drift_chart.png"
test -s "${OUTPUT_DIR}/figures/stage_drift_trajectories.png"
test -s "${OUTPUT_DIR}/figures/layer_drift_trajectories_chart.png"
test -s "${OUTPUT_DIR}/figures/milestone_descriptor_distributions.png"

python - "${OUTPUT_DIR}/validation_checks.csv" <<'PY'
import sys
import pandas as pd

checks = pd.read_csv(sys.argv[1])
severity = checks.get("severity", pd.Series(index=checks.index, dtype=object)).fillna("error")
required = checks[severity != "warning"]
if not bool(required["passed"].all()):
    raise SystemExit("At least one required dynamics validation check failed")
warnings = checks[severity == "warning"]
print(f"required checks passed: {int(required['passed'].sum())}/{len(required)}")
for row in warnings.itertuples():
    print(f"recorded warning: {row.check}: {row.detail}")
PY

echo "[$(date)] Verified LeJEPA dynamics outputs under ${OUTPUT_DIR}"
echo "[$(date)] Done"
