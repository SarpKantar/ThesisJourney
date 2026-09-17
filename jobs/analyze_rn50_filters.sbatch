#!/usr/bin/env bash
#SBATCH --job-name=rn50-filter-analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="${RN50_FILTER_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUTPUT_DIR="${RN50_FILTER_OUTPUT_DIR:-outputs/(1stEXP)rn50_filter_analysis}"
MODEL_DIR="${RN50_FILTER_MODEL_DIR:-models}"
ENV_NAME="${RN50_FILTER_CONDA_ENV:-cnn-filter-db}"
BOOTSTRAP_REPS="${RN50_FILTER_BOOTSTRAP_REPS:-30}"
BOOTSTRAP_SAMPLE_SIZE="${RN50_FILTER_BOOTSTRAP_SAMPLE_SIZE:-100000}"
MAX_SCATTER_POINTS="${RN50_FILTER_MAX_SCATTER_POINTS:-600000}"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Repo: ${REPO_DIR}"
echo "[$(date)] Model dir: ${MODEL_DIR}"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] Conda env: ${ENV_NAME}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -V
python - <<'PY'
import importlib.util
for module in ["torch", "numpy", "scipy", "sklearn", "PIL", "pandas"]:
    print(f"{module}: {bool(importlib.util.find_spec(module))}")
PY

python scripts/analyze_rn50_filters.py \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --bins 70 \
  --bootstrap-reps "${BOOTSTRAP_REPS}" \
  --bootstrap-sample-size "${BOOTSTRAP_SAMPLE_SIZE}" \
  --max-scatter-points "${MAX_SCATTER_POINTS}"

echo "[$(date)] Generated outputs:"
find "${OUTPUT_DIR}" -maxdepth 2 -type f | sort
echo "[$(date)] Done"
