#!/usr/bin/env bash
#SBATCH --job-name=cnn-filter-db-pca
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@180

set -euo pipefail

REPO_DIR="${CNN_FILTER_DB_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATASET_PATH="${CNN_FILTER_DB_DATASET:-${REPO_DIR}/data/dataset.h5}"
OUTPUT_PATH="${CNN_FILTER_DB_PCA_OUTPUT:-outputs/(4thEXP)rn18_cifar10_filter_dynamics/original_cnn_filter_db_pca/cnn_filter_db_full_pca.npz}"
CHECKPOINT_PATH="${CNN_FILTER_DB_PCA_CHECKPOINT:-${OUTPUT_PATH%.npz}.checkpoint.npz}"
ENV_NAME="${CNN_FILTER_DB_CONDA_ENV:-cnn-filter-db}"
CHUNK_SIZE="${CNN_FILTER_DB_PCA_CHUNK_SIZE:-5000000}"
CHECKPOINT_EVERY="${CNN_FILTER_DB_PCA_CHECKPOINT_EVERY:-5}"
MAX_FILTERS="${CNN_FILTER_DB_PCA_MAX_FILTERS:-0}"
MEAN_TOLERANCE="${CNN_FILTER_DB_PCA_MEAN_TOLERANCE:-0.00005}"

cd "${REPO_DIR}"
mkdir -p logs "$(dirname "${OUTPUT_PATH}")" "$(dirname "${CHECKPOINT_PATH}")"

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Repo: ${REPO_DIR}"
echo "[$(date)] Dataset: ${DATASET_PATH}"
echo "[$(date)] Output: ${OUTPUT_PATH}"
echo "[$(date)] Checkpoint: ${CHECKPOINT_PATH}"
echo "[$(date)] Chunk size: ${CHUNK_SIZE}"
echo "[$(date)] Maximum filters: ${MAX_FILTERS}"
df -h "$(dirname "${DATASET_PATH}")" "$(dirname "${OUTPUT_PATH}")"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -V
python - <<'PY'
import h5py
import numpy as np
print("numpy:", np.__version__)
print("h5py:", h5py.__version__)
PY

EXTRA_ARGS=()
if [[ "${CNN_FILTER_DB_PCA_FORCE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--force)
fi
if [[ "${CNN_FILTER_DB_PCA_ALLOW_MEAN_MISMATCH:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--allow-published-mean-mismatch)
fi

python scripts/compute_cnn_filter_db_pca.py \
  --dataset "${DATASET_PATH}" \
  --output "${OUTPUT_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --chunk-size "${CHUNK_SIZE}" \
  --checkpoint-every-chunks "${CHECKPOINT_EVERY}" \
  --max-filters "${MAX_FILTERS}" \
  --published-mean-tolerance "${MEAN_TOLERANCE}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date)] Final artifact:"
ls -lh "${OUTPUT_PATH}" "${CHECKPOINT_PATH}"
echo "[$(date)] Done"
