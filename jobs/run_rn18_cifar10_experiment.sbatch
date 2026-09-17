#!/usr/bin/env bash
#SBATCH --job-name=rn18-cifar10-exp3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUTPUT_DIR="${RN18_CIFAR10_OUTPUT_DIR:-outputs/rn18_cifar10_experiment}"
DATA_DIR="${RN18_CIFAR10_DATA_DIR:-data/cifar10}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

IMAGENET_EPOCHS="${RN18_CIFAR10_IMAGENET_EPOCHS:-20}"
SSL_EPOCHS="${RN18_CIFAR10_SSL_EPOCHS:-40}"
LINEAR_EPOCHS="${RN18_CIFAR10_LINEAR_EPOCHS:-20}"
BATCH_SIZE="${RN18_CIFAR10_BATCH_SIZE:-128}"
SSL_SAVE_EVERY="${RN18_CIFAR10_SSL_SAVE_EVERY:-5}"
QUICK_PROBE_EPOCHS="${RN18_CIFAR10_QUICK_PROBE_EPOCHS:-3}"
QUICK_PROBE_INTERVAL="${RN18_CIFAR10_QUICK_PROBE_INTERVAL:-5}"
SSL_MIN_CHECKPOINTS="${RN18_CIFAR10_SSL_MIN_CHECKPOINTS:-4}"
BINS="${RN18_CIFAR10_ANALYSIS_BINS:-70}"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Repo: ${REPO_DIR}"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] Data dir: ${DATA_DIR}"
echo "[$(date)] Conda env: ${ENV_NAME}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -V
python - <<'PY'
import importlib
for module in ["torch", "torchvision", "numpy", "pandas", "PIL"]:
    mod = importlib.import_module(module)
    print(f"{module}: {getattr(mod, '__version__', 'ok')}")
PY

python scripts/train_rn18_cifar10_experiment.py \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --imagenet-epochs "${IMAGENET_EPOCHS}" \
  --ssl-epochs "${SSL_EPOCHS}" \
  --linear-epochs "${LINEAR_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --ssl-save-every "${SSL_SAVE_EVERY}" \
  --quick-probe-epochs "${QUICK_PROBE_EPOCHS}" \
  --quick-probe-interval "${QUICK_PROBE_INTERVAL}" \
  --ssl-min-checkpoints "${SSL_MIN_CHECKPOINTS}" \
  --device cuda \
  --disable-cudnn \
  --amp

python scripts/analyze_rn18_cifar10_filters.py \
  --experiment-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/analysis" \
  --bins "${BINS}"

echo "[$(date)] Generated outputs:"
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort
echo "[$(date)] Done"
