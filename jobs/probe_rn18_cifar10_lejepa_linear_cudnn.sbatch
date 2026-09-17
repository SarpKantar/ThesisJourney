#!/usr/bin/env bash
#SBATCH --job-name=rn18-cifar10-lejepa-probe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SOURCE_DIR="${RN18_CIFAR10_PROBE_SOURCE_DIR:-outputs/rn18_cifar10_true_lejepa_cudnn_250_noearlystop}"
OUTPUT_DIR="${RN18_CIFAR10_PROBE_OUTPUT_DIR:-outputs/rn18_cifar10_true_lejepa_probe_layernorm_adamw_100}"
IMAGENET_SOURCE_DIR="${RN18_CIFAR10_IMAGENET_SOURCE_DIR:-outputs/rn18_cifar10_experiment_cudnn}"
DATA_DIR="${RN18_CIFAR10_DATA_DIR:-data/cifar10}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"
CHECKPOINT_EPOCHS="${RN18_CIFAR10_PROBE_CHECKPOINT_EPOCHS:-80,90,100,110,120,130,140,150,180,190,220,250}"
LINEAR_EPOCHS="${RN18_CIFAR10_LINEAR_EPOCHS:-100}"
BATCH_SIZE="${RN18_CIFAR10_BATCH_SIZE:-256}"
LINEAR_LR="${RN18_CIFAR10_LINEAR_LR:-1e-3}"
LINEAR_WEIGHT_DECAY="${RN18_CIFAR10_LINEAR_WEIGHT_DECAY:-1e-6}"
LINEAR_OPTIMIZER="${RN18_CIFAR10_LINEAR_OPTIMIZER:-adamw}"
LINEAR_HEAD_TYPE="${RN18_CIFAR10_LINEAR_HEAD_TYPE:-layernorm_linear}"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Source dir: ${SOURCE_DIR}"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] Probe: epochs=${LINEAR_EPOCHS} head=${LINEAR_HEAD_TYPE} opt=${LINEAR_OPTIMIZER} lr=${LINEAR_LR} wd=${LINEAR_WEIGHT_DECAY}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -V
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cudnn version:", torch.backends.cudnn.version())
    x = torch.randn(8, 3, 32, 32, device="cuda")
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    torch.cuda.synchronize()
    print("cudnn probe ok:", tuple(y.shape))
PY

python scripts/probe_rn18_cifar10_lejepa_checkpoints.py \
  --source-experiment-dir "${SOURCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --imagenet-source-dir "${IMAGENET_SOURCE_DIR}" \
  --checkpoint-epochs "${CHECKPOINT_EPOCHS}" \
  --linear-epochs "${LINEAR_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --linear-lr "${LINEAR_LR}" \
  --linear-weight-decay "${LINEAR_WEIGHT_DECAY}" \
  --linear-optimizer "${LINEAR_OPTIMIZER}" \
  --linear-head-type "${LINEAR_HEAD_TYPE}" \
  --device cuda \
  --amp

python scripts/analyze_rn18_cifar10_filters.py \
  --experiment-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/analysis" \
  --bins "${RN18_CIFAR10_ANALYSIS_BINS:-70}"

echo "[$(date)] Done"
