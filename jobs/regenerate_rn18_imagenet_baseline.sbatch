#!/usr/bin/env bash
#SBATCH --job-name=rn18-exp4-imagenet-regen
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
OUTPUT_DIR="${RN18_CIFAR10_IMAGENET_REGEN_OUTPUT_DIR:-outputs/(4thEXP)rn18_cifar10_filter_dynamics/regenerated_imagenet_baseline}"
DATA_DIR="${RN18_CIFAR10_DATA_DIR:-data/cifar10}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Repo: ${REPO_DIR}"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] Data dir: ${DATA_DIR}"
echo "[$(date)] Conda env: ${ENV_NAME}"
echo "[$(date)] Protocol: ImageNet1K_V1 RN18, CIFAR stem, seed=13, batch=128, epochs=20"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python -V
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cudnn available:", torch.backends.cudnn.is_available())
print("cudnn version:", torch.backends.cudnn.version())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("gpu:", torch.cuda.get_device_name(0))
x = torch.randn(8, 3, 32, 32, device="cuda")
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
y = conv(x)
torch.cuda.synchronize()
print("cudnn conv probe ok:", tuple(y.shape))
PY

python scripts/train_rn18_cifar10_experiment.py \
  --imagenet-only \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --seed 13 \
  --imagenet-epochs 20 \
  --imagenet-save-every 1 \
  --batch-size 128 \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --finetune-lr 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --device cuda \
  --amp

test -s "${OUTPUT_DIR}/imagenet_finetune_metrics.csv"
test -s "${OUTPUT_DIR}/summary.md"
CHECKPOINT_COUNT="$(find "${OUTPUT_DIR}/checkpoints" -maxdepth 1 -type f -name 'imagenet_ft_epoch*.pth' | wc -l)"
if [[ "${CHECKPOINT_COUNT}" -ne 20 ]]; then
  echo "Expected 20 ImageNet checkpoints, found ${CHECKPOINT_COUNT}." >&2
  exit 1
fi

echo "[$(date)] Verified ${CHECKPOINT_COUNT} ImageNet checkpoints"
echo "[$(date)] Summary: ${OUTPUT_DIR}/summary.md"
echo "[$(date)] Done"
