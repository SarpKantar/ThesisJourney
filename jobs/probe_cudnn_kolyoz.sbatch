#!/usr/bin/env bash
#SBATCH --job-name=probe-cudnn-kolyoz
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

cd "${REPO_DIR}"
mkdir -p logs

echo "[$(date)] Host: $(hostname)"
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cudnn available:", torch.backends.cudnn.is_available())
print("cudnn enabled:", torch.backends.cudnn.enabled)
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
