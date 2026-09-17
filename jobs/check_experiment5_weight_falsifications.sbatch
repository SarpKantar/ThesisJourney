#!/usr/bin/env bash
#SBATCH --job-name=exp5-falsify
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="${EXP5_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_DIR="${EXP5_DATA_DIR:-data/cifar10}"
ENV_NAME="${EXP5_CONDA_ENV:-cnn-filter-db}"
CHECKPOINT="${EXP5_EXISTING_CHECKPOINT:-outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/checkpoints/lejepa_ssl_epoch110.pth}"

cd "${REPO_DIR}"
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing existing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export PYTHONUNBUFFERED=1

python scripts/check_experiment5_weight_falsifications.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --device cuda
