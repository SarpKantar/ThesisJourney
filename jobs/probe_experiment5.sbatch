#!/usr/bin/env bash
#SBATCH --job-name=exp5-probe
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --array=0-8

set -euo pipefail

REPO_DIR="${EXP5_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_DIR="${EXP5_DATA_DIR:-data/cifar10}"
ENV_NAME="${EXP5_CONDA_ENV:-cnn-filter-db}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"
ARMS=(A B C)
SEEDS=(13 17 23 31 43)
ARM="${ARMS[$((TASK_ID % 3))]}"
SEED="${SEEDS[$((TASK_ID / 3))]}"
CHECKPOINT="outputs/(5thEXP)rn18_cifar10_sigreg_structure/discovery/arm_${ARM}/seed_${SEED}/checkpoints/update_021000.pth"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

cd "${REPO_DIR}"
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export PYTHONUNBUFFERED=1

python scripts/probe_experiment5.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --device cuda
