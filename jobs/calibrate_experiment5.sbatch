#!/usr/bin/env bash
#SBATCH --job-name=exp5-calibrate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G

set -euo pipefail

REPO_DIR="${EXP5_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_DIR="${EXP5_DATA_DIR:-data/cifar10}"
ENV_NAME="${EXP5_CONDA_ENV:-cnn-filter-db}"

cd "${REPO_DIR}"
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export PYTHONUNBUFFERED=1

python scripts/calibrate_experiment5_coefficients.py \
  --data-dir "${DATA_DIR}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --device cuda
