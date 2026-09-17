#!/usr/bin/env bash
#SBATCH --job-name=exp5-toy
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --array=0-44

set -euo pipefail

REPO_DIR="${EXP5_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ENV_NAME="${EXP5_CONDA_ENV:-cnn-filter-db}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"
METHODS=(sigreg vc so)
CONDITIONS=(1 10 100)
METHOD="${METHODS[$((TASK_ID % 3))]}"
CONDITION="${CONDITIONS[$(((TASK_ID / 3) % 3))]}"
SEED="$((TASK_ID / 9))"

cd "${REPO_DIR}"
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export PYTHONUNBUFFERED=1

python scripts/run_experiment5_linear_toy.py \
  --method "${METHOD}" \
  --condition-number "${CONDITION}" \
  --seed "${SEED}" \
  --device cuda
