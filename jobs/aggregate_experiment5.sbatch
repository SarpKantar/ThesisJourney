#!/usr/bin/env bash
#SBATCH --job-name=exp5-aggregate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${EXP5_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ENV_NAME="${EXP5_CONDA_ENV:-cnn-filter-db}"
SEEDS="${EXP5_SEEDS:-13,17,23}"

cd "${REPO_DIR}"
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export PYTHONUNBUFFERED=1

python scripts/aggregate_experiment5.py --seeds "${SEEDS}"
