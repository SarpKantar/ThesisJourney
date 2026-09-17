#!/usr/bin/env bash
#SBATCH --job-name=exp5-abc
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --array=0-8

set -euo pipefail

REPO_DIR="${EXP5_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_DIR="${EXP5_DATA_DIR:-data/cifar10}"
ENV_NAME="${EXP5_CONDA_ENV:-cnn-filter-db}"
LEDGER="${EXP5_COEFFICIENT_LEDGER:-outputs/(5thEXP)rn18_cifar10_sigreg_structure/pilot/coefficient_ledger.json}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"
ARMS=(A B C)
SEEDS=(13 17 23 31 43)
ARM="${ARMS[$((TASK_ID % 3))]}"
SEED="${SEEDS[$((TASK_ID / 3))]}"

if (( TASK_ID < 0 || TASK_ID >= 15 )); then
  echo "Task ID ${TASK_ID} is outside the discovery grid 0..14." >&2
  exit 2
fi

cd "${REPO_DIR}"
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

EXTRA_ARGS=()
if [[ "${ARM}" == "C" ]]; then
  if [[ ! -s "${LEDGER}" ]]; then
    echo "Missing coefficient ledger: ${LEDGER}" >&2
    exit 2
  fi
  VC_ALPHA="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["alpha0"])' "${LEDGER}")"
  EXTRA_ARGS+=(--coefficient "${VC_ALPHA}")
  EXTRA_ARGS+=(--coefficient-ledger "${LEDGER}")
fi

python scripts/train_experiment5.py \
  --arm "${ARM}" \
  --seed "${SEED}" \
  --data-dir "${DATA_DIR}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --device cuda \
  "${EXTRA_ARGS[@]}"
