#!/usr/bin/env bash
#SBATCH --job-name=exp5-pilot-grid
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --array=0-11

set -euo pipefail

REPO_DIR="${EXP5_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_DIR="${EXP5_DATA_DIR:-data/cifar10}"
ENV_NAME="${EXP5_CONDA_ENV:-cnn-filter-db}"
LEDGER="${EXP5_COEFFICIENT_LEDGER:-outputs/(5thEXP)rn18_cifar10_sigreg_structure/pilot/coefficient_ledger.json}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"
SEEDS=(901 902)
SEED="${SEEDS[$((TASK_ID / 6))]}"
VARIANT="$((TASK_ID % 6))"

if [[ ! -s "${LEDGER}" ]]; then
  echo "Missing coefficient ledger: ${LEDGER}" >&2
  exit 2
fi

cd "${REPO_DIR}"
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
export PYTHONUNBUFFERED=1

readarray -t COEFFICIENTS < <(
  python -c '
import json, sys
ledger = json.load(open(sys.argv[1]))
for value in ledger["sigreg_sensitivity_coefficients"] + ledger["vc_candidate_coefficients"]:
    print(value)
' "${LEDGER}"
)

if (( VARIANT < 3 )); then
  ARM=B
  LABEL="sigreg_${VARIANT}"
else
  ARM=C
  LABEL="vc_$((VARIANT - 3))"
fi
COEFFICIENT="${COEFFICIENTS[${VARIANT}]}"
OUTPUT_DIR="outputs/(5thEXP)rn18_cifar10_sigreg_structure/pilot/sensitivity/${LABEL}/seed_${SEED}"

python scripts/train_experiment5.py \
  --arm "${ARM}" \
  --seed "${SEED}" \
  --coefficient "${COEFFICIENT}" \
  --coefficient-ledger "${LEDGER}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --device cuda
