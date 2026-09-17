#!/usr/bin/env bash
#SBATCH --job-name=rn18-cifar10-true-lejepa
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO_DIR="${RN18_CIFAR10_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUTPUT_DIR="${RN18_CIFAR10_LEJEPA_OUTPUT_DIR:-outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong}"
IMAGENET_SOURCE_DIR="${RN18_CIFAR10_IMAGENET_SOURCE_DIR:-outputs/(4thEXP)rn18_cifar10_filter_dynamics/regenerated_imagenet_baseline}"
DATA_DIR="${RN18_CIFAR10_DATA_DIR:-data/cifar10}"
ENV_NAME="${RN18_CIFAR10_CONDA_ENV:-cnn-filter-db}"

SSL_EPOCHS="${RN18_CIFAR10_SSL_EPOCHS:-120}"
LINEAR_EPOCHS="${RN18_CIFAR10_LINEAR_EPOCHS:-50}"
BATCH_SIZE="${RN18_CIFAR10_BATCH_SIZE:-256}"
SSL_SAVE_EVERY="${RN18_CIFAR10_SSL_SAVE_EVERY:-10}"
QUICK_PROBE_EPOCHS="${RN18_CIFAR10_QUICK_PROBE_EPOCHS:-5}"
QUICK_PROBE_INTERVAL="${RN18_CIFAR10_QUICK_PROBE_INTERVAL:-10}"
SSL_MIN_CHECKPOINTS="${RN18_CIFAR10_SSL_MIN_CHECKPOINTS:-999}"
SSL_VIEWS="${RN18_CIFAR10_SSL_VIEWS:-4}"
SSL_GLOBAL_VIEWS="${RN18_CIFAR10_SSL_GLOBAL_VIEWS:-4}"
SSL_PROJ_DIM="${RN18_CIFAR10_SSL_PROJ_DIM:-64}"
SSL_HIDDEN_DIM="${RN18_CIFAR10_SSL_HIDDEN_DIM:-2048}"
SSL_CROP_MIN="${RN18_CIFAR10_SSL_CROP_MIN:-0.08}"
SSL_CROP_MAX="${RN18_CIFAR10_SSL_CROP_MAX:-1.0}"
SSL_JITTER_BRIGHTNESS="${RN18_CIFAR10_SSL_JITTER_BRIGHTNESS:-0.8}"
SSL_JITTER_CONTRAST="${RN18_CIFAR10_SSL_JITTER_CONTRAST:-0.8}"
SSL_JITTER_SATURATION="${RN18_CIFAR10_SSL_JITTER_SATURATION:-0.8}"
SSL_JITTER_HUE="${RN18_CIFAR10_SSL_JITTER_HUE:-0.2}"
SSL_GRAYSCALE_P="${RN18_CIFAR10_SSL_GRAYSCALE_P:-0.20}"
SSL_BLUR_P="${RN18_CIFAR10_SSL_BLUR_P:-0.5}"
SSL_SOLARIZE_P="${RN18_CIFAR10_SSL_SOLARIZE_P:-0.2}"
LEJEPA_LAMBDA="${RN18_CIFAR10_LEJEPA_LAMBDA:-0.02}"
SIGREG_NUM_SLICES="${RN18_CIFAR10_SIGREG_NUM_SLICES:-1024}"
SIGREG_T_MAX="${RN18_CIFAR10_SIGREG_T_MAX:-3.0}"
SIGREG_N_POINTS="${RN18_CIFAR10_SIGREG_N_POINTS:-17}"
SSL_LR="${RN18_CIFAR10_SSL_LR:-0.002}"
SSL_WEIGHT_DECAY="${RN18_CIFAR10_SSL_WEIGHT_DECAY:-5e-4}"
LINEAR_LR="${RN18_CIFAR10_LINEAR_LR:-0.001}"
LINEAR_WEIGHT_DECAY="${RN18_CIFAR10_LINEAR_WEIGHT_DECAY:-1e-6}"
LINEAR_OPTIMIZER="${RN18_CIFAR10_LINEAR_OPTIMIZER:-adamw}"
LINEAR_HEAD_TYPE="${RN18_CIFAR10_LINEAR_HEAD_TYPE:-layernorm_linear}"
BINS="${RN18_CIFAR10_ANALYSIS_BINS:-70}"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Repo: ${REPO_DIR}"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] ImageNet source dir: ${IMAGENET_SOURCE_DIR}"
echo "[$(date)] Data dir: ${DATA_DIR}"
echo "[$(date)] Conda env: ${ENV_NAME}"
echo "[$(date)] cuDNN mode: enabled; this job intentionally does not pass --disable-cudnn"
echo "[$(date)] LeJEPA: epochs=${SSL_EPOCHS} views=${SSL_VIEWS} lambda=${LEJEPA_LAMBDA} slices=${SIGREG_NUM_SLICES} proj_dim=${SSL_PROJ_DIM}"
echo "[$(date)] Linear probe: epochs=${LINEAR_EPOCHS} head=${LINEAR_HEAD_TYPE} opt=${LINEAR_OPTIMIZER} lr=${LINEAR_LR} wd=${LINEAR_WEIGHT_DECAY}"

source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -V
python - <<'PY'
import importlib

import torch

from scripts.train_rn18_cifar10_experiment import LEJEPA_EppsPulley, LEJEPA_SlicingUnivariateTest

for module in ["torch", "torchvision", "numpy", "pandas", "PIL"]:
    mod = importlib.import_module(module)
    print(f"{module}: {getattr(mod, '__version__', 'ok')}")
print("lejepa components:", LEJEPA_EppsPulley, LEJEPA_SlicingUnivariateTest)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
    print("cudnn available:", torch.backends.cudnn.is_available())
    print("cudnn version:", torch.backends.cudnn.version())
    x = torch.randn(8, 3, 32, 32, device="cuda")
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    torch.cuda.synchronize()
    print("cudnn probe ok:", tuple(y.shape))
PY

python scripts/train_rn18_cifar10_experiment.py \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --skip-imagenet-train \
  --imagenet-source-dir "${IMAGENET_SOURCE_DIR}" \
  --ssl-epochs "${SSL_EPOCHS}" \
  --linear-epochs "${LINEAR_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --ssl-save-every "${SSL_SAVE_EVERY}" \
  --quick-probe-epochs "${QUICK_PROBE_EPOCHS}" \
  --quick-probe-interval "${QUICK_PROBE_INTERVAL}" \
  --ssl-min-checkpoints "${SSL_MIN_CHECKPOINTS}" \
  --ssl-views "${SSL_VIEWS}" \
  --ssl-global-views "${SSL_GLOBAL_VIEWS}" \
  --ssl-proj-dim "${SSL_PROJ_DIM}" \
  --ssl-hidden-dim "${SSL_HIDDEN_DIM}" \
  --ssl-crop-min "${SSL_CROP_MIN}" \
  --ssl-crop-max "${SSL_CROP_MAX}" \
  --ssl-jitter-brightness "${SSL_JITTER_BRIGHTNESS}" \
  --ssl-jitter-contrast "${SSL_JITTER_CONTRAST}" \
  --ssl-jitter-saturation "${SSL_JITTER_SATURATION}" \
  --ssl-jitter-hue "${SSL_JITTER_HUE}" \
  --ssl-grayscale-p "${SSL_GRAYSCALE_P}" \
  --ssl-blur-p "${SSL_BLUR_P}" \
  --ssl-solarize-p "${SSL_SOLARIZE_P}" \
  --lejepa-lambda "${LEJEPA_LAMBDA}" \
  --sigreg-num-slices "${SIGREG_NUM_SLICES}" \
  --sigreg-t-max "${SIGREG_T_MAX}" \
  --sigreg-n-points "${SIGREG_N_POINTS}" \
  --ssl-lr "${SSL_LR}" \
  --ssl-weight-decay "${SSL_WEIGHT_DECAY}" \
  --linear-lr "${LINEAR_LR}" \
  --linear-weight-decay "${LINEAR_WEIGHT_DECAY}" \
  --linear-optimizer "${LINEAR_OPTIMIZER}" \
  --linear-head-type "${LINEAR_HEAD_TYPE}" \
  --device cuda \
  --amp

PAIR_OUTPUT="${OUTPUT_DIR}/analysis_selected_pair"

python scripts/analyze_rn18_cifar10_kernel_distributions.py \
  --experiment-dir "${OUTPUT_DIR}" \
  --output-dir "${PAIR_OUTPUT}" \
  --bins 120 \
  --max-stat-samples 250000

python scripts/analyze_rn18_cifar10_filters.py \
  --experiment-dir "${OUTPUT_DIR}" \
  --output-dir "${PAIR_OUTPUT}/pca_shape_analysis" \
  --no-include-stem \
  --bins "${BINS}" \
  --max-density-points 400000

echo "[$(date)] Generated outputs:"
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort
echo "[$(date)] Done"
