#!/usr/bin/env bash
#SBATCH --job-name=cnn-filter-db-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${CNN_FILTER_DB_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_DIR="${CNN_FILTER_DB_DATA_DIR:-${REPO_DIR}/data}"
RECORD_ID="6371680"
API_URL="https://zenodo.org/api/records/${RECORD_ID}"
FALLBACK_URL="https://zenodo.org/record/${RECORD_ID}/files/dataset.h5?download=1"
ENV_NAME="${CNN_FILTER_DB_CONDA_ENV:-cnn-filter-db}"

mkdir -p "${DATA_DIR}" "${REPO_DIR}/logs"
cd "${DATA_DIR}"

echo "[$(date)] Dataset directory: ${DATA_DIR}"
df -h "${DATA_DIR}"

echo "[$(date)] Discovering Zenodo file metadata..."
python - <<'PY' > download_info.tsv || true
import json
import urllib.request

api_url = "https://zenodo.org/api/records/6371680"
with urllib.request.urlopen(api_url, timeout=60) as response:
    record = json.load(response)

files = record.get("files", [])
preferred = None
for candidate in files:
    key = candidate.get("key", "")
    if key == "dataset.h5":
        preferred = candidate
        break
    if key.startswith("dataset.h5"):
        preferred = candidate

if preferred is None and files:
    preferred = files[0]

if preferred is None:
    raise SystemExit("No files found in Zenodo record")

links = preferred.get("links", {})
download_url = links.get("self") or links.get("download")
if not download_url:
    raise SystemExit("No download URL found in Zenodo record")

print(preferred.get("key", "dataset.h5"), preferred.get("size", ""), preferred.get("checksum", ""), download_url, sep="\t")
PY

if [[ -s download_info.tsv ]]; then
    IFS=$'\t' read -r FILE_NAME EXPECTED_SIZE CHECKSUM DOWNLOAD_URL < download_info.tsv
else
    FILE_NAME="dataset.h5"
    EXPECTED_SIZE=""
    CHECKSUM=""
    DOWNLOAD_URL="${FALLBACK_URL}"
    echo "Zenodo API discovery failed; using fallback URL: ${DOWNLOAD_URL}"
fi

echo "[$(date)] File: ${FILE_NAME}"
echo "[$(date)] Expected bytes: ${EXPECTED_SIZE:-unknown}"
echo "[$(date)] Checksum: ${CHECKSUM:-unknown}"
echo "[$(date)] URL: ${DOWNLOAD_URL}"

NEEDS_DOWNLOAD=1
if [[ -f "${FILE_NAME%.xz}" && "${FILE_NAME}" == *.xz ]]; then
    echo "[$(date)] ${FILE_NAME%.xz} already exists; skipping download."
    NEEDS_DOWNLOAD=0
elif [[ -f "${FILE_NAME}" && -n "${EXPECTED_SIZE}" ]]; then
    ACTUAL_SIZE="$(stat -c%s "${FILE_NAME}")"
    if [[ "${ACTUAL_SIZE}" == "${EXPECTED_SIZE}" ]]; then
        echo "[$(date)] ${FILE_NAME} already exists and matches expected size."
        NEEDS_DOWNLOAD=0
    else
        echo "[$(date)] ${FILE_NAME} exists but is ${ACTUAL_SIZE}/${EXPECTED_SIZE} bytes; resuming download."
    fi
elif [[ -f "${FILE_NAME}" ]]; then
    echo "[$(date)] ${FILE_NAME} already exists; trying a resumable verification/download."
fi

if [[ "${NEEDS_DOWNLOAD}" == 1 ]]; then
    echo "[$(date)] Downloading with resume support..."
    if command -v wget >/dev/null 2>&1; then
        wget -c --progress=dot:giga --tries=20 --timeout=60 --waitretry=30 -O "${FILE_NAME}" "${DOWNLOAD_URL}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --retry 20 --retry-delay 30 --connect-timeout 60 -C - -o "${FILE_NAME}" "${DOWNLOAD_URL}"
    else
        echo "Neither wget nor curl is available." >&2
        exit 1
    fi
fi

if [[ -n "${EXPECTED_SIZE}" && -f "${FILE_NAME}" ]]; then
    ACTUAL_SIZE="$(stat -c%s "${FILE_NAME}")"
    echo "[$(date)] Actual bytes: ${ACTUAL_SIZE}"
    if [[ "${ACTUAL_SIZE}" != "${EXPECTED_SIZE}" ]]; then
        echo "Downloaded size does not match Zenodo metadata." >&2
        exit 1
    fi
fi

if [[ "${CHECKSUM}" == md5:* && -f "${FILE_NAME}" ]]; then
    EXPECTED_MD5="${CHECKSUM#md5:}"
    ACTUAL_MD5="$(md5sum "${FILE_NAME}" | awk '{print $1}')"
    echo "[$(date)] Actual md5: ${ACTUAL_MD5}"
    if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
        echo "Downloaded md5 does not match Zenodo metadata." >&2
        exit 1
    fi
fi

if [[ "${FILE_NAME}" == *.xz ]]; then
    echo "[$(date)] Decompressing ${FILE_NAME}..."
    xz -dv -T "${SLURM_CPUS_PER_TASK:-1}" "${FILE_NAME}"
    FILE_NAME="${FILE_NAME%.xz}"
fi

echo "[$(date)] Validating HDF5 contents..."
source "${REPO_DIR}/jobs/_common.sh"
activate_conda_env "${ENV_NAME}"
python - <<PY
import h5py
from pathlib import Path

path = Path("${DATA_DIR}") / "${FILE_NAME}"
with h5py.File(path, "r") as handle:
    print("HDF5 keys:", list(handle.keys()))
    if "filters" in handle:
        print("/filters shape:", handle["filters"].shape, "dtype:", handle["filters"].dtype)
PY

echo "[$(date)] Done. Use this dataset path:"
echo "${DATA_DIR}/${FILE_NAME}"
