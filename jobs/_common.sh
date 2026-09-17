#!/usr/bin/env bash
# Shared, portable Conda activation for Slurm launchers.
#
# Override CNN_FILTER_DB_CONDA_SH when `conda` is not on PATH in batch jobs.
# Set CNN_FILTER_DB_SKIP_CONDA=1 when the submitted environment is already
# active on compute nodes.

activate_conda_env() {
  local env_name="${1:?Conda environment name is required}"
  local conda_sh="${CNN_FILTER_DB_CONDA_SH:-}"

  if [[ "${CNN_FILTER_DB_SKIP_CONDA:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    return 0
  fi

  if [[ -z "${conda_sh}" && -n "${CONDA_EXE:-}" ]]; then
    conda_sh="$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  fi
  if [[ -z "${conda_sh}" ]] && command -v conda >/dev/null 2>&1; then
    conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
  fi
  if [[ -z "${conda_sh}" && -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  if [[ -z "${conda_sh}" && -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
  if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
    echo "Could not locate conda.sh. Set CNN_FILTER_DB_CONDA_SH or CNN_FILTER_DB_SKIP_CONDA=1." >&2
    return 2
  fi

  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${env_name}"
}
