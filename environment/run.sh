#!/usr/bin/env bash
set -euo pipefail
rtcl_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rtcl_conda=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "$rtcl_conda" && -x "$HOME/miniconda3/bin/conda" ]]; then
    rtcl_conda="$HOME/miniconda3/bin/conda"
fi
if [[ -z "$rtcl_conda" ]]; then
    echo "Conda not found; run environment/setup.sh first." >&2
    exit 1
fi
cd "$rtcl_root"
exec "$rtcl_conda" run --no-capture-output -n "${RTCL_ENV_NAME:-rtcl}" python "$@"
