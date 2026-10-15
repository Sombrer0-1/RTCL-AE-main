#!/usr/bin/env bash
set -euo pipefail
rtcl_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rtcl_name=${RTCL_ENV_NAME:-rtcl}
rtcl_conda=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "$rtcl_conda" && -x "$HOME/miniconda3/bin/conda" ]]; then
    rtcl_conda="$HOME/miniconda3/bin/conda"
fi
if [[ -z "$rtcl_conda" ]]; then
    echo "Install Conda/Miniconda for aarch64 first." >&2
    exit 1
fi
if [[ $(uname -m) != aarch64 ]]; then
    echo "This environment is configured for aarch64 NVIDIA Thor." >&2
    exit 1
fi
rtcl_source=""
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || $1 != --source-env ]]; then
        echo "Usage: bash environment/setup.sh [--source-env orion]" >&2
        exit 1
    fi
    rtcl_source=$2
fi
if ! "$rtcl_conda" run -n "$rtcl_name" python -c 'import sys' >/dev/null 2>&1; then
    if [[ -n "$rtcl_source" ]]; then
        "$rtcl_conda" create -n "$rtcl_name" --clone "$rtcl_source" --yes
    else
        "$rtcl_conda" create -n "$rtcl_name" python=3.11.16 pip --yes
    fi
fi
rtcl_python() { "$rtcl_conda" run --no-capture-output -n "$rtcl_name" python "$@"; }
rtcl_python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
if ! rtcl_python -c 'import torch, torchvision; assert torch.__version__ == "2.14.0+cu132" and torchvision.__version__ == "0.29.0+cu132"'; then
    rtcl_python -m pip install --index-url https://download.pytorch.org/whl/cu132 \
        'torch==2.14.0+cu132' 'torchvision==0.29.0+cu132'
fi
rtcl_python -m pip install -c "$rtcl_root/environment/constraints-thor.txt" \
    -c "$rtcl_root/environment/requirements-thor.lock.txt" \
    -r "$rtcl_root/environment/requirements.txt"
rtcl_python "$rtcl_root/environment/repair_wheel_metadata.py"
rtcl_python "$rtcl_root/environment/install_avalanche.py"
rtcl_python -m pip check
rtcl_python "$rtcl_root/environment/check.py" --output "$rtcl_root/output/environment.json"
echo "Ready: conda activate $rtcl_name"
