# Source from experiment shell scripts to select the project interpreter.
rtcl_environment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${RTCL_PYTHON:-} ]]; then
    RTCL_PYTHON=$(bash "$rtcl_environment_dir/run.sh" -c 'import sys; print(sys.executable)') || return 1
fi
if [[ ! -x "$RTCL_PYTHON" ]]; then
    echo "RTCL_PYTHON is not an executable interpreter: $RTCL_PYTHON" >&2
    return 1
fi
export RTCL_PYTHON
export PATH="$(dirname -- "$RTCL_PYTHON"):$PATH"
