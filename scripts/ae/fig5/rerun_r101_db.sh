#!/usr/bin/env bash
set -u
ulimit -n 65535
BR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$BR" || exit 1
source "$BR/environment/shell_env.sh" || exit 1
PY="$RTCL_PYTHON"
LD="${RTCL_OUTPUT_ROOT:-$BR/output/thor-ae}/ae_logs_fig5"
mkdir -p "$LD"
MASTER="$LD/fig5_master.log"
TIMEOUT=14400
run(){ local name="$1"; shift; local log="$LD/$name.log"
  local work="$LD/work/$name"; mkdir -p "$work"
  local attempt rc t0 t1
  for attempt in 1 2; do
    echo "[RUN $name] START attempt=$attempt $(date -Is)" >> "$MASTER"; t0=$(date +%s)
    (cd "$work" && timeout -s KILL "$TIMEOUT" "$@") > "$log" 2>&1; rc=$?; t1=$(date +%s)
    echo "[RUN $name] END attempt=$attempt $(date -Is) rc=$rc dur=$((t1-t0))s" >> "$MASTER"
    [ "$rc" = "0" ] && grep -q "Overall Streaming Accuracy" "$log" && return 0
    echo "[CLEANUP $name] rc=$rc" >> "$MASTER"
    sleep 8
  done
  return 1; }
echo "R101 DB RERUN START $(date -Is) ulimit=$(ulimit -n)" >> "$MASTER"
run recl_resnet101 $PY -u "$BR/main.py" --benchmark split_cifar10 --algorithm replay --model resnet101 --global_scheduler_mode recl_sched --eval_bs 16 --enable_double_buffer
run ours_resnet101 $PY -u "$BR/main.py" --benchmark split_cifar10 --algorithm replay --model resnet101 --global_scheduler_mode adaptocl --training_bs 16 --eval_bs 16 --enable_double_buffer
echo "R101 DB RERUN DONE $(date -Is)" >> "$MASTER"; touch "$LD/r101_rerun.done"
