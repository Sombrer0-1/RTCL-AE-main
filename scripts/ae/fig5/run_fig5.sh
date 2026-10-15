#!/usr/bin/env bash
set -u
BR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$BR" || exit 1
source "$BR/environment/shell_env.sh" || exit 1
PY="$RTCL_PYTHON"
LD="${RTCL_OUTPUT_ROOT:-$BR/output/thor-ae}/ae_logs_fig5"; mkdir -p "$LD"
MASTER="$LD/fig5_master.log"
TIMEOUT=${TIMEOUT:-14400}
run(){ local name="$1"; shift; local log="$LD/$name.log"
  local work="$LD/work/$name"; mkdir -p "$work"
  if grep -q "\[RUN $name\] END .* rc=0" "$MASTER" 2>/dev/null; then echo "[SKIP $name] done" >> "$MASTER"; return 0; fi
  local attempt rc t0 t1
  for attempt in 1 2; do
    echo "[RUN $name] START attempt=$attempt $(date -Is)" >> "$MASTER"; t0=$(date +%s)
    (cd "$work" && timeout -s KILL "$TIMEOUT" "$@") > "$log" 2>&1; rc=$?; t1=$(date +%s)
    echo "[RUN $name] END attempt=$attempt $(date -Is) rc=$rc dur=$((t1-t0))s" >> "$MASTER"
    [ "$rc" = "0" ] && return 0
    echo "[CLEANUP $name] rc=$rc" >> "$MASTER"
    sleep 8
  done
  return "$rc"; }
echo "FIG5 SWEEP START $(date -Is)" >> "$MASTER"
for M in resnet50 resnet101 mobilenetv1 vit_tiny; do
  run vanilla_$M $PY "$BR/test_avalanche_lib.py" --benchmark split_cifar10 --algorithm replay --model $M
  run da_$M      $PY -u "$BR/main.py" --benchmark split_cifar10 --algorithm replay --model $M --global_scheduler_mode default --training_bs 16 --eval_bs 16
  run ekya_$M    $PY -u "$BR/main.py" --benchmark split_cifar10 --algorithm replay --model $M --global_scheduler_mode ekya --training_bs 16 --eval_bs 16
  run recl_$M    $PY -u "$BR/main.py" --benchmark split_cifar10 --algorithm replay --model $M --global_scheduler_mode recl_sched --eval_bs 16 --enable_double_buffer
  run fp_$M      $PY -u "$BR/main.py" --benchmark split_cifar10 --algorithm replay --model $M --global_scheduler_mode fully_parallel --training_bs 16 --eval_bs 16
  run ours_$M    $PY -u "$BR/main.py" --benchmark split_cifar10 --algorithm replay --model $M --global_scheduler_mode adaptocl --training_bs 16 --eval_bs 16 --enable_double_buffer
done
echo "FIG5 SWEEP DONE $(date -Is)" >> "$MASTER"; touch "$LD/fig5.done"
