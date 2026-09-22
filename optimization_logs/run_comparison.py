#!/usr/bin/env python3
"""Serial, seed-paired comparisons; preserve full commands and raw outputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MODES = ['fully_parallel', 'adaptocl', 'freshness']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', default=MODES)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--training-bs', type=int, default=64)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(args.seeds):
        # Rotate order to reduce systematic warmup/order bias.
        offset = index % len(args.modes)
        order = args.modes[offset:] + args.modes[:offset]
        for mode in order:
            name = f'{mode}_seed{seed}'
            work = out / 'work' / name
            work.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / 'main.py'), '--benchmark', 'split_cifar10',
                   '--algorithm', 'replay', '--model', 'resnet20', '--training_bs', str(args.training_bs),
                   '--eval_bs', '16', '--global_scheduler_mode', mode, '--seed', str(seed),
                   '--max_runtime', '240']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='1',
                       PYTHONHASHSEED=str(seed))
            log = out / f'{name}.log'
            started = time.perf_counter()
            with log.open('w') as f:
                f.write(json.dumps({'command': cmd, 'seed': seed, 'gpu': 1, 'cpu_threads': 1}) + '\n')
                f.flush()
                proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
            wall = time.perf_counter() - started
            contents = log.read_text()
            cycles = [json.loads(x) for x in re.findall(r'\[CycleMetrics\] (\{[^\n]+\})', contents)]
            complete = len(re.findall(r'Training Experience \d+ completed', contents)) == 10
            errors = bool(re.search(r'Traceback|\[ERROR\]|Maximum runtime|Training failed', contents))
            valid = rc == 0 and complete and not errors and cycles and (cycles[-1]['final_at_load'] or cycles[-1].get('final_snapshot', False)) and cycles[-1]['samples'] == 10000
            row = dict(mode=mode, seed=seed, wall_sec=wall, qps=50000/wall if valid else '',
                       final_accuracy=cycles[-1]['accuracy'] if cycles else '',
                       eval_cycles=len(cycles), eval_samples=sum(c['samples'] for c in cycles),
                       final_p99_ms=cycles[-1]['batch_p99_ms'] if cycles else '',
                       max_cycle_p99_ms=max((c['batch_p99_ms'] for c in cycles), default=0),
                       return_code=rc, valid=bool(valid), log=log.name)
            rows.append(row)
            with (out / 'results.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows)
            print(json.dumps(row), flush=True)
            # These are generated per-run weights, not user artifacts.
            for checkpoint in work.glob('shared_model_*.pth'):
                checkpoint.unlink()

if __name__ == '__main__':
    main()
