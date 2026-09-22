#!/usr/bin/env python3
"""Frozen, serial GPU matrix; all requests and failed runs remain auditable."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pilot', 'main'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.csv').exists():
        p.error('existing results: choose a new directory')
    if a.phase == 'pilot':
        cases = [('pilot', 'resnet20', 1, 'steady', 100, 300, 3)]
        seeds = [7]
        modes = ['fifo', 'debt', 'neural', 'boundary']
    else:
        cases = [('longer', 'resnet20', 2, 'steady', 100, 6000, 30),
                 ('deeper_bursty', 'resnet56', 1, 'bursty', 33, 6000, 30)]
        seeds = [101, 202]
        modes = ['fifo', 'slack', 'debt', 'neural']
    rows = []
    paired = {}
    for ci, (case, model, epochs, pattern, slo, count, duration) in enumerate(cases):
        for si, seed in enumerate(seeds):
            offset = (ci*2+si) % len(modes)
            for mode in modes[offset:]+modes[:offset]:
                name = f'{case}_{mode}_seed{seed}'
                work = out/'work'/name
                work.mkdir(parents=True, exist_ok=False)
                cmd = [sys.executable, str(ROOT/'main.py'), '--benchmark', 'split_cifar10',
                       '--algorithm', 'replay', '--model', model, '--epoch', str(epochs),
                       '--training_bs', '64', '--eval_bs', '32', '--global_scheduler_mode',
                       'stream_'+mode, '--seed', str(seed), '--max_runtime', '480',
                       '--record_model_hash', '--stream_requests', str(count),
                       '--stream_duration', str(duration), '--stream_slo_ms', str(slo),
                       '--stream_pattern', pattern]
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='1', PYTHONHASHSEED=str(seed))
                load_start = os.getloadavg()
                log = out/(name+'.log')
                started = time.perf_counter()
                with log.open('w') as f:
                    f.write(json.dumps(dict(command=cmd, load_start=load_start,
                                             timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))+'\n')
                    f.flush()
                    proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=f,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        rc = proc.wait(timeout=540)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        rc = proc.wait()
                wall = time.perf_counter()-started
                contents = log.read_text()
                hashes = re.findall(r'\[ModelHash\] ([0-9a-f]+)', contents)
                final = re.findall(r'\[FinalSnapshot\] (\{[^\n]+\})', contents)
                metric_file = work/'stream_metrics.json'
                metric = json.loads(metric_file.read_text()) if metric_file.exists() else {}
                complete = len(re.findall(r'Training Experience \d+ completed', contents)) == 10
                errors = bool(re.search(r'Traceback|\[ERROR\]|Maximum runtime|Training failed', contents))
                valid = rc == 0 and complete and not errors and bool(final) and bool(hashes) and bool(metric)
                trace_hash = ''
                if metric:
                    valid = valid and metric['train_updates'] == 790*epochs and metric['dropped'] == 0
                    requests = json.loads((work/'stream_requests.json').read_text())
                    trace = json.loads((work/'stream_arrivals.json').read_text())
                    trace_hash = hashlib.sha256((work/'stream_arrivals.json').read_bytes()).hexdigest()
                    valid = valid and len(requests) == count and len({r['sample'] for r in requests}) == count
                    valid = valid and [r['id'] for r in requests] == list(range(count))
                    valid = valid and all(r['latency'] >= 0 and r['sample'] == trace[r['id']]['sample'] for r in requests)
                    artifact = out/name
                    artifact.mkdir()
                    for source in work.glob('stream_*.json'):
                        shutil.copy2(source, artifact/source.name)
                snapshot = json.loads(final[-1]) if final else {}
                valid = valid and snapshot.get('total_samples') == 10000
                key = (case, seed)
                identity = (hashes[-1] if hashes else '', trace_hash)
                pair_match = identity == paired.setdefault(key, identity)
                valid = valid and pair_match
                row = dict(case=case, mode=mode, seed=seed, model=model, epochs=epochs,
                           wall_sec=wall, training_samples_per_wall_sec=50000*epochs/wall,
                           final_accuracy=snapshot.get('streaming_accuracy', ''),
                           model_hash=identity[0], trace_hash=trace_hash, pair_match=pair_match,
                           load_start=load_start[0], load_end=os.getloadavg()[0],
                           return_code=rc, valid=bool(valid), log=log.name)
                # Fixed schema even if a run fails before metrics are written.
                fields = ['requests','unique_samples','dropped','miss_rate','p50_ms','p95_ms','p99_ms',
                          'goodput','stream_seconds','train_completion_seconds','online_accuracy',
                          'max_train_pause_seconds','train_updates','controller_seconds','decisions',
                          'guard_overrides','neural_parameters','neural_updates']
                row.update({k: metric.get(k, '') for k in fields})
                rows.append(row)
                with (out/'results.csv').open('w') as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                    w.writerows(rows)
                print(json.dumps(row), flush=True)
                if not valid:
                    raise RuntimeError(f'Invalid run {name}; preserved logs, stop for diagnosis')

if __name__ == '__main__':
    main()
