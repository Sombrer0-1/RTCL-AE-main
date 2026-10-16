#!/usr/bin/env python3
"""Run AOCL and R4 on periodic full-test passes, serially on one physical GPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from environment.experiment_runtime import experiment_env, record_environment

MODES = {
    'fully_parallel': ['--global_scheduler_mode', 'fully_parallel'],
    'aocl': ['--global_scheduler_mode', 'adaptocl', '--enable_double_buffer'],
    'fully_parallel_certified': ['--global_scheduler_mode', 'fully_parallel',
                                '--host_preparation', 'certified'],
    'r4_cooperative': ['--global_scheduler_mode', 'r4_full_eval',
                       '--host_preparation', 'cooperative'],
}


def run_cell(out, phase, seed, mode, gpu):
    work = out / phase / 'work' / f'{mode}_seed{seed}'
    work.mkdir(parents=True, exist_ok=False)
    log = out / phase / f'{mode}_seed{seed}.log'
    command = [sys.executable, str(ROOT / 'main.py'), '--benchmark', 'split_cifar10',
        '--algorithm', 'replay', '--model', 'resnet20', '--epoch', '1',
        '--training_bs', '64', '--eval_bs', '16', '--seed', str(seed),
        '--max_runtime', '300', '--record_model_hash', '--record_eval_batches',
        '--eval_cycles', '5' if phase == 'matched' else '0', *MODES[mode]]
    started = time.perf_counter()
    with log.open('w') as stream:
        stream.write(json.dumps(dict(command=command, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            load_start=os.getloadavg())) + '\n')
        stream.flush()
        proc = subprocess.Popen(command, cwd=work, env=experiment_env(gpu, seed),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=360)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
    wall = time.perf_counter() - started
    contents = log.read_text()
    cycles = [json.loads(item) for item in re.findall(r'\[CycleMetrics\] (\{[^\n]+\})', contents)]
    timing = [json.loads(item) for item in re.findall(r'\[TrainTiming\] (\{[^\n]+\})', contents)]
    hashes = re.findall(r'\[ModelHash\] ([0-9a-f]+)', contents)
    batches = []
    for path in sorted(work.glob('eval_batches_cycle*.json')):
        batches.extend(json.loads(path.read_text()))
    errors = bool(re.search(r'Traceback|\[ERROR\]|Maximum runtime|Training failed', contents))
    valid = (rc == 0 and not errors and
        len(re.findall(r'Training Experience \d+ completed', contents)) == 10 and
        len(hashes) == 1 and bool(timing) and bool(cycles) and
        (cycles[-1]['final_at_load'] or cycles[-1].get('final_snapshot', False)) and
        all(cycle['samples'] == 10000 for cycle in cycles) and
        sum(row['samples'] for row in batches) == 10000 * len(cycles))
    if phase == 'matched':
        valid = valid and len(cycles) == 5
    if mode != 'aocl':
        valid = valid and bool(timing) and timing[-1]['train_updates'] == 790
    prep_path = work / 'full_eval_preparation.json'
    prep = json.loads(prep_path.read_text()) if prep_path.exists() else {}
    if mode in ('fully_parallel_certified', 'r4_cooperative'):
        valid = valid and prep.get('certified_epochs') == 10 and prep.get('fallback_epochs') == 0
    if mode == 'r4_cooperative':
        valid = valid and prep.get('service_samples', 0) > 0
    latencies = np.array([row['latency_ms'] for row in batches])
    row = dict(phase=phase, mode=mode, seed=seed, wall_sec=wall,
        qps=50000 / wall, final_accuracy=cycles[-1]['accuracy'] if cycles else None,
        eval_cycles=len(cycles), eval_samples=sum(c['samples'] for c in cycles),
        eval_versions=','.join(str(c['version_lower_bound']) for c in cycles),
        train_updates=timing[-1]['train_updates'] if timing else None,
        train_seconds=timing[-1]['train_seconds'] if timing else None,
        batch_p95_ms=float(np.percentile(latencies, 95)) if len(latencies) else None,
        batch_p99_ms=float(np.percentile(latencies, 99)) if len(latencies) else None,
        batch_miss_33ms=float(np.mean(latencies > 33)) if len(latencies) else None,
        mean_cycle_sec=float(np.mean([c['duration'] for c in cycles])) if cycles else None,
        max_cycle_sec=max((c['duration'] for c in cycles), default=None),
        service_samples=prep.get('service_samples', 0),
        model_hash=hashes[-1] if hashes else '', return_code=rc, valid=bool(valid),
        log=str(log.relative_to(out)))
    # Retain raw timing and weights: they also identify failed publication runs.
    (work / 'cycles.json').write_text(json.dumps(cycles, indent=2) + '\n')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['native', 'matched', 'both'], default='both')
    parser.add_argument('--seeds', type=int, nargs='+', default=[211, 307, 419])
    parser.add_argument('--modes', nargs='+', choices=list(MODES))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.csv').exists():
        parser.error('results.csv already exists; use a new directory')
    record_environment(out, args.gpu)
    source = out / 'source'
    for relative in ['main.py', 'src/config.py', 'src/workers/train_worker.py',
            'src/workers/eval_worker.py', 'src/schedulers/full_eval_scheduler.py',
            'src/schedulers/boundary_scheduler.py',
            'src/schedulers/host_preparation.py', 'src/schedulers/adaptocl_scheduler.py',
            'src/schedulers/model_scheduler.py', 'experiments/optimization/r4_aocl/DESIGN.md',
            'experiments/optimization/r4_aocl/run_comparison.py']:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    context = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), arguments=vars(args) | {'output': str(out)},
        power=subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,power.draw',
            '--format=csv'], capture_output=True, text=True).stdout)
    (out / 'run_context.json').write_text(json.dumps(context, indent=2) + '\n')
    phases = ['native', 'matched'] if args.phase == 'both' else [args.phase]
    rows = []
    fixed_hashes = {}
    for phase in phases:
        modes = args.modes or (list(MODES) if phase == 'native' else
            ['aocl', 'fully_parallel_certified', 'r4_cooperative'])
        for index, seed in enumerate(args.seeds):
            offset = index % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                print(json.dumps(dict(starting=mode, phase=phase, seed=seed)), flush=True)
                row = run_cell(out, phase, seed, mode, args.gpu)
                if mode != 'aocl' and row['valid']:
                    expected = fixed_hashes.setdefault(seed, row['model_hash'])
                    row['valid'] = row['model_hash'] == expected
                rows.append(row)
                with (out / 'results.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(row), flush=True)
                if not row['valid']:
                    raise RuntimeError(f'Invalid cell {phase}/{mode}/{seed}; raw evidence preserved')


if __name__ == '__main__':
    main()
