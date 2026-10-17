#!/usr/bin/env python3
"""Paired real-GPU tests of future arrivals at the R4 training-start gate."""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from environment.experiment_runtime import experiment_env, record_environment

CASES = {
    'steady_r20': dict(model='resnet20', epochs=2, pattern='steady', slo=100),
    'bursty_r56': dict(model='resnet56', epochs=1, pattern='bursty', slo=33),
}
MODES = {
    'r4': ('immediate', 100),
    'reactive100': ('reactive', 100),
    'oracle100': ('oracle', 100),
    'oracle_unlimited': ('oracle', 0),
}
QUOTA_MODES = {'warm_r4': ('immediate', 0, 0)}
QUOTA_MODES.update({f'q{rate}_{policy}': (policy, 0, rate)
    for rate in (8, 9) for policy in ('immediate', 'uniform', 'reactive', 'phase', 'oracle', 'planned')})
QUOTA_MODES['warm_phase'] = ('phase', 0, 0)
RECENT_MODES = {f'q{rate}_recent': ('planned', 0, rate) for rate in (8, 9)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'main', 'quota_pilot', 'quota_planner_pilot', 'quota',
                                          'quota_recent_pilot', 'quota_recent'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        parser.error('choose a fresh output directory')
    record_environment(out, args.gpu)
    quota_experiment = args.phase.startswith('quota')
    cases = {'bursty_r56': CASES['bursty_r56']} if quota_experiment else CASES
    modes_config = QUOTA_MODES if quota_experiment else MODES
    warmup = 100 if quota_experiment else 20
    recent = args.phase.startswith('quota_recent')
    if recent:
        modes_config = RECENT_MODES
        seeds = [19] if args.phase == 'quota_recent_pilot' else [211, 307, 419]
        modes = ['q8_recent'] if args.phase == 'quota_recent_pilot' else list(RECENT_MODES)
        cells = [('bursty_r56', mode, seed) for si, seed in enumerate(seeds)
                 for mode in (modes if si % 2 == 0 else modes[::-1])]
    elif args.phase == 'quota_pilot':
        cells = [('bursty_r56', mode, 19) for mode in ['q9_immediate', 'q9_oracle']]
    elif args.phase == 'quota_planner_pilot':
        cells = [('bursty_r56', 'q8_planned', 19)]
    elif args.phase == 'quota':
        cells = []
        # Front-loading a whole quota at the burst boundary is a weak control;
        # the formal no-forecast control spaces its updates uniformly instead.
        modes = [mode for mode in QUOTA_MODES if not mode.endswith('_immediate')]
        modes_config = {mode: QUOTA_MODES[mode] for mode in modes}
        for si, seed in enumerate([211, 307, 419]):
            offset = si*3 % len(modes)
            cells.extend(('bursty_r56', mode, seed) for mode in modes[offset:]+modes[:offset])
    elif args.phase == 'pilot':
        cells = [('bursty_r56', mode, 19) for mode in ['r4', 'oracle100']]
    else:
        cells = []
        for ci, case in enumerate(CASES):
            modes = ['r4', 'oracle100'] if case == 'steady_r20' else list(MODES)
            for si, seed in enumerate([211, 307, 419]):
                offset = (ci+si) % len(modes)
                cells.extend((case, mode, seed) for mode in modes[offset:]+modes[:offset])
    (out/'plan.json').write_text(json.dumps(dict(phase=args.phase, cells=cells,
        cases=cases, modes=modes_config, requests=6000, duration=30, warmup_updates=warmup,
        quota_window_seconds=1,
        planner_cost_history=16 if recent else 64,
        train_bound='1.1 * rolling p90 of last 64 completed compute segments',
        infer_bound='unchanged R4 online cost bound'), indent=2)+'\n')
    # Preserve exactly the executable experiment source alongside each matrix.
    for relative in ['main.py', 'src', 'modified', 'environment',
                     'experiments/optimization/arrival_oracle']:
        source = ROOT/relative
        target = out/'source'/relative
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (out/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    (out/'power_mode.txt').write_text(subprocess.check_output(['nvpmodel', '-q'], text=True))
    rows, paired = [], {}
    for case, mode, seed in cells:
        setting = cases[case]
        mode_config = modes_config[mode]
        policy, budget = mode_config[:2]
        quota = mode_config[2] if len(mode_config) == 3 else 0
        name = f'{case}_{mode}_seed{seed}'
        work = out/'work'/name
        work.mkdir(parents=True)
        entry = Path(__file__).with_name('recent_start.py') if recent else ROOT/'main.py'
        cmd = [sys.executable, str(entry), '--benchmark', 'split_cifar10',
            '--algorithm', 'replay', '--model', setting['model'], '--epoch', str(setting['epochs']),
            '--training_bs', '64', '--eval_bs', '32', '--global_scheduler_mode', 'stream_fifo',
            '--host_preparation', 'cooperative', '--seed', str(seed), '--max_runtime', '480',
            '--record_model_hash', '--stream_requests', '6000', '--stream_duration', '30',
            '--stream_slo_ms', str(setting['slo']), '--stream_pattern', setting['pattern'],
            '--stream_start_policy', policy, '--stream_start_max_wait_ms', str(budget),
            '--stream_train_quota', str(quota), '--stream_warmup_updates', str(warmup)]
        log = out/(name+'.log')
        started = time.perf_counter()
        with log.open('w') as handle:
            handle.write(json.dumps(dict(command=cmd, load_start=os.getloadavg(),
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))+'\n')
            handle.flush()
            process = subprocess.Popen(cmd, cwd=work, env=experiment_env(args.gpu, seed),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = process.wait(timeout=540)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                rc = process.wait()
        wall = time.perf_counter()-started
        contents = log.read_text()
        if rc or re.search(r'Traceback|\[ERROR\]|Training failed', contents):
            raise RuntimeError(f'Failed run {name}; log and work directory retained')
        metric = json.loads((work/'stream_metrics.json').read_text())
        requests = json.loads((work/'stream_requests.json').read_text())
        trace = json.loads((work/'stream_arrivals.json').read_text())
        training = json.loads((work/'stream_training.json').read_text())
        starts = json.loads((work/'stream_train_starts.json').read_text())
        prep = json.loads((work/'stream_preparation.json').read_text())
        snapshot = json.loads(re.findall(r'\[FinalSnapshot\] (\{[^\n]+\})', contents)[-1])
        model_hash = re.findall(r'\[ModelHash\] ([0-9a-f]+)', contents)[-1]
        identity = (model_hash, trace)
        assert identity == paired.setdefault((case, seed), identity), name
        assert metric['train_updates'] == len(training) == 790*setting['epochs'], name
        assert metric['dropped'] == 0 and len(requests) == 6000, name
        assert [r['id'] for r in requests] == list(range(6000)), name
        assert len({r['sample'] for r in requests}) == 6000, name
        for r in requests:
            assert r['sample'] == trace[r['id']]['sample'] and r['arrival'] == trace[r['id']]['arrival']
            assert abs(r['latency']-(r['completed']-r['arrival'])) < 1e-9
            assert r['on_time'] == (r['latency'] <= setting['slo']/1000)
        assert snapshot['total_samples'] == 10000, name
        assert prep['certified_epochs'] == 10*setting['epochs'] and prep['fallback_epochs'] == 0
        row = dict(case=case, mode=mode, seed=seed, wall_sec=wall, **metric,
            on_time_accuracy=sum(r['correct'] and r['on_time'] for r in requests)/len(requests),
            mean_request_train_updates=sum(r['train_updates'] for r in requests)/len(requests),
            updates_by_stream_end=max((r['update'] for r in training
                if r['completed'] is not None and r['completed'] <= 30), default=warmup),
            predicted_wait_events=sum(r['action'] == 'wait' for r in starts),
            forced_starts=sum(r['budget_exhausted'] for r in starts),
            quota_forced_starts=sum(r.get('quota_forced', False) for r in starts),
            final_accuracy=snapshot['streaming_accuracy'], model_hash=model_hash,
            valid=True, log=log.name)
        rows.append(row)
        with (out/'results.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
    (out/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))


if __name__ == '__main__':
    main()
