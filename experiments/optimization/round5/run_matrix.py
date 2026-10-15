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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from environment.experiment_runtime import experiment_env, record_environment


def load_completed(out, resume, scheduled):
    """Resume only a validated prefix of the frozen execution order."""
    path = out/'results.csv'
    if not path.exists():
        return [], {}
    if not resume:
        raise ValueError('existing results: use --resume or choose a new directory')
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    expected = [(case, mode, str(seed)) for case, mode, seed in scheduled]
    actual = [(r['case'], r['mode'], r['seed']) for r in rows]
    if actual != expected[:len(actual)] or len(rows) > len(expected):
        raise ValueError('saved results do not match the frozen matrix prefix')
    paired = {}
    for row in rows:
        if row['valid'] != 'True' or row['return_code'] != '0':
            raise ValueError('saved failed run requires diagnosis; cannot silently skip it')
        artifact = out/row['log'].removesuffix('.log')
        trace_file = artifact/'stream_arrivals.json'
        if hashlib.sha256(trace_file.read_bytes()).hexdigest() != row['trace_hash']:
            raise ValueError('saved request trace changed')
        identity = row['model_hash'], row['trace_hash']
        if paired.setdefault((row['case'], int(row['seed'])), identity) != identity:
            raise ValueError('saved paired model/request identities disagree')
    return rows, paired


def preserve_unrecorded(out, name):
    """Keep orphaned logs/work after interruption; their wall time is unknown."""
    sources = [out/'work'/name, out/(name+'.log'), out/name]
    sources = [source for source in sources if source.exists()]
    if not sources:
        return
    parent = out/'interrupted'/name
    attempt = 1
    while (parent/f'attempt{attempt:03d}').exists():
        attempt += 1
    target = parent/f'attempt{attempt:03d}'
    target.mkdir(parents=True)
    for source in sources:
        label = 'work' if source.parent.name == 'work' else 'artifacts' if source.is_dir() else source.name
        shutil.move(str(source), str(target/label))
    (target/'README.txt').write_text('Interrupted coordinator: original artifacts preserved. '
        'Exact parent-measured wall time was not committed; this cell is rerun, not reconstructed from log timestamps.\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pilot', 'main', 'holdout'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0, help='Physical GPU index (Thor: 0)')
    p.add_argument('--resume', action='store_true', help='Continue a validated matrix prefix; preserve orphaned artifacts')
    p.add_argument('--dry-run', action='store_true', help='Validate the saved prefix and list remaining cells without GPU execution')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = [('longer', 'resnet20', 2, 'steady', 100, 6000, 30),
             ('deeper_bursty', 'resnet56', 1, 'bursty', 33, 6000, 30)]
    if a.phase == 'pilot':
        cases = cases[:1]
        seeds = [19]
        modes = ['cooperative', 'predictive', 'predictive_neural']
    elif a.phase == 'holdout':
        seeds = [419]
        modes = ['predictive', 'predictive_neural']
    else:
        seeds = [211, 307]
        modes = ['certified', 'cooperative', 'predictive', 'predictive_neural']
    mode_policy = {'certified': 'stream_fifo', 'cooperative': 'stream_fifo',
                   'predictive': 'stream_predictive', 'predictive_neural': 'stream_predictive_neural'}
    preparation = {mode: 'certified' if mode == 'certified' else 'cooperative' for mode in modes}
    scheduled = []
    for ci, (case, *_rest) in enumerate(cases):
        for si, seed in enumerate(seeds):
            offset = (ci*2+si) % len(modes)
            scheduled.extend((case, mode, seed) for mode in modes[offset:]+modes[:offset])
    rows, paired = load_completed(out, a.resume, scheduled)
    done = {(r['case'], r['mode'], int(r['seed'])) for r in rows}
    if a.dry_run:
        print(json.dumps(dict(phase=a.phase, recorded=len(rows), planned=len(scheduled),
            remaining=[dict(case=case, mode=mode, seed=seed) for case, mode, seed in scheduled
                       if (case, mode, seed) not in done]), indent=2))
        return
    if len(done) == len(scheduled):
        print('All scheduled runs are already recorded; nothing to rerun.')
        return
    record_environment(out, a.gpu)
    for ci, (case, model, epochs, pattern, slo, count, duration) in enumerate(cases):
        for si, seed in enumerate(seeds):
            offset = (ci*2+si) % len(modes)
            for mode in modes[offset:]+modes[:offset]:
                if (case, mode, seed) in done:
                    continue
                name = f'{case}_{mode}_seed{seed}'
                if a.resume:
                    preserve_unrecorded(out, name)
                work = out/'work'/name
                work.mkdir(parents=True, exist_ok=False)
                entry = ROOT/'main.py'
                cmd = [sys.executable, str(entry), '--benchmark', 'split_cifar10',
                       '--algorithm', 'replay', '--model', model, '--epoch', str(epochs),
                       '--training_bs', '64', '--eval_bs', '32', '--global_scheduler_mode',
                       mode_policy[mode],
                       '--host_preparation', preparation[mode],
                       '--seed', str(seed), '--max_runtime', '480',
                       '--record_model_hash', '--stream_requests', str(count),
                       '--stream_duration', str(duration), '--stream_slo_ms', str(slo),
                       '--stream_pattern', pattern]
                env = experiment_env(a.gpu, seed)
                load_start = os.getloadavg()
                log = out/(name+'.log')
                started = time.perf_counter()
                with log.open('x') as f:
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
                    for source in work.glob('stream_*'):
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
                          'guard_overrides','neural_parameters','neural_updates', 'predictive_deferrals',
                          'neural_changed_decisions','released_observations','neural_scored',
                          'neural_log_mae','base_log_mae']
                row.update({k: metric.get(k, '') for k in fields})
                prep_file = work/'stream_preparation.json'
                prep = json.loads(prep_file.read_text()) if prep_file.exists() else {}
                if mode in preparation:
                    valid = valid and prep.get('certified_epochs') == 10*epochs and prep.get('fallback_epochs') == 0
                if preparation[mode] == 'cooperative':
                    valid = valid and 0 < prep.get('service_requests', 0) <= count
                row['valid'] = bool(valid)
                row.update({k: prep.get(k, 0) for k in ['certified_epochs','fallback_epochs','service_batches','service_requests','service_seconds','check_seconds']})
                rows.append(row)
                temporary_csv = out/'results.csv.tmp'
                with temporary_csv.open('w') as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                    w.writerows(rows)
                temporary_csv.replace(out/'results.csv')
                print(json.dumps(row), flush=True)
                if not valid:
                    raise RuntimeError(f'Invalid run {name}; preserved logs, stop for diagnosis')

if __name__ == '__main__':
    main()
