"""Recompute all request metrics and report paired learning ablations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def audit(directory, expected):
    rows = list(csv.DictReader((directory/'results.csv').open()))
    assert len(rows) == expected, (directory, len(rows), expected)
    identities = {}
    for row in rows:
        assert row['valid'] == 'True' and row['return_code'] == '0'
        folder = directory/row['log'].removesuffix('.log')
        arrivals = json.loads((folder/'stream_arrivals.json').read_text())
        records = json.loads((folder/'stream_requests.json').read_text())
        assert len(arrivals) == len(records) == 6000
        assert len({r['sample'] for r in records}) == 6000
        assert [r['id'] for r in records] == list(range(6000))
        slo = .033 if row['case'] == 'deeper_bursty' else .1
        assert all(r['arrival'] == arrivals[r['id']]['arrival']
                   and r['sample'] == arrivals[r['id']]['sample']
                   and np.isclose(r['latency'], r['completed']-r['arrival'])
                   and r['latency'] >= 0 and r['on_time'] == (r['latency'] <= slo)
                   for r in records)
        latency = np.array([r['latency'] for r in records])
        assert np.isclose(np.mean(latency > slo), float(row['miss_rate']))
        for q in (50, 95, 99):
            assert np.isclose(np.percentile(latency, q)*1000, float(row[f'p{q}_ms']))
        assert np.isclose(sum(r['on_time'] for r in records)/max(r['completed'] for r in records),
                          float(row['goodput']))
        assert np.isclose(np.mean([r['correct'] for r in records]), float(row['online_accuracy']))
        assert int(row['train_updates']) == 790*int(row['epochs'])
        assert row['trace_hash'] == hashlib.sha256((folder/'stream_arrivals.json').read_bytes()).hexdigest()
        identity = (row['model_hash'], row['trace_hash'], row['final_accuracy'])
        assert identities.setdefault((row['case'], row['seed']), identity) == identity
        log = (directory/row['log']).read_text()
        assert re.findall(r'\[ModelHash\] ([0-9a-f]+)', log)[-1] == row['model_hash']
        final = json.loads(re.findall(r'\[FinalSnapshot\] (\{[^\n]+\})', log)[-1])
        assert final['total_samples'] == 10000
        assert np.isclose(final['streaming_accuracy'], float(row['final_accuracy']))
        assert int(row['certified_epochs']) == 10*int(row['epochs']) and int(row['fallback_epochs']) == 0
        if row['mode'].startswith('predictive'):
            assert int(row['released_observations']) == 6000
        if row['mode'] == 'predictive_neural':
            assert int(row['neural_parameters']) == 133 and int(row['neural_updates']) > 0
        row['inference_batches'] = len({r['completed'] for r in records})
        row['mean_batch'] = 6000/row['inference_batches']
    return rows


def summarize(rows):
    metrics = ['wall_sec', 'train_completion_seconds', 'p95_ms', 'p99_ms', 'miss_rate',
               'goodput', 'online_accuracy', 'controller_seconds', 'mean_batch', 'inference_batches']
    summary = []
    for case in dict.fromkeys(r['case'] for r in rows):
        for mode in dict.fromkeys(r['mode'] for r in rows):
            selected = [r for r in rows if r['case'] == case and r['mode'] == mode]
            if not selected:
                continue
            entry = dict(case=case, mode=mode, runs=len(selected))
            for key in metrics:
                values = [float(r[key]) for r in selected]
                entry[key] = float(np.mean(values))
                entry[key+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(entry)
    pairs = []
    for row in rows:
        if row['mode'] != 'predictive_neural':
            continue
        base = next(r for r in rows if r['case'] == row['case'] and r['seed'] == row['seed']
                    and r['mode'] == 'predictive')
        pairs.append(dict(case=row['case'], seed=int(row['seed']),
            wall_ratio=float(row['wall_sec'])/float(base['wall_sec']),
            p95_ratio=float(row['p95_ms'])/float(base['p95_ms']),
            p99_ratio=float(row['p99_ms'])/float(base['p99_ms']),
            miss_delta_pp=100*(float(row['miss_rate'])-float(base['miss_rate'])),
            online_accuracy_delta_pp=100*(float(row['online_accuracy'])-float(base['online_accuracy'])),
            neural_log_mae=float(row['neural_log_mae']), base_log_mae=float(row['base_log_mae']),
            changed_decisions=int(row['neural_changed_decisions']),
            neural_updates=int(row['neural_updates']),
            load_start=float(row['load_start']), load_end=float(row['load_end'])))
    return summary, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'main', 'holdout'], required=True)
    parser.add_argument('--partial', action='store_true', help='Explicitly audit an incomplete matrix without claiming completion')
    args = parser.parse_args()
    directory = HERE/args.phase
    planned = {'pilot': 3, 'main': 16, 'holdout': 4}[args.phase]
    observed = len(list(csv.DictReader((directory/'results.csv').open())))
    if args.partial:
        assert 0 < observed <= planned
    rows = audit(directory, observed if args.partial else planned)
    summary, pairs = summarize(rows)
    payload = dict(phase=args.phase, planned_runs=planned, runs=len(rows), complete=len(rows)==planned,
                   responses=len(rows)*6000,
                   summary=summary, learning_pairs=pairs)
    prefix = args.phase+('_partial' if len(rows) < planned else '')
    (HERE/(prefix+'_summary.json')).write_text(json.dumps(payload, indent=2))
    cases = list(dict.fromkeys(r['case'] for r in rows))
    modes = [m for m in ['certified', 'cooperative', 'predictive', 'predictive_neural']
             if any(r['mode'] == m for r in rows)]
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 3.5*len(cases)), squeeze=False,
                             constrained_layout=True)
    for ci, case in enumerate(cases):
        for j, (key, label, scale) in enumerate([('wall_sec', 'Total wall time (s)', 1),
                ('p95_ms', 'Request P95 (ms)', 1), ('p99_ms', 'Request P99 (ms)', 1),
                ('miss_rate', 'Deadline misses (%)', 100)]):
            ax = axes[ci, j]
            for mi, mode in enumerate(modes):
                selected = sorted((r for r in rows if r['case'] == case and r['mode'] == mode), key=lambda r:r['seed'])
                vals = [float(r[key])*scale for r in selected]
                ax.bar(mi, np.mean(vals), color=['#64748b', '#059669', '#0284c7', '#d97706'][mi], alpha=.6)
                for si, value in enumerate(vals):
                    ax.scatter(mi+(si-(len(vals)-1)/2)*.12, value, color='black', marker=['o', 'x', '^'][si], s=24)
            ax.set_xticks(range(len(modes)), modes, rotation=18, fontsize=8)
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=.2)
            if j == 0:
                ax.set_title(case)
    fig.suptitle(f'Online batching and neural ablation | {prefix} | {len(rows)}/{planned} runs\n'
                 'All requests retained; bars average per-run metrics; paired model/trace hashes match')
    fig.savefig(HERE/(prefix+'_comparison.png'), dpi=170)
    fig.savefig(HERE/(prefix+'_comparison.pdf'))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
