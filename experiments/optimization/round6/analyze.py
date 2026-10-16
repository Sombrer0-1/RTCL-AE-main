#!/usr/bin/env python3
"""Audit request/decision/cost evidence and compare matched neural ablations."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.optimization.round5.analyze import audit

METRICS = ['wall_sec', 'train_completion_seconds', 'p50_ms', 'p95_ms', 'p99_ms',
           'miss_rate', 'online_accuracy', 'on_time_correct_rate', 'goodput',
           'controller_seconds', 'mean_batch', 'inference_batches']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'main', 'holdout'], required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    planned = 16 if args.phase == 'main' else 4
    observed = len(list(csv.DictReader((directory/'results.csv').open())))
    if args.partial:
        assert 0 < observed <= planned
    rows = audit(directory, observed if args.partial else planned)
    for row in rows:
        folder = directory/row['log'].removesuffix('.log')
        records = json.loads((folder/'stream_requests.json').read_text())
        joint = np.mean([r['correct'] and r['on_time'] for r in records])
        assert np.isclose(joint, float(row['on_time_correct_rate']))
        if row['mode'].startswith('lookahead'):
            decisions = json.loads((folder/'stream_decisions.json').read_text())
            costs = json.loads((folder/'stream_costs.json').read_text())
            assert len(decisions) == int(row['decisions'])
            assert all(0 <= d['action'] <= min(d['queue'], 32) and d['oldest'] >= 0 for d in decisions)
            assert sum(d['action'] != d['plain_action'] for d in decisions) == int(row['neural_changed_decisions'])
            assert sum(0 < d['action'] < min(d['queue'], 32) for d in decisions) == int(row['lookahead_split_batches'])
            assert sum(d['action'] == 0 for d in decisions) == int(row['predictive_deferrals'])
            if row.get('adaptive_preparation') == 'True':
                intervals = json.loads((folder/'stream_intervals.json').read_text())
                assert len(intervals) == sum(c['kind'] == 'infer' for c in costs)
                assert sum(abs(r['interval']-r['plain_interval']) > 1e-6 for r in intervals) == int(row['neural_changed_intervals'])
                assert np.isclose(min(r['interval'] for r in intervals)*1000, float(row['preparation_interval_min_ms']))
                assert np.isclose(max(r['interval'] for r in intervals)*1000, float(row['preparation_interval_max_ms']))
            if row.get('neural_changed_checks') not in (None, ''):
                checks = json.loads((folder/'stream_check_events.json').read_text())
                assert len(checks) == int(row['neural_changed_checks'])
                assert sum(r['requests_waiting'] for r in checks) == int(row['neural_changed_checks_with_requests'])
                assert all((r['elapsed'] >= r['interval']) != (r['elapsed'] >= r['plain_interval']) for r in checks)
            for kind in ('train', 'infer'):
                selected = [c for c in costs if c['kind'] == kind and c['ready']]
                assert len(selected) == int(row[f'{kind}_cost_scored'])
                assert sum(c['duration'] > c['upper'] for c in selected) == int(row[f'{kind}_bound_exceeded'])
                for model in ('base', 'neural'):
                    field = 'point' if model == 'neural' else 'base'
                    error = np.mean([abs(np.log(c['duration']/c[field])) for c in selected])
                    assert np.isclose(error, float(row[f'{kind}_{model}_log_mae']))
            if row['mode'] in ('lookahead_neural', 'lookahead_rls'):
                assert int(row['neural_parameters']) == (146 if row['mode'] == 'lookahead_rls' else 133)
                assert int(row['neural_updates']) > 0
    summary = []
    for case in dict.fromkeys(r['case'] for r in rows):
        for mode in dict.fromkeys(r['mode'] for r in rows):
            selected = [r for r in rows if r['case'] == case and r['mode'] == mode]
            if not selected:
                continue
            entry = dict(case=case, mode=mode, runs=len(selected))
            for key in METRICS:
                values = [float(r[key]) for r in selected]
                entry[key] = float(np.mean(values))
                entry[key+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(entry)
    pairs = []
    for row in rows:
        if row['mode'] not in ('lookahead', 'lookahead_neural', 'lookahead_rls'):
            continue
        baselines = ('cooperative',) if row['mode'] == 'lookahead' else ('lookahead', 'cooperative', 'predictive_neural')
        for baseline in baselines:
            base = next((r for r in rows if r['case'] == row['case'] and r['seed'] == row['seed']
                         and r['mode'] == baseline), None)
            if base is None:
                continue
            pairs.append(dict(case=row['case'], seed=int(row['seed']), method=row['mode'], baseline=baseline,
                wall_change_pct=100*(float(row['wall_sec'])/float(base['wall_sec'])-1),
                p95_change_pct=100*(float(row['p95_ms'])/float(base['p95_ms'])-1),
                p99_change_pct=100*(float(row['p99_ms'])/float(base['p99_ms'])-1),
                miss_delta_pp=100*(float(row['miss_rate'])-float(base['miss_rate'])),
                correct_delta_pp=100*(float(row['on_time_correct_rate'])-float(base['on_time_correct_rate'])),
                online_delta_pp=100*(float(row['online_accuracy'])-float(base['online_accuracy'])),
                changed_decisions=int(row['neural_changed_decisions']),
                split_batches=int(row['lookahead_split_batches'])))
    result = dict(phase=args.phase, complete=len(rows) == planned, runs=len(rows), planned=planned,
                  responses=len(rows)*6000, summary=summary, pairs=pairs)
    (directory/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (directory/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    if pairs:
        with (directory/'paired_comparisons.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    cases = list(dict.fromkeys(r['case'] for r in rows))
    modes = list(dict.fromkeys(r['mode'] for r in rows))
    colors = dict(cooperative='#64748b', predictive_neural='#d97706',
                  lookahead='#0284c7', lookahead_rls='#059669', lookahead_neural='#059669')
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 3.6*len(cases)), squeeze=False,
                             constrained_layout=True)
    for ci, case in enumerate(cases):
        for j, (key, label, scale) in enumerate([
                ('wall_sec', 'Total wall time (s)', 1), ('p95_ms', 'Request P95 (ms)', 1),
                ('miss_rate', 'Deadline misses (%)', 100), ('on_time_correct_rate', 'On-time correct (%)', 100)]):
            ax = axes[ci, j]
            for mi, mode in enumerate(modes):
                values = [float(r[key])*scale for r in rows if r['case'] == case and r['mode'] == mode]
                if not values:
                    continue
                ax.bar(mi, np.mean(values), color=colors[mode], alpha=.65)
                ax.scatter(mi + (np.arange(len(values))-(len(values)-1)/2)*.1, values, color='black', s=18)
            ax.set_xticks(range(len(modes)), modes, rotation=22, fontsize=7)
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=.2)
            if key == 'miss_rate':
                ax.set_ylim(0, max(1, 1.1*max(float(r[key])*scale for r in rows)))
            if j == 0:
                ax.set_title(case)
    fig.suptitle(f'Calibrated neural costs and two-chunk lookahead | {args.phase} | {len(rows)}/{planned} runs')
    fig.savefig(directory/'comparison.png', dpi=170)
    fig.savefig(directory/'comparison.pdf')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
