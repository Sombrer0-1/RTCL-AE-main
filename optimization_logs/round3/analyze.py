#!/usr/bin/env python3
"""Recompute metrics from every response; plot individual seeds, never pool phases."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODES = ['fifo', 'slack', 'debt', 'neural']
CASES = ['longer', 'deeper_bursty']
COLORS = ['#64748b', '#d97706', '#0284c7', '#9333ea']


def analyze():
    rows = list(csv.DictReader((HERE/'main/results.csv').open()))
    assert len(rows) == 16, f'Expected 16 runs, got {len(rows)}'
    groups = {}
    diagnostics = []
    for row in rows:
        assert row['valid'] == 'True'
        assert int(row['requests']) == 6000 and int(row['dropped']) == 0
        group = groups.setdefault((row['case'], row['seed']), [])
        group.append(row)
        work = HERE/'main'/row['log'].removesuffix('.log')
        requests = json.loads((work/'stream_requests.json').read_text())
        arrivals = json.loads((work/'stream_arrivals.json').read_text())
        slo = .1 if row['case'] == 'longer' else .033
        assert len(requests) == 6000
        assert len({r['sample'] for r in requests}) == 6000
        assert [r['id'] for r in requests] == list(range(6000))
        assert all(r['sample'] == arrivals[r['id']]['sample'] and
                   r['arrival'] == arrivals[r['id']]['arrival'] and
                   abs(r['completed']-r['arrival']-r['latency']) < 1e-7 and
                   r['on_time'] == (r['latency'] <= slo) for r in requests)
        lat = np.array([r['latency'] for r in requests])
        assert np.isclose(np.percentile(lat, 95)*1000, float(row['p95_ms']))
        assert np.isclose(np.mean(lat > slo), float(row['miss_rate']))
        assert np.isclose(sum(r['on_time'] for r in requests)/max(r['completed'] for r in requests),
                          float(row['goodput']))
        missed = [r for r in requests if not r['on_time']]
        # Diagnostic association only: callbacks run after host data/experience setup.
        boundary_misses = sum(r['train_updates'] % 79 == 0 for r in missed)
        diagnostics.append(dict(case=row['case'], mode=row['mode'], seed=int(row['seed']),
                                missed=len(missed), missed_at_epoch_boundary=boundary_misses,
                                requests_after_training=sum(r['train_updates'] == int(row['train_updates']) for r in requests),
                                mean_request_batch=float(np.mean([r['batch'] for r in requests])),
                                control_wall_percent=100*float(row['controller_seconds'])/float(row['wall_sec'])))
    for key, group in groups.items():
        assert len(group) == 4 and {r['mode'] for r in group} == set(MODES)
        assert len({r['model_hash'] for r in group}) == 1, key
        assert len({r['trace_hash'] for r in group}) == 1, key
        assert len({r['final_accuracy'] for r in group}) == 1, key
        assert len({r['train_updates'] for r in group}) == 1, key
    (HERE/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    summary = []
    keys = ['wall_sec', 'train_completion_seconds', 'p50_ms', 'p95_ms', 'p99_ms',
            'miss_rate', 'goodput', 'controller_seconds', 'max_train_pause_seconds',
            'online_accuracy', 'final_accuracy']
    for case in CASES:
        for mode in MODES:
            selected = [r for r in rows if r['case'] == case and r['mode'] == mode]
            result = dict(case=case, mode=mode)
            result.update({k:float(np.mean([float(r[k]) for r in selected])) for k in keys})
            speedups = []
            for r in selected:
                fifo = next(x for x in groups[(case, r['seed'])] if x['mode'] == 'fifo')
                speedups.append(float(fifo['wall_sec'])/float(r['wall_sec']))
            result['paired_wall_speedups_vs_fifo'] = speedups
            summary.append(result)
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
    metrics = [('wall_sec','Total wall time (s)',1), ('p95_ms','Request P95 (ms)',1),
               ('miss_rate','Deadline misses (%)',100), ('controller_seconds','Controller CPU time (s)',1)]
    for ci, case in enumerate(CASES):
        for j, (key,label,mult) in enumerate(metrics):
            ax=axes[ci,j]
            for mi, mode in enumerate(MODES):
                selected = sorted([r for r in rows if r['case'] == case and r['mode'] == mode], key=lambda r:r['seed'])
                values = [float(r[key])*mult for r in selected]
                if ci == 0:
                    ax.bar(mi, np.mean(values), color=COLORS[mi], alpha=.55, width=.65)
                for si,value in enumerate(values):
                    ax.scatter(mi+(-.09 if si==0 else .09),value,color=COLORS[mi],
                               marker='o' if si==0 else 'x',s=32,zorder=3)
            ax.set_xticks(range(4), ['FIFO','Slack','Debt','Neural'])
            ax.set_ylabel(label)
            ax.grid(axis='y',alpha=.2)
            if j==0:
                ax.set_title('ResNet20, 2 epochs, steady' if ci==0 else 'ResNet56: host-load confounded')
            if key=='p95_ms':
                ax.axhline(100 if ci==0 else 33,color='black',ls=':',lw=1,label='SLO')
                ax.legend(fontsize=8)
    fig.suptitle('Uncached causal requests | 6,000 unique samples/run | dots: seed 101; crosses: seed 202\nBottom row: host load rose to 55; no pooled bars or causal speedup claim')
    fig.savefig(HERE/'comparison.png',dpi=180)
    fig.savefig(HERE/'comparison.pdf')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    analyze()
