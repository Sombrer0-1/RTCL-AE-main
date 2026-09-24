#!/usr/bin/env python3
"""Independent supplementary audit; never pool its timings with the main matrix."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    result = {}
    for phase, count, slo in [('diagnostic', 1, .1), ('holdout', 4, .033)]:
        rows = list(csv.DictReader((HERE/phase/'results.csv').open()))
        assert len(rows) == count
        assert all(r['valid'] == 'True' for r in rows)
        assert len({r['model_hash'] for r in rows}) == 1
        assert len({r['trace_hash'] for r in rows}) == 1
        assert len({r['final_accuracy'] for r in rows}) == 1
        for row in rows:
            folder = HERE/phase/row['log'].removesuffix('.log')
            requests = json.loads((folder/'stream_requests.json').read_text())
            trace = json.loads((folder/'stream_arrivals.json').read_text())
            assert len(requests) == 6000 and len({r['sample'] for r in requests}) == 6000
            assert [r['id'] for r in requests] == list(range(6000))
            assert all(r['sample'] == trace[r['id']]['sample'] and
                       r['arrival'] == trace[r['id']]['arrival'] and
                       abs(r['completed']-r['arrival']-r['latency']) < 1e-7 and
                       r['on_time'] == (r['latency'] <= slo) for r in requests)
            assert np.isclose(np.mean([not r['on_time'] for r in requests]), float(row['miss_rate']))
            assert np.isclose(np.percentile([r['latency'] for r in requests],95)*1000,float(row['p95_ms']))
            assert np.isclose(sum(r['on_time'] for r in requests)/max(r['completed'] for r in requests),float(row['goodput']))
        result[phase] = rows
    baseline = next(r for r in csv.DictReader((HERE/'main/results.csv').open())
                    if r['case'] == 'longer' and r['mode'] == 'fifo' and r['seed'] == '101')
    assert result['diagnostic'][0]['model_hash'] == baseline['model_hash']
    assert result['diagnostic'][0]['trace_hash'] == baseline['trace_hash']
    stalls = json.loads((HERE/'diagnostic/longer_fifo_seed101/stream_epoch_stalls.json').read_text())
    assert len(stalls) == 20
    assert [r['train_updates'] for r in stalls] == [79*k for k in range(20)]
    warm = [r for r in stalls if r['previous_update_to_first_iteration_ms'] is not None]
    a = [r['before_epoch_to_first_iteration_ms'] for r in warm]
    b = [r['previous_update_to_first_iteration_ms'] for r in warm]
    result['epoch_diagnostic'] = dict(startup_sum_ms=sum(a),gap_sum_ms=sum(b),
                                     startup_fraction=sum(a)/sum(b),
                                     startup_median_ms=float(np.median(a)),startup_max_ms=max(a),
                                     gap_max_ms=max(b))
    rows = sorted(result['holdout'],key=lambda r: ['fifo','slack','debt','neural'].index(r['mode']))
    metrics = [('wall_sec','Total wall time (s)',1),('p95_ms','Request P95 (ms)',1),
               ('miss_rate','Deadline misses (%)',100),('controller_seconds','Controller CPU time (s)',1)]
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),constrained_layout=True)
    for ax,(key,label,mult) in zip(axes,metrics):
        ax.bar(range(4),[float(r[key])*mult for r in rows],color=['#64748b','#d97706','#0284c7','#9333ea'])
        ax.set_xticks(range(4),['FIFO','Slack','Debt','Neural'])
        ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.2)
        if key=='p95_ms':
            ax.axhline(33,color='black',ls=':',lw=1,label='SLO')
            ax.legend(fontsize=8)
    loads = [float(r[k]) for r in rows for k in ['load_start','load_end']]
    fig.suptitle(f'Independent seed 303 | ResNet56, bursty | host load {min(loads):.1f}–{max(loads):.1f}\nOne run per policy; unchanged parameters; separate from main matrix')
    fig.savefig(HERE/'holdout_comparison.png',dpi=180)
    fig.savefig(HERE/'holdout_comparison.pdf')
    (HERE/'supplement_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['epoch_diagnostic'],indent=2))
    print('All supplementary request records and model/trace hashes verified.')

if __name__ == '__main__':
    main()
