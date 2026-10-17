#!/usr/bin/env python3
"""Compare independently archived recent-cost runs with the frozen quota matrix."""
import argparse
from bisect import bisect_right
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
from experiments.optimization.arrival_oracle.analyze import write_csv, COLORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--recent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    directories = [args.main.resolve(), args.recent.resolve()]
    environments = [json.loads((d/'execution_environment.json').read_text()) for d in directories]
    assert environments[0] == environments[1]
    assert (directories[0]/'power_mode.txt').read_text() == (directories[1]/'power_mode.txt').read_text()
    rows, requests, boundaries, identities = [], {}, {}, {}
    for source, directory in zip(['main', 'recent'], directories):
        plan = json.loads((directory/'plan.json').read_text())
        assert plan['warmup_updates'] == 100 and plan['duration'] == 30
        current = list(csv.DictReader((directory/'audited_results.csv').open()))
        assert len(current) == len(plan['cells'])
        for row in current:
            row = dict(row, source=source)
            assert row['valid'] == 'True' and int(row['train_updates']) == 790
            key = (row['mode'], row['seed'])
            work = directory/'work'/row['log'].removesuffix('.log')
            records = json.loads((work/'stream_requests.json').read_text())
            trace = json.loads((work/'stream_arrivals.json').read_text())
            identity = (row['model_hash'], trace)
            assert identity == identities.setdefault(row['seed'], identity)
            assert len(records) == 6000 and [r['id'] for r in records] == list(range(6000))
            assert np.isclose(np.mean([r['correct'] and r['on_time'] for r in records]), float(row['on_time_accuracy']))
            training = json.loads((work/'stream_training.json').read_text())
            ended = [r['completed'] for r in training if r['completed'] is not None]
            boundaries[key] = [100+bisect_right(ended, t) for t in range(1, 31)]
            requests[key] = records
            rows.append(row)
    indexed = {(r['mode'], r['seed']): r for r in rows}
    pairs = []
    for row in [r for r in rows if r['source'] == 'recent']:
        prefix = row['mode'].split('_')[0]
        for baseline in [prefix+'_uniform', prefix+'_reactive', prefix+'_phase', prefix+'_oracle',
                         prefix+'_planned', 'warm_r4', 'warm_phase']:
            base = indexed[(baseline, row['seed'])]
            a = requests[(row['mode'], row['seed'])]
            b = requests[(baseline, row['seed'])]
            hybrid = np.mean([r['on_time'] and s['correct'] for r, s in zip(a, b)])
            pairs.append(dict(mode=row['mode'], baseline=baseline, seed=int(row['seed']),
                miss_delta_pp=100*(float(row['miss_rate'])-float(base['miss_rate'])),
                correct_delta_pp=100*(float(row['on_time_accuracy'])-float(base['on_time_accuracy'])),
                online_delta_pp=100*(float(row['online_accuracy'])-float(base['online_accuracy'])),
                wall_change_pct=100*(float(row['wall_sec'])/float(base['wall_sec'])-1),
                end_progress_delta=int(row['updates_by_stream_end'])-int(base['updates_by_stream_end']),
                max_boundary_progress_difference=max(abs(x-y) for x, y in zip(
                    boundaries[(row['mode'], row['seed'])], boundaries[(baseline, row['seed'])])),
                timing_component_pp=100*(hybrid-float(base['on_time_accuracy'])),
                prediction_component_pp=100*(float(row['on_time_accuracy'])-hybrid),
                deadline_rescued=sum(r['on_time'] and not s['on_time'] for r, s in zip(a, b)),
                new_deadline_misses=sum(not r['on_time'] and s['on_time'] for r, s in zip(a, b))))
    summary = []
    fields = ['miss_rate', 'p95_ms', 'p99_ms', 'online_accuracy', 'on_time_accuracy', 'wall_sec',
              'updates_by_stream_end', 'mean_request_train_updates', 'quota_end_shortfall',
              'max_boundary_shortfall', 'intentional_idle_seconds', 'planning_seconds',
              'controller_seconds', 'plan_infeasible_windows']
    for mode in dict.fromkeys(r['mode'] for r in rows):
        selected = [r for r in rows if r['mode'] == mode]
        entry = dict(mode=mode, source=selected[0]['source'], runs=len(selected))
        for field in fields:
            values = [float(r[field]) for r in selected]
            entry[field] = float(np.mean(values))
            entry[field+'_std'] = float(np.std(values, ddof=1))
        summary.append(entry)
    write_csv(out/'paired_comparisons.csv', pairs)
    write_csv(out/'summary.csv', summary)
    result = dict(runs=len(rows), requests=len(rows)*6000, directories=list(map(str, directories)),
                  environment=environments[0], summary=summary, pairs=pairs)
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, x, y, ylabel, scale in [
            (axes[0], 'updates_by_stream_end', 'miss_rate', 'Deadline misses (%)', 100),
            (axes[1], 'updates_by_stream_end', 'on_time_accuracy', 'On-time correct (%)', 100),
            (axes[2], 'wall_sec', 'on_time_accuracy', 'On-time correct (%)', 100)]:
        for policy in ['uniform', 'reactive', 'phase', 'oracle', 'planned', 'recent']:
            selected = [r for r in summary if r['mode'] in (f'q8_{policy}', f'q9_{policy}')]
            selected.sort(key=lambda r: r[x])
            if not selected:
                continue
            color = COLORS[selected[0]['mode']]
            label = {'planned': 'window / 64 costs', 'recent': 'window / 16 costs'}.get(policy, policy)
            ax.plot([r[x] for r in selected], [r[y]*scale for r in selected], '-o', color=color, label=label)
            raw = [r for r in rows if r['mode'] in (f'q8_{policy}', f'q9_{policy}')]
            ax.scatter([float(r[x]) for r in raw], [float(r[y])*scale for r in raw], color=color, alpha=.25, s=14)
        for mode in ['warm_r4', 'warm_phase']:
            item = next(r for r in summary if r['mode'] == mode)
            ax.scatter(item[x], item[y]*scale, marker='*', s=140, color=COLORS[mode], label=mode)
        ax.set_xlabel('Training updates by 30 s' if x == 'updates_by_stream_end' else 'Total wall time (s)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out/'information_frontier.png', dpi=170)
    fig.savefig(out/'information_frontier.pdf')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
