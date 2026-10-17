#!/usr/bin/env python3
"""Recompute latency, accuracy and training-progress tradeoffs from raw requests."""
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
from experiments.optimization.arrival_oracle.run_matrix import CASES, MODES

FIELDS = ['wall_sec', 'train_completion_seconds', 'miss_rate', 'p95_ms', 'p99_ms',
    'online_accuracy', 'on_time_accuracy', 'intentional_idle_seconds',
    'mean_request_train_updates', 'updates_by_stream_end', 'controller_seconds',
    'max_train_pause_seconds', 'forced_starts', 'compute_p50_ms', 'compute_p90_ms',
    'misses_during_compute', 'misses_outside_compute', 'mean_batch', 'inference_batches',
    'inference_seconds', 'compute_seconds', 'forced_start_misses', 'underestimated_compute_misses']
FIELDS += ['quota_idle_seconds', 'information_idle_seconds', 'quota_forced_starts',
           'quota_end_shortfall', 'max_boundary_shortfall', 'planning_seconds',
           'plan_infeasible_windows', 'plan_expanded_states']
COLORS = dict(r4='#64748b', reactive100='#d97706', oracle100='#0284c7', oracle_unlimited='#059669')
COLORS['warm_r4'] = COLORS['r4']
COLORS['warm_phase'] = '#7e22ce'
for rate in (8, 9):
    for policy, color in dict(immediate='#94a3b8', uniform='#94a3b8', reactive='#d97706', phase='#9333ea', oracle='#0284c7', planned='#059669', recent='#dc2626').items():
        COLORS[f'q{rate}_{policy}'] = color


def write_csv(path, rows):
    if rows:
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan = json.loads((directory/'plan.json').read_text())
    case_config = plan.get('cases', CASES)
    modes_config = plan.get('modes', MODES)
    rows = list(csv.DictReader((directory/'results.csv').open()))
    assert len(rows) == len(plan['cells']), 'incomplete matrix'
    pair_identities, progress, requests_by_run, boundaries = {}, {}, {}, {}
    for row in rows:
        assert row['valid'] == 'True'
        work = directory/'work'/row['log'].removesuffix('.log')
        records = json.loads((work/'stream_requests.json').read_text())
        requests_by_run[(row['case'], row['mode'], row['seed'])] = records
        trace = json.loads((work/'stream_arrivals.json').read_text())
        training = json.loads((work/'stream_training.json').read_text())
        warmup = int(row.get('origin_steps', plan.get('warmup_updates', 20)))
        quota = int(row.get('train_quota', 0))
        row.setdefault('quota_idle_seconds', 0)
        row.setdefault('information_idle_seconds', row['intentional_idle_seconds'])
        row.setdefault('quota_forced_starts', 0)
        row.setdefault('planning_seconds', 0)
        controls = json.loads((work/'stream_train_starts.json').read_text())
        plan_path = work/'stream_window_plans.json'
        window_plans = json.loads(plan_path.read_text()) if plan_path.exists() else []
        row['plan_infeasible_windows'] = sum(not p['feasible'] for p in window_plans)
        row['plan_expanded_states'] = sum(p['expanded_states'] for p in window_plans)
        decisions = {r['update']+1: r for r in controls if r['action'] == 'train'}
        identity = (row['model_hash'], trace)
        assert identity == pair_identities.setdefault((row['case'], row['seed']), identity)
        slo = case_config[row['case']]['slo']/1000
        assert len(records) == len(trace) == 6000
        assert [r['id'] for r in records] == list(range(6000))
        assert len(training) == int(row['train_updates']) == 790*case_config[row['case']]['epochs']
        latency = np.array([r['completed']-r['arrival'] for r in records])
        for r, arrival, dt in zip(records, trace, latency):
            assert r['sample'] == arrival['sample'] and r['arrival'] == arrival['arrival']
            assert dt >= 0 and np.isclose(dt, r['latency']) and r['on_time'] == (dt <= slo)
        recomputed = dict(miss_rate=float(np.mean(latency > slo)),
            p95_ms=float(np.percentile(latency, 95)*1000),
            p99_ms=float(np.percentile(latency, 99)*1000),
            online_accuracy=float(np.mean([r['correct'] for r in records])),
            on_time_accuracy=float(np.mean([r['correct'] and r['on_time'] for r in records])))
        for key, value in recomputed.items():
            assert np.isclose(value, float(row[key])), (row['log'], key)
        row.update(recomputed)
        intervals = [r for r in training if r['started'] is not None]
        starts = [r['started'] for r in intervals]
        during = []
        forced_misses = underestimated_misses = 0
        for record in records:
            index = bisect_right(starts, record['arrival'])-1
            active = index >= 0 and record['arrival'] < intervals[index]['completed']
            during.append(active)
            if active and not record['on_time']:
                interval = intervals[index]
                decision = decisions.get(interval['update'])
                if decision:
                    forced_misses += bool(decision['budget_exhausted'])
                    underestimated_misses += interval['duration'] > decision['train_bound']
        row['forced_start_misses'] = forced_misses
        row['underestimated_compute_misses'] = underestimated_misses
        row['misses_during_compute'] = sum(flag and dt > slo for flag, dt in zip(during, latency))
        row['misses_outside_compute'] = sum(not flag and dt > slo for flag, dt in zip(during, latency))
        durations = [r['duration']*1000 for r in training[warmup:]]
        row['compute_p50_ms'], row['compute_p90_ms'] = map(float, np.percentile(durations, [50, 90]))
        services = {r['service_started']: r['completed'] for r in records}
        batches = len(services)
        row['inference_batches'] = batches
        row['inference_seconds'] = sum(ended-started for started, ended in services.items())
        row['compute_seconds'] = sum(r['duration'] for r in training)
        row['mean_batch'] = len(records)/batches
        completed = [r['completed'] for r in intervals]
        key = (row['case'], row['mode'], row['seed'])
        progress[key] = [warmup+bisect_right(completed, t)
            for t in np.linspace(0, 30, 301)]
        boundaries[key] = [warmup+bisect_right(completed, t) for t in range(1, 31)]
        row['quota_end_shortfall'] = max(0, warmup+30*quota-boundaries[key][-1]) if quota else 0
        row['max_boundary_shortfall'] = max(max(0, warmup+t*quota-v)
            for t, v in enumerate(boundaries[key], 1)) if quota else 0
    summary = []
    for case in case_config:
        for mode in modes_config:
            selected = [r for r in rows if r['case'] == case and r['mode'] == mode]
            if not selected:
                continue
            result = dict(case=case, mode=mode, runs=len(selected))
            for field in FIELDS:
                values = np.array([float(r[field]) for r in selected])
                result[field] = float(values.mean())
                result[field+'_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
            summary.append(result)
    pairs = []
    for row in rows:
        if row['mode'] in ('r4', 'warm_r4'):
            continue
        if row['mode'] == 'warm_phase':
            baselines = ['warm_r4']
        elif row['mode'].startswith('q'):
            prefix, policy = row['mode'].split('_', 1)
            baselines = ['warm_r4']
            if policy not in ('immediate', 'uniform'):
                baselines += [prefix+'_uniform', prefix+'_immediate']
            if policy in ('phase', 'oracle', 'planned', 'recent'):
                baselines += [prefix+'_reactive']
            if policy in ('oracle', 'planned', 'recent'):
                baselines += [prefix+'_phase']
            if policy in ('planned', 'recent'):
                baselines += [prefix+'_oracle']
            if policy == 'recent':
                baselines += [prefix+'_planned']
        else:
            baselines = ['r4'] + (['reactive100'] if row['mode'] == 'oracle100' else [])
            if row['mode'] == 'oracle_unlimited':
                baselines += ['oracle100']
        for baseline in baselines:
            base = next((r for r in rows if r['case'] == row['case'] and r['seed'] == row['seed']
                         and r['mode'] == baseline), None)
            if base is None:
                continue
            current_requests = requests_by_run[(row['case'], row['mode'], row['seed'])]
            base_requests = requests_by_run[(row['case'], baseline, row['seed'])]
            hybrid_correct = np.mean([a['on_time'] and b['correct']
                for a, b in zip(current_requests, base_requests)])
            boundary_difference = max(abs(a-b) for a, b in zip(
                boundaries[(row['case'], row['mode'], row['seed'])],
                boundaries[(row['case'], baseline, row['seed'])]))
            pairs.append(dict(case=row['case'], seed=int(row['seed']), mode=row['mode'], baseline=baseline,
                wall_change_pct=100*(float(row['wall_sec'])/float(base['wall_sec'])-1),
                miss_delta_pp=100*(row['miss_rate']-base['miss_rate']),
                p95_change_pct=100*(row['p95_ms']/base['p95_ms']-1),
                p99_change_pct=100*(row['p99_ms']/base['p99_ms']-1),
                correct_delta_pp=100*(row['on_time_accuracy']-base['on_time_accuracy']),
                online_delta_pp=100*(row['online_accuracy']-base['online_accuracy']),
                deadline_rescued=sum(a['on_time'] and not b['on_time']
                    for a, b in zip(current_requests, base_requests)),
                new_deadline_misses=sum(not a['on_time'] and b['on_time']
                    for a, b in zip(current_requests, base_requests)),
                timing_component_pp=100*(hybrid_correct-base['on_time_accuracy']),
                prediction_component_pp=100*(row['on_time_accuracy']-hybrid_correct),
                max_boundary_progress_difference=boundary_difference,
                updates_by_stream_end_delta=float(row['updates_by_stream_end'])-float(base['updates_by_stream_end']),
                idle_delta_sec=float(row['intentional_idle_seconds'])-float(base['intentional_idle_seconds'])))
    result = dict(phase=plan['phase'], runs=len(rows), responses=len(rows)*6000,
        summary=summary, pairs=pairs, audit='raw requests, complete updates, matching final model and arrival trace')
    (directory/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    write_csv(directory/'summary.csv', summary)
    write_csv(directory/'paired_comparisons.csv', pairs)
    write_csv(directory/'audited_results.csv', rows)
    cases = [case for case in case_config if any(r['case'] == case for r in rows)]
    if plan['phase'].startswith('quota'):
        groups = [(f'bursty_r56 / {rate} updates per second', 'bursty_r56',
            ['warm_r4', 'warm_phase']+[mode for mode in modes_config if mode.startswith(f'q{rate}_')])
            for rate in (8, 9) if any(r['mode'].startswith(f'q{rate}_') for r in rows)]
    else:
        groups = [(case, case, [mode for mode in modes_config
            if any(r['case'] == case and r['mode'] == mode for r in rows)]) for case in cases]
    fig, axes = plt.subplots(len(groups), 5, figsize=(17, 3.8*len(groups)), squeeze=False,
                            constrained_layout=True)
    for ci, (label, case, modes) in enumerate(groups):
        modes = [mode for mode in modes if any(r['case'] == case and r['mode'] == mode for r in rows)]
        for j, (field, title, scale) in enumerate([
                ('miss_rate', 'Deadline misses (%)', 100), ('p95_ms', 'P95 latency (ms)', 1),
                ('on_time_accuracy', 'On-time correct (%)', 100), ('wall_sec', 'Total wall time (s)', 1),
                ('updates_by_stream_end', 'Training updates by 30 s', 1)]):
            ax = axes[ci, j]
            for mi, mode in enumerate(modes):
                values = [float(r[field])*scale for r in rows if r['case'] == case and r['mode'] == mode]
                ax.bar(mi, np.mean(values), color=COLORS[mode], alpha=.7)
                ax.scatter(mi+(np.arange(len(values))-(len(values)-1)/2)*.1, values, c='black', s=16)
            ax.set_xticks(range(len(modes)), modes, rotation=30, fontsize=8)
            ax.set_ylabel(title)
            ax.set_title(label)
            ax.grid(axis='y', alpha=.2)
            if field == 'miss_rate':
                ax.set_ylim(0, max(1, 1.1*max(float(r[field])*scale
                    for r in rows if r['case'] == case)))
    fig.savefig(directory/'comparison.png', dpi=170)
    fig.savefig(directory/'comparison.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 4), squeeze=False, constrained_layout=True)
    times = np.linspace(0, 30, 301)
    for ci, (label, case, modes) in enumerate(groups):
        ax = axes[0, ci]
        for mode in modes:
            values = [v for (c, m, seed), v in progress.items() if c == case and m == mode]
            if values:
                data = np.array(values)
                ax.plot(times, data.mean(axis=0), color=COLORS[mode], label=mode)
                ax.fill_between(times, data.min(axis=0), data.max(axis=0), color=COLORS[mode], alpha=.12)
        ax.set(xlabel='Time since stream start (s)', ylabel='Completed training updates', title=label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(directory/'training_progress.png', dpi=170)
    fig.savefig(directory/'training_progress.pdf')
    plt.close(fig)
    if plan['phase'].startswith('quota'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
        for ax, x, y, ylabel, scale in [
                (axes[0], 'updates_by_stream_end', 'miss_rate', 'Deadline misses (%)', 100),
                (axes[1], 'updates_by_stream_end', 'on_time_accuracy', 'On-time correct (%)', 100),
                (axes[2], 'wall_sec', 'on_time_accuracy', 'On-time correct (%)', 100)]:
            for policy in ('uniform', 'reactive', 'phase', 'oracle', 'planned', 'recent'):
                selected = [r for r in summary if r['mode'] in (f'q8_{policy}', f'q9_{policy}')]
                selected.sort(key=lambda r: r[x])
                if selected:
                    color = COLORS[selected[0]['mode']]
                    ax.plot([r[x] for r in selected], [r[y]*scale for r in selected],
                            '-o', color=color, label=policy)
                    raw = [r for r in rows if r['mode'] in (f'q8_{policy}', f'q9_{policy}')]
                    ax.scatter([float(r[x]) for r in raw], [float(r[y])*scale for r in raw],
                               color=color, alpha=.25, s=14)
            for mode in ('warm_r4', 'warm_phase'):
                selected = next((r for r in summary if r['mode'] == mode), None)
                if selected:
                    ax.scatter(selected[x], selected[y]*scale, marker='*', s=140,
                               color=COLORS[mode], label=mode)
            ax.set_xlabel('Training updates by 30 s' if x == 'updates_by_stream_end' else 'Total wall time (s)')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.savefig(directory/'information_frontier.png', dpi=170)
        fig.savefig(directory/'information_frontier.pdf')
        plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
