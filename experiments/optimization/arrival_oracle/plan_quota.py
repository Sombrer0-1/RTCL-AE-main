#!/usr/bin/env python3
"""Full-window planning reference in an identical, known fixed-cost plant.

All comparisons know the same service costs. Arrival timestamps are the only
additional information available to the planner. State times are binned, so the
reported solution is feasible in the model, not a continuous-time optimum proof.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.optimization.arrival_oracle.simulate_quota import simulate
from src.schedulers.window_planner import FixedPlant, plan_window, replay_schedule



def planned(data, quota, horizon=30, resolution=.002):
    plant = FixedPlant(data)
    now, cursor, missed, latency, expanded = 0.0, 0, 0, 0.0, 0
    training, windows = [], []
    for window in range(horizon):
        result, feasible, nodes = plan_window(plant, now, cursor, window, quota, resolution)
        now, cursor, m, progress, l, starts = result
        missed += m
        latency += l
        expanded += nodes
        windows.append(dict(window=window, feasible=feasible, misses=m, ready=now, expanded=nodes))
        training.extend(dict(started=t, completed=t+plant.compute) for t in starts)
    now, cursor, m, l = plant.wait_to(now, cursor, data['trace'][-1]['arrival'])
    missed += m
    latency += l
    assert cursor == len(data['trace'])
    assert len(training) == horizon*quota
    audit_missed, audit_latency, audit_cursor, audit_completed = replay_schedule(
        plant, [t['started'] for t in training], horizon)
    assert audit_missed == missed and audit_cursor == cursor
    assert abs(audit_latency-latency) < 1e-7
    assert all(abs(t['completed']-end) < 1e-8 for t, end in zip(training, audit_completed))
    return dict(policy='window_oracle', quota=quota, miss_rate=missed/len(data['trace']),
        updates_by_stream_end=100+sum(t['completed'] <= horizon for t in training),
        infeasible_windows=sum(not w['feasible'] for w in windows),
        expanded_states=expanded, response_latency_sum=latency,
        resolution_ms=resolution*1000), training, windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'plan_quota.py')
    (out/'plan.json').write_text(json.dumps(dict(seeds=[211, 307, 419], quotas=[8, 9],
        resolution_ms=2, input=str(args.input.resolve()),
        cost_information='identical known constants for every controller',
        objective='min misses, then earlier training completion, then response latency',
        semantics='timing-only model; no predicted accuracy'), indent=2)+'\n')
    rows = []
    for seed in [211, 307, 419]:
        profile = json.loads((args.input/f'profile_seed{seed}.json').read_text())
        # Fixed constants remove runtime-cost privilege from information comparisons.
        compute = float(np.median(profile['compute'][100:]))
        prep = float(np.median(profile['prep'][100:]))
        data = dict(profile, compute=[compute]*790, prep=[prep]*790)
        (out/f'fixed_profile_seed{seed}.json').write_text(json.dumps(data))
        for quota in (8, 9):
            for policy in ('immediate', 'uniform', 'reactive', 'phase', 'oracle'):
                row, responses, training = simulate(data, policy, quota, exact_costs=True)
                rows.append(dict(seed=seed, policy=policy, quota=quota,
                    miss_rate=row['miss_rate'], updates_by_stream_end=row['updates_by_stream_end'],
                    infeasible_windows=None, expanded_states=0, response_latency_sum=sum(r['latency'] for r in responses),
                    resolution_ms=None, planning_seconds=0))
            tick = time.perf_counter()
            result, training, windows = planned(data, quota)
            row = dict(seed=seed, **result, planning_seconds=time.perf_counter()-tick)
            rows.append(row)
            (out/f'q{quota}_seed{seed}.json').write_text(json.dumps(dict(summary=row,
                training=training, windows=windows)))
            print(json.dumps(row), flush=True)
    with (out/'results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
