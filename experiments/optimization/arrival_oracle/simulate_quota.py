#!/usr/bin/env python3
"""Timing-only counterfactuals using measured GPU compute, host work and arrivals.

The executor reuses measured costs as a deterministic plant. Controllers see
only completed costs, and only the exact-arrival oracle sees future timestamps.
No classification outcomes are simulated.
"""
import argparse
from bisect import bisect_right
from collections import deque
import csv
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.schedulers.queue_control import CostModel
from src.schedulers.train_start import start_wait, training_quota, uniform_start_wait


def profile(work):
    training = json.loads((work/'stream_training.json').read_text())
    requests = json.loads((work/'stream_requests.json').read_text())
    trace = json.loads((work/'stream_arrivals.json').read_text())
    services = {r['service_started']: (r['completed'], r['batch']) for r in requests}
    services = sorted((a, b, n) for a, (b, n) in services.items())
    prep = []
    for previous, current in zip(training, training[1:]):
        if previous['completed'] is None or current['started'] is None:
            prep.append(0.0)
            continue
        a, b = previous['completed'], current['started']
        inference = sum(max(0, min(b, end)-max(a, start)) for start, end, n in services)
        prep.append(max(0, b-a-inference))
    # prep[i] is host work before training update i+1.
    prep.insert(0, 0.0)
    observed = services[5:]
    design = np.array([[1, n] for a, b, n in observed])
    costs = np.array([b-a for a, b, n in observed])
    intercept, slope = np.linalg.lstsq(design, costs, rcond=None)[0]
    return dict(compute=[r['duration'] for r in training], prep=prep, trace=trace,
                infer_intercept=max(.0001, float(intercept)), infer_slope=max(0.0, float(slope)))


def simulate(data, policy, rate, warmup=100, horizon=30, slo=.033, exact_costs=False):
    arrivals = [r['arrival'] for r in data['trace']]
    costs = CostModel()
    history = deque(data['compute'][max(0, warmup-64):warmup], maxlen=64)
    for i in range(max(0, warmup-64), warmup):
        costs.observe('train', 1, data['compute'][i]+data['prep'][i])
    now, next_arrival, steps = 0.0, 0, warmup
    pending, responses, training = [], [], []
    idle = quota_idle = 0.0

    def admit():
        nonlocal next_arrival
        end = bisect_right(arrivals, now)
        pending.extend(range(next_arrival, end))
        next_arrival = end

    def infer():
        nonlocal now
        selected = pending[:32]
        duration = data['infer_intercept']+data['infer_slope']*len(selected)
        started = now
        now += duration
        costs.observe('infer', len(selected), duration)
        for index in selected:
            responses.append(dict(id=index, arrival=arrivals[index], completed=now,
                latency=now-arrivals[index], on_time=now-arrivals[index] <= slo,
                train_updates=steps, batch=len(selected), service_started=started))
        del pending[:len(selected)]

    for index in range(warmup, len(data['compute'])):
        prep = data['prep'][index] if index > warmup else 0.0
        remaining = prep
        while remaining > 1e-12:
            chunk = min(.008, remaining)
            now += chunk
            remaining -= chunk
            admit()
            if pending:
                infer()
        if index > warmup:
            costs.observe('train', 1, data['compute'][index-1]+prep)
        while True:
            admit()
            if pending:
                infer()
                continue
            bound = data['compute'][index] if exact_costs else 1.1*float(np.quantile(history, .9))
            total = bound+data['prep'][index] if exact_costs else costs.mean('train', 1)
            inference = (lambda b: data['infer_intercept']+data['infer_slope']*b) if exact_costs else (lambda b: costs.bound('infer', b))
            release, slack = training_quota(now, steps-warmup, rate, horizon,
                                            bound, total)
            if release:
                if release < 1e-10:
                    now += release
                    continue
                delay = min(release, .01)
                quota_idle += delay
            elif next_arrival < len(arrivals):
                last = responses[-1]['completed'] if responses else float('-inf')
                if policy == 'uniform':
                    wait = uniform_start_wait(now, steps-warmup, rate, bound)
                else:
                    wait, risk = start_wait(policy, now, last, bound, slo, inference,
                        arrivals if policy == 'oracle' else (), next_arrival)
                delay = min(wait, slack, .01)
            else:
                delay = 0.0
            if delay <= 1e-12:
                break
            now += delay
            idle += delay
        started = now
        now += data['compute'][index]
        steps += 1
        history.append(data['compute'][index])
        training.append(dict(update=steps, started=started, completed=now))
    finished = now
    while pending or next_arrival < len(arrivals):
        admit()
        if pending:
            infer()
        else:
            now = max(now, arrivals[next_arrival])
    assert len(responses) == len(arrivals)
    assert [r['id'] for r in responses] == list(range(len(arrivals)))
    assert steps == len(data['compute'])
    latency = np.array([r['latency'] for r in responses])
    completed = [r['completed'] for r in training]
    boundaries = [warmup+bisect_right(completed, t) for t in range(1, int(horizon)+1)]
    summary = dict(policy=policy, quota=rate, requests=len(responses),
        miss_rate=float(np.mean(latency > slo)), p95_ms=float(np.percentile(latency, 95)*1000),
        p99_ms=float(np.percentile(latency, 99)*1000),
        updates_by_stream_end=warmup+bisect_right(completed, horizon),
        mean_request_train_updates=float(np.mean([r['train_updates'] for r in responses])),
        quota_end_shortfall=max(0, warmup+rate*horizon-boundaries[-1]) if rate else 0,
        max_boundary_shortfall=max(max(0, warmup+rate*t-v) for t, v in enumerate(boundaries, 1)) if rate else 0,
        intentional_idle_seconds=idle, quota_idle_seconds=quota_idle,
        train_completion_seconds=finished, train_updates=steps)
    return summary, responses, training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'output/thor-arrival-oracle-20261017/main')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(seeds=[211, 307, 419], rates=[0, 8, 9, 10],
        policies=['immediate', 'uniform', 'reactive', 'phase', 'oracle'], warmup_updates=100,
        input=str(args.input.resolve()), python=platform.python_version(), numpy=np.__version__,
        semantics='deterministic timing replay; no accuracy estimates or new GPU timings')
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    shutil.copy2(__file__, out/'simulate_quota.py')
    rows = []
    for seed in plan['seeds']:
        data = profile(args.input/'work'/f'bursty_r56_r4_seed{seed}')
        (out/f'profile_seed{seed}.json').write_text(json.dumps(data))
        for rate in plan['rates']:
            for policy in plan['policies']:
                if policy == 'uniform' and not rate:
                    continue
                result, responses, training = simulate(data, policy, rate)
                row = dict(seed=seed, **result)
                rows.append(row)
                name = f'q{rate}_{policy}_seed{seed}'
                (out/(name+'.json')).write_text(json.dumps(dict(summary=row,
                    responses=responses, training=training)))
                print(json.dumps(row), flush=True)
    with (out/'results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
