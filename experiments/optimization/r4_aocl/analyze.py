#!/usr/bin/env python3
"""Summarize paired runs without mixing native and fixed-workload phases."""
import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = ['wall_sec', 'qps', 'final_accuracy', 'eval_cycles', 'eval_samples',
           'eval_unique_versions', 'train_updates', 'train_seconds', 'batch_p95_ms', 'batch_p99_ms', 'batch_miss_33ms',
           'mean_cycle_sec', 'max_cycle_sec', 'service_samples']


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(directory, output):
    rows = list(csv.DictReader((directory / 'results.csv').open()))
    if not rows or any(row['valid'] != 'True' for row in rows):
        raise ValueError('The matrix contains incomplete or invalid cells')
    grouped = {}
    for row in rows:
        row['eval_unique_versions'] = len(set(row['eval_versions'].split(',')))
        grouped.setdefault((row['phase'], row['mode']), []).append(row)
    summaries = []
    for (phase, mode), group in grouped.items():
        record = dict(phase=phase, mode=mode, runs=len(group))
        for key in METRICS:
            values = [float(row[key]) for row in group]
            record[key] = statistics.mean(values)
            record[key + '_std'] = statistics.stdev(values) if len(values) > 1 else ''
        summaries.append(record)
    paired = []
    for row in rows:
        if row['mode'] != 'r4_cooperative':
            continue
        for baseline in rows:
            if baseline['mode'] == 'r4_cooperative' or (baseline['phase'], baseline['seed']) != (row['phase'], row['seed']):
                continue
            paired.append(dict(phase=row['phase'], seed=row['seed'],
                candidate='r4_cooperative', baseline=baseline['mode'],
                wall_change_pct=100 * (float(row['wall_sec']) / float(baseline['wall_sec']) - 1),
                qps_change_pct=100 * (float(row['qps']) / float(baseline['qps']) - 1),
                accuracy_change_pp=100 * (float(row['final_accuracy']) - float(baseline['final_accuracy'])),
                eval_cycles_change=int(row['eval_cycles']) - int(baseline['eval_cycles']),
                batch_p95_change_pct=100 * (float(row['batch_p95_ms']) / float(baseline['batch_p95_ms']) - 1),
                cycle_duration_change_pct=100 * (float(row['mean_cycle_sec']) / float(baseline['mean_cycle_sec']) - 1)))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'paired_comparisons.csv', paired)
    for row in summaries:
        print(json.dumps({key: row[key] for key in ['phase', 'mode', 'runs', 'wall_sec',
            'qps', 'final_accuracy', 'eval_cycles', 'train_updates', 'batch_p95_ms',
            'batch_p99_ms', 'mean_cycle_sec']}))
    return summaries, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.directory, args.output or args.directory)


if __name__ == '__main__':
    main()
