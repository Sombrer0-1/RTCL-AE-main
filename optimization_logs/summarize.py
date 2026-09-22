#!/usr/bin/env python3
"""Aggregate successful full runs only; show means, sample SD and paired ratios."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=HERE)
    folder = parser.parse_args().directory
    rows = list(csv.DictReader((folder / 'results.csv').open()))
    groups = {}
    for row in rows:
        if row['valid'] == 'True':
            groups.setdefault(row['mode'], []).append(row)
    summary = []
    for mode, runs in groups.items():
        out = {'mode': mode, 'n': len(runs)}
        for key in ('wall_sec', 'qps', 'final_accuracy', 'eval_samples', 'max_cycle_p99_ms'):
            vals = [float(r[key]) for r in runs]
            out[key + '_mean'] = st.mean(vals)
            out[key + '_sd'] = st.stdev(vals) if len(vals) > 1 else 0
        out['wall_cv_pct'] = out['wall_sec_sd']/out['wall_sec_mean']*100
        summary.append(out)
    if summary:
        with (folder/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))
    by_seed = {(r['mode'], r['seed']): r for r in rows if r['valid'] == 'True'}
    for baseline in ('adaptocl', 'fully_parallel'):
        ratios = [float(by_seed[baseline, seed]['wall_sec'])/float(r['wall_sec'])
                  for (mode, seed), r in by_seed.items()
                  if mode == 'freshness' and (baseline,seed) in by_seed]
        print('freshness paired speedup vs', baseline, ratios)

if __name__ == '__main__':
    main()
