#!/usr/bin/env python3
"""Create an exportable figure from the archived raw-run tables."""
import csv
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = []
for folder, batch in [(HERE, 64), (HERE/'adaptive64', 64), (HERE/'batch16', 16)]:
    if not (folder/'results.csv').exists():
        continue
    for row in csv.DictReader((folder/'results.csv').open()):
        if row['valid'] == 'True':
            rows.append(dict(row, training_bs=batch))
groups = {}
for row in rows:
    groups.setdefault((row['mode'], row['training_bs']), []).append(row)
labels = [f'{mode}\ntrain B={batch}' for mode, batch in groups]
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
for ax, metric, title in zip(axes.flat,
        ['wall_sec', 'final_accuracy', 'eval_samples', 'max_cycle_p99_ms'],
        ['End-to-end time (s), lower is better', 'Final full-test accuracy',
         'Evaluated samples (includes repeated full-test passes)', 'Maximum per-cycle batch P99 (ms)']):
    values = [[float(r[metric]) for r in runs] for runs in groups.values()]
    means = [st.mean(v) for v in values]
    sd = [st.stdev(v) if len(v)>1 else 0 for v in values]
    ax.bar(range(len(labels)), means, yerr=sd, capsize=4, color=['#7299cf','#ce9575','#6bad8a','#9c85c0','#7299cf','#9c85c0'][:len(labels)])
    for i, vals in enumerate(values):
        ax.scatter([i]*len(vals), vals, color='black', s=18, zorder=3)
    ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=15)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=.2)
fig.suptitle('Reduced scheduler experiments: 3 paired seeds; bars = mean ± sample SD\nRTX 5090 / SplitCIFAR10 / ResNet-20 / Replay', fontsize=14)
fig.savefig(HERE/'comparison.png', dpi=180)
fig.savefig(HERE/'comparison.pdf')
