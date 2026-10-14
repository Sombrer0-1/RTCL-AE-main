#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import statistics as st
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
HERE = parser.parse_args().directory
rows = list(csv.DictReader((HERE/'results.csv').open()))
groups = {}
for row in rows:
    if row['valid'] != 'True':
        raise ValueError(f"Invalid run: {row['log']}")
    groups.setdefault(row['mode'], []).append(row)
by_seed = {}
for row in rows:
    by_seed.setdefault(row['seed'], []).append(row)
for seed, runs in by_seed.items():
    assert len({r['model_hash'] for r in runs}) == 1 and runs[0]['model_hash'], seed
summary=[]
for mode, runs in groups.items():
    out={'mode':mode,'n':len(runs)}
    for key in ('wall_sec','qps','train_seconds','final_accuracy','eval_cycles','max_cycle_p99_ms'):
        values=[float(r[key]) for r in runs]
        out[key+'_mean']=st.mean(values)
        out[key+'_sd']=st.stdev(values) if len(values)>1 else 0
    out['wall_cv_pct']=100*out['wall_sec_sd']/out['wall_sec_mean']
    summary.append(out)
with (HERE/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
paired=[]
for seed,runs in by_seed.items():
    modes={r['mode']:r for r in runs}
    if 'boundary_cached' not in modes: continue
    candidate=modes['boundary_cached']
    for baseline in ('fully_parallel','freshness_adaptive','boundary','freshness_cached'):
        if baseline in modes:
            paired.append({'seed':seed,'baseline':baseline,'speedup':float(modes[baseline]['wall_sec'])/float(candidate['wall_sec'])})
(HERE/'paired_speedups.json').write_text(json.dumps(paired,indent=2))
fig, axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
labels=list(groups)
for ax,metric,title in zip(axes.flat,['wall_sec','train_seconds','eval_cycles','final_accuracy'],
                          ['End-to-end time (s)','Training calls only (s)','Full-test evaluations','Final test accuracy']):
    vals=[[float(r[metric]) for r in groups[m]] for m in labels]
    ax.bar(range(len(labels)),[st.mean(v) for v in vals],yerr=[st.stdev(v) if len(v)>1 else 0 for v in vals],capsize=4)
    for i,v in enumerate(vals): ax.scatter([i]*len(v),v,color='black',s=20,zorder=3)
    ax.set_xticks(range(len(labels)),labels,rotation=15,fontsize=8)
    ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.suptitle('Boundary scheduling and bounded evaluation cache\nSplitCIFAR10 / ResNet-20 / Replay / train batch 16; mean ± sample SD')
fig.savefig(HERE/'comparison.png',dpi=180)
fig.savefig(HERE/'comparison.pdf')
print(json.dumps(summary,indent=2))
print(json.dumps(paired,indent=2))
