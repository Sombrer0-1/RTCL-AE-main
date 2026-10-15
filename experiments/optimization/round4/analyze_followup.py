"""Separate audit and visualization of the post-matrix combination experiment."""
import json
import matplotlib.pyplot as plt
from analyze import audit, HERE, CASES

rows=audit('followup',6)
summary=[]
fig,axes=plt.subplots(2,4,figsize=(13,6.5),constrained_layout=True)
modes=['baseline','certified','certified_slack']
metrics=[('wall_sec','Total wall time (s)',1),('p95_ms','Request P95 (ms)',1),
         ('p99_ms','Request P99 (ms)',1),('miss_rate','Deadline misses (%)',100)]
for ci,case in enumerate(CASES):
    selected=[next(r for r in rows if r['case']==case and r['mode']==mode) for mode in modes]
    for r in selected:
        record=dict(case=case,mode=r['mode'])
        for key in ['wall_sec','p95_ms','p99_ms','miss_rate','controller_seconds','final_accuracy','load_start','load_end']:
            record[key]=float(r[key])
        rec=json.loads((HERE/'followup'/r['log'].removesuffix('.log')/'stream_requests.json').read_text())
        record['inference_batches']=len({x['completed'] for x in rec})
        record['mean_batch']=6000/record['inference_batches']
        record['wall_ratio_vs_baseline']=float(r['wall_sec'])/float(selected[0]['wall_sec'])
        summary.append(record)
    for j,(key,label,mult) in enumerate(metrics):
        ax=axes[ci,j]
        ax.bar(range(3),[float(r[key])*mult for r in selected],color=['#64748b','#0284c7','#d97706'])
        ax.set_xticks(range(3),['FIFO','Cert. FIFO','Cert. Slack'],rotation=12)
        ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.2)
        if j==0:
            ax.set_title('ResNet20 / steady / 100 ms' if ci==0 else 'ResNet56 / bursty / 33 ms')
fig.suptitle('Separate combination experiment | fresh seed 149 | unchanged control parameters\nOne run per cell; do not pool with the main matrix')
fig.savefig(HERE/'followup_comparison.png',dpi=180)
fig.savefig(HERE/'followup_comparison.pdf')
(HERE/'followup_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
