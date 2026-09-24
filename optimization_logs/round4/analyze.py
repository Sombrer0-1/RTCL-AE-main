"""Audit every response and plot paired ablations, retaining host-load evidence."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
MODES=['baseline','certified','cooperative']
CASES=['longer','deeper_bursty']

def audit(phase, expected):
    rows=list(csv.DictReader((HERE/phase/'results.csv').open()))
    assert len(rows)==expected
    paired={}
    for row in rows:
        assert row['valid']=='True'
        folder=HERE/phase/row['log'].removesuffix('.log')
        trace=json.loads((folder/'stream_arrivals.json').read_text())
        rec=json.loads((folder/'stream_requests.json').read_text())
        slo=.033 if row['case']=='deeper_bursty' else .1
        assert len(rec)==6000 and len({r['sample'] for r in rec})==6000
        assert [r['id'] for r in rec]==list(range(6000))
        assert all(r['sample']==trace[r['id']]['sample'] and r['arrival']==trace[r['id']]['arrival']
                   and abs(r['latency']-(r['completed']-r['arrival']))<1e-7
                   and r['on_time']==(r['latency']<=slo) for r in rec)
        lat=np.array([r['latency'] for r in rec])
        assert np.isclose(np.mean(lat>slo),float(row['miss_rate']))
        for q in (50,95,99):
            assert np.isclose(np.percentile(lat,q)*1000,float(row[f'p{q}_ms']))
        assert np.isclose(sum(r['on_time'] for r in rec)/max(r['completed'] for r in rec),float(row['goodput']))
        assert int(row['train_updates'])==790*int(row['epochs'])
        pair=(row['case'],row['seed'])
        identity=(row['model_hash'],row['trace_hash'],row['final_accuracy'])
        assert identity==paired.setdefault(pair,identity)
        if row['mode'] in ('certified','cooperative'):
            assert int(row['certified_epochs'])==10*int(row['epochs']) and int(row['fallback_epochs'])==0
        if row['mode']=='cooperative':
            assert 0<int(row['service_requests'])<=6000
    return rows

def main():
    audit('pilot',3)
    rows=audit('main',12)
    summary=[]
    for case in CASES:
        for mode in MODES:
            selected=[r for r in rows if r['case']==case and r['mode']==mode]
            record=dict(case=case,mode=mode)
            for key in ['wall_sec','p95_ms','p99_ms','miss_rate','goodput','max_train_pause_seconds',
                        'controller_seconds','service_requests','service_seconds','check_seconds']:
                record[key]=float(np.mean([float(r[key]) for r in selected]))
            record['paired_wall_ratio_vs_baseline']=[]
            for r in selected:
                base=next(x for x in rows if x['case']==case and x['seed']==r['seed'] and x['mode']=='baseline')
                record['paired_wall_ratio_vs_baseline'].append(float(r['wall_sec'])/float(base['wall_sec']))
            summary.append(record)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(2,4,figsize=(13,6.5),constrained_layout=True)
    metrics=[('wall_sec','Total wall time (s)',1),('p95_ms','Request P95 (ms)',1),
             ('p99_ms','Request P99 (ms)',1),('miss_rate','Deadline misses (%)',100)]
    for ci,case in enumerate(CASES):
        for j,(metric,label,mult) in enumerate(metrics):
            ax=axes[ci,j]
            for mi,mode in enumerate(MODES):
                vals=[float(r[metric])*mult for r in sorted(rows,key=lambda r:r['seed']) if r['case']==case and r['mode']==mode]
                ax.bar(mi,np.mean(vals),width=.6,color=['#64748b','#0284c7','#059669'][mi],alpha=.6)
                for si,value in enumerate(vals):
                    ax.scatter(mi+(-.07 if si==0 else .07),value,marker='o' if si==0 else 'x',color='black',s=22,zorder=3)
            ax.set_xticks(range(3),['FIFO','Certificate','Cooperative'],rotation=12)
            ax.set_ylabel(label)
            ax.grid(axis='y',alpha=.2)
            if j==0:
                ax.set_title('ResNet20 / 2 epochs / steady' if ci==0 else 'ResNet56 / 1 epoch / bursty')
    fig.suptitle('Host preparation ablation | 6,000 unique uncached requests/run\nDots: seed 47; crosses: seed 83. Bars average run metrics; no significance claim.')
    fig.savefig(HERE/'comparison.png',dpi=180)
    fig.savefig(HERE/'comparison.pdf')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
