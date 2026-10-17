#!/usr/bin/env python3
"""Audit released-label CL traces and compute paired quality/service comparisons."""
import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

FIELDS=['miss_rate','p95_ms','online_accuracy','on_time_accuracy','current_accuracy',
        'early_accuracy','historical_online_accuracy','final_accuracy','oldest_retention',
        'forgetting','train_updates','training_seconds','wall_seconds','online_end',
        'adaptation_seconds','adaptation_unreached','budget_overruns','gain_model_mae_pp','gain_model_bias_pp','boundary_seen_accuracy','boundary_current_accuracy','boundary_historical_accuracy']
COLORS=dict(r4='#64748b',phase='#9333ea',oracle_latency='#0284c7',causal_quality='#d97706',oracle_quality='#059669',uniform_work='#a16207',oracle_work='#db2777')


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def adaptation(records,seconds):
    result=[]
    for task in range(1,5):
        curve=[]
        for second in range(int(seconds)):
            group=[r for r in records if r['task']==task and r['current'] and
                   second<=r['arrival']-(task-1)*seconds<second+1]
            curve.append(float(np.mean([r['correct'] for r in group])) if group else 0)
        reached=next((i+1 for i in range(1,len(curve)) if curve[i-1]>=.6 and curve[i]>=.6),None)
        result.append(dict(task=task,reached=reached,restricted_seconds=seconds if reached is None else reached,curve=curve))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    directory=args.directory.resolve();plan=json.loads((directory/'plan.json').read_text())
    rows=list(csv.DictReader((directory/'results.csv').open()));assert len(rows)==len(plan['cells'])
    identities={};records={};curves={};adaptations={};evaluations={};calibration=[];boundary_quality=[];window_statistics=[]
    for row in rows:
        work=directory/'work'/row['work'];data=json.loads((work/'requests.json').read_text())
        trace=json.loads((work/'arrivals.json').read_text());train=json.loads((work/'training.json').read_text())
        probes=json.loads((work/'probes.json').read_text());split=json.loads((work/'split.json').read_text())
        evaluation=json.loads((work/'evaluations.json').read_text());key=(row['pattern'],row['policy'],row['seed'])
        if (work/'window_plans.json').exists():
            plans=json.loads((work/'window_plans.json').read_text())
            window_statistics.append(dict(pattern=row['pattern'],policy=row['policy'],seed=int(row['seed']),
                windows=len(plans),infeasible_windows=sum(not p['feasible'] for p in plans),
                planning_seconds=sum(p['planning_seconds'] for p in plans),
                expanded_states=sum(p['expanded'] for p in plans)))
        ident=(trace,split);assert identities.setdefault((row['pattern'],row['seed']),ident)==ident
        assert len(data)==len(trace)==plan['requests'] and [r['id'] for r in data]==list(range(len(data)))
        for r,t in zip(data,trace):
            assert all(r[k]==t[k] for k in ['id','sample','arrival','task','current'])
            assert r['task']==min(4,1+int(r['arrival']/plan['phase_seconds']))
            assert abs(r['latency']-(r['completed']-r['arrival']))<1e-8
            assert r['on_time']==(r['latency']<=plan['slo_ms']/1000)
        assert all(not(set(a)&set(b)) for a,b in zip(split['train'],split['validation']))
        assert len(set(sum(split['train'],[])))==5*2048
        assert len(set(sum(split['validation'],[])))==5*256
        for r in train:
            assert r['started']>=(r['task']-1)*plan['phase_seconds']+plan['label_delay']
            assert abs(r['duration']-(r['completed']-r['started']))<1e-8
        for r in probes:assert r['time']>=(r['task']-1)*plan['phase_seconds']+plan['label_delay']
        run_calibration=[]
        for before,after in zip(probes,probes[1:]):
            if before['task']!=after['task']:continue
            steps=after['step']-before['step']
            predicted=100*(min(1.0,before['accuracy']+max(0.0,before['gain'])*steps)-before['accuracy'])
            observed=100*(after['accuracy']-before['accuracy'])
            entry=dict(pattern=row['pattern'],policy=row['policy'],seed=int(row['seed']),
                task=before['task'],step=before['step'],predicted_gain_pp=predicted,
                observed_accuracy_gain_pp=observed,
                observed_probability_gain_pp=100*(after['probability']-before['probability']))
            calibration.append(entry);run_calibration.append(entry)
        row['gain_model_mae_pp']=float(np.mean([abs(r['predicted_gain_pp']-r['observed_accuracy_gain_pp']) for r in run_calibration])) if run_calibration else 0.0
        row['gain_model_bias_pp']=float(np.mean([r['predicted_gain_pp']-r['observed_accuracy_gain_pp'] for r in run_calibration])) if run_calibration else 0.0
        stage_quality=[]
        for e in evaluation[1:]:
            task=e['task'];parts=e['per_task']
            point=dict(pattern=row['pattern'],policy=row['policy'],seed=int(row['seed']),task=task,
                nominal_time=task*plan['phase_seconds'],snapshot_time=e['time'],
                seen_accuracy=float(np.mean(parts[:task+1])),current_accuracy=parts[task],
                historical_accuracy=float(np.mean(parts[:task])))
            stage_quality.append(point);boundary_quality.append(point)
        for field in ['seen_accuracy','current_accuracy','historical_accuracy']:
            row['boundary_'+field]=float(np.mean([p[field] for p in stage_quality]))
        assert len(train)==int(row['train_updates'])==sum(json.loads(row['updates']))
        latency=np.array([r['latency'] for r in data]);current=[r for r in data if r['current']]
        early=[r for r in current if r['arrival']%plan['phase_seconds']<2]
        old=[r for r in data if not r['current']]
        recompute=dict(miss_rate=float(np.mean(latency>plan['slo_ms']/1000)),p95_ms=float(np.percentile(latency,95)*1000),
            online_accuracy=float(np.mean([r['correct'] for r in data])),
            on_time_accuracy=float(np.mean([r['correct'] and r['on_time'] for r in data])),
            current_accuracy=float(np.mean([r['correct'] for r in current])),early_accuracy=float(np.mean([r['correct'] for r in early])),
            historical_online_accuracy=float(np.mean([r['correct'] for r in old])))
        for k,v in recompute.items():assert np.isclose(v,float(row[k])),(key,k)
        adapt=adaptation(data,plan['phase_seconds']);adaptations[key]=adapt
        row['adaptation_seconds']=float(np.mean([a['restricted_seconds'] for a in adapt]))
        row['adaptation_unreached']=sum(a['reached'] is None for a in adapt)
        row['budget_overruns']=sum(r['completed']>r['task']*plan['phase_seconds'] for r in train)
        records[key]=data;evaluations[key]=evaluation
        curves[key]=[float(np.mean([r['correct'] for r in current if second<=r['arrival']<second+1]))
                     for second in range(int(plan['horizon_seconds']))]
    write_csv(directory/'audited_results.csv',rows)
    write_csv(directory/'gain_calibration.csv',calibration)
    write_csv(directory/'boundary_learning.csv',boundary_quality)
    write_csv(directory/'window_statistics.csv',window_statistics)
    summary=[]
    for pattern in dict.fromkeys(r['pattern'] for r in rows):
        for policy in dict.fromkeys(r['policy'] for r in rows):
            group=[r for r in rows if r['pattern']==pattern and r['policy']==policy]
            if not group:continue
            item=dict(pattern=pattern,policy=policy,runs=len(group))
            for field in FIELDS:
                values=[float(r[field]) for r in group];item[field]=statistics.mean(values)
                item[field+'_std']=statistics.stdev(values) if len(values)>1 else 0
            summary.append(item)
    pairs=[]
    index={(r['pattern'],r['policy'],r['seed']):r for r in rows}
    for row in rows:
        for baseline in ['r4','phase','causal_quality','uniform_work']:
            bkey=(row['pattern'],baseline,row['seed'])
            if bkey not in index or row['policy']==baseline:continue
            base=index[bkey];key=(row['pattern'],row['policy'],row['seed'])
            pair=dict(pattern=row['pattern'],policy=row['policy'],baseline=baseline,seed=int(row['seed']))
            for field in ['miss_rate','online_accuracy','on_time_accuracy','early_accuracy','final_accuracy','oldest_retention','forgetting','boundary_seen_accuracy','boundary_current_accuracy','boundary_historical_accuracy']:
                pair[field+'_delta_pp']=100*(float(row[field])-float(base[field]))
            for field in ['train_updates','wall_seconds','adaptation_seconds']:
                pair[field+'_delta']=float(row[field])-float(base[field])
            a,b=records[key],records[bkey]
            hybrid=np.mean([r['on_time'] and s['correct'] for r,s in zip(a,b)])
            pair['timing_component_pp']=100*(hybrid-float(base['on_time_accuracy']))
            pair['prediction_component_pp']=100*(float(row['on_time_accuracy'])-hybrid)
            pairs.append(pair)
    write_csv(directory/'summary.csv',summary);write_csv(directory/'paired_comparisons.csv',pairs)
    analysis=dict(runs=len(rows),requests=sum(int(r['requests']) for r in rows),summary=summary,pairs=pairs,
                  adaptation={str(k):v for k,v in adaptations.items()})
    (directory/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    patterns=list(dict.fromkeys(r['pattern'] for r in rows))
    fig,axes=plt.subplots(len(patterns),2,figsize=(12,4.3*len(patterns)),squeeze=False,constrained_layout=True)
    for i,pattern in enumerate(patterns):
        for policy,color in COLORS.items():
            group=[r for r in summary if r['pattern']==pattern and r['policy']==policy]
            if not group:continue
            r=group[0]
            for ax,field,label in [(axes[i,0],'online_accuracy','Online accuracy (%)'),(axes[i,1],'final_accuracy','Final full-test accuracy (%)')]:
                ax.scatter(100*r['miss_rate'],100*r[field],s=70,color=color,label=policy)
                for source in [s for s in rows if s['pattern']==pattern and s['policy']==policy]:
                    ax.scatter(100*float(source['miss_rate']),100*float(source[field]),color=color,alpha=.3,s=25)
                ax.set_xlabel('Deadline misses (%)');ax.set_ylabel(label);ax.set_title(pattern);ax.grid(alpha=.2)
        axes[i,0].legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(directory/f'quality_service.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(len(patterns),1,figsize=(13,3.7*len(patterns)),squeeze=False,constrained_layout=True)
    for i,pattern in enumerate(patterns):
        ax=axes[i,0]
        for policy,color in COLORS.items():
            selected=[v for k,v in curves.items() if k[:2]==(pattern,policy)]
            if not selected:continue
            values=100*np.array(selected);mean=values.mean(0);t=np.arange(len(mean))+.5
            ax.plot(t,mean,label=policy,color=color);ax.fill_between(t,values.min(0),values.max(0),color=color,alpha=.1)
        for boundary in range(0,int(plan['horizon_seconds']),int(plan['phase_seconds'])):
            ax.axvline(boundary,color='gray',ls='--',lw=.6);ax.axvspan(boundary,boundary+plan['label_delay'],color='gray',alpha=.12)
        ax.set(xlabel='Time since first class change (s)',ylabel='Current-task accuracy (%)',title=pattern,ylim=(0,100));ax.legend(fontsize=8,ncol=5);ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(directory/f'adaptation.{ext}',dpi=160)
    plt.close(fig)
    print(json.dumps(dict(runs=len(rows),summary=summary),indent=2))

if __name__=='__main__':main()
