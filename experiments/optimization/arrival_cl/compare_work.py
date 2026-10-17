#!/usr/bin/env python3
"""Cross-stage paired audit for the independent common-work experiment."""
import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.optimization.arrival_cl.analyze import write_csv


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--main',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    roots=[args.main.resolve(),args.work.resolve()]
    assert json.loads((roots[0]/'execution_environment.json').read_text())==json.loads((roots[1]/'execution_environment.json').read_text())
    assert (roots[0]/'power_mode.txt').read_text()==(roots[1]/'power_mode.txt').read_text()
    indexed={};traces={};data={};training={};locations={};summary=[]
    for root in roots:
        summary.extend(csv.DictReader((root/'summary.csv').open()))
        for row in csv.DictReader((root/'audited_results.csv').open()):
            key=(row['pattern'],row['policy'],row['seed']);work=root/'work'/row['work']
            identity=(json.loads((work/'arrivals.json').read_text()),json.loads((work/'split.json').read_text()))
            assert traces.setdefault((row['pattern'],row['seed']),identity)==identity
            indexed[key]=row;locations[key]=work
            data[key]=json.loads((work/'requests.json').read_text())
            training[key]=json.loads((work/'training.json').read_text())
    pairs=[];matches=[]
    for seed in ['211','307','419']:
        key=('bursty','oracle_work',seed);row=indexed[key]
        for baseline in ['uniform_work','r4','phase','causal_quality','oracle_quality']:
            bkey=('bursty',baseline,seed);base=indexed[bkey]
            pair=dict(seed=int(seed),policy='oracle_work',baseline=baseline)
            for field in ['miss_rate','online_accuracy','on_time_accuracy','current_accuracy','early_accuracy','final_accuracy','oldest_retention','forgetting','boundary_seen_accuracy','boundary_current_accuracy','boundary_historical_accuracy']:
                pair[field+'_delta_pp']=100*(float(row[field])-float(base[field]))
            pair['train_updates_delta']=int(row['train_updates'])-int(base['train_updates'])
            pair['adaptation_seconds_delta']=float(row['adaptation_seconds'])-float(base['adaptation_seconds'])
            a,b=data[key],data[bkey]
            hybrid=np.mean([r['on_time'] and s['correct'] for r,s in zip(a,b)])
            pair['timing_component_pp']=100*(hybrid-float(base['on_time_accuracy']))
            pair['prediction_component_pp']=100*(float(row['on_time_accuracy'])-hybrid)
            diffs=[]
            for task in range(1,5):
                aa=[r['completed'] for r in training[key] if r['task']==task]
                bb=[r['completed'] for r in training[bkey] if r['task']==task]
                diffs.extend(abs(bisect_right(aa,(task-1)*12+t)-bisect_right(bb,(task-1)*12+t)) for t in range(1,13))
            pair['max_boundary_step_difference']=max(diffs)
            shifts=[]
            for task in range(1,5):
                aa=[r['completed'] for r in training[key] if r['task']==task]
                bb=[r['completed'] for r in training[bkey] if r['task']==task]
                shifts.extend(x-y for x,y in zip(aa,bb))
            pair['mean_completion_shift_ms']=1000*float(np.mean(shifts))
            pair['max_completion_delay_ms']=1000*max(shifts)
            pair['fraction_updates_earlier']=float(np.mean(np.array(shifts)<0))
            pairs.append(pair)
        bkey=('bursty','uniform_work',seed)
        counts=json.loads(row['updates']);basecounts=json.loads(indexed[bkey]['updates'])
        aa=torch.load(locations[key]/'horizon_model.pt',map_location='cpu',weights_only=True)['model']
        bb=torch.load(locations[bkey]/'horizon_model.pt',map_location='cpu',weights_only=True)['model']
        equal=all(torch.equal(v,bb[k]) for k,v in aa.items())
        if counts==basecounts:assert equal
        matches.append(dict(seed=int(seed),oracle_updates=counts,uniform_updates=basecounts,
                            equal_stage_counts=counts==basecounts,equal_final_model=equal))
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_comparisons.csv',pairs)
    (out/'analysis.json').write_text(json.dumps(dict(pairs=pairs,work_matches=matches),indent=2)+'\n')
    print(json.dumps(dict(work_matches=matches,pairs=pairs),indent=2))

if __name__=='__main__':main()
