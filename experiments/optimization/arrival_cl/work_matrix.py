#!/usr/bin/env python3
"""Six independent work-matched runs using the primary stage's checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from environment.experiment_runtime import experiment_env,record_environment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--main',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();main=args.main.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise ValueError('Use a new output directory')
    record_environment(out,args.gpu)
    assert json.loads((main/'execution_environment.json').read_text())==json.loads((out/'execution_environment.json').read_text())
    plan=json.loads((main/'plan.json').read_text());plan['phase']='work'
    plan['rate']=10;plan['stage_updates']=114;plan['objective']='expected timely correct count; same cumulative work release'
    plan['cells']=[('bursty',p,seed) for i,seed in enumerate([211,307,419])
        for p in (['uniform_work','oracle_work'] if i%2==0 else ['oracle_work','uniform_work'])]
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    shutil.copytree(ROOT/'experiments/optimization/arrival_cl',out/'source',ignore=shutil.ignore_patterns('__pycache__','thor-20261017'))
    for cmd,name in [(['nvidia-smi'],'gpu_before.txt'),(['nvpmodel','-q'],'power_mode.txt')]:
        (out/name).write_text(subprocess.check_output(cmd,text=True))
    rows=[]
    for pattern,policy,seed in plan['cells']:
        name=f'{pattern}_{policy}_seed{seed}';work=out/'work'/name;work.mkdir(parents=True)
        command=[sys.executable,str(ROOT/'experiments/optimization/arrival_cl/work_control.py'),
            '--checkpoint',str(main/'checkpoints'/f'seed{seed}.pt'),'--seed',str(seed),'--policy',policy]
        began=time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            log.write(json.dumps(dict(command=command,timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))+'\n');log.flush()
            subprocess.run(command,cwd=work,env=experiment_env(args.gpu,seed),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        row=json.loads((work/'summary.json').read_text());row['wall_seconds']=time.perf_counter()-began;row['work']=name
        for field in ['updates','final_per_task']:row[field]=json.dumps(row[field])
        rows.append(row)
        with (out/'results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print(json.dumps(row),flush=True)
    (out/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))

if __name__=='__main__':main()
