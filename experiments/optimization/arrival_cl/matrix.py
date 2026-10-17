#!/usr/bin/env python3
"""Serial GPU matrix; each child has an isolated working directory."""
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['pilot','main'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise ValueError('Use a new output directory')
    record_environment(out,args.gpu)
    policies=['r4','phase','oracle_latency','causal_quality','oracle_quality']
    seeds=[19] if args.phase=='pilot' else [211,307,419]
    cells=[]
    for index,seed in enumerate(seeds):
        order=policies[index:]+policies[:index]
        cells.extend(('bursty',p,seed) for p in order)
        if args.phase=='main':cells.extend(('steady',p,seed) for p in ['r4','oracle_quality'])
    plan=dict(phase=args.phase,cells=cells,phase_seconds=12,label_delay=.5,requests=9600,
              slo_ms=33,pretrain_epochs=5,train_per_task=2048,validation_per_task=256,
              replay_memory=2000,quality_probe_interval=4,horizon_seconds=48,
              predictor_horizon_seconds=1,cost_history=16,warmup_rounds=5,queue_service='FIFO for all policies')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for relative in ['experiments/optimization/arrival_cl','src/models/model_init.py',
                     'src/schedulers/boundary_scheduler.py','src/schedulers/train_start.py',
                     'src/schedulers/stream_scheduler.py','src/utils/reproducibility.py',
                     'environment/experiment_runtime.py']:
        source=ROOT/relative;target=out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
        else:shutil.copy2(source,target)
    for cmd,name in [(['nvidia-smi'],'gpu_before.txt'),(['nvpmodel','-q'],'power_mode.txt')]:
        (out/name).write_text(subprocess.check_output(cmd,text=True))
    rows=[];identities={};runner=ROOT/'experiments/optimization/arrival_cl/run.py'
    checkpoint_dir=out/'checkpoints';checkpoint_dir.mkdir()
    for seed in seeds:
        checkpoint=checkpoint_dir/f'seed{seed}.pt'
        command=[sys.executable,str(runner),'--seed',str(seed),'--checkpoint',str(checkpoint),'--prepare']
        with (out/f'prepare_seed{seed}.log').open('w') as log:
            subprocess.run(command,cwd=out,env=experiment_env(args.gpu,seed),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    for pattern,policy,seed in cells:
        name=f'{pattern}_{policy}_seed{seed}';work=out/'work'/name;work.mkdir(parents=True)
        command=[sys.executable,str(runner),'--seed',str(seed),'--checkpoint',str(checkpoint_dir/f'seed{seed}.pt'),
                 '--pattern',pattern,'--policy',policy]
        started=time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            log.write(json.dumps(dict(command=command,timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))+'\n');log.flush()
            subprocess.run(command,cwd=work,env=experiment_env(args.gpu,seed),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        row=json.loads((work/'summary.json').read_text());row['wall_seconds']=time.perf_counter()-started;row['work']=name
        trace=json.loads((work/'arrivals.json').read_text())
        assert identities.setdefault((pattern,seed),trace)==trace
        row['updates']=json.dumps(row['updates']);row['final_per_task']=json.dumps(row['final_per_task'])
        rows.append(row)
        with (out/'results.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps(row),flush=True)
    (out/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))

if __name__=='__main__':main()
