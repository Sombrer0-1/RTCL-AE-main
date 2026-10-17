#!/usr/bin/env python3
"""Timed class-incremental GPU experiment with delayed supervision.

Uses the project ResNet56, Adam and explicit 64-current/64-replay updates.
This isolated clock-driven executor is not the Avalanche experience driver.
"""
import argparse
from bisect import bisect_right
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
# Preserve the project's Avalanche import order.
from avalanche.benchmarks.classic import SplitCIFAR10
from src.models.model_init import initialize_model
from src.schedulers.boundary_scheduler import evaluation_state
from src.schedulers.stream_scheduler import make_trace
from src.schedulers.train_start import blocking_risk, start_wait
from src.utils.reproducibility import seed_worker
from experiments.optimization.arrival_cl.control import quality_train, causal_times
from experiments.optimization.arrival_cl.warmup import warm_compute

MEAN = torch.tensor([.4914,.4822,.4465]).view(1,3,1,1)
STD = torch.tensor([.2023,.1994,.2010]).view(1,3,1,1)


def load_data(seed):
    benchmark = SplitCIFAR10(n_experiences=5, fixed_class_order=list(range(10)),
                           shuffle=False, return_task_id=False)
    # Find the original torchvision leaf without assuming the user's data root.
    def leaf(node):
        from torchvision.datasets import CIFAR10
        if isinstance(node, CIFAR10):
            return node
        if hasattr(node, '_flat_data'):
            return leaf(node._flat_data)
        if hasattr(node, '_datasets'):
            for child in node._datasets:
                result = leaf(child)
                if result is not None:
                    return result
        if hasattr(node, 'dataset'):
            return leaf(node.dataset)
        return None
    train = leaf(benchmark.train_stream[0].dataset)
    test = leaf(benchmark.test_stream[0].dataset)
    assert train is not None and test is not None
    x = torch.from_numpy(train.data.copy()).permute(0,3,1,2).contiguous()
    y = torch.tensor(train.targets)
    tx = torch.from_numpy(test.data.copy()).permute(0,3,1,2).contiguous()
    ty = torch.tensor(test.targets)
    rng = np.random.default_rng(seed+4501)
    pools, probes = [], []
    for task in range(5):
        ids, val = [], []
        for label in [2*task, 2*task+1]:
            choices = rng.permutation(np.flatnonzero(y.numpy()==label))
            ids.extend(choices[:1024]); val.extend(choices[1024:1152])
        pools.append(np.array(ids)); probes.append(np.random.default_rng(seed+task+7919).permutation(val))
    eval_ids = np.concatenate([np.flatnonzero(ty.numpy()==c)[:200] for c in range(10)])
    return x,y,tx,ty,pools,probes,eval_ids


def inputs(raw, ids, seed=None):
    value = raw[torch.as_tensor(np.asarray(ids).copy())].float()/255
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        flip = torch.rand(len(ids), generator=generator)<.5
        value[flip] = value[flip].flip(-1)
        padded = torch.nn.functional.pad(value,(4,4,4,4))
        offset = torch.randint(0,9,(len(ids),2), generator=generator)
        value = torch.stack([padded[i,:,int(a):int(a)+32,int(b):int(b)+32]
                             for i,(a,b) in enumerate(offset)])
    return (value-MEAN)/STD


def evaluate(model, raw, labels, ids, device):
    correct = np.zeros(len(ids), dtype=bool)
    with evaluation_state(model), torch.no_grad():
        model.eval()
        for start in range(0,len(ids),128):
            selected = ids[start:start+128]
            pred = model(inputs(raw,selected).to(device)).argmax(1).cpu()
            correct[start:start+len(selected)] = (pred==labels[selected]).numpy()
    return dict(accuracy=float(correct.mean()),
        per_task=[float(correct[labels[ids].numpy()//2==t].mean())
                  if np.any(labels[ids].numpy()//2==t) else None for t in range(5)])


def build_model(device):
    return initialize_model(SimpleNamespace(model='resnet56'), 'split_cifar10').to(device)


def prepare(args):
    seed_worker(args.seed)
    torch.set_num_threads(1)
    device=torch.device('cuda:0')
    x,y,tx,ty,pools,probes,eval_ids=load_data(args.seed)
    model=build_model(device); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    began=time.perf_counter(); steps=0
    for epoch in range(5):
        order=np.random.default_rng(args.seed+epoch*1009).permutation(pools[0])
        for start in range(0,len(order),64):
            ids=order[start:start+64]
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(inputs(x,ids,args.seed+steps).to(device)),y[ids].to(device))
            loss.backward();optimizer.step(); steps+=1
    torch.cuda.synchronize()
    snapshot=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),seed=args.seed,
        initial=evaluate(model,tx,ty,eval_ids,device),steps=steps,
        pretrain_seconds=time.perf_counter()-began)
    torch.save(snapshot,args.checkpoint)
    print(json.dumps({k:v for k,v in snapshot.items() if k not in ['model','optimizer']}),flush=True)


class Experiment:
    def __init__(self,args):
        self.args=args; seed_worker(args.seed);torch.set_num_threads(1)
        self.device=torch.device('cuda:0')
        self.x,self.y,self.tx,self.ty,self.pools,self.probes,self.eval_ids=load_data(args.seed)
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert checkpoint['seed']==args.seed
        self.initial=checkpoint['initial']
        self.model=build_model(self.device);self.model.load_state_dict(checkpoint['model'])
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.duration=4*args.phase_seconds
        # Same timestamps and sample sequence for every policy of a given seed/load.
        base=make_trace(args.seed,args.requests,self.duration,args.pattern,max(args.requests,10000))
        rng=np.random.default_rng(args.seed+913)
        self.trace=[]
        for i,row in enumerate(base):
            task=min(4,1+int(row['arrival']/args.phase_seconds))
            current=rng.random()<.8
            choices=np.flatnonzero((self.ty.numpy()//2==task) if current else (self.ty.numpy()//2<task))
            self.trace.append(dict(id=i,arrival=row['arrival'],sample=int(rng.choice(choices)),
                                   task=task,current=bool(current)))
        self.times=[r['arrival'] for r in self.trace]
        self.cursor=0;self.pending=deque();self.records=[];self.training=[];self.controls=[]
        self.costs=deque(maxlen=16);self.infer_costs=deque(maxlen=64)
        self.task=1;self.steps=[0]*5;self.gain=0.0;self.accuracy=0.0;self.last_probability=None
        self.probe_records=[];self.snapshots=[];self.probed=False;self.preparing=False
        self.memory={}
        for task in range(1,5):
            previous=[]
            for old in range(task):
                previous.extend(np.random.default_rng(args.seed+old+131).permutation(self.pools[old])[:2000//task])
            self.memory[task]=np.array(previous)
        warm=warm_compute(self.model,self.optimizer,self.device)
        self.costs.extend(warm[-4:])
        Path('warmup.json').write_text(json.dumps(dict(durations=warm,history=warm[-4:],restored=True)))
        # Warm inference using the restored model without altering training state.
        with evaluation_state(self.model),torch.no_grad():
            self.model.eval();self.model(torch.zeros(32,3,32,32,device=self.device))
            self.model(torch.zeros(64,3,32,32,device=self.device))
            self.model(torch.zeros(128,3,32,32,device=self.device))
        torch.cuda.synchronize()
        Path('split.json').write_text(json.dumps(dict(train=[p.tolist() for p in self.pools],
            validation=[p.tolist() for p in self.probes],eval_ids=self.eval_ids.tolist(),
            label_release=[(t-1)*args.phase_seconds+args.label_delay for t in range(1,5)])))
        self.origin=time.perf_counter()

    def now(self): return time.perf_counter()-self.origin

    def admit(self):
        now=self.now()
        while self.cursor<len(self.trace) and self.times[self.cursor]<=now:
            self.pending.append(self.cursor);self.cursor+=1

    def infer_bound(self,b):
        if not self.infer_costs: return .006*(1+b/32)
        # Causal per-batch bound; no future measured durations.
        return float(np.quantile(self.infer_costs,.8))

    def infer(self):
        self.admit();selected=list(self.pending)[:32]
        if not selected:return
        started=self.now()
        ids=np.array([self.trace[i]['sample'] for i in selected])
        with evaluation_state(self.model),torch.no_grad():
            self.model.eval();pred=self.model(inputs(self.tx,ids).to(self.device)).argmax(1).cpu()
            correct=(pred==self.ty[ids]).tolist()
        end=self.now();self.infer_costs.append(end-started)
        for i,ok in zip(selected,correct):
            r=dict(self.trace[i],correct=bool(ok),completed=end,service_started=started,
                   latency=end-self.times[i],on_time=end-self.times[i]<=self.args.slo_ms/1000,
                   updates=list(self.steps),batch=len(selected))
            self.records.append(r);assert self.pending.popleft()==i

    def probe(self):
        # All labels come from released training-domain validation pools.
        task=self.task;assert self.now()>=(task-1)*self.args.phase_seconds+self.args.label_delay
        old=np.concatenate(self.probes[:task])
        past=np.random.default_rng(self.args.seed+task).choice(old,64,replace=False)
        ids=np.concatenate([self.probes[task][:64],past])
        scores=[];correct=[]
        with evaluation_state(self.model),torch.no_grad():
            self.model.eval()
            for start in [0,64]:
                part=ids[start:start+64];labels=self.y[part].to(self.device)
                output=self.model(inputs(self.x,part).to(self.device))
                scores.append(float(output.softmax(1).gather(1,labels[:,None]).mean().cpu()))
                correct.append(float((output.argmax(1)==labels).float().mean().cpu()))
                self.admit()
                if self.pending:self.infer()
        probability=.8*scores[0]+.2*scores[1]
        accuracy=.8*correct[0]+.2*correct[1]
        if self.last_probability is not None:
            gain=(probability-self.last_probability)/4
            self.gain=.5*self.gain+.5*max(-.05,min(.05,gain))
        self.last_probability=probability;self.accuracy=accuracy;self.probed=True
        self.probe_records.append(dict(time=self.now(),task=task,step=self.steps[task],
            probability=probability,accuracy=accuracy,gain=self.gain))

    def should_train(self):
        now=self.now();task=self.task;end=task*self.args.phase_seconds
        if now<(task-1)*self.args.phase_seconds+self.args.label_delay:return False
        compute=1.1*float(np.quantile(self.costs,.9)) if self.costs else .060
        if now+compute+.008>=end:return False
        policy=self.args.policy
        if policy=='r4':return not self.pending
        if policy=='phase':
            if self.pending:return False
            wait,_=start_wait('phase',now,0,compute,self.args.slo_ms/1000,self.infer_bound)
            return wait<=0
        if policy=='oracle_latency':
            if self.pending:return False
            risk=blocking_risk(self.times,self.cursor,now,compute,self.args.slo_ms/1000,self.infer_bound)
            return risk==0
        if self.pending:return False
        known=[]
        horizon=min(now+1.0,end)
        if policy=='oracle_quality':
            future=self.times[self.cursor:bisect_right(self.times,horizon)]
            times=known+future
        else:
            times=[t for t in causal_times(self.times[bisect_right(self.times,now-1,hi=self.cursor):self.cursor],known,now) if t<horizon]
        decision,a,b=quality_train(times,now,compute,self.infer_bound,
            self.args.slo_ms/1000,self.accuracy,max(0.0,self.gain))
        self.controls.append(dict(now=now,task=task,train=decision,score_train=a,score_wait=b,
            accuracy=self.accuracy,gain=self.gain,pending=len(known),compute=compute))
        return decision

    def train(self):
        task=self.task;step=self.steps[task]
        rng=np.random.default_rng(self.args.seed+task*1000003+step*97)
        ids=np.concatenate([rng.choice(self.pools[task],64),rng.choice(self.memory[task],64)])
        prepared=[]
        for start in range(0,128,8):
            prepared.append(inputs(self.x,ids[start:start+8],self.args.seed+task*1000003+step*1009+start))
            self.admit()
            if self.pending:self.infer()
        now=self.now()
        compute=1.1*float(np.quantile(self.costs,.9)) if self.costs else .060
        if now+compute>=task*self.args.phase_seconds:return
        self.model.train();started=self.now()
        self.optimizer.zero_grad(set_to_none=True)
        logits=self.model(torch.cat(prepared).to(self.device))
        loss=torch.nn.functional.cross_entropy(logits,self.y[ids].to(self.device))
        loss.backward();self.optimizer.step();torch.cuda.synchronize()
        ended=self.now();self.steps[task]+=1;self.costs.append(ended-started)
        self.training.append(dict(task=task,update=self.steps[task],started=started,completed=ended,
                                  duration=ended-started,loss=float(loss.detach().cpu())))
        if self.steps[task]%4==0:self.probe()

    def run(self):
        while self.now()<self.duration:
            task=min(4,1+int(self.now()/self.args.phase_seconds))
            if task!=self.task:
                self.snapshots.append((self.task,self.now(),{k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}))
                self.task=task;self.gain=0.0;self.last_probability=None;self.probed=False
            self.admit()
            release=(task-1)*self.args.phase_seconds+self.args.label_delay
            if not self.probed and self.now()>=release:self.probe()
            decision=self.should_train()
            if decision:self.train()
            elif self.pending:self.infer()
            else:time.sleep(.001)
        self.snapshots.append((self.task,self.now(),{k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}))
        online_end=self.now()
        while self.pending or self.cursor<len(self.trace):
            self.admit()
            if self.pending:self.infer()
            else:time.sleep(.001)
        end_state=self.snapshots[-1][2]
        torch.save(dict(model=end_state,steps=self.steps), 'horizon_model.pt')
        final=evaluate(self.model,self.tx,self.ty,np.arange(len(self.ty)),self.device)
        evaluations=[dict(task=0,time=0,**self.initial)]
        for task,when,state in self.snapshots:
            self.model.load_state_dict(state)
            evaluations.append(dict(task=task,time=when,**evaluate(self.model,self.tx,self.ty,self.eval_ids,self.device)))
        self.model.load_state_dict(end_state)
        for name,data in [('requests',self.records),('training',self.training),('controls',self.controls),
                          ('probes',self.probe_records),('evaluations',evaluations),('arrivals',self.trace)]:
            Path(name+'.json').write_text(json.dumps(data))
        latency=np.array([r['latency'] for r in self.records])
        current=[r for r in self.records if r['current']]
        early=[r for r in current if r['arrival']%self.args.phase_seconds<2]
        old=[r for r in self.records if not r['current']]
        last=evaluations[-1]['per_task']
        forgetting=float(np.mean([max(e['per_task'][t] for e in evaluations if e['task']>=t)-last[t] for t in range(4)]))
        summary=dict(policy=self.args.policy,seed=self.args.seed,pattern=self.args.pattern,
            online_budget=self.duration,online_end=online_end,requests=len(self.records),
            miss_rate=float(np.mean(latency>self.args.slo_ms/1000)),p95_ms=float(np.percentile(latency,95)*1000),
            online_accuracy=float(np.mean([r['correct'] for r in self.records])),
            on_time_accuracy=float(np.mean([r['correct'] and r['on_time'] for r in self.records])),
            current_accuracy=float(np.mean([r['correct'] for r in current])),
            early_accuracy=float(np.mean([r['correct'] for r in early])),
            historical_online_accuracy=float(np.mean([r['correct'] for r in old])),
            final_accuracy=final['accuracy'],final_per_task=final['per_task'],
            initial_task_accuracy=self.initial['per_task'][0],oldest_retention=last[0],
            forgetting=forgetting,updates=list(self.steps),train_updates=sum(self.steps),
            probe_calls=len(self.probe_records),training_seconds=sum(r['duration'] for r in self.training))
        Path('summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--policy',choices=['r4','phase','oracle_latency','causal_quality','oracle_quality'],default='r4')
    parser.add_argument('--pattern',choices=['steady','bursty'],default='bursty')
    parser.add_argument('--phase-seconds',type=float,default=12)
    parser.add_argument('--label-delay',type=float,default=.5)
    parser.add_argument('--requests',type=int,default=9600)
    parser.add_argument('--slo-ms',type=float,default=33)
    args=parser.parse_args()
    if args.prepare:prepare(args)
    else:Experiment(args).run()

if __name__=='__main__':main()
