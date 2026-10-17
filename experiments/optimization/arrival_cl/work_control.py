#!/usr/bin/env python3
"""Independent common-work diagnostic; the primary timed executor is unchanged."""
import argparse
from pathlib import Path
from bisect import bisect_right
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.optimization.arrival_cl.run import Experiment
from experiments.optimization.arrival_cl.learning_window import LearningPlant,plan_learning_window,reference_deadlines


class WorkExperiment(Experiment):
    def __init__(self,args):
        super().__init__(args)
        self.work_plan={};self.work_window=None;self.window_records=[]

    def should_train(self):
        now=self.now();phase_start=(self.task-1)*self.args.phase_seconds
        release=phase_start+self.args.label_delay
        if now<release or self.pending:return False
        compute=1.1*float(np.quantile(self.costs,.9))
        if now+compute+.008>=self.task*self.args.phase_seconds:return False
        window=int(now);deadline=min(window+1,self.task*self.args.phase_seconds)
        target=min(114,int(round((deadline-release)*self.args.rate)))
        remaining=target-self.steps[self.task]
        if remaining<=0:return False
        prep=.008
        slack=max(0.0,deadline-now-compute-prep-(remaining-1)*(compute+prep))
        if self.args.policy=='uniform_work':
            due=release+(self.steps[self.task]+1)/self.args.rate-compute-prep
            return due<=now or slack==0
        key=(self.task,window)
        if key!=self.work_window:
            seen=set();services=[]
            for request in reversed(self.records):
                started=request['service_started']
                if started in seen:continue
                seen.add(started);services.append((request['batch'],request['completed']-started))
                if len(services)==64:break
            if len(set(b for b,t in services))>1:
                matrix=np.array([[1,b] for b,t in services]);cost=np.array([t for b,t in services])
                intercept,slope=np.linalg.lstsq(matrix,cost,rcond=None)[0]
                intercept=max(.001,float(intercept));slope=max(0.0,float(slope))
            else:intercept=self.infer_bound(1);slope=0.0
            plant=LearningPlant(dict(trace=self.trace[:bisect_right(self.times,deadline)],compute=[float(np.median(self.costs))],prep=[prep],
                infer_intercept=intercept,infer_slope=slope),self.args.slo_ms/1000,
                self.accuracy,max(0.0,self.gain))
            began=time.perf_counter()
            due=[release+(self.steps[self.task]+i+1)/self.args.rate for i in range(remaining)]
            due=reference_deadlines(plant,now+prep,self.cursor,due)
            result,feasible,expanded=plan_learning_window(plant,now+prep,self.cursor,window,remaining,completion_deadlines=due)
            self.work_plan={self.steps[self.task]+i:start for i,start in enumerate([t-prep for t in result[5]])}
            self.work_window=key
            self.window_records.append(dict(task=self.task,window=window,now=now,remaining=remaining,
                accuracy=self.accuracy,gain=self.gain,feasible=feasible,expanded=expanded,
                planning_seconds=time.perf_counter()-began,expected_errors=result[2],starts=list(result[5]),completion_deadlines=due,infer_intercept=intercept,infer_slope=slope))
        return self.work_plan.get(self.steps[self.task],now)<=self.now() or slack==0

    def run(self):
        super().run()
        import json
        Path('window_plans.json').write_text(json.dumps(self.window_records))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--policy',choices=['uniform_work','oracle_work'],required=True)
    parser.add_argument('--pattern',default='bursty')
    parser.add_argument('--phase-seconds',type=float,default=12)
    parser.add_argument('--label-delay',type=float,default=.5)
    parser.add_argument('--requests',type=int,default=9600)
    parser.add_argument('--slo-ms',type=float,default=33)
    parser.add_argument('--rate',type=int,default=10)
    WorkExperiment(parser.parse_args()).run()

if __name__=='__main__':main()
