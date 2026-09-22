"""Real GPU execution of a causal open-loop arrival trace, without input caching."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from src.schedulers.boundary_scheduler import evaluation_state
from src.schedulers.queue_control import QueueController
from src.utils.logging_utils import log_info


def make_trace(seed, count, duration, pattern, dataset_size):
    if not 0 < count <= dataset_size or duration <= 0:
        raise ValueError('request count must fit unique test samples; duration must be positive')
    rng = np.random.default_rng(seed+8128)
    if pattern == 'steady':
        releases = np.sort(rng.uniform(0,duration,count))
    elif pattern == 'bursty':
        # 80% of arrivals in each second's 200 ms burst, no oracle at policy side.
        slots = rng.integers(0,max(1,int(duration)),count)
        burst = rng.random(count) < .8
        offset = np.where(burst,rng.uniform(0,.2,count),rng.uniform(.2,1,count))
        releases = np.sort(np.minimum(slots+offset,np.nextafter(duration,0)))
    else:
        raise ValueError('unknown arrival pattern')
    ids = rng.permutation(dataset_size)[:count]
    return [{'arrival':float(t),'sample':int(i)} for t,i in zip(releases,ids)]


def request_metrics(records):
    """Request goodput ends at the last response, not at later training completion."""
    latency = np.array([r['latency'] for r in records])*1000
    elapsed = max(r['completed'] for r in records)
    on_time = sum(r['on_time'] for r in records)
    return {'miss_rate':1-on_time/len(records),
            'p50_ms':float(np.percentile(latency,50)),
            'p95_ms':float(np.percentile(latency,95)),
            'p99_ms':float(np.percentile(latency,99)),
            'goodput':on_time/elapsed,'stream_seconds':elapsed}


class StreamSchedulerPlugin:
    def __init__(self, args, test_stream):
        self.args = args
        self.dataset = ConcatDataset([exp.dataset for exp in test_stream])
        self.trace = make_trace(args.seed or 0,args.stream_requests,args.stream_duration,
                                args.stream_pattern,len(self.dataset))
        self.controller = QueueController(args.global_scheduler_mode.removeprefix('stream_'),
                                          args.stream_slo_ms/1000)
        self.origin = None
        self.next_arrival = 0
        self.pending = []
        self.records = []
        self.steps = 0
        self.previous_train = None
        self.control_seconds = 0.0
        self.decisions = 0
        self.max_train_pause = 0.0
        self.last_train_end = None
        self.first = time.perf_counter()

    def _observe_train(self, now):
        if self.previous_train is not None:
            started,features,base = self.previous_train
            if self.steps > 5:  # cold CUDA startup does not seed the service predictor
                tick=time.perf_counter()
                self.controller.observe('train',1,max(now-started,1e-9),features,base)
                self.control_seconds+=time.perf_counter()-tick
            self.previous_train=None

    def _admit(self, now):
        if self.origin is None:
            return
        elapsed=now-self.origin
        while self.next_arrival<len(self.trace) and self.trace[self.next_arrival]['arrival']<=elapsed:
            self.pending.append(self.next_arrival)
            self.next_arrival+=1

    def before_training_iteration(self, strategy, **kwargs):
        now=time.perf_counter()
        self._observe_train(now)
        if self.origin is None and self.steps>=20:
            self.origin=now
        if self.origin is not None:
            self._drive(strategy,allow_train=True)
        now=time.perf_counter()
        if self.last_train_end is not None:
            self.max_train_pause=max(self.max_train_pause,now-self.last_train_end)
        tick=time.perf_counter()
        self.previous_train=(now,self.controller.features('train',1,len(self.pending)),
                             self.controller.costs.mean('train',1))
        self.control_seconds+=time.perf_counter()-tick

    def after_training_iteration(self,strategy,**kwargs):
        if strategy.device.type == 'cuda':
            torch.cuda.synchronize(strategy.device)
        self.steps+=1
        self.last_train_end=time.perf_counter()

    def after_training_exp(self,strategy,**kwargs):
        self._observe_train(time.perf_counter())
        if self.controller.policy=='boundary' and self.origin is not None:
            self._drive(strategy,allow_train=False)

    def _drive(self,strategy,allow_train):
        while True:
            now=time.perf_counter()
            self._admit(now)
            if not self.pending:
                return
            tick=time.perf_counter()
            arrivals=[self.trace[i]['arrival'] for i in self.pending]
            b=self.controller.choose(now-self.origin,arrivals,allow_train)
            self.control_seconds+=time.perf_counter()-tick
            self.decisions+=1
            if b==0:
                return
            self._infer(strategy,b)

    def _infer(self,strategy,b):
        selected=self.pending[:b]
        tick=time.perf_counter()
        features=self.controller.features('infer',b,len(self.pending))
        base=self.controller.costs.mean('infer',b)
        self.control_seconds+=time.perf_counter()-tick
        started=time.perf_counter()
        with evaluation_state(strategy.model),torch.no_grad():
            strategy.model.eval()
            items=[self.dataset[self.trace[i]['sample']] for i in selected]
            x=torch.stack([item[0] for item in items]).to(strategy.device)
            y=torch.as_tensor([int(item[1]) for item in items],device=strategy.device)
            predicted=strategy.model(x).argmax(dim=1)
            correct=(predicted==y).cpu().tolist()  # synchronizes completed results
        ended=time.perf_counter()
        tick=time.perf_counter()
        self.controller.observe('infer',b,ended-started,features,base)
        self.control_seconds+=time.perf_counter()-tick
        for index,ok in zip(selected,correct):
            request=self.trace[index]
            latency=ended-self.origin-request['arrival']
            self.records.append({'id':index,'sample':request['sample'],
                'arrival':request['arrival'],'completed':ended-self.origin,
                'latency':latency,'on_time':latency<=self.controller.slo,
                'correct':bool(ok),'train_updates':self.steps,'batch':b})
        del self.pending[:b]

    def finish(self,strategy):
        self._observe_train(time.perf_counter())
        train_finished=time.perf_counter()
        if self.origin is None:
            self.origin=train_finished
        while self.next_arrival<len(self.trace) or self.pending:
            self._drive(strategy,allow_train=False)
            if not self.pending and self.next_arrival<len(self.trace):
                wait=self.origin+self.trace[self.next_arrival]['arrival']-time.perf_counter()
                if wait>0:
                    time.sleep(min(wait,.01))
        assert len(self.records)==len(self.trace)
        assert len({r['sample'] for r in self.records})==len(self.trace)
        summary={'requests':len(self.records),'unique_samples':len(self.trace),'dropped':0,
                 **request_metrics(self.records),
                 'train_completion_seconds':train_finished-self.first,
                 'online_accuracy':sum(r['correct'] for r in self.records)/len(self.records),
                 'max_train_pause_seconds':self.max_train_pause,'train_updates':self.steps,
                 'controller_seconds':self.control_seconds,'decisions':self.decisions,
                 'guard_overrides':self.controller.guard_overrides,
                 'neural_parameters':65 if self.controller.net is not None else 0,
                 'neural_updates':self.controller.net.updates if self.controller.net is not None else 0}
        Path('stream_metrics.json').write_text(json.dumps(summary,indent=2))
        Path('stream_requests.json').write_text(json.dumps(self.records))
        Path('stream_arrivals.json').write_text(json.dumps(self.trace))
        log_info('[StreamMetrics] '+json.dumps(summary))
        return summary
