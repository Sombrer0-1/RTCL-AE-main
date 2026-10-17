"""Open-loop GPU requests with causal policies and an explicit arrival oracle."""
import json
import time
from collections import deque
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
        # 80% of arrivals in each second's 200 ms burst.
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
        self.preparation = getattr(args, 'host_preparation', 'baseline')
        self.prep_last_check = self.first
        self.prep_stats = dict(certified_epochs=0, fallback_epochs=0, service_batches=0,
                               service_requests=0, service_seconds=0.0, check_seconds=0.0)
        self.start_policy = getattr(args, 'stream_start_policy', 'immediate')
        self.start_budget = getattr(args, 'stream_start_max_wait_ms', 100.0) / 1000
        self.arrival_times = [request['arrival'] for request in self.trace]
        self.compute_costs = deque(maxlen=64)
        self.compute_started = None
        self.training_records = []
        self.start_records = []
        self.idle_seconds = 0.0
        self.train_quota = getattr(args, 'stream_train_quota', 0)
        self.warmup_updates = getattr(args, 'stream_warmup_updates', 20)
        self.origin_steps = self.warmup_updates
        self.quota_idle_seconds = 0.0
        self.window_plan = {}
        self.plan_window_index = None
        self.plan_records = []
        self.planning_seconds = 0.0

    def before_training_epoch(self, strategy, **kwargs):
        if self.preparation != 'baseline':
            from src.schedulers.host_preparation import install_preparation
            service = (lambda: self._preparation_service(strategy)) if self.preparation == 'cooperative' else None
            install_preparation(strategy.dataloader, service, self.prep_stats)

    def _preparation_service(self, strategy):
        started = time.perf_counter()
        if self.origin is None or started-self.prep_last_check < min(.008, self.controller.slo/4):
            self.prep_stats['check_seconds'] += time.perf_counter()-started
            return
        self.prep_last_check = started
        self._admit(started)
        self.prep_stats['check_seconds'] += time.perf_counter()-started
        if not self.pending:
            return
        tick = time.perf_counter()
        b = min(len(self.pending), self.controller.max_batch)
        self._infer(strategy, b)
        elapsed = time.perf_counter()-tick
        # Inference is already observed separately; don't charge it to train again.
        if self.previous_train is not None:
            t, features, base = self.previous_train
            self.previous_train = (t+elapsed, features, base)
        self.prep_stats['service_batches'] += 1
        self.prep_stats['service_requests'] += b
        self.prep_stats['service_seconds'] += elapsed
        self.prep_last_check = time.perf_counter()

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
        if self.origin is None and self.steps>=self.warmup_updates:
            self.origin=now
            self.origin_steps=self.steps
        if self.origin is not None:
            self._drive(strategy,allow_train=True)
        now=time.perf_counter()
        if self.last_train_end is not None:
            self.max_train_pause=max(self.max_train_pause,now-self.last_train_end)
        tick=time.perf_counter()
        self.previous_train=(now,self.controller.features('train',1,len(self.pending)),
                             self.controller.costs.mean('train',1))
        self.control_seconds+=time.perf_counter()-tick
        self.compute_started = time.perf_counter()

    def after_training_iteration(self,strategy,**kwargs):
        if strategy.device.type == 'cuda':
            torch.cuda.synchronize(strategy.device)
        self.steps+=1
        self.last_train_end=time.perf_counter()
        duration = self.last_train_end - self.compute_started
        self.compute_costs.append(duration)
        self.training_records.append(dict(update=self.steps, duration=duration,
            started=self.compute_started-self.origin if self.origin is not None else None,
            completed=self.last_train_end-self.origin if self.origin is not None else None))

    def after_training_exp(self,strategy,**kwargs):
        self._observe_train(time.perf_counter())
        if self.controller.policy=='boundary' and self.origin is not None:
            self._drive(strategy,allow_train=False)

    def _train_bound(self):
        duration = (float(np.quantile(self.compute_costs, .9)) if self.compute_costs
                    else self.controller.costs.mean('train', 1))
        return 1.1*duration

    def _drive(self,strategy,allow_train):
        idle_used = 0.0
        while True:
            now=time.perf_counter()
            self._admit(now)
            if not self.pending:
                quota_slack = float('inf')
                if allow_train and self.train_quota:
                    from src.schedulers.train_start import training_quota
                    tick = time.perf_counter()
                    train_cost = self._train_bound()
                    release, quota_slack = training_quota(now-self.origin,
                        self.steps-self.origin_steps, self.train_quota, self.args.stream_duration,
                        train_cost, self.controller.costs.mean('train', 1))
                    self.control_seconds += time.perf_counter()-tick
                    if release > 0:
                        tick = time.perf_counter()
                        time.sleep(min(release, .01))
                        slept = time.perf_counter()-tick
                        self.quota_idle_seconds += slept
                        self.idle_seconds += slept
                        continue
                if allow_train and self.start_policy != 'immediate' and self.next_arrival < len(self.trace):
                    from src.schedulers.train_start import start_wait
                    tick = time.perf_counter()
                    train_cost = self._train_bound()
                    elapsed = now-self.origin
                    last_response = self.records[-1]['completed'] if self.records else float('-inf')
                    # The reactive control receives no unreleased timestamps.
                    future = self.arrival_times if self.start_policy == 'oracle' else ()
                    if self.start_policy == 'uniform':
                        from src.schedulers.train_start import uniform_start_wait
                        wait = uniform_start_wait(elapsed, self.steps-self.origin_steps,
                                                  self.train_quota, train_cost)
                        risk = 0
                    elif self.start_policy == 'planned':
                        wait = self._planned_wait(elapsed)
                        risk = 0
                        self._admit(time.perf_counter())
                        if self.pending:
                            self.control_seconds += time.perf_counter()-tick
                            continue
                    else:
                        wait, risk = start_wait(self.start_policy, elapsed, last_response,
                            train_cost, self.controller.slo,
                            lambda b: self.controller.costs.bound('infer', b),
                            future, self.next_arrival, self.controller.max_batch)
                    remaining = max(0.0, self.start_budget-idle_used) if self.start_budget else float('inf')
                    delay = min(wait, remaining, quota_slack, .01)
                    self.start_records.append(dict(now=elapsed, update=self.steps,
                        train_bound=train_cost, extra_misses=risk,
                        action='wait' if delay > 0 else 'train', idle_used=idle_used,
                        quota_slack=quota_slack if self.train_quota else None,
                        quota_forced=bool(wait > 0 and quota_slack == 0),
                        budget_exhausted=bool(wait > 0 and remaining == 0)))
                    self.control_seconds += time.perf_counter()-tick
                    if delay > 0:
                        tick = time.perf_counter()
                        time.sleep(delay)
                        slept = time.perf_counter()-tick
                        idle_used += slept
                        self.idle_seconds += slept
                        continue
                return
            tick=time.perf_counter()
            arrivals=[self.trace[i]['arrival'] for i in self.pending]
            b=self.controller.choose(now-self.origin,arrivals,allow_train)
            self.control_seconds+=time.perf_counter()-tick
            self.decisions+=1
            if b==0:
                return
            self._infer(strategy,b)

    def _planned_wait(self, elapsed):
        """Replan one work window from completed costs, keeping labels private."""
        from src.schedulers.window_planner import FixedPlant, plan_window
        window = int(elapsed)
        if window != self.plan_window_index:
            tick = time.perf_counter()
            compute = float(np.median(self.compute_costs)) if self.compute_costs else self.controller.costs.mean('train', 1)
            prep = max(0, self.controller.costs.mean('train', 1)-compute)
            infer = self.controller.costs.means['infer']
            plant = FixedPlant(dict(trace=self.trace, compute=[compute], prep=[prep],
                                    infer_intercept=infer, infer_slope=infer/32), self.controller.slo)
            target = int(min(window+1, self.args.stream_duration)*self.train_quota)
            count = max(0, target-(self.steps-self.origin_steps))
            if count:
                result, feasible, nodes = plan_window(plant, elapsed,
                    self.next_arrival, window, count)
                self.window_plan = {self.steps+i: start for i, start in enumerate(result[5])}
                self.plan_records.append(dict(window=window, now=elapsed, update=self.steps,
                    updates=count, compute=compute, prep=prep, infer=infer,
                    predicted_misses=result[2], feasible=feasible, expanded_states=nodes,
                    starts=list(result[5])))
            else:
                self.window_plan = {}
            self.plan_window_index = window
            self.planning_seconds += time.perf_counter()-tick
        return max(0.0, self.window_plan.get(self.steps, elapsed)-(time.perf_counter()-self.origin))

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
            self.records[-1]['service_started'] = started-self.origin
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
                 'start_policy':self.start_policy, 'intentional_idle_seconds':self.idle_seconds,
                 'train_quota':self.train_quota, 'origin_steps':self.origin_steps,
                 'quota_idle_seconds':self.quota_idle_seconds,
                 'information_idle_seconds':self.idle_seconds-self.quota_idle_seconds,
                 'planning_seconds':self.planning_seconds, 'plan_windows':len(self.plan_records),
                 'guard_overrides':self.controller.guard_overrides,
                 'neural_parameters':65 if self.controller.net is not None else 0,
                 'neural_updates':self.controller.net.updates if self.controller.net is not None else 0}
        Path('stream_metrics.json').write_text(json.dumps(summary,indent=2))
        Path('stream_requests.json').write_text(json.dumps(self.records))
        Path('stream_arrivals.json').write_text(json.dumps(self.trace))
        Path('stream_preparation.json').write_text(json.dumps(self.prep_stats, indent=2))
        Path('stream_training.json').write_text(json.dumps(self.training_records))
        Path('stream_train_starts.json').write_text(json.dumps(self.start_records))
        Path('stream_window_plans.json').write_text(json.dumps(self.plan_records))
        log_info('[StreamMetrics] '+json.dumps(summary))
        return summary
