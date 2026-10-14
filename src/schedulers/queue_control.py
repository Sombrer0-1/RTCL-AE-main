"""Causal minibatch scheduling with measured slack and virtual training debt.

Bounds are empirical risk estimates, NOT WCET guarantees. No request is dropped.
The tiny neural residual only ranks actions admitted by the analytic filter.
"""
from collections import deque
import math
import numpy as np


class TinyResidual:
    """65 parameters, CPU only; online supervision uses completed service costs."""
    def __init__(self):
        rng = np.random.default_rng(1729)
        self.w1 = rng.normal(0, .15, (8, 6))
        self.b1 = np.zeros(8)
        self.w2 = np.zeros(8)
        self.b2 = 0.0
        self.seen = 0
        self.updates = 0

    def residual(self, x):
        return .25 * np.tanh(self.w2 @ np.tanh(self.w1 @ x + self.b1) + self.b2)

    def observe(self, x, target):
        self.seen += 1
        if self.seen % 8:
            return
        h = np.tanh(self.w1 @ x + self.b1)
        z = np.tanh(self.w2 @ h + self.b2)
        delta = (.25*z - np.clip(target, -.25, .25)) * .25 * (1-z*z)
        hidden_delta = delta * self.w2 * (1-h*h)
        self.w2 -= .02 * delta * h
        self.b2 -= .02 * delta
        self.w1 -= .02 * np.outer(hidden_delta, x)
        self.b1 -= .02 * hidden_delta
        self.updates += 1


class CostModel:
    def __init__(self):
        self.means = {'train': .015, 'infer': .002}
        self.history = {kind: deque(maxlen=64) for kind in self.means}
        self.upper = {kind: 2*mean for kind, mean in self.means.items()}

    @staticmethod
    def scale(kind, batch):
        return 1.0 if kind == 'train' else 1 + batch/32

    def mean(self, kind, batch):
        return self.means[kind] * self.scale(kind, batch)

    def bound(self, kind, batch):
        return self.upper[kind]*self.scale(kind, batch)

    def observe(self, kind, batch, duration):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('cost must be finite and positive')
        scaled = duration / self.scale(kind, batch)
        self.means[kind] = .9*self.means[kind] + .1*scaled
        self.history[kind].append(scaled)
        values = self.history[kind]
        if len(values) < 4:
            self.upper[kind] = 2*self.means[kind]
        else:
            a = np.asarray(values)
            self.upper[kind] = 1.15*max(self.means[kind]+2*float(a.std()),
                                        float(np.quantile(a, .9)))


class QueueController:
    def __init__(self, policy, slo, max_batch=32, train_share=.7):
        if policy not in ('fifo', 'slack', 'debt', 'neural', 'boundary'):
            raise ValueError('unknown policy')
        if slo <= 0 or max_batch < 1 or not 0 < train_share < 1:
            raise ValueError('invalid controller bounds')
        self.policy, self.slo, self.max_batch = policy, slo, max_batch
        self.rho = train_share
        self.debt = 0.0
        self.costs = CostModel()
        self.net = TinyResidual() if policy == 'neural' else None
        self.guard_overrides = 0
        self.observations = 0

    def features(self, kind, batch, queue_size):
        return np.array([float(kind == 'train'), math.log2(max(batch,1))/5,
                         min(self.costs.means['train']/self.slo, 3),
                         min(self.costs.means['infer']/self.slo, 3),
                         min(queue_size/32, 3), min(self.debt/self.slo, 3)])

    def prediction(self, kind, batch, queue_size):
        base = self.costs.mean(kind, batch)
        if self.net is None or self.observations < 32:
            return base
        return base * math.exp(float(self.net.residual(self.features(kind,batch,queue_size))))

    def observe(self, kind, batch, duration, features=None, base=None):
        if self.net is not None and features is not None:
            self.net.observe(features, math.log(max(duration,1e-9)/max(base,1e-9)))
        self.costs.observe(kind,batch,duration)
        self.debt = max(0, self.debt + (self.rho - (kind == 'train'))*duration)
        self.observations += 1

    def choose(self, now, arrivals, allow_train=True):
        """Return 0 for one train update, or FIFO inference batch length.

        arrivals contains ONLY released requests, expressed on the same clock.
        """
        if not len(arrivals):
            return 0
        ages = now - np.asarray(arrivals, dtype=float)
        if ages.min() < -1e-9:
            raise ValueError('controller must not see future arrivals')
        n = min(len(ages),self.max_batch)
        if self.policy == 'fifo' or not allow_train:
            return n
        if self.policy == 'boundary':
            return 0
        slack = self.slo - ages[0]
        train_ok = slack > self.costs.bound('train',1) + self.costs.bound('infer',n)
        candidates = sorted({min(b,n) for b in (1,4,8,16,32,self.max_batch)})
        feasible = [b for b in candidates if self.costs.bound('infer',b) <= slack]
        # No admissible batch means overload: finish work, never drop late items.
        if not feasible:
            self.guard_overrides += 1
            return n
        if self.policy == 'slack':
            return 0 if train_ok else max(feasible)
        if not train_ok:
            self.guard_overrides += 1
        actions = ([0] if train_ok else []) + feasible
        w = ages/self.slo
        sum_w, sum_w2 = float(w.sum()),float(w@w)
        debt = self.debt/self.slo
        scores=[]
        for b in actions:
            kind = 'train' if b == 0 else 'infer'
            dt = self.prediction(kind,max(b,1),len(ages))
            d = dt/self.slo
            debt_next = max(0,debt+(self.rho-(b==0))*d)
            remain_w = sum_w-float(w[:b].sum())
            remain_w2 = sum_w2-float(w[:b]@w[:b])
            next_wait = remain_w2+2*d*remain_w+(len(w)-b)*d*d
            drift = .5*(debt_next**2-debt**2+next_wait-sum_w2)
            scores.append((drift/max(dt,1e-9),b))
        return min(scores)[1]
