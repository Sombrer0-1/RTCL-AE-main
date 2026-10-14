"""One-step batching lookahead and an optional online neural service predictor.

Only released arrivals and completed operation costs are used. The learned
predictor changes the amortization decision, never the empirical deadline guard.
"""
from collections import deque
import math

import numpy as np


class OnlineServiceNetwork:
    """9 -> 12 -> 1 CPU MLP, 133 parameters, bounded log-cost residual.

    Replay SGD uses a private RNG and completed costs, with prequential error
    measured before each update. No training-model data or accuracy labels enter.
    """
    parameters = 133

    def __init__(self):
        self.rng = np.random.default_rng(2718)
        self.w1 = self.rng.normal(0, .2, (12, 9))
        self.b1 = np.zeros(12)
        self.w2 = self.rng.normal(0, .03, 12)
        self.b2 = 0.0
        self.history = deque(maxlen=256)
        self.updates = 0
        self.seen = 0
        self.neural_error = 0.0
        self.base_error = 0.0
        self.scored = 0

    def residual(self, x):
        return float(np.clip(self.w2 @ np.tanh(self.w1 @ x + self.b1) + self.b2,
                             -math.log(4), math.log(4)))

    def observe(self, x, target):
        if not np.isfinite(x).all() or not math.isfinite(target):
            raise ValueError('nonfinite service observation')
        if self.seen >= 32:
            self.neural_error += abs(self.residual(x) - target)
            self.base_error += abs(target)
            self.scored += 1
        self.history.append((np.array(x, copy=True), float(np.clip(target, -math.log(4), math.log(4)))))
        self.seen += 1
        if self.seen % 8:
            return
        indices = self.rng.choice(len(self.history), min(16, len(self.history)), replace=False)
        x_batch = np.stack([self.history[int(i)][0] for i in indices])
        targets = np.array([self.history[int(i)][1] for i in indices])
        hidden = np.tanh(x_batch @ self.w1.T + self.b1)
        predicted = hidden @ self.w2 + self.b2
        delta = np.clip(predicted - targets, -.5, .5) / len(indices)
        hidden_delta = delta[:, None] * self.w2 * (1-hidden*hidden)
        self.w2 -= .03 * (hidden.T @ delta)
        self.b2 -= .03 * float(delta.sum())
        self.w1 -= .03 * (hidden_delta.T @ x_batch)
        self.b1 -= .03 * hidden_delta.sum(axis=0)
        self.updates += 1


class PredictiveBatchController:
    """Compare serving now with one more productive, bounded service vacation.

    Costs amortize across the expected batch after one training/preparation step.
    A quadratic response-time penalty and a 0.35*SLO service envelope prevent
    waiting until the full deadline just to improve occupancy. This envelope
    is an empirical target; uninterruptible host/GPU stalls can exceed it.
    """
    def __init__(self, policy, slo, max_batch=32, train_share=.7):
        # Imported here to avoid a module cycle with the controller factory.
        from src.schedulers.queue_control import CostModel
        if policy not in ('predictive', 'predictive_neural'):
            raise ValueError('unknown predictive policy')
        if slo <= 0 or max_batch < 1 or not 0 < train_share < 1:
            raise ValueError('invalid controller bounds')
        self.policy, self.slo, self.max_batch = policy, slo, max_batch
        self.rho, self.debt = train_share, 0.0
        self.costs = CostModel()
        self.net = OnlineServiceNetwork() if policy.endswith('_neural') else None
        self.guard_overrides = 0
        self.observations = 0
        self.releases = deque()
        self.release_count = 0
        self.rate = 0.0
        self.stage = 'iteration'
        self.deferred = 0
        self.neural_changed = 0

    def admit(self, arrivals, now):
        if any(t > now+1e-9 for t in arrivals):
            raise ValueError('cannot observe future arrivals')
        self.releases.extend(arrivals)
        self.release_count += len(arrivals)
        while self.releases and self.releases[0] < now-.2:
            self.releases.popleft()
        # Empty periods decay naturally; the policy never sees the full trace.
        self.rate = len(self.releases)/min(.2, max(now, .02))

    def features(self, kind, batch, queue_size):
        return np.array([float(kind == 'train'), batch/self.max_batch,
                         (batch/self.max_batch)**2, math.log2(max(batch, 1))/5,
                         min(self.costs.means['train']/self.slo, 2),
                         min(self.costs.means['infer']/self.slo, 2),
                         min(self.rate*self.slo/self.max_batch, 2),
                         min(queue_size/self.max_batch, 2),
                         float(self.stage == 'preparation')])

    def prediction(self, kind, batch, queue_size, learned=True):
        base = self.costs.mean(kind, batch)
        if learned and kind == 'infer' and self.net is not None and self.net.seen >= 32:
            return base*math.exp(self.net.residual(self.features(kind, batch, queue_size)))
        return base

    def observe(self, kind, batch, duration, features=None, base=None):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('cost must be finite and positive')
        if self.net is not None and kind == 'infer' and features is not None:
            self.net.observe(features, math.log(duration/max(base, 1e-9)))
        self.costs.observe(kind, batch, duration)
        self.debt = max(0, self.debt+(self.rho-(kind == 'train'))*duration)
        self.observations += 1

    def _defer_is_better(self, age, n, delay, future_batch, learned):
        budget = .35*self.slo
        unit = self.costs.mean('infer', 1)
        def score(wait, batch):
            duration = self.prediction('infer', batch, n, learned=learned)
            return duration/batch + .1*unit*((age+wait+duration)/budget)**2
        return bool(score(delay, future_batch) < score(0, n))

    def choose(self, now, arrivals, allow_train=True, stage='iteration', check_interval=.008):
        if not arrivals:
            return 0
        ages = now-np.asarray(arrivals, dtype=float)
        if ages.min() < -1e-9:
            raise ValueError('controller must not see future arrivals')
        n = min(len(ages), self.max_batch)
        if not allow_train or n == self.max_batch:
            return n
        self.stage = stage
        # At preparation checks we can revisit before running a GPU update.
        # At the iteration boundary an entire next train operation must fit.
        delay = check_interval if stage == 'preparation' else self.costs.mean('train', 1)
        risk_delay = check_interval if stage == 'preparation' else self.costs.bound('train', 1)
        future_batch = min(self.max_batch, max(n, int(n+self.rate*delay)))
        if (ages[0]+risk_delay+self.costs.bound('infer', future_batch) >= .35*self.slo
                or ages[0]+risk_delay+self.costs.bound('infer', n) >= self.slo):
            self.guard_overrides += 1
            return n
        decision = self._defer_is_better(ages[0], n, delay, future_batch, True)
        if self.net is not None and self.net.seen >= 32:
            self.neural_changed += decision != self._defer_is_better(ages[0], n, delay, future_batch, False)
        self.deferred += int(decision)
        return 0 if decision else n

    def diagnostics(self):
        result = dict(predictive_deferrals=self.deferred, neural_changed_decisions=self.neural_changed,
                      released_observations=self.release_count)
        if self.net is not None:
            result.update(neural_scored=self.net.scored,
                          neural_log_mae=self.net.neural_error/max(1, self.net.scored),
                          base_log_mae=self.net.base_error/max(1, self.net.scored))
        return result
