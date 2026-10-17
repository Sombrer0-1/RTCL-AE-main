"""Finite-window FIFO scheduling with known arrivals and fixed service estimates."""
from bisect import bisect_right
import numpy as np


class FixedPlant:
    def __init__(self, data, slo=.033):
        self.arrivals = [r['arrival'] for r in data['trace']]
        self.compute = data['compute'][0]
        self.prep = data['prep'][0]
        self.intercept = data['infer_intercept']
        self.slope = data['infer_slope']
        self.slo = slo

    def drain(self, now, cursor, one=False):
        missed, latency = 0, 0.0
        arrivals = self.arrivals
        while cursor < len(arrivals) and arrivals[cursor] <= now:
            count = min(32, bisect_right(arrivals, now)-cursor)
            now += self.intercept+self.slope*count
            for arrival in arrivals[cursor:cursor+count]:
                missed += now-arrival > self.slo
                latency += now-arrival
            cursor += count
            if one:
                break
        return now, cursor, missed, latency

    def train(self, now, cursor):
        completed = now+self.compute
        now = completed
        remaining, missed, latency = self.prep, 0, 0.0
        while remaining > 1e-12:
            chunk = min(.008, remaining)
            now += chunk
            remaining -= chunk
            now, cursor, m, l = self.drain(now, cursor, one=True)
            missed += m
            latency += l
        now, cursor, m, l = self.drain(now, cursor)
        return now, cursor, missed+m, latency+l, completed

    def wait_to(self, now, cursor, target):
        missed, latency = 0, 0.0
        while cursor < len(self.arrivals) and self.arrivals[cursor] <= target:
            now = max(now, self.arrivals[cursor])
            now, cursor, m, l = self.drain(now, cursor)
            missed += m
            latency += l
        return max(now, target), cursor, missed, latency


def plan_window(plant, ready, cursor, window, quota, resolution=.002):
    deadline = window+1.0
    ready, cursor, initial_m, initial_l = plant.wait_to(ready, cursor, float(window))
    # State: ready time, unserved cursor, misses, sum training completions,
    # summed response latency, tuple of planned start times.
    initial = (ready, cursor, initial_m, 0.0, initial_l, ())
    count = int(np.ceil(1/resolution))+2
    buckets = [dict() for _ in range(count)]
    def put(k, state):
        now = state[0]
        remaining = quota-k
        if now+remaining*plant.compute+max(0, remaining-1)*plant.prep > deadline+1e-10:
            return
        tick = max(0, int(np.ceil((now-window)/resolution-1e-10)))
        if tick >= count:
            return
        old = buckets[tick].get(k)
        key = lambda s: (s[2], s[3], s[4], s[0])
        if old is None or key(state) < key(old):
            buckets[tick][k] = state
    put(0, initial)
    best, expanded = None, 0
    for bucket in buckets:
        for k, state in list(bucket.items()):
            expanded += 1
            now, cursor, missed, progress, latency, starts = state
            after, j, m, l, completed = plant.train(now, cursor)
            if completed <= deadline+1e-10:
                trained = (after, j, missed+m, progress+completed, latency+l, starts+(now,))
                if k+1 == quota:
                    end, jj, mm, ll = plant.wait_to(after, j, deadline)
                    result = (end, jj, trained[2]+mm, trained[3], trained[4]+ll, trained[5])
                    key = lambda s: (s[2], s[3], s[4], s[0])
                    if best is None or key(result) < key(best):
                        best = result
                else:
                    put(k+1, trained)
            if cursor < len(plant.arrivals) and plant.arrivals[cursor] < deadline:
                end, j, m, l = plant.wait_to(now, cursor, plant.arrivals[cursor])
                put(k, (end, j, missed+m, progress, latency+l, starts))
    feasible = best is not None
    if best is None:
        # Earliest service-priority execution preserves prescribed training work.
        now, cursor, missed, progress, latency, starts = initial
        for _ in range(quota):
            start = now
            now, cursor, m, l, completed = plant.train(now, cursor)
            missed += m
            latency += l
            progress += completed
            starts += (start,)
        now, cursor, m, l = plant.wait_to(now, cursor, deadline)
        best = (now, cursor, missed+m, progress, latency+l, starts)
    return best, feasible, expanded


def replay_schedule(plant, starts, horizon):
    """Independently execute selected starts to audit the search cost bookkeeping."""
    now, cursor, missed, latency = 0.0, 0, 0, 0.0
    completed = []
    for start in starts:
        now, cursor, m, l = plant.wait_to(now, cursor, start)
        assert abs(now-start) < 1e-8, (now, start)
        missed += m
        latency += l
        now, cursor, m, l, end = plant.train(now, cursor)
        missed += m
        latency += l
        completed.append(end)
    now, cursor, m, l = plant.wait_to(now, cursor, horizon)
    return missed+m, latency+l, cursor, completed

