"""Whole-window learned-utility planning under a common update budget."""
import numpy as np
from src.schedulers.window_planner import FixedPlant


class LearningPlant(FixedPlant):
    def __init__(self,data,slo,accuracy,gain):
        super().__init__(data,slo)
        self.accuracy=accuracy;self.gain=gain;self.completed=0

    def drain(self,now,cursor,one=False):
        end,index,missed,latency=super().drain(now,cursor,one=one)
        probability=min(1.0,max(0.0,self.accuracy+self.completed*self.gain))
        # Each served request costs one minus its expected timely correctness.
        errors=(index-cursor)*(1-probability)+missed*probability
        return end,index,errors,latency


def plan_learning_window(plant, ready, cursor, window, quota, resolution=.002, completion_deadlines=None):
    deadline = window+1.0
    plant.completed=0
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
            plant.completed=k+1
            after, j, m, l, completed = plant.train(now, cursor)
            update_deadline=deadline if completion_deadlines is None else min(deadline,completion_deadlines[k])
            if completed <= update_deadline+1e-10:
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
                plant.completed=k
                end, j, m, l = plant.wait_to(now, cursor, plant.arrivals[cursor])
                put(k, (end, j, missed+m, progress, latency+l, starts))
    feasible = best is not None
    if best is None:
        # Earliest service-priority execution preserves prescribed training work.
        now, cursor, missed, progress, latency, starts = initial
        for step in range(quota):
            plant.completed=step+1
            start = now
            now, cursor, m, l, completed = plant.train(now, cursor)
            missed += m
            latency += l
            progress += completed
            starts += (start,)
        now, cursor, m, l = plant.wait_to(now, cursor, deadline)
        best = (now, cursor, missed+m, progress, latency+l, starts)
    return best, feasible, expanded



def reference_deadlines(plant,ready,cursor,nominal,tolerance=.002):
    """Feasible FIFO uniform-pacing reference from the current state."""
    result=[]
    for due in nominal:
        ready,cursor,_,_=plant.wait_to(ready,cursor,max(ready,due-plant.compute))
        ready,cursor,_,_,completed=plant.train(ready,cursor)
        result.append(completed+tolerance)
    return result
