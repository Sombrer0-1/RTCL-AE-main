"""Arrival-information controls for the training-start headroom experiment."""
from bisect import bisect_right
import math


def uniform_start_wait(now, completed, rate, compute_cost):
    return max(0.0, (completed+1)/rate-compute_cost-now)


def training_quota(now, completed, rate, horizon, compute_cost, total_cost):
    """Return mandatory release wait and discretionary slack in a 1 s work window.

    All policies receive the same cumulative update quota. If behind schedule,
    the next update starts without discretionary waiting; late work carries over.
    """
    if not rate or now >= horizon:
        return 0.0, float('inf')
    deadline = min(math.floor(now) + 1, horizon)
    target = int(deadline * rate)
    remaining = target - completed
    if remaining <= 0:
        return max(0.0, deadline-now), 0.0
    work = compute_cost + (remaining-1)*total_cost
    return 0.0, max(0.0, deadline-now-work)


def fifo_misses(arrivals, available, slo, infer_cost, max_batch=32):
    """Serve a fixed timestamp list, admitting arrivals during each FIFO batch."""
    index = 0
    misses = 0
    while index < len(arrivals):
        available = max(available, arrivals[index])
        end = min(bisect_right(arrivals, available), index + max_batch)
        available += infer_cost(end - index)
        misses += sum(available - arrival > slo for arrival in arrivals[index:end])
        index = end
    return misses


def blocking_risk(arrivals, next_arrival, now, train_cost, slo, infer_cost, max_batch=32):
    """Extra misses predicted from one nonpreemptive update at an empty queue.

    Both alternatives evaluate the same future requests, including those arriving
    during the first inference batch after training. Only timestamps are read.
    """
    end = bisect_right(arrivals, now + train_cost + infer_cost(max_batch))
    future = arrivals[next_arrival:end]
    immediate = fifo_misses(future, now, slo, infer_cost, max_batch)
    blocked = fifo_misses(future, now + train_cost, slo, infer_cost, max_batch)
    return max(0, blocked - immediate)


def start_wait(policy, now, last_response, train_cost, slo, infer_cost,
               arrivals=(), next_arrival=0, max_batch=32):
    """Return (idle seconds, extra misses); the reactive control sees no future."""
    if policy == 'immediate':
        return 0.0, 0
    if policy == 'reactive':
        quiet = max(0.0, train_cost + infer_cost(1) - slo)
        return max(0.0, last_response + quiet - now), 0
    if policy == 'phase':
        vulnerable = max(0.0, train_cost + infer_cost(1) - slo)
        phase = now % 1
        if vulnerable and (phase < .2 or phase+vulnerable >= 1):
            return (.2-phase if phase < .2 else 1.2-phase), 0
        return 0.0, 0
    risk = blocking_risk(arrivals, next_arrival, now, train_cost, slo,
                         infer_cost, max_batch)
    wait = max(0.0, arrivals[next_arrival] - now) if risk else 0.0
    return wait, risk
