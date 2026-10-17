"""One-update lookahead; inputs contain timestamps and released-label estimates."""
from bisect import bisect_right


def score_update(arrivals, now, compute, infer, slo, accuracy, gain, train_first):
    """Expected timely correct results; both alternatives perform one update.

    train_first executes now. The alternative serves one FIFO batch (waiting
    for its first arrival if needed), then updates. Beyond that update both
    paths use the same estimated accuracy. Future labels are never an input.
    """
    times = list(arrivals)
    cursor, score, time = 0, 0.0, now
    if not train_first and times:
        time = max(time, times[0])
        count = min(32, bisect_right(times, time))
        time += infer(count)
        score += accuracy*sum(time-t <= slo for t in times[:count])
        cursor += count
    time += compute
    updated = min(1.0, max(0.0, accuracy+gain))
    while cursor < len(times):
        time = max(time, times[cursor])
        end = min(cursor+32, bisect_right(times, time))
        time += infer(end-cursor)
        score += updated*sum(time-t <= slo for t in times[cursor:end])
        cursor = end
    return score


def quality_train(arrivals, now, compute, infer, slo, accuracy, gain):
    a = score_update(arrivals, now, compute, infer, slo, accuracy, gain, True)
    b = score_update(arrivals, now, compute, infer, slo, accuracy, gain, False)
    return a >= b-1e-10, a, b


def causal_times(history, pending, now, horizon=1.0):
    """Uniform short-range forecast from past arrivals; no future trace access."""
    span = min(1.0, max(.05, now))
    count = sum(now-span < t <= now for t in history)
    rate = max(1.0, count/span)
    step = 1/rate
    return list(pending)+[now+i*step for i in range(1, int(horizon/step)+1)]
