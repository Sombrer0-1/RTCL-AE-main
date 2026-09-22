"""Cooperative evaluation admission, independent of training accuracy/batch size.

A single in-flight evaluation consumes the latest published snapshot. EWMA cost
sets the next idle interval; version lag can override it. No process suspension
is required and final draining always evaluates the last committed snapshot.
"""

class FreshnessPolicy:
    def __init__(self, now, min_interval=1.0, max_interval=5.0, max_lag=2):
        if not 0 < min_interval <= max_interval or max_lag < 1:
            raise ValueError("invalid freshness bounds")
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_lag = max_lag
        self.last_end = now
        self.last_version = 0
        self.cost = None

    @property
    def interval(self):
        return min(self.max_interval, max(self.min_interval, self.cost or self.min_interval))

    def ready(self, now, version, training_active):
        if version <= self.last_version:
            return False
        return (not training_active or now - self.last_end >= self.interval
                or (version - self.last_version >= self.max_lag
                    and now - self.last_end >= self.min_interval))

    def observe(self, now, version, duration):
        if duration < 0 or version <= self.last_version:
            raise ValueError("observation must have positive version progress and nonnegative cost")
        self.cost = duration if self.cost is None else 0.75 * self.cost + 0.25 * duration
        self.last_end = now
        self.last_version = version

class InferenceBatchPolicy:
    """Bounded multiplicative probing with a latency deadband, at cycle boundaries.

    This is a soft measured-latency target, not a hard real-time guarantee.
    Training batch and optimizer updates are never changed.
    """
    def __init__(self, initial, target_ms=10.0, maximum=256):
        if initial < 1 or maximum < initial or target_ms <= 0:
            raise ValueError('invalid inference batch bounds')
        self.minimum = initial
        self.maximum = maximum
        self.batch = initial
        self.target_ms = target_ms

    def observe(self, p95_ms):
        import math
        if not math.isfinite(p95_ms) or p95_ms <= 0:
            return self.batch
        if p95_ms > self.target_ms:
            self.batch = max(self.minimum, self.batch // 2)
        elif p95_ms < self.target_ms / 2:
            self.batch = min(self.maximum, self.batch * 2)
        return self.batch
