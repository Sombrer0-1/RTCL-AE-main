"""Bounded resident cache for deterministic, repeatedly evaluated datasets.

Only the boundary_cached SplitCIFAR10 experiment enables this. Cache is filled
while serving the first pass, so that pass and its warmup cost remain measured.
"""
import torch


class EvaluationCache:
    def __init__(self, max_bytes):
        if max_bytes < 0:
            raise ValueError('cache budget must be nonnegative')
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.entries = {}

    def batches(self, key, loader, batch_size, device):
        if key in self.entries:
            x, y = self.entries[key]
            for start in range(0, len(y), batch_size):
                yield x[start:start + batch_size], y[start:start + batch_size], None
            return
        xs, ys = [], []
        eligible = None
        for x, y, task_id in loader:
            if eligible is None:
                per_sample = (x[0].numel() * x.element_size() +
                              y[0].numel() * y.element_size())
                required = len(loader.dataset) * per_sample
                eligible = required <= self.max_bytes - self.used_bytes
                if torch.device(device).type == 'cuda':
                    free, _ = torch.cuda.mem_get_info(device)
                    # Reserve headroom; concatenate duplicates one experience briefly.
                    eligible = eligible and required * 2 < free // 2
            # First pass retains the original transfer timing. Cache population
            # occurs after consuming the batch and is included in cycle wall time.
            yield x, y, task_id
            if eligible:
                xs.append(x.to(device)); ys.append(y.to(device))
        if eligible and xs:
            x, y = torch.cat(xs), torch.cat(ys)
            size = x.numel() * x.element_size() + y.numel() * y.element_size()
            if size <= self.max_bytes - self.used_bytes:
                self.entries[key] = (x, y)
                self.used_bytes += size
