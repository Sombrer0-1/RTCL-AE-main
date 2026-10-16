"""R4 cooperative execution with AOCL's periodic, full-test snapshot workload."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from src.schedulers.boundary_scheduler import evaluation_state
from src.schedulers.host_preparation import install_preparation
from src.utils.logging_utils import log_info


class CertifiedPreparationPlugin:
    """Share R4's backend optimization with the concurrent baselines."""

    def __init__(self):
        self.stats = dict(certified_epochs=0, fallback_epochs=0, service_batches=0,
                          service_samples=0, service_seconds=0.0)

    def before_training_epoch(self, strategy, **kwargs):
        install_preparation(strategy.dataloader, None, self.stats)

    def after_training(self, strategy, **kwargs):
        Path('full_eval_preparation.json').write_text(json.dumps(self.stats) + '\n')


class FullEvaluationPlugin(CertifiedPreparationPlugin):
    """Alternate whole training minibatches and fixed-snapshot eval minibatches.

    A snapshot is published after each training experience. Evaluation retains
    that version for its entire pass, even when newer snapshots are published.
    """

    def __init__(self, args, test_stream, shared_data):
        super().__init__()
        from src.workers.eval_worker import EVAL_INTERVAL_SECONDS
        self.args = args
        self.test_stream = test_stream
        self.shared_data = shared_data
        self.interval = EVAL_INTERVAL_SECONDS
        self.last_eval_end = time.perf_counter()
        self.last_service = self.last_eval_end
        self.latest = None
        self.latest_version = 0
        self.active_version = 0
        self.active_final_at_load = False
        self.eval_strategy = None
        self.iterator = None
        self.cycles = []
        self.batch_records = []
        self.train_updates = 0
        self.train_completed = False

    def before_training_epoch(self, strategy, **kwargs):
        preparation = self.args.host_preparation
        if preparation != 'baseline':
            service = (lambda: self._preparation_service(strategy)) if preparation == 'cooperative' else None
            install_preparation(strategy.dataloader, service, self.stats)

    def after_training_iteration(self, strategy, **kwargs):
        if strategy.device.type == 'cuda':
            torch.cuda.synchronize(strategy.device)
        self.train_updates += 1

    def before_training_iteration(self, strategy, **kwargs):
        self._service(strategy)

    def publish(self, strategy, version):
        # CPU clones retain the committed version while subsequent training runs.
        self.latest = {key: value.detach().cpu().clone()
                       for key, value in strategy.model.state_dict().items()}
        self.latest_version = version
        self._service(strategy)

    def _start(self, strategy):
        if self.eval_strategy is None:
            self.eval_strategy = SimpleNamespace(model=copy.deepcopy(strategy.model),
                device=strategy.device, eval_mb_size=self.args.eval_bs)
        self.eval_strategy.model.load_state_dict(self.latest)
        self.active_version = self.latest_version
        self.active_final_at_load = self.train_completed
        from src.workers.eval_worker import iter_interruptible_eval
        self.iterator = iter_interruptible_eval(self.eval_strategy, self.test_stream,
            None, self.shared_data, len(self.cycles) + 1, self.args)

    def _service(self, strategy, force=False):
        if self.iterator is None:
            limit = getattr(self.args, 'eval_cycles', 0)
            if self.latest is None or (limit and not self.train_completed and
                    len(self.cycles) >= limit - 1):
                return None
            if not force and time.perf_counter() - self.last_eval_end < self.interval:
                return None
        # Enter and leave per step: neither RNG restoration nor no_grad may span
        # a yield into the training loop.
        with evaluation_state(strategy.model):
            if self.iterator is None:
                self._start(strategy)
            try:
                record = next(self.iterator)
                self.batch_records.append(record)
                return record
            except StopIteration as completed:
                result = completed.value
                self.iterator = None
                if not result or not result['total_samples']:
                    raise RuntimeError('Full snapshot evaluation failed')
                cycle = dict(cycle=len(self.cycles) + 1,
                    version_lower_bound=self.active_version,
                    final_at_load=self.active_final_at_load,
                    final_snapshot=self.active_version == len(self.test_stream),
                    accuracy=result['streaming_accuracy'], samples=result['total_samples'],
                    eval_batch=self.args.eval_bs, cache_bytes=0,
                    duration=result['eval_latency'], batch_p95_ms=result['batch_p95_ms'],
                    batch_p99_ms=result['batch_p99_ms'])
                self.cycles.append(cycle)
                self.last_eval_end = time.perf_counter()
                log_info('[CycleMetrics] ' + json.dumps(cycle))
                return None

    def _preparation_service(self, strategy):
        now = time.perf_counter()
        if now - self.last_service < .008:
            return
        record = self._service(strategy)
        ended = time.perf_counter()
        if record is not None:
            self.stats['service_batches'] += 1
            self.stats['service_samples'] += record['samples']
            self.stats['service_seconds'] += ended - now
        self.last_service = ended

    def finish(self, strategy):
        self.train_completed = True
        # Finish an old in-flight pass, then guarantee a final-model pass.
        while self.iterator is not None:
            self._service(strategy, force=True)
        limit = getattr(self.args, 'eval_cycles', 0)
        target = limit if limit else len(self.cycles) + (not self.cycles or
            not self.cycles[-1]['final_snapshot'])
        while len(self.cycles) < target:
            self._service(strategy, force=True)
        Path('full_eval_preparation.json').write_text(json.dumps(self.stats) + '\n')
        Path('full_eval_cycles.json').write_text(json.dumps(self.cycles) + '\n')
        Path('full_eval_training.json').write_text(json.dumps(dict(
            train_updates=self.train_updates)) + '\n')
