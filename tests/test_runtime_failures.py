from contextlib import ExitStack
import multiprocessing as mp
import os
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import psutil
import torch
from torch.utils.data import TensorDataset
import main
from avalanche.benchmarks import nc_benchmark
from avalanche.training.plugins import EvaluationPlugin

from src.config import parse_arguments
from src.schedulers.dynamic_config import DynamicConfigController
from src.schedulers.timeline_scheduler import GlobalTimelineScheduler, global_scheduler_worker
from src.utils.config_loader import validate_batch_config
from src.workers.config_worker import dynamic_config_worker
from src.workers.eval_worker import eval_worker, request_config_update_handler_eval
from src.workers.train_worker import train_worker, request_config_update_handler


def arguments(*extra):
    with patch('sys.argv', ['test', '--benchmark', 'split_cifar10', '--cuda', '-1', *extra]):
        return parse_arguments()


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Linear(2, 10)

    def forward(self, x):
        return self.output(x)


def stop_again_on_term(ready):
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGSTOP))
    ready.set()
    signal.raise_signal(signal.SIGSTOP)
    while True:
        signal.pause()


class RuntimeFailureTests(unittest.TestCase):
    def tearDown(self):
        main.runtime_state.reset_globals()

    def test_freshness_rechecks_loaded_snapshot_and_reaches_final_version(self):
        for mode in ('freshness', 'freshness_adaptive', 'freshness_cached'):
            with self.subTest(mode=mode), ExitStack() as stack:
                args = arguments('--global_scheduler_mode', mode, '--enable_double_buffer')
                shared = {'train_process_active': False, 'published_version': 1,
                          'total_experiences': 3, 'all_experiences_completed': True}
                model = Classifier()
                states = []
                for version in (1, 1, 3):
                    states.append({'model_version': version, 'model_state_dict': {
                        key: torch.full_like(value, version) for key, value in model.state_dict().items()}})
                scheduler = MagicMock()
                scheduler.read_state_dict.side_effect = states
                strategy = SimpleNamespace(model=model, device=torch.device('cpu'), eval_mb_size=16)
                evaluated = []

                def evaluate(*_args, **_kwargs):
                    evaluated.append(int(model.output.weight[0, 0].item()))
                    shared['published_version'] = 2 if len(evaluated) == 1 else 3
                    return dict(streaming_accuracy=.5, experience_accuracies={}, total_samples=4,
                                batch_p95_ms=1., batch_p99_ms=1.)

                stack.enter_context(patch('src.workers.eval_worker.initialize_model', return_value=model))
                stack.enter_context(patch('src.workers.eval_worker.create_benchmark',
                    return_value=SimpleNamespace(test_stream=[object()])))
                stack.enter_context(patch('src.workers.eval_worker.create_cl_strategy', return_value=strategy))
                stack.enter_context(patch('src.workers.eval_worker.CSVLogger'))
                stack.enter_context(patch('src.workers.eval_worker.signal.signal'))
                stack.enter_context(patch('src.workers.eval_worker.interruptible_eval', side_effect=evaluate))
                stack.enter_context(patch('src.workers.eval_worker.time.sleep',
                    side_effect=lambda _: shared.update(published_version=3)))
                eval_worker(args, torch.device('cpu'), scheduler, None, None, shared)
                self.assertEqual(evaluated, [1, 3])
                self.assertFalse(shared.get('worker_failed', False))

    def run_publication_failure(self, failed_version):
        args = arguments('--algorithm', 'naive', '--training_bs', '2',
                         '--global_scheduler_mode', 'freshness', '--enable_double_buffer')
        data = TensorDataset(torch.randn(12, 2), torch.arange(12) % 3)
        benchmark = nc_benchmark(data, data, n_experiences=3, task_labels=False, shuffle=False)
        shared = {'all_experiences_completed': False}
        scheduler = MagicMock()
        seen_versions = []

        def publish(_state, model_version):
            seen_versions.append(shared.get('published_version', 0))
            if model_version == failed_version:
                raise RuntimeError('injected publication failure')

        scheduler.write_state_dict.side_effect = publish
        with patch('src.workers.train_worker.initialize_model', return_value=Classifier()), \
             patch('src.workers.train_worker.create_benchmark', return_value=benchmark), \
             patch('src.workers.train_worker.CSVLogger'), \
             patch('src.workers.train_worker.create_evaluation_plugin',
                   return_value=EvaluationPlugin(loggers=[])), \
             patch('src.workers.train_worker.signal.signal'):
            train_worker(args, torch.device('cpu'), scheduler, None, None, shared)
        return shared, seen_versions

    def test_failed_publication_does_not_advance_version_and_later_publish_recovers(self):
        shared, seen = self.run_publication_failure(2)
        self.assertEqual(seen, [0, 1, 1])
        self.assertEqual(shared['published_version'], 3)
        self.assertTrue(shared['all_experiences_completed'])
        self.assertFalse(shared.get('worker_failed', False))

    def test_final_publication_failure_is_reported(self):
        shared, _ = self.run_publication_failure(3)
        self.assertEqual(shared['published_version'], 2)
        self.assertTrue(shared['worker_failed'])
        self.assertFalse(shared['all_experiences_completed'])

    def test_reconfiguration_interval_does_not_replace_timeslice(self):
        for file_timeslice in ('absent', 'omitted', None):
            with self.subTest(file_timeslice=file_timeslice), tempfile.TemporaryDirectory() as temporary:
                shared = {}
                controller = DynamicConfigController(shared, arguments('--timeslice', '.125'))
                path = None
                if file_timeslice != 'absent':
                    import json
                    config = {'batches': [dict(index=0, train_batch_size=32, eval_batch_size=8, duration=0)]}
                    if file_timeslice is None:
                        config['timeslice'] = None
                    path = Path(temporary) / 'config.json'
                    path.write_text(json.dumps(config))
                with patch('src.workers.config_worker.safe_kill'), \
                     patch('src.workers.config_worker.log_event'):
                    dynamic_config_worker(controller, 1, 2, 0, path)
                self.assertEqual(shared['timeslice'], .125)

    def test_invalid_file_timeslice_is_rejected(self):
        config = {'batches': [dict(index=0, train_batch_size=32, eval_batch_size=8)]}
        for value in (None, -1, 0, float('inf'), float('nan'), True, '1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'timeslice'):
                validate_batch_config(dict(config, timeslice=value))

    def test_config_notifications_do_not_relay_themselves(self):
        shared = {'CONFIG_UPDATE_REQUESTED': True, 'timeslice': .1}
        delivered, pending = [], []
        rounds = 0

        def send(pid, sig):
            if sig == 0:
                return rounds < 3
            if sig == signal.SIGUSR1:
                delivered.append(pid)
                pending.append(pid)
            return True

        def advance(_):
            nonlocal rounds
            rounds += 1
            while pending:
                handler = request_config_update_handler if pending.pop() == 1 else request_config_update_handler_eval
                handler(signal.SIGUSR1, None, shared)

        with patch('src.schedulers.timeline_scheduler.safe_kill', side_effect=send), \
             patch('src.schedulers.timeline_scheduler.time.sleep', side_effect=advance):
            global_scheduler_worker(GlobalTimelineScheduler(.1, 'fully_parallel'), 1, 2, shared)
        self.assertEqual(delivered, [1, 2])
        self.assertFalse(shared['CONFIG_UPDATE_REQUESTED'])

    def test_cleanup_reaps_worker_that_stops_again_on_sigterm(self):
        context = mp.get_context('spawn')
        ready = context.Event()
        worker = context.Process(target=stop_again_on_term, args=(ready,))
        worker.start()
        try:
            self.assertTrue(ready.wait(20))
            deadline = time.monotonic() + 5
            while psutil.Process(worker.pid).status() != psutil.STATUS_STOPPED:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            main.terminate_process(worker, timeout=.2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(worker.exitcode, -signal.SIGKILL)
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join()

    def test_main_propagates_reported_or_exited_worker_failure(self):
        for reported in (False, True):
            with self.subTest(reported=reported), ExitStack() as stack:
                manager = MagicMock()
                manager.dict.side_effect = lambda **kwargs: dict(kwargs, worker_failed=reported)
                stack.enter_context(patch('main.parse_arguments', return_value=arguments()))
                stack.enter_context(patch('main.create_benchmark'))
                stack.enter_context(patch('main.mp.set_start_method'))
                stack.enter_context(patch('main.mp.Manager', return_value=manager))
                processes = stack.enter_context(patch('main.mp.Process'))
                processes.return_value.is_alive.return_value = False
                processes.return_value.name = 'FailedWorker'
                processes.return_value.exitcode = 0 if reported else 7
                stack.enter_context(patch('main.signal.signal'))
                stack.enter_context(patch('main.time.sleep'))
                with self.assertRaisesRegex(RuntimeError, 'worker|Worker'):
                    main.main()


if __name__ == '__main__':
    unittest.main()
