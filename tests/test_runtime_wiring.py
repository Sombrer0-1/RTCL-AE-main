from contextlib import ExitStack, redirect_stderr
import io
import json
import os
from pathlib import Path
import multiprocessing as mp
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import TensorDataset
import main
from avalanche.benchmarks import nc_benchmark
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, SupervisedPlugin

from src.config import parse_arguments
from src.schedulers.adaptocl_scheduler import AdaptOCLScheduler
from src.schedulers.dynamic_config import DynamicConfigController
from src.schedulers.recl_scheduler import RECLScheduler
from src.schedulers.timeline_scheduler import GlobalTimelineScheduler, global_scheduler_worker
from src.utils import signal_handlers
from src.workers.config_worker import dynamic_config_worker
from src.workers.eval_worker import eval_worker, interruptible_eval
from src.workers.train_worker import create_cl_strategy, create_training_plugins


def arguments(*extra):
    with patch('sys.argv', ['test', '--benchmark', 'split_cifar10', *extra]):
        return parse_arguments()


class Counter(SupervisedPlugin):
    def __init__(self):
        self.iterations = 0

    def after_training_iteration(self, strategy, **kwargs):
        self.iterations += 1


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.output(x)


class RuntimeWiringTests(unittest.TestCase):
    def tearDown(self):
        main.runtime_state.reset_globals()

    def test_replay_has_one_buffer_and_naive_executes_supplied_callbacks(self):
        x, y = torch.randn(8, 2), torch.arange(8) % 2
        benchmark = nc_benchmark(TensorDataset(x, y), TensorDataset(x, y),
            n_experiences=2, task_labels=False, shuffle=False)
        for algorithm in ('naive', 'replay'):
            with self.subTest(algorithm=algorithm):
                args = arguments('--algorithm', algorithm, '--training_bs', '2', '--mem_size', '8')
                counter = Counter()
                model = Classifier()
                strategy = create_cl_strategy(args, model, torch.optim.Adam(model.parameters()),
                    torch.nn.CrossEntropyLoss(), torch.device('cpu'),
                    EvaluationPlugin(loggers=[]), create_training_plugins(args) + [counter])
                buffers = [p for p in strategy.plugins if isinstance(p, ReplayPlugin)]
                self.assertEqual(len(buffers), int(algorithm == 'replay'))
                for experience in benchmark.train_stream:
                    strategy.train(experience)
                self.assertEqual(counter.iterations, strategy.clock.train_iterations)
                self.assertEqual(counter.iterations, 4)
                if buffers:
                    self.assertEqual(len(buffers[0].storage_policy.buffer), 8)

    def test_parent_and_worker_share_cifar100_subset_and_resize(self):
        from src.workers.train_worker import create_benchmark
        args = arguments('--benchmark', 'split_cifar100', '--model', 'resnet50')
        with patch('avalanche.benchmarks.classic.SplitCIFAR100') as constructor:
            main.create_benchmark(args)
            parent = constructor.call_args.kwargs
            create_benchmark(args)
            worker = constructor.call_args.kwargs
        self.assertEqual(parent['fixed_class_order'], list(range(10)))
        self.assertEqual(worker['fixed_class_order'], parent['fixed_class_order'])
        self.assertEqual(parent['train_transform'].transforms[0].size, (224, 224))

    def run_main_setup(self, args):
        manager = MagicMock()
        manager.dict.side_effect = dict
        with ExitStack() as stack:
            stack.enter_context(patch('main.parse_arguments', return_value=args))
            stack.enter_context(patch('main.create_benchmark'))
            stack.enter_context(patch('main.mp.set_start_method'))
            stack.enter_context(patch('main.mp.Manager', return_value=manager))
            processes = stack.enter_context(patch('main.mp.Process'))
            processes.return_value.is_alive.return_value = False
            processes.return_value.exitcode = 0
            stack.enter_context(patch('main.signal.signal'))
            stack.enter_context(patch('main.time.sleep'))
            stack.enter_context(patch('main.timestamp_logger_worker'))
            stack.enter_context(patch('main.analyze_batch_size_changes'))
            main.main()
        calls = {call.kwargs['name']: call.kwargs for call in processes.call_args_list}
        return calls

    def test_aocl_cli_reaches_existing_controls_and_preserves_batch64(self):
        defaults = AdaptOCLScheduler(1)
        for extra in ([], ['--enable_dynamic_reconfiguration'],
                      ['--uam_alpha', '.7', '--uam_delta_acc', '.2',
                          '--latency_budget', '3', '--adaptocl_focus', 'continuous_eval']):
            args = arguments('--global_scheduler_mode', 'adaptocl', *extra)
            calls = self.run_main_setup(args)
            scheduler = calls['AdaptOCLScheduler']['args'][0]
            self.assertEqual(scheduler.la_priority_percent, args.uam_alpha)
            self.assertEqual(scheduler.aa_accuracy_threshold, args.uam_delta_acc)
            self.assertEqual(scheduler.latency_budget, args.latency_budget)
            self.assertEqual(scheduler.operation_focus, args.adaptocl_focus)
            self.assertEqual(calls['TrainWorker']['args'][5]['train_batch_size'], 64)
            if not extra:
                for key in ('la_priority_percent', 'aa_accuracy_threshold', 'latency_budget'):
                    self.assertEqual(getattr(scheduler, key), getattr(defaults, key))

    def test_ekya_options_and_dynamic_config_file_are_forwarded(self):
        args = arguments('--global_scheduler_mode', 'ekya', '--ekya_min_resource', '.2',
            '--ekya_max_resource', '.8', '--ekya_utility_threshold', '2',
            '--enable_dynamic_reconfiguration', '--config_file', 'schedule.json')
        calls = self.run_main_setup(args)
        scheduler = calls['GlobalScheduler']['args'][0].ekya_scheduler
        self.assertEqual((scheduler.min_resource, scheduler.max_resource,
                          scheduler.utility_threshold), (.2, .8, 2))
        self.assertEqual(calls['DynamicConfigWorker']['kwargs']['config_file'], 'schedule.json')
        self.assertEqual(calls['TrainWorker']['args'][5]['train_batch_size'], args.training_bs)

    def test_unimplemented_controls_fail_explicitly(self):
        for option in ('--switch_penalty', '--ekya_profiling_interval'):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    arguments(option, '1')
                self.assertEqual(error.exception.code, 2)

    def test_dynamic_updates_reach_workers_without_reverting_scheduler_changes(self):
        shared = {}
        controller = DynamicConfigController(shared, arguments())
        self.assertEqual(shared['train_batch_size'], 16)
        controller.update_config({'train_batch_size': 64, 'eval_batch_size': 32})
        self.assertEqual(shared['train_batch_size'], 64)
        self.assertEqual(shared['eval_batch_size'], 32)
        shared['train_batch_size'] = 48  # a later AOCL boundary update
        controller.update_config({'timeslice': .2})
        self.assertEqual(controller.get_current_config()['train_batch_size'], 48)
        self.assertEqual(shared['config']['train_batch_size'], 48)

    def test_config_file_timeslice_is_applied(self):
        shared = {}
        controller = DynamicConfigController(shared, arguments())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'schedule.json'
            path.write_text(json.dumps({'timeslice': .07, 'batches': [
                {'index': 0, 'train_batch_size': 32, 'eval_batch_size': 8, 'duration': 0}]}))
            with patch('src.workers.config_worker.safe_kill'), patch('src.workers.config_worker.log_event'):
                dynamic_config_worker(controller, 1, 2, .5, str(path))
        self.assertEqual((shared['train_batch_size'], shared['eval_batch_size'],
                          shared['timeslice']), (32, 8, .07))

    def test_eval_worker_applies_dynamic_batch_at_cycle_boundary(self):
        args = arguments('--eval_cycles', '1', '--global_scheduler_mode', 'fully_parallel')
        shared = {'train_process_active': False, 'all_experiences_completed': True,
                  'total_experiences': 1, 'published_version': 1}
        controller = DynamicConfigController(shared, args)
        controller.update_config({'eval_batch_size': 4})
        model = Classifier()
        model.output = torch.nn.Linear(2, 10)
        strategy = SimpleNamespace(model=model, device=torch.device('cpu'), eval_mb_size=16)
        fake_result = dict(streaming_accuracy=1., experience_accuracies={}, total_samples=4,
                           batch_p95_ms=1., batch_p99_ms=1.)
        with ExitStack() as stack:
            stack.enter_context(patch('src.workers.eval_worker.initialize_model', return_value=model))
            stack.enter_context(patch('src.workers.eval_worker.create_benchmark',
                                     return_value=SimpleNamespace(test_stream=[object()])))
            stack.enter_context(patch('src.workers.eval_worker.create_cl_strategy', return_value=strategy))
            stack.enter_context(patch('src.workers.eval_worker.CSVLogger'))
            stack.enter_context(patch('src.workers.eval_worker.signal.signal'))
            stack.enter_context(patch('torch.load', return_value={
                'model_state_dict': model.state_dict(), 'model_version': 1}))
            evaluate = stack.enter_context(patch('src.workers.eval_worker.interruptible_eval',
                                                 return_value=fake_result))
            from threading import Lock
            eval_worker(args, torch.device('cpu'), None, Lock(), 'unused', shared, controller)
        evaluate.assert_called_once()
        self.assertEqual(strategy.eval_mb_size, 4)

    def test_timeline_uses_updated_timeslice(self):
        scheduler = GlobalTimelineScheduler(1., mode='fully_parallel')
        with patch('src.schedulers.timeline_scheduler.safe_kill', side_effect=[True]*4 + [False]*2), \
             patch('src.schedulers.timeline_scheduler.time.sleep') as sleep:
            global_scheduler_worker(scheduler, 1, 2, {'timeslice': .125})
        sleep.assert_called_once_with(.125)

    def test_recl_receives_manager_feedback_and_consumes_it_once(self):
        model = Classifier()
        strategy = SimpleNamespace(model=model, device=torch.device('cpu'), eval_mb_size=2)
        dataset = TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long),
                                torch.zeros(4, dtype=torch.long))
        stream = [SimpleNamespace(dataset=dataset, current_experience=0)]
        args = arguments('--global_scheduler_mode', 'recl_sched', '--slo_ms', '0')
        with mp.get_context('spawn').Manager() as manager:
            shared = manager.dict(eval_latency=manager.dict(), eval_slo_ms=0)
            result = interruptible_eval(strategy, stream, None, shared, 1, args)
            self.assertEqual(result['total_samples'], 4)
            self.assertGreater(shared['eval_latency'][str(os.getpid())], 0)
            scheduler = RECLScheduler(1, os.getpid(), args, shared)
            scheduler._update_miss_ratio_and_weight()
            self.assertAlmostEqual(scheduler.current_eval_weight, .5)
            scheduler._update_miss_ratio_and_weight()
            self.assertEqual(scheduler.slo_miss_history, [1])
            self.assertAlmostEqual(scheduler.current_eval_weight, .5)

    def test_signal_handlers_update_the_state_read_by_main(self):
        signal_handlers.signal_handler(signal.SIGTERM, None)
        self.assertTrue(main.runtime_state.TERMINATE_SIGNAL)
        signal_handlers.request_config_update_handler(signal.SIGUSR1, None)
        self.assertTrue(main.runtime_state.CONFIG_UPDATE_REQUESTED)


if __name__ == '__main__':
    unittest.main()
