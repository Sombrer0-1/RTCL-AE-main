import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
from collections import deque
from types import SimpleNamespace

from src.schedulers.train_start import blocking_risk, fifo_misses, start_wait, training_quota, uniform_start_wait


class TrainStartTests(unittest.TestCase):
    def test_vulnerable_window_is_shorter_than_compute(self):
        infer = lambda b: .005
        self.assertEqual(blocking_risk([.031, .032, .050], 0, 0, .060, .033, infer), 1)
        self.assertEqual(blocking_risk([.050], 0, 0, .060, .033, infer), 0)

    def test_burst_queue_includes_subsequent_batches(self):
        infer = lambda b: .005
        self.assertEqual(fifo_misses([.040]*65, .060, .033, infer), 1)
        self.assertEqual(blocking_risk([.040]*65, 0, 0, .060, .033, infer), 1)

    def test_arrivals_during_inference_are_admitted(self):
        self.assertEqual(fifo_misses([0, .004, .005], 0, .006, lambda b: .005), 0)

    def test_reactive_and_immediate_do_not_read_future(self):
        class Unreadable:
            def __getitem__(self, key):
                raise AssertionError('causal control read future')
        self.assertEqual(start_wait('immediate', 0, 0, .06, .033,
                                    lambda b: .005, Unreadable()), (0, 0))
        self.assertAlmostEqual(start_wait('phase', .1, 0, .06, .033,
                                         lambda b: .005, Unreadable())[0], .1)
        wait, risk = start_wait('reactive', 0, 0, .06, .033, lambda b: .005, Unreadable())
        self.assertAlmostEqual(wait, .032)
        self.assertEqual(risk, 0)
        self.assertEqual(start_wait('reactive', .040, 0, .06, .033, lambda b: .005), (0, 0))

    def test_oracle_waits_only_when_training_adds_misses(self):
        self.assertEqual(start_wait('oracle', 0, 0, .06, .033, lambda b: .005, [.010]), (.010, 1))
        self.assertEqual(start_wait('oracle', 0, 0, .06, .033, lambda b: .005, [.050]), (0, 0))

    def test_quota_releases_same_work_and_carries_late_updates(self):
        release, slack = training_quota(.2, 2, 8, 30, .06, .08)
        self.assertEqual(release, 0)
        self.assertAlmostEqual(slack, .34)
        self.assertAlmostEqual(training_quota(.7, 8, 8, 30, .06, .08)[0], .3)
        self.assertEqual(training_quota(1.9, 8, 8, 30, .06, .08), (0, 0))
        self.assertEqual(training_quota(30, 240, 8, 30, .06, .08), (0, float('inf')))

    def test_uniform_pacing_needs_only_work_progress_and_clock(self):
        self.assertAlmostEqual(uniform_start_wait(0, 0, 8, .06), .065)
        self.assertEqual(uniform_start_wait(.22, 1, 8, .06), 0)

    def test_phase_control_anticipates_burst_without_exact_arrivals(self):
        cost = lambda b: .005
        self.assertAlmostEqual(start_wait('phase', .99, 0, .06, .033, cost)[0], .21)
        self.assertEqual(start_wait('phase', .5, 0, .06, .033, cost), (0, 0))
        self.assertEqual(start_wait('phase', .1, 0, .01, .1, cost), (0, 0))

    def test_window_planner_moves_equal_training_work_and_replay_matches(self):
        from src.schedulers.window_planner import FixedPlant, plan_window, replay_schedule
        data = dict(trace=[{'arrival': t} for t in [.01, .02, .03, .08, .09, .1, .2]],
                    compute=[.06], prep=[.005], infer_intercept=.003, infer_slope=.0001)
        plant = FixedPlant(data)
        result, feasible, nodes = plan_window(plant, 0, 0, 0, 2)
        self.assertTrue(feasible)
        self.assertEqual(len(result[5]), 2)
        self.assertGreater(nodes, 0)
        missed, latency, cursor, completed = replay_schedule(plant, result[5], 1)
        self.assertEqual(missed, result[2])
        self.assertAlmostEqual(latency, result[4])
        self.assertEqual(cursor, 7)
        self.assertTrue(all(t <= 1 for t in completed))
        first, j, m, l, done = plant.train(0, 0)
        second, j, mm, ll, done = plant.train(first, j)
        self.assertLess(missed, m+mm)

    def test_recent_planner_changes_only_its_point_estimate_history(self):
        import importlib.util
        from pathlib import Path
        import numpy as np
        import src.schedulers.stream_scheduler as stream
        base = stream.StreamSchedulerPlugin
        path = Path(__file__).resolve().parents[1]/'experiments/optimization/arrival_oracle/recent_start.py'
        spec = importlib.util.spec_from_file_location('_recent_start_test', path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            stream.StreamSchedulerPlugin = base
        plugin = module.RecentPlanner.__new__(module.RecentPlanner)
        original = deque([.026]*43+[.053]*21, maxlen=64)
        plugin.compute_costs = original
        observed = []
        def point_estimate(instance, elapsed):
            observed.append(float(np.median(instance.compute_costs)))
            return .123
        with patch.object(base, '_planned_wait', point_estimate):
            self.assertEqual(plugin._planned_wait(0), .123)
        self.assertAlmostEqual(observed[0], .053)
        self.assertIs(plugin.compute_costs, original)
        self.assertEqual(len(plugin.compute_costs), 64)

    def test_timing_simulation_preserves_quotas_and_never_invents_accuracy(self):
        from experiments.optimization.arrival_oracle.simulate_quota import simulate
        data = dict(trace=[{'arrival': .03*i} for i in range(1, 20)],
                    compute=[.01]*60, prep=[.005]*60, infer_intercept=.002, infer_slope=0)
        for policy in ['immediate', 'uniform', 'reactive', 'phase', 'oracle']:
            result, requests, training = simulate(data, policy, 2, warmup=20, horizon=2)
            self.assertEqual(result['updates_by_stream_end'], 24)
            self.assertEqual(result['train_updates'], 60)
            self.assertEqual(len(requests), 19)
            self.assertTrue(all('correct' not in r for r in requests))

    def test_parser_requires_explicit_supported_experiment_path(self):
        from src.config import parse_arguments
        base = ['test', '--benchmark', 'split_cifar10', '--global_scheduler_mode', 'stream_fifo',
                '--host_preparation', 'cooperative', '--stream_start_policy', 'oracle']
        with patch('sys.argv', base):
            self.assertEqual(parse_arguments().stream_start_policy, 'oracle')
        for argv in [base[:-4]+base[-2:], base+['--stream_start_max_wait_ms', '-1']]:
            with patch('sys.argv', argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_arguments()

    def test_wait_budget_preserves_progress_and_unlimited_drains_finite_trace(self):
        from src.schedulers.stream_scheduler import StreamSchedulerPlugin
        for budget in (.1, 0):
            clock = [0.0]
            plugin = StreamSchedulerPlugin.__new__(StreamSchedulerPlugin)
            plugin.trace = [{'arrival': .01*i} for i in range(1, 101)]
            plugin.arrival_times = [r['arrival'] for r in plugin.trace]
            plugin.origin = 0
            plugin.next_arrival = 0
            plugin.pending = []
            plugin.records = []
            plugin.compute_costs = deque([.06])
            plugin.start_policy = 'oracle'
            plugin.start_budget = budget
            plugin.train_quota = 0
            plugin.steps = 20
            plugin.start_records = []
            plugin.idle_seconds = plugin.control_seconds = plugin.decisions = 0
            plugin.controller = SimpleNamespace(slo=.033, max_batch=32,
                costs=SimpleNamespace(bound=lambda kind, batch: .005),
                choose=lambda now, arrivals, allow: min(len(arrivals), 32))
            served = []
            def infer(strategy, batch):
                clock[0] += .005
                served.extend(plugin.pending[:batch])
                del plugin.pending[:batch]
                plugin.records.append({'completed': clock[0]})
            plugin._infer = infer
            def sleep(seconds):
                clock[0] += seconds
            with patch('src.schedulers.stream_scheduler.time.perf_counter', lambda: clock[0]), \
                    patch('src.schedulers.stream_scheduler.time.sleep', sleep):
                plugin._drive(None, allow_train=True)
            self.assertEqual(served, list(range(len(served))))
            if budget:
                self.assertLess(plugin.next_arrival, len(plugin.trace))
                self.assertAlmostEqual(plugin.idle_seconds, budget)
                self.assertTrue(plugin.start_records[-1]['budget_exhausted'])
            else:
                self.assertEqual(len(served), len(plugin.trace))


if __name__ == '__main__':
    unittest.main()
