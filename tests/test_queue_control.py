import unittest
import numpy as np
from src.schedulers.queue_control import QueueController, TinyResidual, CostModel
from src.schedulers.stream_scheduler import make_trace, request_metrics


class QueueTests(unittest.TestCase):
    def test_goodput_uses_last_response_and_includes_late_requests(self):
        records = [{'latency':.01, 'completed':1, 'on_time':True},
                   {'latency':.5, 'completed':2, 'on_time':False}]
        metric = request_metrics(records)
        self.assertEqual(metric['stream_seconds'], 2)
        self.assertEqual(metric['goodput'], .5)
        self.assertEqual(metric['miss_rate'], .5)
        self.assertGreater(metric['p99_ms'], 400)

    def test_trace_is_unique_causal_input_and_reproducible(self):
        for pattern in ('steady', 'bursty'):
            a = make_trace(101, 6000, 30, pattern, 10000)
            self.assertEqual(a, make_trace(101, 6000, 30, pattern, 10000))
            self.assertEqual(len({r['sample'] for r in a}), 6000)
            t = [r['arrival'] for r in a]
            self.assertEqual(t, sorted(t))
            self.assertTrue(0 <= t[0] <= t[-1] < 30)
        with self.assertRaises(ValueError):
            make_trace(1, 10001, 30, 'steady', 10000)

    def test_no_future_information(self):
        for policy in ('fifo', 'slack', 'debt', 'neural'):
            with self.assertRaises(ValueError):
                QueueController(policy, .1).choose(0, [.001])

    def test_urgent_request_blocks_training_even_with_neural_bias(self):
        for policy in ('slack', 'debt', 'neural'):
            c = QueueController(policy, .1)
            c.debt = 100
            if c.net:
                c.net.b2 = -100
                c.observations = 100
            self.assertGreater(c.choose(.09, [0, .01]), 0)
            self.assertGreater(c.choose(1, [0, .01]), 0)

    def test_all_overdue_requests_drain_in_fifo_batches(self):
        for policy in ('fifo', 'slack', 'debt', 'neural', 'boundary'):
            c = QueueController(policy, .1)
            pending = [0]*101
            processed = 0
            while pending:
                b = c.choose(2, pending, allow_train=False)
                self.assertTrue(1 <= b <= min(32, len(pending)))
                del pending[:b]
                processed += b
            self.assertEqual(processed, 101)

    def test_service_debt_accumulates_and_repays(self):
        c = QueueController('debt', .1)
        c.observe('infer', 4, 1)
        self.assertAlmostEqual(c.debt, .7)
        c.observe('train', 1, 1)
        self.assertAlmostEqual(c.debt, .4)
        c.observe('train', 1, 2)
        self.assertEqual(c.debt, 0)

    def test_residual_is_bounded_and_does_not_touch_global_rng(self):
        np.random.seed(123)
        expected = np.random.rand()
        np.random.seed(123)
        net = TinyResidual()
        x = np.array([1, .2, .5, .4, 3, 3])
        for _ in range(800):
            net.observe(x, 100)
        self.assertEqual(net.updates, 100)
        self.assertTrue(0 < net.residual(x) <= .25)
        self.assertEqual(np.random.rand(), expected)
        self.assertEqual(net.w1.size+net.b1.size+net.w2.size+1, 65)

    def test_cost_bounds_track_variability_and_reject_invalid_costs(self):
        c = CostModel()
        for dt in [.01, .01, .01, .05]:
            c.observe('train', 1, dt)
        self.assertGreater(c.bound('train', 1), c.mean('train', 1))
        for dt in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                c.observe('train', 1, dt)

if __name__ == '__main__':
    unittest.main()
