import unittest
import numpy as np

from src.schedulers.predictive_batching import OnlineServiceNetwork, PredictiveBatchController


class PredictiveTests(unittest.TestCase):
    def test_released_arrivals_only_and_empty_window_decay(self):
        c = PredictiveBatchController('predictive', .1)
        with self.assertRaises(ValueError):
            c.admit([.01], 0)
        c.admit([.01, .02, .03], .04)
        self.assertEqual(c.release_count, 3)
        self.assertGreater(c.rate, 0)
        c.admit([], .5)
        self.assertEqual(c.rate, 0)
        with self.assertRaises(ValueError):
            c.choose(0, [.001])

    def test_learning_cannot_override_guard_or_leave_late_requests(self):
        c = PredictiveBatchController('predictive_neural', .033)
        c.net.seen = 100
        c.net.b2 = -100
        self.assertEqual(c.choose(.04, [0]*40, stage='preparation'), 32)
        self.assertEqual(c.choose(.02, [0], stage='preparation'), 1)
        self.assertEqual(c.choose(0, [0]*4, allow_train=False), 4)

    def test_amortizes_preparation_without_waiting_until_deadline(self):
        c = PredictiveBatchController('predictive', .1)
        for _ in range(64):
            c.observe('infer', 4, .002)
            c.observe('train', 1, .012)
        c.admit([i*.002 for i in range(50)], .1)
        self.assertEqual(c.choose(.1, [.1], stage='preparation'), 0)
        self.assertEqual(c.choose(.13, [.1], stage='preparation'), 1)

    def test_network_learns_observed_cost_without_changing_global_rng(self):
        np.random.seed(42)
        expected = np.random.rand()
        np.random.seed(42)
        net = OnlineServiceNetwork()
        x = np.array([0, .25, .0625, .6, .1, .02, .5, .25, 1])
        before = abs(net.residual(x)-.6)
        for _ in range(800):
            net.observe(x, .6)
        self.assertLess(abs(net.residual(x)-.6), before*.2)
        self.assertEqual(np.random.rand(), expected)
        self.assertEqual(net.w1.size+net.b1.size+net.w2.size+1, net.parameters)
        self.assertTrue(abs(net.residual(x*100)) <= np.log(4))

    def test_supported_preparation_paths(self):
        from src.config import parse_arguments
        from unittest.mock import patch
        for policy in ('stream_predictive', 'stream_predictive_neural'):
            for preparation in ('certified', 'cooperative'):
                with patch('sys.argv', ['test', '--global_scheduler_mode', policy,
                                       '--host_preparation', preparation]):
                    self.assertEqual(parse_arguments().global_scheduler_mode, policy)

    def test_learned_diagnostics_are_serializable_after_decisions(self):
        import json
        c = PredictiveBatchController('predictive_neural', .1)
        for _ in range(64):
            x = c.features('infer', 4, 4)
            base = c.costs.mean('infer', 4)
            c.observe('infer', 4, .002, x, base)
        c.admit([i*.002 for i in range(50)], .1)
        c.choose(.1, [.1], stage='preparation')
        self.assertIsInstance(c.neural_changed, int)
        json.dumps(c.diagnostics())


if __name__ == '__main__':
    unittest.main()
