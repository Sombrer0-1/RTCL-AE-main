import unittest
from src.schedulers.freshness_scheduler import FreshnessPolicy

class FreshnessTests(unittest.TestCase):
    def test_no_duplicate_and_final_drain(self):
        p = FreshnessPolicy(0)
        self.assertFalse(p.ready(100, 0, True))
        self.assertTrue(p.ready(1, 1, True))
        p.observe(4, 1, 3)
        self.assertFalse(p.ready(100, 1, False))
        self.assertTrue(p.ready(4, 2, False))

    def test_cost_smoothing_and_lag_override(self):
        p = FreshnessPolicy(0)
        p.observe(3, 1, 3)
        self.assertFalse(p.ready(4, 2, True))
        self.assertTrue(p.ready(4, 3, True))
        self.assertFalse(p.ready(3.9, 3, True))
        self.assertTrue(p.ready(6, 2, True))
        p.observe(10, 3, 7)
        self.assertEqual(p.interval, 4)
        p.observe(100, 4, 100)
        self.assertEqual(p.interval, 5)

    def test_bad_observation(self):
        with self.assertRaises(ValueError):
            FreshnessPolicy(0, min_interval=0)
        with self.assertRaises(ValueError):
            FreshnessPolicy(0).observe(1, 0, 1)


class FreshnessTraceTests(unittest.TestCase):
    def test_slow_updates_wait_at_most_max_interval_after_completion(self):
        p = FreshnessPolicy(0)
        p.observe(10, 1, 100)  # expensive prior evaluation clamps at 5 s
        self.assertFalse(p.ready(14.9, 2, True))
        self.assertTrue(p.ready(15, 2, True))

    def test_final_update_arriving_during_eval_is_drained(self):
        p = FreshnessPolicy(0)
        self.assertTrue(p.ready(1, 1, True))
        p.observe(4, 1, 3)
        self.assertTrue(p.ready(4, 10, False))
        p.observe(7, 10, 3)
        self.assertFalse(p.ready(7, 10, False))

class InferenceBatchTests(unittest.TestCase):
    def test_probe_deadband_backoff_and_bounds(self):
        from src.schedulers.freshness_scheduler import InferenceBatchPolicy
        p = InferenceBatchPolicy(16)
        self.assertEqual(p.observe(2), 32)
        self.assertEqual(p.observe(6), 32)
        self.assertEqual(p.observe(11), 16)
        self.assertEqual(p.observe(11), 16)
        for _ in range(10):
            p.observe(1)
        self.assertEqual(p.batch, 256)
        self.assertEqual(p.observe(float('nan')), 256)
        self.assertEqual(p.observe(0), 256)

if __name__ == '__main__':
    unittest.main()
