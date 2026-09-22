import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from src.schedulers.boundary_scheduler import evaluation_state, run_boundary

class BoundaryTests(unittest.TestCase):
    def test_rng_and_mixed_module_modes_restored_even_on_failure(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        model.train()
        model[1].eval()
        random.seed(12); np.random.seed(12); torch.manual_seed(12)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.seed(12); np.random.seed(12); torch.manual_seed(12)
        before = {k: v.clone() for k,v in model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with evaluation_state(model):
                model.eval()
                random.random(); np.random.rand(); torch.rand(50)
                raise RuntimeError('test failure')
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))
        self.assertTrue(model.training)
        self.assertTrue(model[0].training)
        self.assertFalse(model[1].training)
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))

    def test_timeout_cleans_up_worker(self):
        proc = MagicMock()
        proc.is_alive.return_value = True
        args = SimpleNamespace(global_scheduler_mode="boundary", enable_dynamic_reconfiguration=False,
                               enable_double_buffer=False, max_runtime=0)
        with patch('torch.multiprocessing.Process', return_value=proc):
            with self.assertRaises(TimeoutError):
                run_boundary(args, 'cpu')
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_child_failure_reaches_parent(self):
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 1
        args = SimpleNamespace(global_scheduler_mode="boundary", enable_dynamic_reconfiguration=False,
                               enable_double_buffer=False, max_runtime=1)
        with patch('torch.multiprocessing.Process', return_value=proc):
            with self.assertRaisesRegex(RuntimeError, 'exit code 1'):
                run_boundary(args, 'cpu')

if __name__ == '__main__':
    unittest.main()
