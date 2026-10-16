import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset
from avalanche.benchmarks.classic import SplitCIFAR10  # fork import order

from src.schedulers.full_eval_scheduler import FullEvaluationPlugin
from src.workers.eval_worker import interruptible_eval, iter_interruptible_eval


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.output(x)


class FullEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(semseg=False, debug=False, eval_bs=2,
            host_preparation='cooperative', global_scheduler_mode='r4_full_eval',
            record_eval_batches=False, eval_cycles=0)
        torch.manual_seed(3)
        self.strategy = SimpleNamespace(model=Classifier(), device=torch.device('cpu'),
                                        eval_mb_size=2)
        x = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [0., 2.]])
        self.stream = [SimpleNamespace(current_experience=0,
            dataset=TensorDataset(x, torch.tensor([0, 1, 0, 1]), torch.zeros(4, dtype=torch.long)))]

    def test_incremental_evaluation_matches_full_pass_and_restores_grad(self):
        expected = interruptible_eval(self.strategy, self.stream, None, {}, 1, self.args)
        iterator = iter_interruptible_eval(self.strategy, self.stream, None, {}, 1, self.args)
        batches = []
        while True:
            try:
                batches.append(next(iterator))
                self.assertTrue(torch.is_grad_enabled())
                self.strategy.model(torch.ones(1, 2)).sum().backward()
            except StopIteration as completed:
                actual = completed.value
                break
        self.assertEqual(sum(b['samples'] for b in batches), 4)
        for key in ['total_samples', 'streaming_accuracy', 'experience_accuracies']:
            self.assertEqual(actual[key], expected[key])

    def test_snapshot_stays_fixed_while_training_and_publication_continue(self):
        plugin = FullEvaluationPlugin(self.args, self.stream, {})
        plugin.interval = 0
        self.strategy.model.train()
        self.strategy.model.output.weight.data.copy_(torch.eye(2))
        rng = torch.get_rng_state().clone()
        plugin.publish(self.strategy, 1)  # first half evaluated with correct model
        self.assertTrue(self.strategy.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.strategy.model.output.weight.data.copy_(torch.eye(2).flip(0))
        plugin.publish(self.strategy, 2)  # second half must retain version 1
        plugin._service(self.strategy)
        self.assertEqual(plugin.cycles[0]['version_lower_bound'], 1)
        self.assertEqual(plugin.cycles[0]['accuracy'], 1.)
        self.assertTrue(torch.equal(plugin.latest['output.weight'], torch.eye(2).flip(0)))
        self.assertTrue(self.strategy.model.training)

    def test_fixed_pass_budget_reserves_final_pass_and_drains_all_samples(self):
        self.args.eval_cycles = 3
        plugin = FullEvaluationPlugin(self.args, self.stream, {})
        plugin.interval = 0
        plugin.publish(self.strategy, 1)
        for _ in range(12):
            plugin._service(self.strategy)
        self.assertEqual(len(plugin.cycles), 2)
        self.assertIsNone(plugin.iterator)
        with tempfile.TemporaryDirectory() as temporary:
            original_write = Path.write_text
            def write_here(path, text, *args, **kwargs):
                return original_write(Path(temporary) / path.name, text, *args, **kwargs)
            with patch.object(Path, 'write_text', write_here):
                plugin.finish(self.strategy)
        self.assertEqual(len(plugin.cycles), 3)
        self.assertEqual(sum(row['samples'] for row in plugin.cycles), 12)
        self.assertTrue(plugin.cycles[-1]['final_at_load'])
        self.assertTrue(plugin.cycles[-1]['final_snapshot'])


if __name__ == '__main__':
    unittest.main()
