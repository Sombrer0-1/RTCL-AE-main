import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.schedulers.evaluation_cache import EvaluationCache

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        self.y = torch.arange(10)
        self.loader = DataLoader(TensorDataset(self.x, self.y, self.y), batch_size=4)

    def test_rebatch_without_reading_source(self):
        cache = EvaluationCache(200)
        first = list(cache.batches(0, self.loader, 4, 'cpu'))
        self.assertTrue(torch.equal(torch.cat([b[0] for b in first]), self.x))
        second = list(cache.batches(0, None, 3, 'cpu'))
        self.assertEqual([len(b[1]) for b in second], [3, 3, 3, 1])
        self.assertTrue(torch.equal(torch.cat([b[0] for b in second]), self.x))
        self.assertTrue(torch.equal(torch.cat([b[1] for b in second]), self.y))
        self.assertEqual(cache.used_bytes, 200)

    def test_budget_fallback_and_interrupted_fill(self):
        cache = EvaluationCache(199)
        out = list(cache.batches(0, self.loader, 4, 'cpu'))
        self.assertEqual(sum(len(b[1]) for b in out), 10)
        self.assertEqual(cache.entries, {})
        cache = EvaluationCache(200)
        stream = cache.batches(0, self.loader, 4, 'cpu')
        next(stream); stream.close()
        self.assertEqual(cache.entries, {})
        self.assertEqual(cache.used_bytes, 0)

if __name__ == '__main__':
    unittest.main()
