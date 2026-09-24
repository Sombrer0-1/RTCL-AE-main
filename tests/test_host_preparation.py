import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from avalanche.benchmarks.classic import SplitCIFAR10  # fork import ordering
from avalanche.benchmarks.utils.data_loader import SingleDatasetDataLoader
from avalanche.benchmarks.utils.data import AvalancheDataset
from src.schedulers.host_preparation import certifies_plain_cifar, install_preparation
from src.schedulers.boundary_scheduler import evaluation_state


def dataset():
    d = object.__new__(CIFAR10)
    d.data = np.arange(24*32*32*3, dtype=np.int64).astype(np.uint8).reshape(24,32,32,3)
    d.targets = list(range(24))
    d.transform = ToTensor()
    d.target_transform = None
    return d


class PreparationTests(unittest.TestCase):
    def test_slack_accepts_certificate_but_not_fifo_service_points(self):
        from src.config import parse_arguments
        argv=['test','--global_scheduler_mode','stream_slack','--host_preparation','certified']
        with patch('sys.argv',argv):
            self.assertEqual(parse_arguments().host_preparation,'certified')
        with patch('sys.argv',argv[:-1]+['cooperative']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_arguments()

    def test_certificate_uses_structure_and_rejects_ffcv_and_unknown(self):
        d=dataset()
        wrapped=ConcatDataset([Subset(d,[1,2]),d])
        self.assertTrue(certifies_plain_cifar([wrapped]))
        d.ffcv_info=object()
        self.assertFalse(certifies_plain_cifar([wrapped]))
        self.assertFalse(certifies_plain_cifar([TensorDataset(torch.zeros(3))]))
        self.assertFalse(certifies_plain_cifar([]))

    def test_batches_and_rng_match_original_across_epochs(self):
        d=AvalancheDataset([dataset()])
        def run(cooperative):
            torch.manual_seed(91)
            loader=SingleDatasetDataLoader(d, batch_size=8, shuffle=True, num_workers=0)
            stats={'certified_epochs':0,'fallback_epochs':0}
            calls=[]
            model=torch.nn.Linear(1,1)
            def service():
                calls.append(1)
                with evaluation_state(model):
                    torch.rand(17)
            if cooperative is not None:
                install_preparation(loader, service if cooperative else None, stats)
            batches=[(x.clone(),y.clone()) for _ in range(2) for x,y in loader]
            return batches,torch.get_rng_state(),stats,calls
        expected,rng,_,_=run(None)
        for mode in (False,True):
            actual,actual_rng,stats,calls=run(mode)
            self.assertEqual(stats['certified_epochs'],2)
            for (x,y),(xx,yy) in zip(expected,actual):
                self.assertTrue(torch.equal(x,xx))
                self.assertTrue(torch.equal(y,yy))
            self.assertTrue(torch.equal(rng,actual_rng))
            self.assertEqual(len(calls),6 if mode else 0)

    def test_unknown_backend_falls_back(self):
        loader=SingleDatasetDataLoader(AvalancheDataset([TensorDataset(torch.arange(12))]),batch_size=4)
        stats={'certified_epochs':0,'fallback_epochs':0}
        install_preparation(loader,lambda: self.fail('must not interrupt unknown loader'),stats)
        self.assertEqual(len(list(loader)),3)
        self.assertEqual(stats['fallback_epochs'],1)

if __name__=='__main__':
    unittest.main()
