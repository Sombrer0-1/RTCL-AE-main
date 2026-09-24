"""Opt-in structural backend certificate and cooperative host preparation.

No dataset samples are cached. Unknown structures retain Avalanche's backend
selection. Only synchronous, single-process CIFAR10 preparation is supported.
"""
from types import MethodType
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import CIFAR10


def certifies_plain_cifar(datasets):
    # Imports stay local: Avalanche's package initialization is order-sensitive.
    from avalanche.benchmarks.utils.data import AvalancheDataset
    from avalanche.benchmarks.utils.flat_data import FlatData
    pending = list(datasets)
    seen = set()
    leaves = 0
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if hasattr(node, 'ffcv_info'):
            return False
        if isinstance(node, AvalancheDataset):
            pending.append(node._flat_data)
        elif isinstance(node, FlatData):
            pending.extend(node._datasets)
        elif type(node) is ConcatDataset:
            pending.extend(node.datasets)
        elif type(node) is Subset:
            pending.append(node.dataset)
        elif type(node) is CIFAR10:
            leaves += 1
        else:
            return False
    return leaves > 0


class ServiceDataset(Dataset):
    def __init__(self, dataset, service):
        self.dataset = dataset
        self.service = service
        self.reads = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        self.reads += 1
        if self.reads % 8 == 0:
            self.service()
        return item


def install_preparation(loader, service, stats):
    """Patch this loader instance, preserving sampler construction and RNG draws."""
    from avalanche.benchmarks.utils.data_loader import MultiDatasetDataLoader, MultiDatasetSampler
    from avalanche.benchmarks.utils.data_loader import _make_data_loader_with_batched_sampler
    if getattr(loader, '_rtcl_preparation_installed', False):
        return
    if not isinstance(loader, MultiDatasetDataLoader):
        raise ValueError('Unsupported preparation loader')
    if loader.loader_kwargs.get('num_workers', 0) != 0:
        raise ValueError('Cooperative preparation requires num_workers=0')
    original = loader._get_loader

    def get_loader(instance):
        if not certifies_plain_cifar(instance.datasets):
            stats['fallback_epochs'] += 1
            return original()
        stats['certified_epochs'] += 1
        samplers = instance._create_samplers(instance.datasets, instance.batch_sizes,
                                             instance.distributed_sampling, instance.loader_kwargs)
        sampler = MultiDatasetSampler(instance.datasets, samplers,
            termination_dataset_idx=instance.termination_dataset,
            oversample_small_datasets=instance.oversample_small_datasets,
            never_ending=instance.never_ending)
        dataset = ConcatDataset(instance.datasets)
        if service is not None:
            dataset = ServiceDataset(dataset, service)
        return _make_data_loader_with_batched_sampler(dataset, sampler, instance.loader_kwargs)

    loader._get_loader = MethodType(get_loader, loader)
    loader._rtcl_preparation_installed = True
