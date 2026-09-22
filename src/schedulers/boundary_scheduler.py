"""Cooperative train/eval scheduling at completed experience boundaries.

Evaluation uses the committed training model in the same CUDA context. This
eliminates snapshot IPC and evaluates every version, but requests must wait for
an experience boundary: it is not a replacement for continuous inference.
"""
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def evaluation_state(model):
    """Keep evaluation from changing training RNG streams or module modes."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training


def boundary_worker(args, device):
    from src.workers.train_worker import train_worker
    state = {'train_process_active': True, 'all_experiences_completed': False,
             'TERMINATE_SIGNAL': False, 'CONFIG_UPDATE_REQUESTED': False,
             'train_batch_size': args.training_bs, 'eval_batch_size': args.eval_bs,
             'timeslice': args.timeslice}
    train_worker(args, device, None, None, None, state)
    if state.get('worker_failed', False) or not state['all_experiences_completed']:
        raise RuntimeError('Boundary worker did not complete all training experiences')


def run_boundary(args, device):
    """One compute worker; parent enforces timeout without a Manager/scheduler."""
    if args.enable_dynamic_reconfiguration or args.enable_double_buffer:
        raise ValueError('Boundary scheduling owns configuration and does not use double buffers')
    if args.global_scheduler_mode == "boundary_cached" and (args.benchmark != "split_cifar10" or args.semseg):
        raise ValueError('Resident evaluation cache is validated only for deterministic SplitCIFAR10 evaluation')
    import time
    import torch.multiprocessing as mp
    from src.utils import signal_handlers
    worker = mp.Process(target=boundary_worker, args=(args, device), name='BoundaryWorker')
    worker.start()
    try:
        deadline = time.monotonic() + args.max_runtime
        while worker.is_alive():
            if signal_handlers.TERMINATE_SIGNAL:
                raise KeyboardInterrupt('Boundary scheduling interrupted')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'Boundary worker exceeded {args.max_runtime}s')
            worker.join(timeout=min(0.1, remaining))
        if worker.exitcode != 0:
            raise RuntimeError(f'Boundary worker failed: exit code {worker.exitcode}')
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
