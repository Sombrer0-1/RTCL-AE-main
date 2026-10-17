import multiprocessing as mp
import signal
import time
import unittest

import psutil
import torch

from src.schedulers.model_scheduler import ModelScheduler


class PausedRead:
    def __init__(self, shared, ready):
        self.shared = shared
        self.ready = ready
        self.paused = False

    def __getitem__(self, key):
        return self.shared[key]

    def get(self, key, default=None):
        value = self.shared.get(key, default)
        if torch.is_tensor(value) and not self.paused:
            self.paused = True
            self.ready.set()
            signal.raise_signal(signal.SIGSTOP)
        return value


def paused_reader(shared, ready, result):
    scheduler = ModelScheduler(shared, use_double_buffer=True)
    scheduler.shared_data = PausedRead(shared, ready)
    checkpoint = scheduler.read_state_dict()
    result.put((checkpoint['model_version'],
                {key: tensor.item() for key, tensor in checkpoint['model_state_dict'].items()}))


class ModelSchedulerTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for GPU snapshot transfer')
    def test_cuda_publication_keeps_model_on_device_and_snapshot_unchanged(self):
        tensor = torch.arange(16, device='cuda:0', dtype=torch.float32)
        address = tensor.data_ptr()
        scheduler = ModelScheduler({}, use_double_buffer=True)
        scheduler.write_state_dict({'weight': tensor}, model_version=1)
        tensor.add_(100)
        checkpoint = scheduler.read_state_dict()
        self.assertEqual(tensor.device, torch.device('cuda:0'))
        self.assertEqual(tensor.data_ptr(), address)
        self.assertEqual(checkpoint['model_version'], 1)
        self.assertTrue(torch.equal(checkpoint['model_state_dict']['weight'],
                                    torch.arange(16, dtype=torch.float32)))

    def test_snapshot_owns_cpu_storage_and_preserves_compression(self):
        shared = {}
        scheduler = ModelScheduler(shared, use_double_buffer=True)
        self.assertIsNone(scheduler.read_state_dict())
        state = {'small': torch.tensor([1.234567]),
                 'large': torch.full((262145,), 1.234567),
                 'counter': torch.tensor(7)}
        expected = {key: value.clone() for key, value in state.items()}
        scheduler.write_state_dict(state, model_version=3)
        for value in state.values():
            value.zero_()
        actual = scheduler.read_state_dict()
        self.assertEqual(actual['model_version'], 3)
        self.assertTrue(torch.equal(actual['model_state_dict']['small'], expected['small']))
        self.assertTrue(torch.equal(actual['model_state_dict']['counter'], expected['counter']))
        self.assertTrue(torch.equal(actual['model_state_dict']['large'], expected['large'].half().float()))
        self.assertEqual(shared['bufB_' + str(hash('large'))].dtype, torch.float16)

    def test_reused_buffer_retries_instead_of_returning_mixed_state(self):
        class ReusedDuringRead(dict):
            callback = None

            def get(self, key, default=None):
                value = super().get(key, default)
                if torch.is_tensor(value) and self.callback is not None:
                    callback, self.callback = self.callback, None
                    callback()
                return value

        shared = ReusedDuringRead()
        scheduler = ModelScheduler(shared, use_double_buffer=True)
        scheduler.write_state_dict({'a': torch.tensor(1), 'b': torch.tensor(1)}, 1)

        def reuse():
            scheduler.write_state_dict({'a': torch.tensor(2), 'b': torch.tensor(2)}, 2)
            scheduler.write_state_dict({'c': torch.tensor(3), 'd': torch.tensor(3)}, 3)

        shared.callback = reuse
        actual = scheduler.read_state_dict()
        self.assertEqual(actual['model_version'], 3)
        self.assertEqual(set(actual['model_state_dict']), {'c', 'd'})
        self.assertTrue(all(t.item() == 3 for t in actual['model_state_dict'].values()))

    def test_stopped_reader_does_not_block_publication(self):
        context = mp.get_context('spawn')
        with context.Manager() as manager:
            shared = manager.dict()
            scheduler = ModelScheduler(shared, use_double_buffer=True)
            scheduler.write_state_dict({'a': torch.tensor(1), 'b': torch.tensor(1)}, 1)
            ready, result = context.Event(), context.Queue()
            reader = context.Process(target=paused_reader, args=(shared, ready, result))
            reader.start()
            try:
                self.assertTrue(ready.wait(20), 'Reader did not start')
                deadline = time.monotonic() + 5
                while psutil.Process(reader.pid).status() != psutil.STATUS_STOPPED:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                scheduler.write_state_dict({'a': torch.tensor(2), 'b': torch.tensor(2)}, 2)
                scheduler.write_state_dict({'a': torch.tensor(3), 'b': torch.tensor(3)}, 3)
                psutil.Process(reader.pid).send_signal(signal.SIGCONT)
                self.assertEqual(result.get(timeout=20), (3, {'a': 3, 'b': 3}))
                reader.join(timeout=5)
                self.assertEqual(reader.exitcode, 0)
            finally:
                if reader.is_alive():
                    reader.kill()
                    reader.join(timeout=5)
                result.close()


if __name__ == '__main__':
    unittest.main()
