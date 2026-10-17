"""Check imports and real CUDA/Avalanche execution without downloading datasets."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smoke():
    import numpy as np
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from avalanche.benchmarks.classic import SoftRobot, EndlessCLSim
    from avalanche.benchmarks import nc_benchmark
    from avalanche.training import Replay
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import accuracy_metrics
    from src.models.model_init import initialize_model
    from src.schedulers.stream_scheduler import StreamSchedulerPlugin
    from src.schedulers.boundary_scheduler import evaluation_state
    from src.utils.reproducibility import seed_worker

    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "CUDA unavailable; CPU fallback is not an environment check"
    device = torch.device("cuda:0")
    rng = np.random.default_rng(123)

    def dataset(count, preparation_delay=0):
        # Exact torchvision CIFAR10 leaves exercise the real backend certificate.
        data = CIFAR10.__new__(CIFAR10)
        data.data = rng.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)
        data.targets = (np.arange(count) % 2).tolist()
        def transform(image):
            if preparation_delay:
                time.sleep(preparation_delay)
            return ToTensor()(image)
        data.transform = transform
        data.target_transform = None
        return data

    # Deliberate synthetic host work ensures preparation service points execute.
    train, test = dataset(256, preparation_delay=.001), dataset(64)
    results = []
    for mode, preparation, start_policy, quota in [
            ("baseline", "baseline", "immediate", 0), ("stream_fifo", "certified", "immediate", 0),
            ("stream_fifo", "cooperative", "immediate", 0), ("r4_full_eval", "cooperative", "immediate", 0),
            ("stream_fifo", "cooperative", "reactive", 0), ("stream_fifo", "cooperative", "oracle", 0),
            ("stream_fifo", "cooperative", "immediate", 10), ("stream_fifo", "cooperative", "reactive", 10),
            ("stream_fifo", "cooperative", "phase", 10), ("stream_fifo", "cooperative", "oracle", 10),
            ("stream_fifo", "cooperative", "planned", 10), ("stream_fifo", "cooperative", "uniform", 10)]:
        seed_worker(7)
        args = SimpleNamespace(model="resnet20", global_scheduler_mode=mode,
                               host_preparation=preparation, seed=7, stream_requests=64,
                               stream_duration=.3, stream_slo_ms=100, stream_pattern="steady",
                               eval_bs=8,
                               stream_start_policy=start_policy, stream_start_max_wait_ms=100,
                               stream_train_quota=quota,
                               semseg=False, debug=False, eval_cycles=3, record_eval_batches=True)
        if start_policy == 'phase':
            args.stream_pattern = 'bursty'
        benchmark = nc_benchmark(train, test, n_experiences=2, task_labels=False,
                                 shuffle=False, fixed_class_order=[0, 1])
        model = initialize_model(args, "split_cifar10")
        if mode == 'r4_full_eval':
            from src.schedulers.full_eval_scheduler import FullEvaluationPlugin
            plugin = FullEvaluationPlugin(args, benchmark.test_stream, {})
            plugin.interval = 0
        else:
            plugin = StreamSchedulerPlugin(args, benchmark.test_stream) if mode != "baseline" else None
        strategy = Replay(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=.001),
                          criterion=torch.nn.CrossEntropyLoss(), mem_size=16, train_mb_size=8,
                          train_epochs=1, eval_mb_size=8, device=device,
                          plugins=[plugin] if plugin else [],
                          evaluator=EvaluationPlugin(accuracy_metrics(stream=True), loggers=[]))
        # Stream metrics are generated artifacts; keep this validation self-contained.
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="rtcl-smoke-") as temporary:
            try:
                os.chdir(temporary)
                for experience in benchmark.train_stream:
                    strategy.train(experience)
                    if mode == 'r4_full_eval':
                        plugin.publish(strategy, experience.current_experience + 1)
                if plugin:
                    plugin.finish(strategy)
                    if mode == 'r4_full_eval':
                        assert len(plugin.cycles) == 3
                        assert all(cycle['samples'] == 64 for cycle in plugin.cycles)
                        assert plugin.cycles[-1]['final_snapshot']
                        assert plugin.stats['service_samples'] > 0
                        stats = plugin.stats
                    else:
                        assert len(plugin.records) == 64
                        stats = plugin.prep_stats
                        if preparation == 'cooperative':
                            assert stats['service_requests'] > 0
                    assert stats["certified_epochs"] == 2
                    assert stats["fallback_epochs"] == 0
                with evaluation_state(strategy.model):
                    strategy.eval(benchmark.test_stream)
            finally:
                os.chdir(previous)
        torch.cuda.synchronize()
        digest = hashlib.sha256()
        for name, tensor in sorted(strategy.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        results.append({"mode": mode, "host_preparation": preparation,
                        "stream_start_policy": start_policy,
                        "stream_train_quota": quota,
                        "model_hash": digest.hexdigest(),
                        "train_updates": strategy.clock.train_iterations,
                        "requests": len(plugin.records) if plugin and mode != 'r4_full_eval' else 0,
                        "full_eval_cycles": len(plugin.cycles) if mode == 'r4_full_eval' else 0})
    assert len({row["model_hash"] for row in results}) == 1, "Scheduling changed training weights"
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Write current environment and validation manifest")
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    import torch
    import torchvision
    import avalanche
    import timm
    import tinyimagenet
    import qpsolvers
    assert "quadprog" in qpsolvers.available_solvers
    import numpy as np
    solution = qpsolvers.solve_qp(np.eye(2), np.array([-1., -1.]),
                                 G=-np.eye(2), h=np.zeros(2), solver="quadprog")
    assert np.allclose(solution, [1., 1.]), "GEM's native QP solver failed"
    for entry in ["main.py", "test_avalanche_lib.py", "test_avalanche_lib_mp.py"]:
        subprocess.run([sys.executable, str(ROOT / entry), "--help"],
                       check=True, stdout=subprocess.DEVNULL)
    results = smoke()
    manifest = {"platform": platform.platform(), "architecture": platform.machine(),
                "python": platform.python_version(), "executable": sys.executable,
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "avalanche": avalanche.__version__, "avalanche_source": avalanche.__file__,
                "avalanche_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT / ".deps/avalanche"), "rev-parse", "HEAD"], text=True).strip(),
                "packages": {name: importlib.metadata.version(name)
                             for name in ["numpy", "timm", "pytorchcv", "quadprog", "nvidia-cusparselt-cu13"]},
                "gpu_smoke": results, "validation_data": "synthetic CIFAR10; no benchmark downloads"}
    print(json.dumps(manifest, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
