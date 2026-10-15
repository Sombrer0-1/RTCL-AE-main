"""Portable GPU selection and per-directory experiment environment identity."""
import json
import os
from pathlib import Path
import subprocess
import sys


def experiment_env(gpu=0, seed=None):
    if gpu < 0:
        raise ValueError("Experiment runners require a nonnegative physical GPU index")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=str(gpu))
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    return env


def record_environment(out, gpu=0):
    probe = '''import importlib.metadata as m, json, platform, sys, torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Selected GPU unavailable"
print(json.dumps(dict(python=platform.python_version(), executable=sys.executable,
    platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
    packages={name:m.version(name) for name in ["torchvision","avalanche-lib","numpy","pytorchcv"]})))'''
    result = subprocess.run([sys.executable, "-c", probe], env=experiment_env(gpu),
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("GPU environment validation failed:\n" + result.stderr)
    current = json.loads(result.stdout.strip().splitlines()[-1])
    current["physical_gpu"] = gpu
    out = Path(out)
    path = out / "execution_environment.json"
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError("Experiment environment changed; use a new output directory")
    elif (out / "results.csv").exists():
        raise ValueError("Historical results have no matching environment identity; use a new output directory")
    else:
        path.write_text(json.dumps(current, indent=2) + "\n")
    return current
