import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from environment.experiment_runtime import experiment_env, record_environment


class ExperimentEnvironmentTests(unittest.TestCase):
    def test_physical_gpu_selection_and_cpu_rejection(self):
        env = experiment_env(0, 19)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["PYTHONHASHSEED"], "19")
        with self.assertRaises(ValueError):
            experiment_env(-1)

    def test_resume_accepts_same_environment_and_rejects_changed_environment(self):
        probe = SimpleNamespace(returncode=0, stdout=json.dumps({"gpu": "NVIDIA Thor", "torch": "2.14"}), stderr="")
        with tempfile.TemporaryDirectory() as temporary, patch("subprocess.run", return_value=probe):
            root = Path(temporary)
            record_environment(root)
            (root / "results.csv").write_text("saved timing\n")
            record_environment(root)
            probe.stdout = json.dumps({"gpu": "RTX 5090", "torch": "2.14"})
            with self.assertRaisesRegex(ValueError, "environment changed"):
                record_environment(root)
            self.assertEqual((root / "results.csv").read_text(), "saved timing\n")

    def test_legacy_results_and_unavailable_gpu_are_rejected(self):
        probe = SimpleNamespace(returncode=0, stdout=json.dumps({"gpu": "NVIDIA Thor"}), stderr="")
        with tempfile.TemporaryDirectory() as temporary, patch("subprocess.run", return_value=probe):
            root = Path(temporary)
            (root / "results.csv").write_text("historical timing\n")
            with self.assertRaisesRegex(ValueError, "Historical results"):
                record_environment(root)
            probe.returncode, probe.stderr = 1, "Selected GPU unavailable"
            with self.assertRaisesRegex(RuntimeError, "GPU environment validation failed"):
                record_environment(root)


if __name__ == "__main__":
    unittest.main()
