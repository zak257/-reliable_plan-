"""Small end-to-end equivalence test for disjoint capacity partitions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from polar_reliability_planning.recovery.config import read_recovery_config
from polar_reliability_planning.recovery_cli import ROOT
from scripts.recovery_parallel_grid import toml_text


class ParallelWorkflowTests(unittest.TestCase):
    def test_partitioned_and_serial_full_grid_have_identical_incumbents(self):
        config = read_recovery_config(ROOT / "config/zhongshan_recovery_stress.toml")
        if not Path(config["data_root"]).exists():
            self.skipTest("External station CSV unavailable")
        config["hours"] = 24
        config["recovery"]["warmup_hours"] = 24
        config["grid"].update(diesel=[4], battery_energy=[20], pcs=[4])
        config["reliability"].update(samples=2, validation_samples=0, validation_seed=2026091703,
                                      eens_limit_kwh=100000., cvar_limit_kwh=100000.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "small.toml"
            path.write_text(toml_text(config))
            serial = subprocess.run([sys.executable, "-m", "polar_reliability_planning.recovery_cli", "plan",
                                     "--config", str(path), "--output", str(root / "serial")],
                                    cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(serial.returncode, 0, serial.stdout + serial.stderr)
            parallel = subprocess.run([sys.executable, "-m", "scripts.recovery_parallel_grid",
                                       "--config", str(path), "--output", str(root / "parallel"),
                                       "--samples", "2", "--validation-samples", "0", "--workers", "2"],
                                      cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(parallel.returncode, 0, parallel.stdout + parallel.stderr)
            expected = json.loads((root / "serial" / "summary.json").read_text())
            actual = json.loads((root / "parallel" / "full_grid" / "summary.json").read_text())
            self.assertEqual(expected["incumbent"], actual["incumbent"])
            self.assertEqual(actual["relative_gap"], 0)
            self.assertEqual(expected["total_designs"], actual["total_designs"])


if __name__ == "__main__":
    unittest.main()
