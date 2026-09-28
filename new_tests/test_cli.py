"""Independent CLI behavior and output safety, requiring no solver license."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from certified_reliability_planning.cli import main


class CliTests(unittest.TestCase):
    def call(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_nonempty_output_directory_is_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            (output / "summary.json").write_bytes(b'{"legacy": true}\n')
            (output / "nested").mkdir()
            (output / "nested" / "source.dat").write_bytes(b"preserve existing evidence\x00")
            before = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.rglob("*") if p.is_file()}
            code, stdout, stderr = self.call(["demo", "--output", str(output)])
            after = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in output.rglob("*") if p.is_file()}
            self.assertEqual(code, 1)
            self.assertIn("拒绝覆盖", stderr)
            self.assertEqual(before, after)
            self.assertEqual(stdout, "")

    def test_existing_output_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing.txt"
            output.write_text("existing file", encoding="utf-8")
            code, _, stderr = self.call(["demo", "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertIn("拒绝覆盖", stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing file")

    def test_demo_and_validation_write_successful_auditable_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            for command in ("demo", "validate-small"):
                with self.subTest(command=command):
                    output = Path(temporary) / command
                    code, stdout, stderr = self.call([command, "--output", str(output)])
                    self.assertEqual(code, 0, stderr)
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary["status"], "certified_optimal")
                    self.assertEqual(summary["incumbent"], [2, 1])
                    self.assertTrue(summary["validation"]["certificate_matches_enumeration"])
                    self.assertTrue(summary["legacy_isolation"]["passed"])
                    self.assertLessEqual(summary["absolute_gap_cost"], summary["options"]["epsilon_cost"])
                    self.assertIsNotNone(summary["certificate"])
                    if command == "validate-small":
                        self.assertTrue(summary["validation"]["passed"])
                    for name in ("events.jsonl", "progress.json", "resolved_config.json",
                                 "population_truth.json", "iid_uniform_samples.npy", "certificate_state.npz"):
                        self.assertTrue((output / name).is_file(), name)
                    self.assertIn("certified_optimal", stdout)
                    for line in (output / "events.jsonl").read_text().splitlines():
                        self.assertIn("action", json.loads(line))

    def test_operation_budget_exhaustion_has_exit_four_and_no_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "limited"
            code, _, stderr = self.call(["demo", "--output", str(output), "--max-oracle-calls", "1"])
            self.assertEqual(code, 4, stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "budget_exhausted")
            self.assertEqual(summary["stop_reason"], "operation_call_budget")
            self.assertEqual(summary["counters"]["operation_calls"], 1)
            self.assertIsNone(summary["certificate"])
            self.assertGreater(summary["labels"]["unknown"], 0)

    def test_verify_isolation_succeeds_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            unused = Path(temporary) / "unused"
            code, stdout, stderr = self.call(["verify-isolation", "--output", str(unused)])
            self.assertEqual(code, 0, stderr)
            result = json.loads(stdout)
            self.assertTrue(result["passed"])
            self.assertGreater(result["checked_existing_files"], 0)
            self.assertEqual(result["changed_or_missing"], [])
            self.assertFalse(unused.exists())

    def test_invalid_parameters_produce_error_evidence_without_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, (option, value, message) in enumerate((
                ("--delta", "1", "delta"),
                ("--epsilon-cost", "0", "epsilon_cost"),
                ("--max-oracle-calls", "0", "max_oracle_calls"),
                ("--alpha", "1", "alpha"),
                ("--eens-limit", "-1", "risk limits"),
            )):
                with self.subTest(option=option):
                    output = Path(temporary) / str(index)
                    code, _, stderr = self.call(["demo", "--output", str(output), option, value])
                    self.assertEqual(code, 1)
                    self.assertIn(message, stderr)
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary["status"], "error")
                    self.assertIn(message, summary["error"])
                    self.assertNotIn("certificate", summary)

    def test_module_entry_point_works_without_importing_gurobi(self):
        # Block the optional solver import in a fresh interpreter, then execute
        # exactly the package module entry point used by the public launcher.
        program = """import runpy, sys
sys.modules['gurobipy'] = None
sys.argv = ['certified_reliability_planning', 'verify-isolation']
runpy.run_module('certified_reliability_planning', run_name='__main__')
"""
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", program], cwd=root,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["passed"])


if __name__ == "__main__":
    unittest.main()
