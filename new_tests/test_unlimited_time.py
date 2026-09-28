"""Total wall-clock limits can be disabled without unbounding Oracle calls."""

from contextlib import redirect_stderr
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from certified_reliability_planning.benchmarks import AnalyticInterval
from certified_reliability_planning.cli import options_from, parse_args
from certified_reliability_planning.planner import CertifiedPlanner, PlannerOptions
from new_tests.test_planner import ConstantLawOracle


class UnlimitedTimeTests(unittest.TestCase):
    def simulated_run(self, max_seconds):
        """Each solve advances a private clock by two hours, without sleeping."""
        clock = SimpleNamespace(now=0.0)
        budgets = {"economic": [], "operation": []}
        oracle = ConstantLawOracle([0], [1])

        def economic(point, budget, gap):
            budgets["economic"].append(budget)
            clock.now += 7200
            return AnalyticInterval(1, 1)

        def operation(point, sample, budget, gap):
            budgets["operation"].append(budget)
            clock.now += 7200
            return AnalyticInterval(0, 0)

        options = PlannerOptions(max_seconds=max_seconds, initial_samples=2,
                                 max_samples=4, max_stages=2,
                                 base_call_seconds=3, max_call_seconds=5)
        planner = CertifiedPlanner(oracle.candidates, oracle.cost_lower_bounds,
                                   oracle, operation, economic, options,
                                   eens_limit=0, cvar_limit=0, alpha=.95)
        with patch("certified_reliability_planning.planner.time",
                   SimpleNamespace(perf_counter=lambda: clock.now)):
            result = planner.run()
        return result, budgets

    def test_disabled_total_budget_survives_elapsed_time_and_keeps_finite_calls(self):
        result, budgets = self.simulated_run(0)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["stop_reason"], "stage_or_sample_budget")
        self.assertEqual(result["stages_completed"], 2)
        self.assertGreater(result["elapsed_seconds"], 3600)
        self.assertTrue(budgets["economic"])
        self.assertTrue(budgets["operation"])
        self.assertIn(5, budgets["operation"])
        for kind in budgets:
            self.assertTrue(all(0 < value <= 5 and math.isfinite(value)
                                for value in budgets[kind]))
        self.assertIsNone(result["certificate"])

    def test_finite_total_budget_still_stops_after_elapsed_time(self):
        result, budgets = self.simulated_run(3600)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["stop_reason"], "wall_time_budget")
        self.assertEqual(result["stages_completed"], 1)
        self.assertEqual(budgets["economic"], [3])
        self.assertEqual(budgets["operation"], [])

    def test_invalid_time_limits_are_rejected(self):
        for value in (-1, math.nan, math.inf, -math.inf, True, False):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_seconds"):
                PlannerOptions(max_seconds=value)
        for field in ("base_call_seconds", "max_call_seconds"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                PlannerOptions(**{field: 0})

    def test_no_time_limit_and_zero_override_configured_limit(self):
        for flags in (["--no-time-limit"], ["--max-seconds", "0"]):
            with self.subTest(flags=flags):
                options = options_from(parse_args(["plan", *flags]), {"max_seconds": 3600})
                self.assertEqual(options.max_seconds, 0)

    def test_default_and_explicit_finite_cli_limits_are_preserved(self):
        self.assertEqual(options_from(parse_args(["plan"])).max_seconds, 600)
        self.assertEqual(options_from(parse_args(["plan"]), {"max_seconds": 3600}).max_seconds, 3600)
        self.assertEqual(options_from(parse_args(["plan", "--max-seconds", "15"]),
                                      {"max_seconds": 0}).max_seconds, 15)

    def test_conflicting_cli_time_limits_are_rejected_in_either_order(self):
        for flags in (["--max-seconds", "3600", "--no-time-limit"],
                      ["--no-time-limit", "--max-seconds", "3600"]):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse_args(["plan", *flags])
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
