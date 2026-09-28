"""Relative economic certificates retain risk and global-bound requirements."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from certified_reliability_planning.benchmarks import AnalyticInterval
from certified_reliability_planning.cli import main, options_from, parse_args
from certified_reliability_planning.planner import CertifiedPlanner, PlannerOptions
from new_tests.test_planner import ConstantLawOracle


class RelativeGapTests(unittest.TestCase):
    def options(self, **kwargs):
        return replace(PlannerOptions(relative_gap=.01, epsilon_cost=1000,
                                      initial_samples=1, max_samples=1,
                                      max_stages=1, max_seconds=20), **kwargs)

    def planner(self, lower, upper, *, options=None, loss_bound=0):
        oracle = ConstantLawOracle([0], [(lower + upper) / 2], bound=loss_bound)
        economic = lambda *args: AnalyticInterval(lower, upper)
        return CertifiedPlanner(oracle.candidates, oracle.cost_lower_bounds, oracle,
                                oracle.operation, economic, options or self.options(),
                                eens_limit=0, cvar_limit=0, alpha=.5)

    def test_relative_gap_does_not_accept_large_absolute_epsilon(self):
        result = self.planner(90, 100).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertTrue(result["incumbent_reliability_certified"])
        self.assertLess(result["absolute_gap_cost"], result["options"]["epsilon_cost"])
        self.assertAlmostEqual(result["relative_gap_cost"], .1)
        self.assertIsNone(result["certificate"])

    def test_relative_gap_accepts_large_cost_scale_with_small_absolute_epsilon(self):
        result = self.planner(995000, 1000000,
                              options=self.options(epsilon_cost=1)).run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["absolute_gap_cost"], 5000)
        self.assertAlmostEqual(result["relative_gap_cost"], .005)
        self.assertEqual(result["options"]["relative_gap"], .01)
        self.assertIsInstance(result["optimality_criterion"], dict)
        self.assertTrue(result["optimality_criterion"])
        self.assertTrue(result["certificate"]["global_lower_bound_includes_all_unexcluded_designs"])

    def test_one_percent_equality_is_accepted(self):
        result = self.planner(99, 100).run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertAlmostEqual(result["relative_gap_cost"], .01)

    def test_relative_denominator_is_incumbent_cost(self):
        # (100 - 99) / 100 is 1%; dividing by the lower bound is too strict.
        planner = self.planner(99, 100)
        planner.cost_lower[:] = 99
        planner.cost_upper[:] = 100
        planner.labels[:] = 1
        self.assertEqual(planner._certificate_status(), "certified_optimal")

    def test_economic_gap_alone_cannot_certify_unknown_reliability(self):
        result = self.planner(99.5, 100, loss_bound=1).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["incumbent_reliability_certified"])
        self.assertIsNone(result["incumbent"])
        self.assertIsNone(result["relative_gap_cost"])
        self.assertIsNone(result["certificate"])

    def test_zero_cost_exact_certificate_has_zero_relative_gap(self):
        result = self.planner(0, 0).run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["relative_gap_cost"], 0)

    def test_tiny_positive_cost_cannot_use_an_absolute_denominator_floor(self):
        result = self.planner(0, 1e-10).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["relative_gap_cost"], 1)
        self.assertIsNone(result["certificate"])

    def test_contradictory_bound_is_rejected_even_at_zero_cost(self):
        planner = self.planner(0, 0)
        planner.cost_lower[:] = 1
        planner.cost_upper[:] = 0
        planner.labels[:] = 1
        with self.assertRaisesRegex(ValueError, "contradict"):
            planner._certificate_status()

    def test_default_absolute_behavior_is_preserved(self):
        self.assertIsNone(PlannerOptions().relative_gap)
        loose = self.planner(90, 100, options=self.options(relative_gap=None)).run()
        strict = self.planner(90, 100,
                              options=self.options(relative_gap=None, epsilon_cost=1)).run()
        self.assertEqual(loose["status"], "certified_optimal")
        self.assertEqual(strict["status"], "budget_exhausted")
        self.assertAlmostEqual(loose["relative_gap_cost"], .1)

    def test_cost_tolerance_selects_one_configured_criterion(self):
        self.assertEqual(self.options().cost_tolerance(100), 1)
        self.assertEqual(self.options().cost_tolerance(-100), 1)
        self.assertEqual(self.options().cost_tolerance(0), 0)
        self.assertEqual(self.options(relative_gap=None).cost_tolerance(100), 1000)

    def test_competitor_filter_uses_relative_slack(self):
        oracle = ConstantLawOracle([0, 0, 0], [100, 99.4, 99.6], bound=0)
        planner = CertifiedPlanner(oracle.candidates, oracle.cost_lower_bounds, oracle,
                                   oracle.operation, oracle.economic,
                                   self.options(), 0, 0, .5)
        planner.labels[0] = 1
        planner.cost_upper[0] = 100
        planner.cost_lower[:] = [90, 99.4, 99.6]
        np.testing.assert_array_equal(planner._competitive(), [True, True, False])
        # The cost-priority filter must not remove the third design's lower
        # bound from the overall economic certificate.
        planner.cost_lower[0] = 99.8
        self.assertEqual(planner._global_lower(), 99.4)

    def test_invalid_relative_tolerances_are_rejected(self):
        for value in (0, -0.01, 1, 1.01, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "relative_gap"):
                    PlannerOptions(relative_gap=value)


class RelativeGapCliTests(unittest.TestCase):
    def call(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_both_cli_spellings_override_configured_relative_gap(self):
        for spelling in ("--relative-gap", "--mip-gap"):
            with self.subTest(spelling=spelling):
                args = parse_args(["plan", spelling, ".01"])
                options = options_from(args, {"relative_gap": .05})
                self.assertEqual(options.relative_gap, .01)

    def test_demo_produces_auditable_one_percent_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "relative"
            code, _, stderr = self.call(["demo", "--output", str(output),
                                         "--mip-gap", ".01"])
            self.assertEqual(code, 0, stderr)
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["status"], "certified_optimal")
            self.assertLessEqual(result["relative_gap_cost"], .01)
            self.assertEqual(result["options"]["relative_gap"], .01)
            self.assertTrue(result["validation"]["certificate_matches_enumeration"])
            resolved = json.loads((output / "resolved_config.json").read_text())
            self.assertEqual(resolved["options"]["relative_gap"], .01)

    def test_invalid_cli_relative_gap_writes_error_without_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, value in enumerate(("0", "-0.01", "1", "nan", "inf")):
                with self.subTest(value=value):
                    output = Path(temporary) / str(index)
                    code, _, stderr = self.call(["demo", "--output", str(output),
                                                 "--relative-gap", value])
                    self.assertEqual(code, 1)
                    self.assertIn("relative_gap", stderr)
                    result = json.loads((output / "summary.json").read_text())
                    self.assertEqual(result["status"], "error")
                    self.assertNotIn("certificate", result)


class RelativeGapSolverTests(unittest.TestCase):
    """Observe parameters at the actual Gurobi optimization boundary."""

    def test_nominal_relative_gap_does_not_relax_operation_loss_solves(self):
        import gurobipy as gp
        from certified_reliability_planning.oracles import MicrogridOracle
        from polar_reliability_planning.config import SolverOptions
        from polar_reliability_planning.data import COMPONENTS
        from polar_reliability_planning.validation.small_system import small_system

        data, pool, _ = small_system(True)
        options = SolverOptions(mip_gap=.01, threads=1, output_flag=False)
        observed = []
        optimize = gp.Model.optimize

        def record(model, *args, **kwargs):
            observed.append((model.ModelName, model.Params.MIPGap,
                             model.Params.MIPGapAbs))
            return optimize(model, *args, **kwargs)

        with gp.Env(params={"OutputFlag": 0}) as env:
            oracle = MicrogridOracle(data, options, env)
            with patch.object(gp.Model, "optimize", record):
                nominal_units = dict(zip(COMPONENTS, [0, 0, 1, 0, 0]))
                nominal = oracle.economic(nominal_units, 20, 0)
                units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
                operation = oracle.operation(units, pool.scenario(data, units, 1),
                                             20, .125)
        self.assertEqual(observed, [("certified_nominal_cost", .01, 0),
                                    ("certified_uc_loss", 0, .125)])
        self.assertTrue(math.isfinite(nominal.upper))
        self.assertGreater(operation.lower, 0)
        self.assertLessEqual(operation.lower, operation.upper)

    def test_regional_master_receives_one_percent_solver_tolerance(self):
        import gurobipy as gp
        from certified_reliability_planning.economic_search import EconomicSearch
        from polar_reliability_planning.config import SolverOptions
        from polar_reliability_planning.validation.small_system import small_system

        data, _, _ = small_system(True)
        options = SolverOptions(mip_gap=.01, threads=1, output_flag=False)
        observed = []
        optimize = gp.Model.optimize

        def record(model, *args, **kwargs):
            observed.append((model.Params.MIPGap, model.Params.MIPGapAbs))
            return optimize(model, *args, **kwargs)

        with gp.Env(params={"OutputFlag": 0}) as env:
            search = EconomicSearch(data, options, env)
            try:
                with patch.object(gp.Model, "optimize", record):
                    result = search.solve(20, 0)
            finally:
                search.close()
        self.assertEqual(observed, [(.01, 0)])
        self.assertIsNotNone(result.point)
        self.assertLessEqual(result.lower, result.upper)


if __name__ == "__main__":
    unittest.main()
