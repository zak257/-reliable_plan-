"""End-to-end certificate checks, including deliberately unresolved problems."""

from dataclasses import replace
import math
import unittest

import numpy as np

from certified_reliability_planning.benchmarks import AnalyticInterval, TinyCapacityOracle
from certified_reliability_planning.planner import CertifiedPlanner, PlannerOptions


class ConstantLawOracle(TinyCapacityOracle):
    """A degenerate iid law with configurable monotone loss and arbitrary costs."""

    def __init__(self, losses, costs, *, bound=1.0, economic_floor=0.0):
        super().__init__(economic_floor=economic_floor)
        self.candidates = [(i,) for i in range(len(losses))]
        self.capital_lower_bounds = {point: 0.0 for point in self.candidates}
        self.cost_lower_bounds = [0.0 for _ in self.candidates]
        self.costs = dict(zip(self.candidates, costs))
        self.losses = dict(zip(self.candidates, losses))
        self.loss_bound = bound

    def true_loss(self, point):
        value = self.losses[self._point(point)]
        return (value, value), (.5, .5)


class PlannerTests(unittest.TestCase):
    def options(self, **kwargs):
        base = PlannerOptions(epsilon_cost=.05, initial_samples=32, max_samples=4096,
                              max_stages=12, max_oracle_calls=50000,
                              max_economic_calls=500, max_seconds=20)
        return replace(base, **kwargs)

    def planner(self, oracle, options=None, *, eens=.3, cvar=.5, alpha=.5):
        return CertifiedPlanner(oracle.candidates, oracle.cost_lower_bounds, oracle,
                                oracle.operation, oracle.economic,
                                options or self.options(), eens, cvar, alpha)

    def test_analytic_population_optimum_and_global_cost_certificate(self):
        oracle = TinyCapacityOracle(7)
        planner = self.planner(oracle)
        result = planner.run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["incumbent"], [2, 1])
        self.assertLessEqual(result["lower_bound_cost"], 7.0)
        self.assertGreaterEqual(result["upper_bound_cost"], 7.0)
        self.assertLessEqual(result["absolute_gap_cost"], .05)
        self.assertTrue(result["incumbent_reliability_certified"])
        self.assertTrue(result["certificate"]["global_lower_bound_includes_all_unexcluded_designs"])
        # Finishing need not classify an expensive boundary capacity.  Its
        # cost lower bound, rather than an invented risk label, removes it
        # from contention while it remains in the reported global minimum.
        self.assertGreater(result["labels"]["unknown"], 0)
        unexcluded = (planner.labels != -1) & (planner.labels != 2)
        self.assertEqual(result["lower_bound_cost"], float(np.min(planner.cost_lower[unexcluded])))
        for point, label in zip(oracle.candidates, planner.labels):
            truth = oracle.true_metrics(point)
            if label == 1:
                self.assertLessEqual(truth["eens"], .3 + 1e-12)
                self.assertLessEqual(truth["cvar"], .5 + 1e-12)
            elif label == -1:
                self.assertTrue(truth["eens"] > .3 or truth["cvar"] > .5)
        with self.assertRaises(RuntimeError):
            planner.run()

    def test_feasibility_upper_orthant_does_not_prune_cheaper_larger_design(self):
        oracle = ConstantLawOracle([0, 0, 0], [10, 1, 12], bound=0)
        planner = self.planner(oracle, eens=0, cvar=0)
        result = planner.run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["incumbent"], [1])
        self.assertLessEqual(result["lower_bound_cost"], 1.0)
        self.assertGreaterEqual(result["upper_bound_cost"], 1.0)
        # A single risk certificate covers all greater designs; economics
        # must still locate the nonmonotone minimum inside that region.
        np.testing.assert_array_equal(planner.labels, [1, 1, 1])
        self.assertEqual(result["counters"]["operation_calls"], 0)
        self.assertGreaterEqual(result["counters"]["economic_calls"], 2)

    def test_boundary_competitor_keeps_global_gap_open(self):
        oracle = ConstantLawOracle([.5, 0], [1, 2])
        planner = self.planner(oracle, self.options(initial_samples=256, max_samples=256,
                                                   max_stages=8), eens=.5, cvar=.5)
        result = planner.run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertIsNone(result["certificate"])
        self.assertEqual(result["incumbent"], [1])
        self.assertTrue(result["incumbent_reliability_certified"])
        self.assertEqual(planner.labels[0], 0)
        self.assertLessEqual(result["lower_bound_cost"], 1.0)
        self.assertGreater(result["absolute_gap_cost"], .9)

    def test_expensive_unresolved_record_does_not_block_sampling_of_competitor(self):
        # Once design 2 establishes a cost-2 incumbent, design 0 is too
        # expensive to matter.  Its coarse loss bracket must not prevent the
        # cost-1 design from collecting enough observations for certification.
        oracle = ConstantLawOracle([.551, .5, 0], [5, 1, 2])
        planner = self.planner(oracle, self.options(initial_samples=128, max_samples=8192,
                                                   max_stages=20, max_oracle_calls=100000,
                                                   max_economic_calls=1000),
                               eens=.55, cvar=.55, alpha=0)
        result = planner.run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["incumbent"], [1])
        self.assertGreater(result["counters"]["annual_samples"], 128)

    def test_one_oracle_call_cannot_certify_unresolved_sample_prefix(self):
        oracle = ConstantLawOracle([0], [1])
        planner = self.planner(oracle, self.options(initial_samples=32, max_samples=32,
                                                   max_oracle_calls=1), eens=.3, cvar=.5)
        result = planner.run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertIsNone(result["certificate"])
        self.assertIsNone(result["incumbent"])
        self.assertEqual(result["counters"]["operation_calls"], 1)
        self.assertEqual(result["labels"]["unknown"], 1)
        record = planner.records[(0,)]
        self.assertEqual(len(record.lower), 32)
        self.assertEqual(len(record.upper), 32)
        self.assertGreaterEqual(float(np.mean(record.upper)), 31 / 32)
        self.assertEqual(int(np.sum(record.upper == 1)), 31)

    def test_solver_gap_is_preserved_and_prevents_false_optimality(self):
        oracle = ConstantLawOracle([0], [4], bound=0, economic_floor=2)
        result = self.planner(oracle, self.options(max_stages=4), eens=0, cvar=0).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["incumbent"], [0])
        self.assertTrue(result["incumbent_reliability_certified"])
        self.assertEqual(result["lower_bound_cost"], 3)
        self.assertEqual(result["upper_bound_cost"], 5)
        self.assertEqual(result["absolute_gap_cost"], 2)
        self.assertIsNone(result["certificate"])

    def test_all_risk_failures_can_certify_infeasibility(self):
        oracle = ConstantLawOracle([1, .9, .8], [1, 2, 3])
        planner = self.planner(oracle, eens=.2, cvar=.3)
        result = planner.run()
        self.assertEqual(result["status"], "certified_infeasible")
        np.testing.assert_array_equal(planner.labels, [-1, -1, -1])
        self.assertIsNone(result["incumbent"])
        self.assertIsNotNone(result["certificate"])

    def test_proven_nominal_infeasibility_excludes_only_that_design(self):
        oracle = ConstantLawOracle([0, 0], [1, 3], bound=0)
        ordinary_economic = oracle.economic

        def economic(point, budget_seconds, absolute_gap):
            if point == (0,):
                return AnalyticInterval(math.inf, math.inf, status="nominal_infeasible")
            return ordinary_economic(point, budget_seconds, absolute_gap)

        planner = self.planner(oracle, eens=0, cvar=0)
        planner.economic = economic
        result = planner.run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["incumbent"], [1])
        np.testing.assert_array_equal(planner.labels, [2, 1])

    def test_invalid_oracle_bounds_abort_without_certificate(self):
        oracle = ConstantLawOracle([0], [1])
        planner = self.planner(oracle)
        planner.operation = lambda *args: AnalyticInterval(0, 1.1)
        with self.assertRaisesRegex(ValueError, "loss bound"):
            planner.run()


if __name__ == "__main__":
    unittest.main()
