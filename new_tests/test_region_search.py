"""Regional dual evidence, risk cuts, and fair-search integration checks."""

from dataclasses import dataclass
import math
import unittest

from certified_reliability_planning.benchmarks import AnalyticInterval
from certified_reliability_planning.planner import CertifiedPlanner, PlannerOptions
from new_tests.test_planner import ConstantLawOracle


@dataclass
class RegionEvidence:
    lower: float
    point: tuple | None = None
    upper: float | None = None
    dispatch: object = None
    work: float = 0
    runtime_seconds: float = 0
    status: str = "finite_reference"
    infeasible: bool = False


class FiniteRegion:
    """Independent exhaustive nominal master for the finite test domain."""

    def __init__(self, oracle, response=None):
        self.oracle, self.response = oracle, response
        self.cuts = []
        self.solve_snapshots = []

    def add_failure(self, point):
        self.cuts.append(tuple(point))

    def solve(self, budget_seconds, absolute_gap):
        self.solve_snapshots.append(tuple(self.cuts))
        if self.response is not None:
            return self.response
        remaining = [point for point in self.oracle.candidates
                     if not any(all(a <= b for a, b in zip(point, cut)) for cut in self.cuts)]
        if not remaining:
            return RegionEvidence(math.inf, infeasible=True)
        point = min(remaining, key=self.oracle.costs.get)
        cost = self.oracle.costs[point]
        return RegionEvidence(cost, point, cost)


class RegionIntegrationTests(unittest.TestCase):
    def planner(self, oracle, region, *, eens=.3, cvar=.5, stages=10,
                economic=None, samples=128):
        options = PlannerOptions(epsilon_cost=.05, initial_samples=samples,
                                 max_samples=4096, max_stages=stages,
                                 max_oracle_calls=30000, max_economic_calls=100,
                                 max_seconds=20)
        return CertifiedPlanner(oracle.candidates, oracle.cost_lower_bounds, oracle,
                                oracle.operation, economic or oracle.economic,
                                options, eens, cvar, .5, region_search=region)

    def test_master_is_resolved_after_certified_risk_cut(self):
        oracle = ConstantLawOracle([1, 0], [1, 2])
        region = FiniteRegion(oracle)
        result = self.planner(oracle, region).run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["incumbent"], [1])
        self.assertEqual(region.cuts, [(0,)])
        self.assertIn(((0,),), region.solve_snapshots)
        self.assertEqual(result["lower_bound_cost"], 2)
        self.assertEqual(result["upper_bound_cost"], 2)

    def test_unknown_and_feasible_points_do_not_enter_failure_cuts(self):
        oracle = ConstantLawOracle([.5, 0], [1, 2])
        region = FiniteRegion(oracle)
        result = self.planner(oracle, region, eens=.5, cvar=.5, stages=6).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["incumbent"], [1])
        self.assertEqual(region.cuts, [])
        self.assertEqual(result["labels"]["unknown"], 1)
        self.assertEqual(result["labels"]["risk_feasible"], 1)
        self.assertGreaterEqual(result["absolute_gap_cost"], 1)

    def test_no_incumbent_still_reports_useful_regional_dual_bound(self):
        oracle = ConstantLawOracle([0], [4], bound=0)
        region = FiniteRegion(oracle, RegionEvidence(3, status="time_limit_no_incumbent"))
        economic = lambda *args: AnalyticInterval(3, math.inf, status="time_limit")
        result = self.planner(oracle, region, eens=0, cvar=0, stages=3, economic=economic).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["lower_bound_cost"], 3)
        self.assertIsNone(result["upper_bound_cost"])
        self.assertIsNone(result["incumbent"])
        self.assertIsNone(result["certificate"])

    def test_region_incumbent_is_never_used_as_global_lower_bound(self):
        oracle = ConstantLawOracle([0], [20], bound=0)
        region = FiniteRegion(oracle, RegionEvidence(0, (0,), 100))
        economic = lambda *args: AnalyticInterval(0, 100, status="time_limit")
        result = self.planner(oracle, region, eens=0, cvar=0, stages=3, economic=economic).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["lower_bound_cost"], 0)
        self.assertEqual(result["upper_bound_cost"], 100)
        self.assertEqual(result["absolute_gap_cost"], 100)
        self.assertIsNone(result["certificate"])

    def test_nominal_infeasibility_does_not_generate_a_risk_downset_cut(self):
        oracle = ConstantLawOracle([0, 0], [1, 2], bound=0)
        region = FiniteRegion(oracle, RegionEvidence(0))
        ordinary = oracle.economic

        def economic(point, budget, gap):
            return (AnalyticInterval(math.inf, math.inf, status="infeasible")
                    if point == (0,) else ordinary(point, budget, gap))

        result = self.planner(oracle, region, eens=0, cvar=0, economic=economic).run()
        self.assertEqual(result["status"], "certified_optimal")
        self.assertEqual(result["incumbent"], [1])
        self.assertEqual(result["labels"]["nominal_infeasible"], 1)
        self.assertEqual(region.cuts, [])

    def test_region_infeasibility_proof_can_end_without_pointwise_search(self):
        oracle = ConstantLawOracle([0, 0], [1, 2], bound=0)
        region = FiniteRegion(oracle, RegionEvidence(math.inf, infeasible=True))
        result = self.planner(oracle, region).run()
        self.assertEqual(result["status"], "certified_infeasible")
        self.assertIsNotNone(result["certificate"])
        self.assertEqual(result["counters"]["operation_calls"], 0)
        self.assertIsNone(result["incumbent"])

    def test_contradictory_region_cost_does_not_produce_certificate(self):
        oracle = ConstantLawOracle([0], [1], bound=0)
        region = FiniteRegion(oracle, RegionEvidence(2, (0,), 1))
        with self.assertRaisesRegex(ValueError, "upper bound"):
            self.planner(oracle, region).run()


if __name__ == "__main__":
    unittest.main()
