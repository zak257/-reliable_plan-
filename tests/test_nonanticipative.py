from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import gurobipy as gp
import numpy as np

from polar_reliability_planning.config import ReliabilityLimits, SolverOptions, UnitCommitmentOptions
from polar_reliability_planning.data import CaseData, COMPONENTS
from polar_reliability_planning.data.cap_plan_loader import FAILABLE_COMPONENTS
from polar_reliability_planning.scenario_generation import ScenarioPool
from polar_reliability_planning.reliability.monte_carlo import ReliabilityOracle
from polar_reliability_planning.reliability.nonanticipative import (
    NonanticipativeOracle, NonanticipativeOptions, observation_nodes, audit_nonanticipativity)
from polar_reliability_planning.validation.small_system import small_system, enumerate_grid
from polar_reliability_planning.planning.optimizer import optimize_reliability
from polar_reliability_planning.cli import parse_args, main


def branching_system():
    # Starting at t=0 entails running at t=1. Future low load cannot absorb
    # minimum diesel output. Perfect foresight starts only in the high branch;
    # a causal policy must make the same t=0 decision in both branches.
    bounds = {k: (0, 2 if k == "diesel" else 0) for k in COMPONENTS}
    data = CaseData("future_load_branch", np.ones(2), np.zeros(2), np.zeros(2),
                    {k: 1.0 for k in COMPONENTS}, bounds, {k: 0.0 for k in COMPONENTS},
                    1.0, 1.0, 0.0, 1.0,
                    unit_commitment=UnitCommitmentOptions(True, 1.0, 2, 1, 0))
    availability = {k: np.ones((2, bounds[k][1], 2), dtype=np.uint8) for k in FAILABLE_COMPONENTS}
    factors = np.ones((2, 2))
    load = np.array([[1., 1.], [1., 0.]])
    pool = ScenarioPool(availability, np.zeros((2, 2), dtype=np.uint8), np.array([.5, .5]),
                        factors, factors, load)
    units = {k: int(k == "diesel") for k in COMPONENTS}
    return data, pool, units


class NonanticipativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(params={"OutputFlag": 0})
        cls.options = SolverOptions(time_limit=30, oracle_time_limit=30, mip_gap=0)

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def test_shared_past_prevents_future_dependent_start(self):
        data, pool, units = branching_system()
        limits = ReliabilityLimits(.75, 10, .5)
        old = ReliabilityOracle(data, pool, limits, self.options, self.env)
        causal = NonanticipativeOracle(data, pool, limits, self.options, self.env,
                                      dispatch_options=NonanticipativeOptions(initial_soc=0))
        try:
            previous = old.evaluate(units)
            result = causal.evaluate(units)
            self.assertTrue(previous.feasible)
            self.assertAlmostEqual(previous.eens_kwh, .5)
            self.assertFalse(result.feasible)
            self.assertAlmostEqual(result.eens_kwh, 1.)
            self.assertAlmostEqual(result.information["joint_violation_bound_kwh"], .25)
            self.assertIsNone(result.standard_error_kwh)
            for d in causal.last_dispatches:
                self.assertAlmostEqual(d["diesel_unit_online"][0, 0], 0)
                self.assertAlmostEqual(d["shed_kw"][0], 1)
            # Controls can react after the different t=1 load is observed.
            self.assertAlmostEqual(causal.last_dispatches[0]["diesel_kw"][1], 1)
            self.assertAlmostEqual(causal.last_dispatches[1]["diesel_kw"][1], 0)
            with tempfile.TemporaryDirectory() as d:
                causal.save_policy(Path(d) / "policy.npz")
                with np.load(Path(d) / "policy.npz") as p:
                    self.assertEqual(p["nodes"][0, 0], p["nodes"][1, 0])
                    self.assertNotEqual(p["nodes"][0, 1], p["nodes"][1, 1])
            bad = [{k: v.copy() for k, v in d.items()} for d in causal.last_dispatches]
            bad[1]["diesel_unit_online"][0, 0] = 1
            self.assertFalse(audit_nonanticipativity(bad, causal.last_nodes, 0)["passed"])
        finally:
            old.close()
            causal.close()

    def test_histories_do_not_merge_and_uninstalled_modules_not_observed(self):
        data, pool, units = branching_system()
        paths = {k: np.repeat(a[:, :, :1], 3, axis=2) for k, a in pool.availability.items()}
        paths["diesel"][1, 1, 0] = 0  # An uninstalled module conveys no information.
        load = np.array([[1., 1., 1.], [1., 0., 1.]])
        one = np.ones((2, 3))
        p = ScenarioPool(paths, np.zeros((2, 3), dtype=np.uint8), np.array([.5, .5]), one, one, load)
        nodes = observation_nodes(p, units)
        self.assertEqual(nodes[0, 0], nodes[1, 0])
        self.assertNotEqual(nodes[0, 1], nodes[1, 1])
        self.assertNotEqual(nodes[0, 2], nodes[1, 2])
        expanded = observation_nodes(p, {**units, "diesel": 2})
        self.assertNotEqual(expanded[0, 0], expanded[1, 0])

    def test_initial_energy_common_and_policy_physics(self):
        data, pool, _ = small_system(True)
        units = {k: hi for k, (_, hi) in data.unit_bounds.items()}
        oracle = NonanticipativeOracle(data, pool, ReliabilityLimits(100, 100), self.options, self.env,
                                      dispatch_options=NonanticipativeOptions(initial_soc=.4))
        result = oracle.evaluate(units)
        self.assertTrue(result.feasible)
        self.assertTrue(result.information["nonanticipativity_audit"]["passed"])
        for d in oracle.last_dispatches:
            self.assertAlmostEqual(d["usable_energy_kwh"][0], 1.6)
            self.assertAlmostEqual(d["usable_energy_kwh"][-1], 1.6)
        oracle.close()

    def test_export_uses_requested_cached_capacity_policy(self):
        data, pool, units = branching_system()
        oracle = NonanticipativeOracle(data, pool, ReliabilityLimits(.75, 10, .5), self.options, self.env,
                                      dispatch_options=NonanticipativeOptions(initial_soc=0))
        try:
            oracle.evaluate(units)
            oracle.evaluate({**units, "diesel": 2})
            oracle.evaluate(units)  # Cache hit does not replace last_dispatches.
            with tempfile.TemporaryDirectory() as d:
                oracle.save_policy(Path(d) / "old_policy.npz", units)
                with np.load(Path(d) / "old_policy.npz") as p:
                    self.assertEqual(p["units"][COMPONENTS.index("diesel")], 1)
                    self.assertTrue(np.all(p["diesel_unit_online"][:, 1] == 0))
        finally:
            oracle.close()

    def test_planning_cuts_match_exhaustive_joint_feasibility(self):
        data, pool, limits = small_system(True)
        make = lambda: NonanticipativeOracle(data, pool, limits, self.options, self.env,
                                             dispatch_options=NonanticipativeOptions(initial_soc=.5))
        search, grid = make(), make()
        result = optimize_reliability(data, search, self.options, max_iterations=48, lift_cuts=True, env=self.env)
        truth = enumerate_grid(data, grid, self.options, self.env)
        self.assertIsNotNone(result.solution)
        self.assertAlmostEqual(result.solution.objective_yuan, truth["best"]["objective_yuan"], places=5)
        for cut in result.cuts:
            self.assertEqual(cut.information["failure_proof"], "positive_joint_violation_lower_bound")
            self.assertGreater(cut.information["joint_violation_bound_kwh"], limits.tolerance_kwh)
            for row in truth["rows"]:
                if row["reliability_feasible"]:
                    self.assertFalse(cut.excludes(row["units"]))
        search.close()
        grid.close()

    def test_default_config_and_refuse_false_holdout_and_oversized_tree(self):
        self.assertEqual(parse_args([]).config.name, "zhongshan_nonanticipative.toml")
        with tempfile.TemporaryDirectory() as d:
            code = main(["plan", "--hours", "2", "--samples", "2", "--validation-samples", "1", "--output", d])
            self.assertEqual(code, 1)
            self.assertIn("not validate", (Path(d) / "error.json").read_text())
        with self.assertRaisesRegex(ValueError, "scenario-hours"):
            NonanticipativeOracle.check_size(20000, 8760, NonanticipativeOptions())


if __name__ == "__main__":
    unittest.main()
