from __future__ import annotations

from dataclasses import replace
import importlib.util
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.config import ReliabilityLimits, SolverOptions
from polar_reliability_planning.data import CaseData, COMPONENTS, load_case
from polar_reliability_planning.data.cap_plan_loader import FAILABLE_COMPONENTS
from polar_reliability_planning.planning.master_milp import MasterMILP, MasterInfeasible
from polar_reliability_planning.planning.optimizer import optimize_reliability
from polar_reliability_planning.planning.reliability_cuts import ReliabilityCut, add_reliability_cut
from polar_reliability_planning.reliability import ReliabilityOracle
from polar_reliability_planning.reliability.cvar_calculation import calculate_CVaR
from polar_reliability_planning.reliability.eens_calculation import calculate_EENS
from polar_reliability_planning.reliability.operation_model import OperationModel
from polar_reliability_planning.scenario_generation import Scenario, ScenarioPool, generate_pool
from polar_reliability_planning.scenario_generation.climate_generator import generate_weather
from polar_reliability_planning.scenario_generation.failure_generator import FailureParameters, generate_failure
from polar_reliability_planning.validation.monte_carlo_validation import audit_nominal_solution
from polar_reliability_planning.validation.small_system import small_system, validate_small_system
from polar_reliability_planning.validation.sensitivity_analysis import run_sensitivity


class MetricsTests(unittest.TestCase):
    def test_weighted_eens(self):
        self.assertAlmostEqual(calculate_EENS([0, 10, 100], [0.5, 0.4, 0.1]), 14)

    def test_fractional_cvar_mass_and_no_input_mutation(self):
        losses = [100, 0, 10]
        self.assertAlmostEqual(calculate_CVaR(losses, 0.85), 100)
        self.assertEqual(losses, [100, 0, 10])
        self.assertAlmostEqual(calculate_CVaR([0, 10, 100], 0.85, [0.5, 0.4, 0.1]), 70)
        # ceil/floor slicing is wrong here: the 25% tail is 20%*100 + 5%*10.
        self.assertAlmostEqual(calculate_CVaR([0, 0, 0, 10, 100], 0.75), 82)

    def test_cvar_alpha_zero_is_mean_and_ties(self):
        self.assertAlmostEqual(calculate_CVaR([2, 2, 9], 0, [0.3, 0.2, 0.5]), 5.5)
        self.assertAlmostEqual(calculate_CVaR([5, 5, 5], 0.9999), 5)

    def test_invalid_metrics_fail_explicitly(self):
        for losses, p in [([], None), ([float("nan")], None), ([-1], None), ([1, 2], [0.2, 0.2])]:
            with self.assertRaises(ValueError):
                calculate_EENS(losses, p)
        with self.assertRaises(ValueError):
            calculate_CVaR([1], 1)
        with self.assertRaises(ValueError):
            ReliabilityLimits(-1)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.data, _, _ = small_system()

    def test_zero_failure_rates_and_persistent_save_load(self):
        pool = generate_pool(self.data, 3, 9, {})
        for a in pool.availability.values():
            self.assertTrue(np.all(a == 1))
            self.assertFalse(a.flags.writeable)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.npz"
            pool.save(path)
            restored = ScenarioPool.load(path)
            self.assertEqual(pool.fingerprint, restored.fingerprint)
            self.assertEqual(pool.metadata, restored.metadata)

    def test_streams_remain_identical_when_pool_expands(self):
        failures = {k: {"normal_rate_per_hour": 0.3, "mean_repair_hours": 2} for k in FAILABLE_COMPONENTS}
        small = generate_pool(self.data, 2, 42, failures)
        larger_data = replace(self.data, unit_bounds={k: (0, hi + 2) for k, (_, hi) in self.data.unit_bounds.items()})
        large = generate_pool(larger_data, 4, 42, failures)
        for k in FAILABLE_COMPONENTS:
            np.testing.assert_array_equal(small.availability[k], large.availability[k][:2, :self.data.unit_bounds[k][1]])

    def test_repair_returns_equipment_to_service(self):
        path = generate_failure(np.zeros(200, dtype=np.uint8), FailureParameters(1, 1, 1), np.random.default_rng(3))
        self.assertTrue(np.any((path[:-1] == 1) & (path[1:] == 0)))
        self.assertTrue(np.any((path[:-1] == 0) & (path[1:] == 1)))

    def test_climate_conditions_failure_probability(self):
        params = FailureParameters(normal_rate_per_hour=0, extreme_rate_per_hour=10, mean_repair_hours=10000)
        normal = generate_failure(np.zeros(30, dtype=np.uint8), params, np.random.default_rng(7))
        extreme = generate_failure(np.ones(30, dtype=np.uint8), params, np.random.default_rng(7))
        self.assertTrue(normal.all())
        self.assertEqual(int(extreme.sum()), 0)
        weather = generate_weather(50, 1, np.random.default_rng(2), {"enabled": True,
            "normal_to_extreme_rate_per_hour": 10, "extreme_to_normal_rate_per_hour": 0})
        self.assertTrue(weather.all())

    def test_battery_failure_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            generate_pool(self.data, 1, 1, {"battery_energy": {"normal_rate_per_hour": 0.1}})


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(params={"OutputFlag": 0})
        cls.options = SolverOptions(mip_gap=0, time_limit=30, oracle_time_limit=30, threads=1)

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def test_analytical_shortage_without_storage(self):
        data, pool, limits = small_system()
        units = {k: 0 for k in COMPONENTS}
        units["diesel"] = 1
        units["wind"] = 1
        oracle = ReliabilityOracle(data, pool, limits, self.options, self.env)
        try:
            result = oracle.evaluate(units)
            expected = []
            for s in range(pool.samples):
                scenario = pool.scenario(data, units, s)
                expected.append(float(np.maximum(0, data.load_kw - scenario.wind_available_kw - scenario.diesel_available_kw).sum()))
            np.testing.assert_allclose(result.losses_kwh, expected, atol=1e-8)
        finally:
            oracle.close()

    def test_cyclic_soc_cannot_create_free_energy(self):
        data, _, _ = small_system()
        units = {k: 0 for k in COMPONENTS}
        units.update(battery_energy=1, pcs=1)
        operation = OperationModel(data, self.options, self.env)
        try:
            loss, dispatch = operation.solve(units, Scenario.nominal(data, units), return_dispatch=True)
            self.assertAlmostEqual(loss, data.demand_kwh)
            self.assertAlmostEqual(dispatch["stored_energy_kwh"][0], dispatch["stored_energy_kwh"][-1])
        finally:
            operation.close()

    def test_storage_shift_efficiency_and_pcs_failure(self):
        data, _, _ = small_system()
        data = replace(data, load_kw=np.array([0, 2]), wind_pu=np.array([2, 0]), pv_pu=np.zeros(2), efficiency=0.5)
        units = {k: 0 for k in COMPONENTS}
        units.update(wind=1, battery_energy=1, pcs=1)
        scenario = Scenario.nominal(data, units)
        operation = OperationModel(data, self.options, self.env)
        try:
            # Charge <= 2 kW for one hour, round-trip 0.25 -> at most 0.5 kWh supplied.
            loss, dispatch = operation.solve(units, scenario, True)
            self.assertAlmostEqual(loss, 1.5)
            blocked = replace(scenario, pcs_available_kw=np.zeros(2))
            self.assertAlmostEqual(operation.solve(units, blocked)[0], 2.0)
            self.assertLessEqual(float((dispatch["charge_kw"] + dispatch["discharge_kw"]).max()), 2 + 1e-8)
        finally:
            operation.close()

    def test_oracle_monotone_in_each_of_five_dimensions(self):
        data, pool, limits = small_system()
        oracle = ReliabilityOracle(data, pool, limits, self.options, self.env)
        try:
            points = list(product(*(range(data.unit_bounds[k][1] + 1) for k in COMPONENTS)))
            losses = {point: oracle.evaluate(dict(zip(COMPONENTS, point))).losses_kwh for point in points}
            for point in points:
                for j, key in enumerate(COMPONENTS):
                    if point[j] == data.unit_bounds[key][1]:
                        continue
                    greater = list(point)
                    greater[j] += 1
                    self.assertTrue(np.all(np.array(losses[tuple(greater)]) <= np.array(losses[point]) + 1e-7), (point, key))
            before = oracle.operation.solve_count
            oracle.evaluate(dict(zip(COMPONENTS, points[0])))
            self.assertEqual(before, oracle.operation.solve_count)
        finally:
            oracle.close()

    def test_disjunctive_cut_matches_entire_grid(self):
        data, pool, _ = small_system()
        bad = (0, 0, 1, 0, 0)
        cut = ReliabilityCut(bad, pool.fingerprint, 1, 1)
        with gp.Model(env=self.env) as model:
            v = {k: model.addVar(lb=0, ub=data.unit_bounds[k][1], vtype=GRB.INTEGER) for k in COMPONENTS}
            add_reliability_cut(model, v, data.unit_bounds, cut, 0)
            for values in product(*(range(data.unit_bounds[k][1] + 1) for k in COMPONENTS)):
                point = dict(zip(COMPONENTS, values))
                for k, value in point.items():
                    v[k].LB = value
                    v[k].UB = value
                model.optimize()
                self.assertEqual(model.Status == GRB.INFEASIBLE, cut.excludes(point))

    def test_all_max_cut_makes_model_infeasible(self):
        data, pool, _ = small_system()
        cut = ReliabilityCut(tuple(data.unit_bounds[k][1] for k in COMPONENTS), pool.fingerprint, 1, 1)
        master = MasterMILP(data, self.options, [cut], self.env)
        try:
            with self.assertRaises(MasterInfeasible):
                master.solve()
        finally:
            master.close()

    def test_exact_grid_and_boundary_costs_agree(self):
        result = validate_small_system(self.env)
        self.assertTrue(result["passed"])
        self.assertEqual(result["grid"]["grid_points"], 48)
        self.assertGreater(len(result["boundary"]["cuts"]), 0)

    def test_lifted_cuts_keep_exact_optimum(self):
        data, pool, limits = small_system()
        oracle = ReliabilityOracle(data, pool, limits, self.options, self.env)
        try:
            result = optimize_reliability(data, oracle, self.options, 48, True, env=self.env)
            self.assertIsNotNone(result.solution)
            self.assertAlmostEqual(result.solution.objective_yuan, 11.3)
        finally:
            oracle.close()

    def test_iteration_limit_does_not_claim_a_feasible_plan(self):
        data, pool, limits = small_system()
        oracle = ReliabilityOracle(data, pool, limits, self.options, self.env)
        try:
            result = optimize_reliability(data, oracle, self.options, max_iterations=1, env=self.env)
            self.assertEqual(result.status, "iteration_limit")
            self.assertIsNone(result.solution)
        finally:
            oracle.close()

    def test_optional_cvar_constraint_changes_feasibility(self):
        data, pool, _ = small_system()
        units = dict(zip(COMPONENTS, (1, 1, 1, 0, 0)))
        mean_only = ReliabilityOracle(data, pool, ReliabilityLimits(1.0), self.options, self.env)
        mean_tail = ReliabilityOracle(data, pool, ReliabilityLimits(1.0, 1.0), self.options, self.env)
        try:
            self.assertTrue(mean_only.evaluate(units).feasible)
            self.assertFalse(mean_tail.evaluate(units).feasible)
        finally:
            mean_only.close()
            mean_tail.close()

    def test_sensitivity_uses_same_pool_and_distinguishes_limits(self):
        data, pool, limits = small_system()
        limits = replace(limits, cvar_kwh=None)  # Isolate EENS in this comparison.
        history = []
        rows = run_sensitivity(data, pool, limits, [0.05, 1.0], self.options,
                               env=self.env, on_iteration=history.append)
        self.assertAlmostEqual(rows[0]["solution"]["objective_yuan"], 11.3)
        self.assertAlmostEqual(rows[1]["solution"]["objective_yuan"], 10.5)
        self.assertEqual({r["reliability"]["sample_fingerprint"] for r in rows}, {pool.fingerprint})
        self.assertEqual({r["sensitivity_eens_limit_kwh"] for r in history}, {0.05, 1.0})


@unittest.skipUnless(Path("/home/yzk/cap_plan/data/changcheng/input/curve.csv").is_file(), "Optional local cap_plan data is unavailable")
class CapPlanIntegrationTests(unittest.TestCase):
    def test_module_upper_override_preserves_inputs_and_costs(self):
        baseline = load_case("/home/yzk/cap_plan/data", "zhongshan", hours=24)
        expanded = load_case("/home/yzk/cap_plan/data", "zhongshan", hours=24,
                             max_units={"wind": 10, "pv": 10})
        self.assertEqual(expanded.unit_bounds["wind"], (0, 10))
        self.assertEqual(expanded.unit_bounds["pv"], (0, 10))
        for key in ("diesel", "battery_energy", "pcs"):
            self.assertEqual(expanded.unit_bounds[key], baseline.unit_bounds[key])
        self.assertEqual(expanded.annual_cost_per_unit, baseline.annual_cost_per_unit)
        self.assertEqual(expanded.manifest, baseline.manifest)
        np.testing.assert_array_equal(expanded.load_kw, baseline.load_kw)
        for invalid in ({"wind": -1}, {"wind": 1.5}, {"wind": True}, {"typo": 10}):
            with self.assertRaises(ValueError):
                load_case("/home/yzk/cap_plan/data", "zhongshan", hours=24, max_units=invalid)

    def test_loader_matches_reference_full_year_inputs_and_costs(self):
        source = Path("/home/yzk/cap_plan/cap_plan_dual_bound_gurobi.py")
        if not source.is_file():
            self.skipTest("cap_plan reference not installed")
        spec = importlib.util.spec_from_file_location("cap_plan_test_reference", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for case in ("changcheng", "zhongshan"):
            reference = module.CaseData(case, 0, 365, Path("/home/yzk/cap_plan/data"))
            actual = load_case("/home/yzk/cap_plan/data", case)
            np.testing.assert_allclose(actual.load_kw, reference.load_kw)
            np.testing.assert_allclose(actual.wind_pu, reference.wind_pu)
            np.testing.assert_allclose(actual.pv_pu, reference.pv_pu)
            for key in COMPONENTS:
                reference_key = "ess" if key == "battery_energy" else key
                self.assertEqual(actual.unit_bounds[key], reference.unit_bounds[reference_key])
                self.assertAlmostEqual(actual.period_cost_per_unit[key], reference.capacity_cost_per_unit[reference_key])
            self.assertEqual(actual.efficiency, reference.ess_efficiency)
            self.assertEqual(actual.fuel_cost_per_kwh, reference.fuel_cost_per_kwh)

    def test_nominal_master_and_fixed_capacity_dispatch_agree(self):
        data = load_case("/home/yzk/cap_plan/data", "changcheng", hours=48)
        options = SolverOptions(mip_gap=0)
        with gp.Env(params={"OutputFlag": 0}) as env:
            master = MasterMILP(data, options, env=env)
            try:
                solution = master.solve()
                audit = audit_nominal_solution(data, solution, options, env)
                self.assertTrue(audit["no_shedding_passed"])
                self.assertTrue(audit["dispatch_cost_consistent"])
            finally:
                master.close()

    def test_invalid_hour_range_and_off_grid_capacity_rejected(self):
        with self.assertRaises(ValueError):
            load_case("/home/yzk/cap_plan/data", "changcheng", start_hour=8759, hours=2)
        data, _, _ = small_system()
        with self.assertRaises(ValueError):
            data.validate_units({k: 0.5 for k in COMPONENTS})


if __name__ == "__main__":
    unittest.main()
