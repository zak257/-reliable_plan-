from dataclasses import replace
from itertools import product
import unittest

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.config import SolverOptions, UnitCommitmentOptions, ReliabilityLimits
from polar_reliability_planning.data import COMPONENTS
from polar_reliability_planning.reliability import ReliabilityOracle
from polar_reliability_planning.reliability.operation_model import OperationModel, OptimizationError, add_dispatch
from polar_reliability_planning.reliability.unit_commitment import add_unit_commitment, audit_commitment
from polar_reliability_planning.scenario_generation import Scenario
from polar_reliability_planning.validation.small_system import small_system
from polar_reliability_planning.validation.monte_carlo_validation import audit_physical_dispatch
from polar_reliability_planning.planning.optimizer import optimize_reliability


class UnitCommitmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(params={"OutputFlag": 0})
        cls.options = SolverOptions(mip_gap=0, time_limit=30, oracle_time_limit=30)

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def schedule_status(self, online, available, up=3, down=3, power=None):
        data, _, _ = small_system(True)
        h = len(online)
        data = replace(data, load_kw=np.ones(h), wind_pu=np.zeros(h), pv_pu=np.zeros(h),
                       unit_bounds={**data.unit_bounds, "diesel": (0, 1)},
                       unit_commitment=UnitCommitmentOptions(min_up_hours=up, min_down_hours=down))
        with gp.Model(env=self.env) as model:
            p = model.addMVar(h, lb=0)
            block = add_unit_commitment(model, data, p)
            block.update_availability(np.array([available]))
            model.addConstr(block.online == np.array([online]))
            if power is not None:
                model.addConstr(p == power)
            model.optimize()
            return model.Status

    def test_voluntary_shutdown_cannot_break_minimum_uptime(self):
        self.assertEqual(self.schedule_status([1, 0, 0, 0], [1, 1, 1, 1]), GRB.INFEASIBLE)
        self.assertEqual(self.schedule_status([1, 1, 1, 0], [1, 1, 1, 1]), GRB.OPTIMAL)

    def test_forced_trip_overrides_uptime_but_not_downtime(self):
        available = [1, 0, 1, 1, 1, 1]
        self.assertEqual(self.schedule_status([1, 0, 0, 0, 1, 1], available), GRB.OPTIMAL)
        self.assertEqual(self.schedule_status([1, 0, 1, 1, 1, 1], available), GRB.INFEASIBLE)

    def test_old_trip_does_not_waive_a_later_startup_obligation(self):
        self.assertEqual(self.schedule_status([1, 0, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1], up=5, down=1), GRB.INFEASIBLE)

    def test_minimum_output_and_offline_output(self):
        self.assertEqual(self.schedule_status([1, 1, 1], [1, 1, 1], power=np.array([0.7, 1, 1])), GRB.INFEASIBLE)
        self.assertEqual(self.schedule_status([0, 0, 0], [1, 1, 1], power=np.array([1, 0, 0])), GRB.INFEASIBLE)

    def test_simultaneous_charge_discharge_is_forbidden(self):
        data, _, _ = small_system(True)
        data = replace(data, efficiency=1.0)
        with gp.Model(env=self.env) as model:
            block = add_dispatch(model, data, np.zeros(4), np.zeros(4), 0, 2, 4, np.zeros(4), False)
            model.addConstr(block.variables["charge_kw"][0] == 1)
            model.addConstr(block.variables["discharge_kw"][0] == 1)
            model.optimize()
            self.assertEqual(model.Status, GRB.INFEASIBLE)

    def test_zero_loss_shortcut_returns_an_audited_uc_dispatch(self):
        data, _, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 2, 0, 0]))
        operation = OperationModel(data, self.options, self.env)
        try:
            scenario = Scenario.nominal(data, units)
            loss, dispatch = operation.solve(units, scenario, True)
            self.assertEqual(loss, 0)
            self.assertEqual(operation.zero_loss_certificates, 1)
            self.assertTrue(audit_commitment(data, units, scenario, dispatch)["passed"])
        finally:
            operation.close()

    def test_uc_losses_are_monotone_on_all_five_capacity_dimensions(self):
        data, pool, limits = small_system(True)
        oracle = ReliabilityOracle(data, pool, limits, self.options, self.env)
        try:
            points = list(product(*(range(data.unit_bounds[k][1] + 1) for k in COMPONENTS)))
            losses = {point: np.array(oracle.evaluate(dict(zip(COMPONENTS, point))).losses_kwh) for point in points}
            for point in points:
                for j, key in enumerate(COMPONENTS):
                    if point[j] < data.unit_bounds[key][1]:
                        more = list(point)
                        more[j] += 1
                        self.assertTrue(np.all(losses[tuple(more)] <= losses[point] + 1e-7), (point, key))
        finally:
            oracle.close()

    def test_cvar_can_reject_a_uc_plan_that_passes_eens(self):
        data, pool, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        oracle = ReliabilityOracle(data, pool, ReliabilityLimits(2.0, 1.0), self.options, self.env)
        try:
            result = oracle.evaluate(units)
            self.assertTrue(result.eens_feasible)
            self.assertFalse(result.cvar_feasible)
            self.assertEqual(result.summary()["violated_constraints"], ["CVaR"])
        finally:
            oracle.close()

    def test_startup_cost_is_charged_on_each_actual_start(self):
        data, _, _ = small_system(True)
        data = replace(data, unit_commitment=replace(data.unit_commitment, startup_cost_yuan=7.0))
        units = dict(zip(COMPONENTS, [0, 0, 1, 0, 0]))
        operation = OperationModel(data, self.options, self.env)
        try:
            cost, dispatch = operation.solve(units, Scenario.nominal(data, units), True, objective="nominal_cost")
            self.assertAlmostEqual(cost, data.demand_kwh * data.fuel_cost_per_kwh + 7)
            self.assertEqual(round(float(dispatch["diesel_startup_units"].sum())), 1)
        finally:
            operation.close()

    def test_limited_economic_incumbent_is_usable_but_never_labelled_optimal(self):
        data, _, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 2, 1, 1]))
        operation = OperationModel(data, self.options, self.env)
        try:
            scenario = Scenario.nominal(data, units)
            _, start = operation.solve(units, scenario, True)
            # A deterministic solver limit after the feasible MIP start avoids
            # a flaky wall-clock timeout in this regression.
            operation.model.Params.Presolve = 0
            operation.model.Params.SolutionLimit = 1
            cost, dispatch = operation.solve(units, scenario, True, objective="nominal_cost", warm_start=start)
            self.assertEqual(operation.last_solve_summary["status"], GRB.SOLUTION_LIMIT)
            self.assertFalse(operation.last_solve_summary["optimal"])
            self.assertTrue(audit_physical_dispatch(data, units, scenario, dispatch)["passed"])
            self.assertAlmostEqual(cost, data.fuel_cost_per_kwh * data.dt_hours * dispatch["diesel_kw"].sum())
        finally:
            operation.close()

    def test_limited_positive_loss_cannot_be_used_as_exact_reliability(self):
        data, pool, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        operation = OperationModel(data, self.options, self.env)
        try:
            operation.model.Params.Presolve = 0
            operation.model.Params.SolutionLimit = 1
            with self.assertRaisesRegex(OptimizationError, "optimal losses are required"):
                operation.solve(units, pool.scenario(data, units, 1))
            self.assertIsNone(operation.last_solve_summary)
        finally:
            operation.close()

    def test_physical_audit_detects_corrupted_power_and_energy(self):
        data, _, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 2, 1, 1]))
        operation = OperationModel(data, self.options, self.env)
        try:
            scenario = Scenario.nominal(data, units)
            _, dispatch = operation.solve(units, scenario, True)
            self.assertTrue(audit_physical_dispatch(data, units, scenario, dispatch)["passed"])
            for key in ("diesel_kw", "usable_energy_kwh"):
                corrupt = {k: v.copy() for k, v in dispatch.items()}
                corrupt[key][1] += 0.25
                self.assertFalse(audit_physical_dispatch(data, units, scenario, corrupt)["passed"])
        finally:
            operation.close()

    def test_partial_sample_bound_rejects_safely_and_cache_can_be_completed(self):
        data, pool, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        oracle = ReliabilityOracle(data, pool, ReliabilityLimits(2.0, 1.0), self.options, self.env)
        try:
            partial = oracle.evaluate(units, certify_infeasible_early=True)
            self.assertTrue(partial.metrics_are_lower_bounds)
            self.assertFalse(partial.feasible)
            self.assertIsNone(partial.eens_feasible)
            self.assertFalse(partial.cvar_feasible)
            self.assertIsNone(partial.losses_kwh[-1])
            count = oracle.operation.solve_count
            complete = oracle.evaluate(units)
            self.assertFalse(complete.metrics_are_lower_bounds)
            self.assertTrue(complete.eens_feasible)
            self.assertFalse(complete.cvar_feasible)
            self.assertGreaterEqual(complete.eens_kwh + 1e-7, partial.eens_kwh)
            self.assertGreaterEqual(complete.cvar_kwh + 1e-7, partial.cvar_kwh)
            expected_new = len(partial.losses_kwh) if partial.losses_are_relaxation_bounds else sum(q is None for q in partial.losses_kwh)
            self.assertEqual(oracle.operation.solve_count, count + expected_new)
        finally:
            oracle.close()

    def test_relaxation_may_only_certify_failure(self):
        data, pool, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        oracle = ReliabilityOracle(data, pool, ReliabilityLimits(2.0, 0.5), self.options, self.env)
        try:
            certificate = oracle.certify_failure(units)
            self.assertIsNotNone(certificate)
            self.assertTrue(certificate.losses_are_relaxation_bounds)
            self.assertFalse(certificate.feasible)
            self.assertEqual(oracle.operation.solve_count, 0)
            full = oracle.evaluate(units)
            self.assertFalse(full.losses_are_relaxation_bounds)
            self.assertGreaterEqual(full.cvar_kwh, certificate.cvar_kwh - 1e-7)
        finally:
            oracle.close()

    def test_lifted_uc_search_preserves_exhaustive_optimum(self):
        data, pool, limits = small_system(True)
        oracle = ReliabilityOracle(data, pool, limits, self.options, self.env)
        try:
            result = optimize_reliability(data, oracle, self.options, max_iterations=48, lift_cuts=True, env=self.env)
            self.assertIsNotNone(result.solution)
            self.assertTrue(result.reliability.feasible)
            self.assertFalse(result.reliability.metrics_are_lower_bounds)
            self.assertAlmostEqual(result.solution.objective_yuan, 12.1)
        finally:
            oracle.close()


if __name__ == "__main__":
    unittest.main()
