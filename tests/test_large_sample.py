from dataclasses import replace
import math
import unittest

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.config import SolverOptions
from polar_reliability_planning.data import COMPONENTS
from polar_reliability_planning.reliability import ReliabilityOracle
from polar_reliability_planning.reliability.operation_model import OperationModel
from polar_reliability_planning.reliability.dispatch_audit import audit_physical_dispatch
from polar_reliability_planning.reliability.unit_commitment import standby_commitment
from polar_reliability_planning.scenario_generation import Scenario
from polar_reliability_planning.scenario_generation.failure_generator import FailureParameters, generate_failure
from polar_reliability_planning.validation.small_system import small_system


class EventSimulationTests(unittest.TestCase):
    def test_event_failure_simulation_preserves_original_hourly_random_paths(self):
        for seed in range(12):
            weather = np.random.default_rng(seed).integers(0, 2, 2000, dtype=np.uint8)
            for params in (FailureParameters(1e-4, 5e-4, 48, 2), FailureParameters(0.5, 0.01, 0.8, 3), FailureParameters()):
                dt = 0.5
                pf = [-math.expm1(-rate * dt) for rate in (params.normal_rate_per_hour, params.extreme_rate_per_hour)]
                pr = [-math.expm1(-dt / repair) for repair in (params.mean_repair_hours, params.mean_repair_hours * params.extreme_repair_multiplier)]
                rng = np.random.default_rng(seed)
                w = int(weather[0])
                available = rng.random() >= pf[w] / (pf[w] + pr[w])
                draws = rng.random(len(weather) - 1)
                expected = [available]
                for t, draw in enumerate(draws, start=1):
                    w = int(weather[t])
                    if draw < (pf[w] if available else pr[w]):
                        available = not available
                    expected.append(available)
                actual = generate_failure(weather, params, np.random.default_rng(seed), dt)
                np.testing.assert_array_equal(actual, expected)

    def test_event_standby_matches_hourly_policy_including_restart_deadlines(self):
        for seed in range(10):
            availability = np.random.default_rng(seed).integers(0, 2, (5, 100))
            for down in (1, 3, 20):
                for target in (0, 1, 2, 6):
                    active = np.zeros(5, dtype=bool)
                    restart = np.zeros(5, dtype=int)
                    expected = np.zeros_like(availability)
                    for t in range(100):
                        trips = active & (availability[:, t] == 0)
                        restart[trips] = t + down
                        active[trips] = False
                        for i in range(5):
                            if active.sum() >= target:
                                break
                            if not active[i] and availability[i, t] and t >= restart[i]:
                                active[i] = True
                        expected[:, t] = active
                    np.testing.assert_array_equal(standby_commitment(availability, down, target), expected)


class LargeSampleSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(params={"OutputFlag": 0})
        cls.options = SolverOptions(time_limit=30, oracle_time_limit=30, mip_gap=0.01)

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def test_direct_zero_certificate_is_physically_valid_without_solver_search(self):
        data, _, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 2, 0, 0]))
        operation = OperationModel(data, self.options, self.env)
        try:
            scenario = Scenario.nominal(data, units)
            loss, dispatch = operation.solve(units, scenario, True)
            self.assertEqual(loss, 0)
            self.assertEqual(operation.solver_calls, 0)
            self.assertEqual(operation.model.Status, GRB.LOADED)
            self.assertTrue(audit_physical_dispatch(data, units, scenario, dispatch)["passed"])
            self.assertIsNone(operation.last_solve_summary["status"])
        finally:
            operation.close()

    def test_storage_certificate_supplies_a_peak_above_generator_rating(self):
        data, _, _ = small_system(True)
        data = replace(data, load_kw=np.array([3., 5., 3., 3.]))
        units = dict(zip(COMPONENTS, [0, 0, 1, 1, 1]))
        operation = OperationModel(data, self.options, self.env)
        try:
            scenario = Scenario.nominal(data, units)
            loss, dispatch = operation.solve(units, scenario, True)
            self.assertEqual(loss, 0)
            self.assertEqual(operation.storage_zero_loss_certificates, 1)
            self.assertEqual(operation.full_mip_solves, 0)
            self.assertGreaterEqual(dispatch["discharge_kw"][1], 1 - 1e-7)
            self.assertTrue(audit_physical_dispatch(data, units, scenario, dispatch)["passed"])
        finally:
            operation.close()

    def test_economic_gap_does_not_relax_positive_loss_oracle(self):
        data, pool, _ = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        operation = OperationModel(data, self.options, self.env)
        try:
            operation.solve(units, Scenario.nominal(data, units), objective="nominal_cost")
            self.assertEqual(operation.last_solve_summary["requested_mip_gap"], 0.01)
            loss, _ = operation.solve(units, pool.scenario(data, units, 1))
            self.assertGreater(loss, 0)
            self.assertEqual(operation.last_solve_summary["requested_mip_gap"], 0)
            self.assertTrue(operation.last_solve_summary["optimal"])
        finally:
            operation.close()

    def test_checkpoint_callback_records_each_exact_scenario_once(self):
        data, pool, limits = small_system(True)
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        records = []
        oracle = ReliabilityOracle(data, pool, limits, self.options, self.env,
                                   on_scenario=lambda *row: records.append(row))
        try:
            result = oracle.evaluate(units)
            self.assertEqual([row[1] for row in records], list(range(pool.samples)))
            self.assertEqual([row[3] for row in records], list(result.losses_kwh))
            oracle.evaluate(units)
            self.assertEqual(len(records), pool.samples)
        finally:
            oracle.close()
