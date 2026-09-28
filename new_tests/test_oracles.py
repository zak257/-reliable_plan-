from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np

from certified_reliability_planning.oracles import Interval, MicrogridOracle
from certified_reliability_planning.sampling import SharedScenarioStream
from polar_reliability_planning.config import SolverOptions
from polar_reliability_planning.data import COMPONENTS
from polar_reliability_planning.data.cap_plan_loader import FAILABLE_COMPONENTS
from polar_reliability_planning.reliability.operation_model import OperationModel
from polar_reliability_planning.scenario_generation import Scenario, ScenarioPool, generate_pool
from polar_reliability_planning.validation.small_system import small_system


class SharedStreamTests(unittest.TestCase):
    def setUp(self):
        self.data, _, _ = small_system(True)
        self.failures = {k: {"normal_rate_per_hour": 0.3, "extreme_rate_per_hour": 0.9,
                             "mean_repair_hours": 2} for k in FAILABLE_COMPONENTS}
        self.weather = {"enabled": True, "normal_to_extreme_rate_per_hour": 0.4,
                        "extreme_to_normal_rate_per_hour": 0.1,
                        "extreme_load_factor": 1.3, "extreme_wind_factor": 0.4}

    def test_extension_preserves_paths_and_matches_legacy_generator(self):
        stream = SharedScenarioStream(self.data, 72, self.failures, self.weather)
        stream.extend(2)
        prefix = stream.to_pool()
        from certified_reliability_planning import sampling
        original = sampling.generate_failure
        with patch.object(sampling, "generate_failure", wraps=original) as generator:
            stream.extend(5)
            self.assertEqual(generator.call_count, 3 * sum(self.data.unit_bounds[k][1]
                                                           for k in FAILABLE_COMPONENTS))
            stream.extend(3)
            self.assertEqual(stream.samples, 5)
        reference = generate_pool(self.data, 5, 72, self.failures, self.weather)
        self.assertEqual(stream.fingerprint, reference.fingerprint)
        for key in FAILABLE_COMPONENTS:
            np.testing.assert_array_equal(prefix.availability[key], stream.to_pool().availability[key][:2])
        np.testing.assert_array_equal(prefix.weather, stream.to_pool().weather[:2])

    def test_shared_module_prefix_and_deterministic_weather_bound(self):
        stream = SharedScenarioStream(self.data, 72, self.failures, self.weather).extend(5)
        small = {k: 0 for k in COMPONENTS}
        small["diesel"] = 1
        large = dict(small, diesel=2)
        for index in range(5):
            a, b = stream.scenario(small, index), stream.scenario(large, index)
            np.testing.assert_array_equal(a.diesel_module_availability, b.diesel_module_availability[:1])
            self.assertLessEqual(float(a.load_kw.sum() * self.data.dt_hours), stream.loss_bound + 1e-12)
        self.assertAlmostEqual(stream.loss_bound, self.data.demand_kwh * 1.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            stream.save(path)
            restored = ScenarioPool.load(path)
            self.assertEqual(restored.fingerprint, stream.fingerprint)
            self.assertEqual(restored.metadata["samples"], 5)

    def test_global_bound_covers_rounding_in_pointwise_weather_loads(self):
        rng = np.random.default_rng(18)
        load = rng.uniform(0, 100, 8760)
        data = replace(self.data, load_kw=load, wind_pu=np.zeros(len(load)), pv_pu=np.zeros(len(load)))
        stream = SharedScenarioStream(data, 72, {}, {"extreme_load_factor": 1.3})
        all_extreme_loss = float(np.sum(data.load_kw * 1.3) * data.dt_hours)
        self.assertGreaterEqual(stream.loss_bound, all_extreme_loss)
        self.assertLess(stream.loss_bound - all_extreme_loss, 1e-4)


class IntervalOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(params={"OutputFlag": 0})
        cls.options = SolverOptions(mip_gap=0, threads=1, oracle_time_limit=30)

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def setUp(self):
        self.data, self.pool, _ = small_system(True)
        self.oracle = MicrogridOracle(self.data, self.options, self.env)

    def test_zero_budget_has_feasible_all_shed_upper(self):
        units = {k: 0 for k in COMPONENTS}
        result = self.oracle.operation(units, Scenario.nominal(self.data, units), 0, 0)
        self.assertEqual((result.lower, result.upper), (0, self.data.demand_kwh))
        self.assertEqual(result.work, 0)
        self.assertTrue(self.oracle.last_audit["physical"]["passed"])
        self.assertTrue(self.oracle.last_audit["commitment"]["passed"])

    def test_constructive_zero_requires_actual_no_shedding_witness(self):
        units = dict(zip(COMPONENTS, [1, 1, 2, 0, 0]))
        result = self.oracle.operation(units, Scenario.nominal(self.data, units), 0, 0)
        self.assertEqual((result.lower, result.upper), (0, 0))
        self.assertEqual(result.status, "constructed_zero_loss")
        self.assertTrue(np.all(result.dispatch["shed_kw"] == 0))

    def test_positive_uc_loss_encloses_legacy_optimum_and_tracks_work(self):
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        scenario = self.pool.scenario(self.data, units, 1)
        old = OperationModel(self.data, self.options, self.env)
        try:
            truth, _ = old.solve(units, scenario)
        finally:
            old.close()
        result = self.oracle.operation(units, scenario, 30, 0)
        self.assertGreater(truth, 0)
        self.assertLessEqual(result.lower, truth)
        self.assertGreaterEqual(result.upper, truth)
        self.assertLess(result.width, 1e-4)
        self.assertGreater(result.work, 0)
        self.assertGreater(result.runtime_seconds, 0)

    def test_limited_incumbent_remains_an_interval_instead_of_exact_loss(self):
        from certified_reliability_planning import oracles
        original = oracles.configure_model

        def limited(model, options, oracle=False):
            original(model, options, oracle)
            model.Params.Presolve = 0
            model.Params.SolutionLimit = 1

        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        scenario = self.pool.scenario(self.data, units, 1)
        old = OperationModel(self.data, self.options, self.env)
        try:
            truth, _ = old.solve(units, scenario)
        finally:
            old.close()
        with patch.object(oracles, "configure_model", side_effect=limited):
            result = self.oracle.operation(units, scenario, 30, 0)
        self.assertEqual(result.status, "solution_limit")
        self.assertLessEqual(result.lower, truth)
        self.assertGreaterEqual(result.upper, truth)
        self.assertGreater(result.width, 1)

    def test_invalid_solver_incumbent_is_discarded(self):
        from certified_reliability_planning import oracles
        original = oracles.dispatch_values

        def broken(block, data, energy):
            dispatch = original(block, data, energy)
            dispatch["shed_kw"] = np.full(data.hours, -100)
            return dispatch

        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        scenario = self.pool.scenario(self.data, units, 1)
        with patch.object(oracles, "dispatch_values", side_effect=broken):
            result = self.oracle.operation(units, scenario, 30, 0)
        self.assertIn("incumbent_audit_failed", result.status)
        self.assertEqual(result.upper, float(scenario.load_kw.sum() * self.data.dt_hours))
        np.testing.assert_array_equal(result.dispatch["shed_kw"], scenario.load_kw)

    def test_aggregate_failure_cannot_replace_unit_identity(self):
        units = dict(zip(COMPONENTS, [1, 1, 1, 0, 0]))
        scenario = self.pool.scenario(self.data, units, 1)
        with self.assertRaises(ValueError):
            self.oracle.operation(units, replace(scenario, diesel_module_availability=None), 30, 0)

    def test_nominal_economic_cost_includes_capital_fuel_and_starts(self):
        data = replace(self.data, unit_commitment=replace(self.data.unit_commitment, startup_cost_yuan=7))
        oracle = MicrogridOracle(data, self.options, self.env)
        units = dict(zip(COMPONENTS, [0, 0, 1, 0, 0]))
        truth = data.period_cost_per_unit["diesel"] + data.demand_kwh * data.fuel_cost_per_kwh + 7
        result = oracle.economic(units, 30, 0)
        self.assertLessEqual(result.lower, truth)
        self.assertGreaterEqual(result.upper, truth)
        self.assertLess(result.width, 1e-4)
        self.assertTrue(oracle.last_audit["no_shedding"])

    def test_nominal_infeasible_is_distinguished_from_timeout(self):
        units = {k: 0 for k in COMPONENTS}
        uncertain = self.oracle.economic(units, 0, 0)
        self.assertEqual(uncertain.lower, 0)
        self.assertEqual(uncertain.upper, float("inf"))
        proved = self.oracle.economic(units, 30, 0)
        self.assertEqual((proved.lower, proved.upper), (float("inf"), float("inf")))
        self.assertEqual(proved.status, "nominal_infeasible")

    def test_positive_tiny_loss_is_never_reported_as_exact_zero(self):
        data = replace(self.data, load_kw=np.full(4, 1e-10))
        oracle = MicrogridOracle(data, self.options, self.env)
        units = {k: 0 for k in COMPONENTS}
        result = oracle.operation(units, Scenario.nominal(data, units), 30, 0)
        self.assertGreater(result.upper, 0)
        self.assertGreaterEqual(result.upper, data.demand_kwh)

    def test_continuous_dispatch_also_returns_an_interval(self):
        data, _, _ = small_system(False)
        oracle = MicrogridOracle(data, self.options, self.env)
        units = dict(zip(COMPONENTS, [1, 0, 0, 0, 0]))
        scenario = Scenario.nominal(data, units)
        truth = float(np.maximum(0, scenario.load_kw - scenario.wind_available_kw).sum())
        result = oracle.operation(units, scenario, 30, 0)
        self.assertLessEqual(result.lower, truth)
        self.assertGreaterEqual(result.upper, truth)
        self.assertLess(result.width, 1e-4)

    def test_interval_rejects_invalid_endpoints(self):
        for endpoints in ((1, 0), (float("nan"), 1), (-1, 2)):
            with self.assertRaises(ValueError):
                Interval(*endpoints)


if __name__ == "__main__":
    unittest.main()
