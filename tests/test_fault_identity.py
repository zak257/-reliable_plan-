"""Verify that contingency identity and simultaneous outages reach UC dispatch."""
from dataclasses import replace
from itertools import combinations
import unittest

import gurobipy as gp
import numpy as np

from polar_reliability_planning.config import SolverOptions
from polar_reliability_planning.data.cap_plan_loader import COMPONENTS, FAILABLE_COMPONENTS
from polar_reliability_planning.reliability.operation_model import OperationModel
from polar_reliability_planning.scenario_generation import ScenarioPool, generate_pool
from polar_reliability_planning.validation.small_system import small_system


class FaultIdentityTests(unittest.TestCase):
    def test_each_component_and_module_has_its_own_failure_trajectory(self):
        data, _, _ = small_system(True)
        data = replace(data, unit_bounds={key: (0, 3) for key in COMPONENTS})
        failures = {key: {"normal_rate_per_hour": 0.2, "mean_repair_hours": 3}
                    for key in FAILABLE_COMPONENTS}
        pool = generate_pool(data, 200, 20260908, failures)
        for key in FAILABLE_COMPONENTS:
            paths = pool.availability[key]
            for module in range(3):
                self.assertTrue(np.any(paths[:, module] == 0), (key, module))
                self.assertTrue(np.any(paths[:, module] == 1), (key, module))
            self.assertFalse(np.array_equal(paths[:, 0], paths[:, 1]), key)
            self.assertTrue(np.any((paths[:, 0] == 0) & (paths[:, 1] == 0)), key)

    def test_all_installed_module_identities_reach_available_capacity(self):
        data, _, _ = small_system(True)
        data = replace(data, unit_bounds={key: (0, 3) for key in COMPONENTS})
        one = np.ones((1, data.hours))
        units = {key: 2 for key in COMPONENTS}
        for key in FAILABLE_COMPONENTS:
            for failed_module in range(3):
                with self.subTest(component=key, failed_module=failed_module):
                    availability = {k: np.ones((1, 3, data.hours), dtype=np.uint8)
                                    for k in FAILABLE_COMPONENTS}
                    availability[key][0, failed_module] = 0
                    pool = ScenarioPool(availability, np.zeros_like(one), np.ones(1), one, one, one)
                    scenario = pool.scenario(data, units, 0)
                    surviving = 1 if failed_module < 2 else 2
                    factor = data.wind_pu if key == "wind" else data.pv_pu if key == "pv" else 1
                    expected = surviving * data.module_sizes[key] * factor
                    np.testing.assert_allclose(getattr(scenario, f"{key}_available_kw"), expected)
                    if key == "diesel":
                        np.testing.assert_array_equal(scenario.diesel_module_availability,
                                                      availability[key][0, :2])

    def test_uc_honors_any_single_double_and_triple_generator_outage(self):
        data, _, _ = small_system(True)
        data = replace(data, load_kw=np.full(data.hours, 6.),
                       unit_bounds={**data.unit_bounds, "diesel": (0, 3)})
        units = {key: 0 for key in COMPONENTS}
        units["diesel"] = 3  # Each unit is 4 kW; two units can meet the 6 kW load.
        one = np.ones((1, data.hours))
        with gp.Env(params={"OutputFlag": 0}) as env:
            operation = OperationModel(data, SolverOptions(time_limit=30, oracle_time_limit=30), env)
            try:
                for count in (1, 2, 3):
                    for failed in combinations(range(3), count):
                        with self.subTest(failed=failed):
                            availability = {key: np.ones((1, data.unit_bounds[key][1], data.hours), dtype=np.uint8)
                                            for key in FAILABLE_COMPONENTS}
                            availability["diesel"][0, list(failed)] = 0
                            pool = ScenarioPool(availability, np.zeros_like(one), np.ones(1), one, one, one)
                            loss, dispatch = operation.solve(units, pool.scenario(data, units, 0), True)
                            expected = max(0, 6 - 4 * (3 - count)) * data.hours
                            self.assertAlmostEqual(loss, expected, places=7)
                            self.assertTrue(np.all(dispatch["diesel_unit_online"][list(failed)] == 0))
            finally:
                operation.close()
