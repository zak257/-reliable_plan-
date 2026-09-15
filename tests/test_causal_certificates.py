from dataclasses import replace
import unittest
import numpy as np
import gurobipy as gp

from polar_reliability_planning.config import SolverOptions, ReliabilityLimits, UnitCommitmentOptions
from polar_reliability_planning.data import CaseData, COMPONENTS
from polar_reliability_planning.data.cap_plan_loader import FAILABLE_COMPONENTS
from polar_reliability_planning.scenario_generation import ScenarioPool
from polar_reliability_planning.validation.small_system import small_system
from polar_reliability_planning.reliability.causal_certificates import (
    ColdReserveController, evaluate_controller, NonanticipativeCertificateOracle)
from polar_reliability_planning.reliability.nonanticipative import NonanticipativeOptions
from polar_reliability_planning.reliability.operation_model import OptimizationError
from polar_reliability_planning.reliability.dispatch_audit import audit_physical_dispatch
from polar_reliability_planning.reliability.unit_commitment import audit_commitment


class CausalCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(params={"OutputFlag": 0})

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def test_future_changes_do_not_change_controls_before_observation(self):
        data, pool, _ = small_system(True)
        units = {k: hi for k, (_, hi) in data.unit_bounds.items()}
        controller = ColdReserveController(data, units)
        a = pool.scenario(data, units, 0)
        paths = a.diesel_module_availability.copy()
        paths[:, 2] = 0
        b = replace(a, diesel_module_availability=paths,
                    diesel_available_kw=paths.sum(axis=0) * data.module_sizes["diesel"],
                    load_kw=np.array([3., 4., 3.5, 3.5]))
        da, db = controller.dispatch(a), controller.dispatch(b)
        for k in da:
            np.testing.assert_array_equal(da[k][..., :2], db[k][..., :2])
        self.assertFalse(np.array_equal(da["diesel_unit_online"], db["diesel_unit_online"]))
        self.assertTrue(audit_commitment(data, units, b, db)["passed"])
        self.assertTrue(audit_physical_dispatch(data, units, b, db)["passed"])

    def test_cyclic_soc_and_same_rule_in_holdout(self):
        data, pool, _ = small_system(True)
        units = {k: hi for k, (_, hi) in data.unit_bounds.items()}
        limits = ReliabilityLimits(100, 100)
        result = evaluate_controller(data, pool, limits, units)
        validation = evaluate_controller(data, pool, limits, units, sample_role="independent_holdout")
        self.assertEqual(result.information["controller_sha256"], validation.information["controller_sha256"])
        self.assertEqual(result.losses_kwh, validation.losses_kwh)
        controller = ColdReserveController(data, units)
        for s in range(pool.samples):
            dispatch = controller.dispatch(pool.scenario(data, units, s))
            np.testing.assert_array_equal(dispatch["usable_energy_kwh"], np.full(data.hours + 1, .8 * 4))
        self.assertIsNone(result.standard_error_kwh)

    def test_perfect_foresight_pass_cannot_accept_failing_controller(self):
        sizes = dict(zip(COMPONENTS, (2., 1., 1., 1., 1.)))
        bounds = {k: (0, int(k in ("wind", "battery_energy", "pcs"))) for k in COMPONENTS}
        data = CaseData("charging_opportunity", np.ones(2), np.array([1., 0.]), np.zeros(2),
                        sizes, bounds, {k: 0. for k in COMPONENTS}, 1., 1., 0., 1.,
                        unit_commitment=UnitCommitmentOptions(enabled=True))
        availability = {k: np.ones((1, bounds[k][1], 2), dtype=np.uint8) for k in FAILABLE_COMPONENTS}
        one = np.ones((1, 2))
        pool = ScenarioPool(availability, np.zeros((1, 2), dtype=np.uint8), np.ones(1), one, one, one)
        units = {k: b[1] for k, b in bounds.items()}
        oracle = NonanticipativeCertificateOracle(data, pool, ReliabilityLimits(.2, .5), SolverOptions(), self.env,
                      dispatch_options=NonanticipativeOptions(initial_soc=0, max_scenario_hours=1))
        try:
            with self.assertRaisesRegex(OptimizationError, "unresolved"):
                oracle.evaluate(units)
            self.assertFalse(oracle.cache)
            self.assertEqual(oracle.lower.evaluate(units).eens_kwh, 0.)
        finally:
            oracle.close()

    def test_joint_fallback_when_bounds_are_inconclusive(self):
        data, pool, limits = small_system(True)
        units = {k: hi for k, (_, hi) in data.unit_bounds.items()}
        oracle = NonanticipativeCertificateOracle(data, pool, limits, SolverOptions(), self.env)
        try:
            result = oracle.evaluate(units)
            if result.feasible:
                self.assertIn(result.information["information_structure"],
                              ("observed_history_scenario_tree", "causal_policy_valid_on_unseen_histories"))
            else:
                self.assertTrue(result.metrics_are_lower_bounds)
        finally:
            oracle.close()

    def test_failure_uses_lower_bound_not_controller_loss(self):
        data, pool, limits = small_system(True)
        oracle = NonanticipativeCertificateOracle(data, pool, limits, SolverOptions(), self.env)
        try:
            result = oracle.evaluate({k: 0 for k in COMPONENTS})
            self.assertFalse(result.feasible)
            self.assertTrue(result.losses_are_relaxation_bounds)
            self.assertEqual(result.information["failure_proof"],
                             "perfect_foresight_LP_risk_lower_bound_exceeds_limit")
        finally:
            oracle.close()

    def test_impossible_minimum_output_is_rejected(self):
        data, _, _ = small_system(True)
        units = {k: hi for k, (_, hi) in data.unit_bounds.items()}
        data = replace(data, unit_commitment=replace(data.unit_commitment, min_output_fraction=1.))
        with self.assertRaisesRegex(ValueError, "minimum diesel output"):
            ColdReserveController(data, units)


if __name__ == "__main__":
    unittest.main()
