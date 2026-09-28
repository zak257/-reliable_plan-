"""Deterministic checks of distribution bands, oracle gaps, and exact CVaR."""

import math
import unittest

import numpy as np

from certified_reliability_planning.risk import empirical_cvar, risk_bounds


class EmpiricalCvarTests(unittest.TestCase):
    def test_fractional_boundary_atom(self):
        # The worst 40% contains all of the 10 atom and 15% of the 4 atom.
        self.assertAlmostEqual(empirical_cvar([0, 0, 4, 10], 0.6), 7.75)
        self.assertAlmostEqual(empirical_cvar([0, 4, 10], 0.6, [2, 1, 1]), 7.75)

    def test_mean_and_tail_endpoints(self):
        self.assertAlmostEqual(empirical_cvar([0, 4, 10], 0.0, [2, 1, 1]), 3.5)
        self.assertEqual(empirical_cvar([0, 4, 10], 0.99999), 10.0)
        self.assertEqual(empirical_cvar([17] * 19, 0.931), 17.0)
        self.assertEqual(empirical_cvar([-10, -3, 2], 0.5, [0, 1, 0]), -3.0)

    def test_matches_continuous_eta_piecewise_linear_minimum(self):
        values = np.array([0.0, 1.0, 1.0, 3.0, 11.0])
        weights = np.array([1.0, 2.0, 7.0, 9.0, 3.0]) / 22.0
        for alpha in [0.0, 0.1, 0.47, 0.8, 0.99]:
            expected = min(
                eta + np.dot(weights, np.maximum(values - eta, 0)) / (1 - alpha)
                for eta in values
            )
            self.assertAlmostEqual(empirical_cvar(values, alpha, weights), expected)

    def test_invalid_distributions(self):
        cases = [([], 0.5, None), ([[1]], 0.5, None), ([math.inf], 0.5, None),
                 ([1], 1.0, None), ([1], -0.1, None), ([1], 0.5, [0]),
                 ([1, 2], 0.5, [1]), ([1, 2], 0.5, [1, -1])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                empirical_cvar(*args)


class RiskBoundsTests(unittest.TestCase):
    def test_allocation_is_over_capacity_and_every_prefix(self):
        m, k, delta = 100, 37, 0.03
        result = risk_bounds(np.zeros(m), np.ones(m), 2, 0.8, delta, k)
        allocation = 6 * delta / (math.pi ** 2 * k * m ** 2)
        expected = math.sqrt(math.log(2 / allocation) / (2 * m))
        self.assertAlmostEqual(result.radius, expected)
        self.assertEqual(result.sample_count, m)
        # The analytical infinite union is K * sum_m delta_nm = delta.
        self.assertAlmostEqual(k * 6 * delta / (math.pi ** 2 * k) * math.pi ** 2 / 6, delta)
        self.assertGreater(risk_bounds(np.zeros(m), np.ones(m), 2, .8, delta, 2*k).radius, result.radius)

    def test_unresolved_scenarios_keep_full_denominator(self):
        low = np.r_[np.full(10, 2.0), np.zeros(90)]
        high = np.r_[np.full(10, 2.0), np.full(90, 10.0)]
        result = risk_bounds(low, high, 10, 0.8, 0.05, 1)
        self.assertEqual(result.sample_count, 100)
        self.assertAlmostEqual(result.empirical_eens_lower, 0.2)
        self.assertAlmostEqual(result.empirical_eens_upper, 9.2)
        self.assertAlmostEqual(result.mean_oracle_width, 9.0)
        self.assertEqual(result.eens_lower, 0.0)
        self.assertEqual(result.cvar_upper, 10.0)

    def test_endpoint_atoms_and_fractional_tail(self):
        zeros, ones = np.zeros(1000), np.ones(1000)
        zero_result = risk_bounds(zeros, zeros, 10, 0.8, 0.05, 1)
        r = zero_result.radius
        self.assertLess(r, 0.2)
        self.assertEqual(zero_result.eens_lower, 0.0)
        self.assertEqual(zero_result.cvar_lower, 0.0)
        self.assertAlmostEqual(zero_result.eens_upper, 10 * r)
        self.assertAlmostEqual(zero_result.cvar_upper, 10 * r / 0.2)
        full_result = risk_bounds(10*ones, 10*ones, 10, 0.0, 0.05, 1)
        self.assertAlmostEqual(full_result.eens_lower, 10 * (1-r))
        self.assertAlmostEqual(full_result.cvar_lower, 10 * (1-r))
        self.assertAlmostEqual(full_result.eens_upper, 10)
        self.assertAlmostEqual(full_result.cvar_upper, 10)

    def test_small_sample_vacuous_band_and_zero_support(self):
        result = risk_bounds([2], [2], 10, 0.95, 0.05, 100)
        self.assertGreater(result.radius, 1)
        self.assertEqual((result.eens_lower, result.cvar_lower), (0, 0))
        self.assertEqual((result.eens_upper, result.cvar_upper), (10, 10))
        result = risk_bounds([0], [0], 0, .95, .05, 1)
        self.assertEqual((result.eens_lower, result.eens_upper, result.cvar_lower, result.cvar_upper), (0, 0, 0, 0))

    def test_exact_cdf_integration_against_independent_formula(self):
        low = np.tile([0., 1., 2., 5., 7.], 40)
        high = np.minimum(low + np.tile([0., .5, 2., 2., 3.], 40), 10)
        alpha = 0.61
        result = risk_bounds(low, high, 10, alpha, 0.05, 3)
        support = np.unique(np.r_[0, low, high, 10])
        dx = np.diff(support)
        f_upper = np.minimum(np.mean(low[:, None] <= support[:-1], axis=0) + result.radius, 1)
        f_lower = np.maximum(np.mean(high[:, None] <= support[:-1], axis=0) - result.radius, 0)
        self.assertAlmostEqual(result.eens_lower, np.dot(1 - f_upper, dx))
        self.assertAlmostEqual(result.eens_upper, np.dot(1 - f_lower, dx))
        # CVaR = min_eta eta + integral_eta^B (1-F(x)) dx / (1-alpha).
        for cdf, obtained in [(f_upper, result.cvar_lower), (f_lower, result.cvar_upper)]:
            candidates = [eta + np.dot(1-cdf[i:], dx[i:]) / (1-alpha)
                          for i, eta in enumerate(support)]
            self.assertAlmostEqual(obtained, min(candidates))

    def test_refinement_monotonicity_and_width_decomposition(self):
        truth = np.linspace(0, 10, 500)
        coarse = risk_bounds(np.maximum(0, truth-2), np.minimum(10, truth+2), 10, .73, .05, 6)
        tight = risk_bounds(np.maximum(0, truth-.5), np.minimum(10, truth+.5), 10, .73, .05, 6)
        self.assertGreaterEqual(tight.eens_lower, coarse.eens_lower)
        self.assertGreaterEqual(tight.cvar_lower, coarse.cvar_lower)
        self.assertLessEqual(tight.eens_upper, coarse.eens_upper)
        self.assertLessEqual(tight.cvar_upper, coarse.cvar_upper)
        for result in [coarse, tight]:
            self.assertAlmostEqual(result.eens_width, result.mean_oracle_width + result.eens_sampling_width)
            self.assertAlmostEqual(result.cvar_width, result.cvar_oracle_width + result.cvar_sampling_width)
            self.assertLessEqual(result.eens_sampling_width, 2 * 10 * result.radius + 1e-12)
            self.assertLessEqual(result.cvar_sampling_width, 2 * 10 * result.radius / .27 + 1e-12)
            self.assertLessEqual(result.cvar_oracle_width, result.mean_oracle_width / .27 + 1e-12)
            self.assertLessEqual(result.eens_lower, result.empirical_eens_lower)
            self.assertGreaterEqual(result.eens_upper, result.empirical_eens_upper)

    def test_large_search_space_log_allocation(self):
        result = risk_bounds([0, 1], [0, 1], 1, .5, 1e-300, 10**1000)
        self.assertTrue(math.isfinite(result.radius))
        self.assertEqual(result.eens_lower, 0)
        self.assertEqual(result.eens_upper, 1)

    def test_invalid_intervals_and_parameters(self):
        valid = dict(lower=[0, 1], upper=[1, 2], bound=2, alpha=.9, delta=.05, cardinality=1)
        invalid = [dict(lower=[]), dict(lower=[0]), dict(lower=[0, 3]),
                   dict(lower=[-1, 0]), dict(upper=[1, 3]), dict(upper=[1, math.nan]),
                   dict(bound=-1), dict(bound=math.inf), dict(alpha=1), dict(alpha=-1),
                   dict(delta=0), dict(delta=1), dict(cardinality=0),
                   dict(cardinality=1.2), dict(cardinality=True)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                risk_bounds(**(valid | change))


if __name__ == "__main__":
    unittest.main()
