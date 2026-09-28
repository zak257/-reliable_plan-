"""Population references and genuine common sampling for the analytic example."""

import math
import unittest

from certified_reliability_planning.benchmarks import TinyCapacityOracle


class TinyBenchmarkTests(unittest.TestCase):
    def test_known_population_optimum_with_nonmonotone_cost(self):
        oracle = TinyCapacityOracle()
        rows = oracle.truth_table()
        best = min((row for row in rows if row["feasible"]), key=lambda row: row["cost"])
        self.assertEqual(best["point"], [2, 1])
        self.assertAlmostEqual(best["cost"], 7.0)
        self.assertAlmostEqual(best["eens"], 0.025)
        self.assertAlmostEqual(best["cvar"], 0.05)
        # A feasible point does not justify discarding its upper orthant.
        self.assertTrue(next(row for row in rows if row["point"] == [1, 1])["feasible"])
        self.assertLess(oracle.costs[(2, 1)], oracle.costs[(1, 1)])
        self.assertLess(oracle.costs[(2, 1)], oracle.costs[(2, 2)])
        for row in rows:
            self.assertGreaterEqual(row["cost"], oracle.capital_lower_bounds[tuple(row["point"])])

    def test_sample_prefix_is_stable_and_shared_across_capacities(self):
        growing, once = TinyCapacityOracle(19), TinyCapacityOracle(19)
        growing.extend(23)
        prefix = growing.draws
        growing.extend(117)
        once.extend(117)
        self.assertEqual(growing.draws[:23], prefix)
        self.assertEqual(growing.draws, once.draws)
        self.assertEqual(growing.samples, 117)
        self.assertNotEqual(growing.draws[:23], growing.draws[23:46])
        for index in range(growing.samples):
            for x in range(2):
                small = growing.operation((x, 1), index, 1, 0)
                large = growing.operation((x + 1, 1), index, 1, 0)
                self.assertEqual(small.lower, small.upper)
                self.assertLessEqual(large.upper, small.lower)

    def test_refined_and_limited_brackets_contain_true_optimum(self):
        oracle = TinyCapacityOracle(refinable=True, interval_floor=0.02, economic_floor=0.1)
        oracle.extend(1)
        values, _ = oracle.true_loss((1, 1))
        truth = values[int(oracle.draws[0] >= .5)]
        previous = math.inf
        for _ in range(8):
            interval = oracle.operation((1, 1), 0, 1, 0)
            self.assertLessEqual(interval.lower, truth)
            self.assertGreaterEqual(interval.upper, truth)
            self.assertLessEqual(interval.upper - interval.lower, previous + 1e-15)
            previous = interval.upper - interval.lower
        self.assertAlmostEqual(previous, .02)
        economic = oracle.economic((1, 1), 1, 0)
        self.assertLessEqual(economic.lower, oracle.costs[(1, 1)])
        self.assertGreaterEqual(economic.upper, oracle.costs[(1, 1)])
        self.assertGreater(economic.upper - economic.lower, 0)

    def test_no_budget_keeps_trivial_bounds(self):
        oracle = TinyCapacityOracle()
        oracle.extend(1)
        interval = oracle.operation((2, 2), 0, 0, 0)
        self.assertEqual((interval.lower, interval.upper), (0, 1))
        economic = oracle.economic((2, 2), 0, 0)
        self.assertEqual(economic.lower, oracle.capital_lower_bounds[(2, 2)])
        self.assertEqual(economic.upper, math.inf)
        self.assertEqual(oracle.operation_calls, 0)
        self.assertEqual(oracle.economic_calls, 0)


if __name__ == "__main__":
    unittest.main()
