import unittest

from experiments.brain_runtime.stats import FRESH_HIDDEN_SEEDS, gap_significant, wilson_interval


class WilsonIntervalTests(unittest.TestCase):
    def test_known_value(self):
        low, high = wilson_interval(8, 10)
        self.assertAlmostEqual(low, 0.49, delta=0.02)
        self.assertAlmostEqual(high, 0.94, delta=0.02)

    def test_edges_stay_in_unit_interval(self):
        low, high = wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = wilson_interval(20, 20)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)
        with self.assertRaises(ValueError):
            wilson_interval(-1, 10)
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)


class GapSignificantTests(unittest.TestCase):
    def test_large_gap_significant(self):
        self.assertTrue(gap_significant(120, 120, 60, 120))

    def test_small_gap_not_significant(self):
        self.assertFalse(gap_significant(100, 120, 98, 120))


class FreshSeedsTests(unittest.TestCase):
    def test_registry_avoids_burned_seeds(self):
        burned = {101, 103, 107}
        for version, seeds in FRESH_HIDDEN_SEEDS.items():
            self.assertEqual(len(seeds), 3, version)
            self.assertFalse(burned & set(seeds), version)


if __name__ == "__main__":
    unittest.main()
