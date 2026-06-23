import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from stats_methods.support_stats_methods import SupportingFunctions as SF


class TestCalculateStdDev(unittest.TestCase):
    def test_uses_valid_count_not_total_length(self):
        # 3 валидных значения (10,12,14) + 2 NaN (выбывшие животные)
        values = [10, 12, 14, float("nan"), float("nan")]
        std = SF.calculate_std_dev(values, mean_value=12.0)
        self.assertAlmostEqual(std, math.sqrt(8 / 2), places=9)  # n=3 -> /(3-1)

    def test_no_nan_matches_naive_formula(self):
        values = [10, 12, 14]
        std = SF.calculate_std_dev(values, mean_value=12.0)
        self.assertAlmostEqual(std, math.sqrt(8 / 2), places=9)

    def test_fewer_than_two_valid_returns_nan(self):
        self.assertTrue(math.isnan(SF.calculate_std_dev([5.0, float("nan")], 5.0)))
        self.assertTrue(math.isnan(SF.calculate_std_dev([], 0.0)))


class TestCalculateErrorMargin(unittest.TestCase):
    def test_correct_n_gives_wider_interval_than_buggy_total_n(self):
        std = math.sqrt(8 / 2)  # = 2.0, n_valid=3
        sem_correct = SF.calculate_error_margin(std, 3)
        sem_with_old_buggy_total_n = SF.calculate_error_margin(std, 5)
        self.assertAlmostEqual(sem_correct, 2.0 / math.sqrt(3), places=9)
        self.assertGreater(sem_correct, sem_with_old_buggy_total_n)

    def test_zero_n_returns_nan_instead_of_crashing(self):
        self.assertTrue(math.isnan(SF.calculate_error_margin(1.0, 0)))


class TestCountAtRisk(unittest.TestCase):
    def test_counts_non_nan_only(self):
        self.assertEqual(SF.count_at_risk([1.0, 2.0, float("nan"), 3.0, float("nan")]), 3)

    def test_all_valid(self):
        self.assertEqual(SF.count_at_risk([1.0, 2.0, 3.0]), 3)

    def test_all_nan(self):
        self.assertEqual(SF.count_at_risk([float("nan"), float("nan")]), 0)


if __name__ == "__main__":
    unittest.main()
