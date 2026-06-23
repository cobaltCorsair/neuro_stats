import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")

from data_processing.excel_data_processor import RatSurvivalEvent
from stats_methods.kaplan_meier import kaplan_meier_estimate, log_rank_test, plot_kaplan_meier


def _events(spec):
    """spec: список (label, day, event_observed)."""
    return [RatSurvivalEvent(label, day, observed, "death" if observed else "censored") for label, day, observed in spec]


class TestKaplanMeierEstimate(unittest.TestCase):
    def test_textbook_example_deaths_and_censoring(self):
        # A:death@2, B:censored@3, C:death@5, D:death@7, E:censored@9
        events = _events([("A", 2.0, True), ("B", 3.0, False), ("C", 5.0, True),
                           ("D", 7.0, True), ("E", 9.0, False)])
        km = kaplan_meier_estimate(events)
        self.assertEqual(km.times, (0.0, 2.0, 3.0, 5.0, 7.0, 9.0))
        expected_survival = [1.0, 0.8, 0.8, 0.8 * (2 / 3), 0.8 * (2 / 3) * 0.5, 0.8 * (2 / 3) * 0.5]
        for actual, expected in zip(km.survival, expected_survival):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(km.n_at_risk, (5, 5, 4, 3, 2, 1))
        self.assertEqual(km.n_events, (0, 1, 0, 1, 1, 0))
        self.assertEqual(km.n_censored, (0, 0, 1, 0, 0, 1))
        self.assertEqual(km.median_survival, 7.0)

    def test_no_events_returns_degenerate_result(self):
        km = kaplan_meier_estimate([])
        self.assertEqual(km.times, (0.0,))
        self.assertEqual(km.survival, (1.0,))
        self.assertIsNone(km.median_survival)

    def test_events_with_missing_day_are_excluded(self):
        events = _events([("A", 2.0, True)]) + [RatSurvivalEvent("B", None, False, "")]
        km = kaplan_meier_estimate(events)
        self.assertEqual(km.n_at_risk[0], 1)

    def test_all_survive_median_is_none(self):
        events = _events([("A", 10.0, False), ("B", 12.0, False)])
        km = kaplan_meier_estimate(events)
        self.assertEqual(km.survival[-1], 1.0)
        self.assertIsNone(km.median_survival)


class TestLogRankTest(unittest.TestCase):
    def test_identical_groups_give_p_near_one(self):
        events = _events([("A", 2.0, True), ("B", 5.0, True), ("C", 9.0, False)])
        chi2_stat, p_value = log_rank_test(events, events)
        self.assertAlmostEqual(chi2_stat, 0.0, places=6)
        self.assertAlmostEqual(p_value, 1.0, places=6)

    def test_clearly_different_groups_give_small_p(self):
        fast_death = _events([(f"F{i}", 1.0, True) for i in range(10)])
        long_alive = _events([(f"L{i}", 100.0, False) for i in range(10)])
        chi2_stat, p_value = log_rank_test(fast_death, long_alive)
        self.assertGreater(chi2_stat, 10.0)
        self.assertLess(p_value, 0.01)

    def test_no_overlap_in_time_does_not_crash(self):
        chi2_stat, p_value = log_rank_test([], [])
        self.assertEqual(chi2_stat, 0.0)
        self.assertEqual(p_value, 1.0)


class TestPlotKaplanMeier(unittest.TestCase):
    def test_plot_runs_without_error_for_single_and_multiple_groups(self):
        group_a = _events([("A1", 2.0, True), ("A2", 9.0, False)])
        group_b = _events([("B1", 4.0, True), ("B2", 6.0, True)])
        fig, ax = plot_kaplan_meier({"Group A": group_a, "Group B": group_b}, show_ci=True)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
