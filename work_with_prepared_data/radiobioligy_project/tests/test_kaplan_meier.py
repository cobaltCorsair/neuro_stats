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
from stats_methods.kaplan_meier import (
    format_calculation_steps,
    hazard_ratio_log_rank,
    kaplan_meier_estimate,
    log_rank_test,
    max_observed_day,
    median_survival_ci,
    n_at_risk_table,
    plot_kaplan_meier,
    restricted_mean_survival_time,
    risk_table_time_points,
)


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


class TestMedianSurvivalCi(unittest.TestCase):
    def test_lower_bound_uses_lower_band_upper_bound_uses_upper_band(self):
        # Klein & Moeschberger 4.4: L = inf{t: нижняя граница ДИ S(t) <= 0.5},
        # U = inf{t: верхняя граница ДИ S(t) <= 0.5}. С точечной медианой 7.0 нижняя
        # граница ДИ должна быть <= 7.0 (находится раньше или совпадает с точкой).
        events = _events([("A", 2.0, True), ("B", 3.0, False), ("C", 5.0, True),
                           ("D", 7.0, True), ("E", 9.0, False)])
        km = kaplan_meier_estimate(events)
        lower, upper = median_survival_ci(km)
        self.assertIsNotNone(lower)
        self.assertLessEqual(lower, km.median_survival)
        if upper is not None:
            self.assertGreaterEqual(upper, km.median_survival)

    def test_no_events_gives_both_bounds_none(self):
        km = kaplan_meier_estimate([])
        self.assertEqual(median_survival_ci(km), (None, None))


class TestRestrictedMeanSurvivalTime(unittest.TestCase):
    def test_matches_manual_step_integral(self):
        events = _events([("A", 2.0, True), ("B", 3.0, False), ("C", 5.0, True),
                           ("D", 7.0, True), ("E", 9.0, False)])
        km = kaplan_meier_estimate(events)
        rmst = restricted_mean_survival_time(km)
        # площадь по ступеням: 1.0*(2-0) + 0.8*(3-2) + 0.8*(5-3) + 0.5333*(7-5) + 0.2667*(9-7)
        manual = 1.0 * 2 + 0.8 * 1 + 0.8 * 2 + (0.8 * 2 / 3) * 2 + (0.8 * 2 / 3 * 0.5) * 2
        self.assertAlmostEqual(rmst, manual, places=6)

    def test_all_survive_rmst_equals_full_range(self):
        events = _events([("A", 10.0, False), ("B", 12.0, False)])
        km = kaplan_meier_estimate(events)
        rmst = restricted_mean_survival_time(km)
        self.assertAlmostEqual(rmst, 12.0, places=6)

    def test_degenerate_single_point_gives_zero(self):
        km = kaplan_meier_estimate([])
        self.assertEqual(restricted_mean_survival_time(km), 0.0)


class TestHazardRatioLogRank(unittest.TestCase):
    def test_identical_groups_give_hazard_ratio_one(self):
        events = _events([("A", 2.0, True), ("B", 5.0, True), ("C", 9.0, False)])
        result = hazard_ratio_log_rank(events, events)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.hazard_ratio, 1.0, places=6)
        self.assertLess(result.ci_lower, 1.0)
        self.assertGreater(result.ci_upper, 1.0)

    def test_worse_group_gives_hazard_ratio_above_one(self):
        worse = _events([(f"W{i}", 2.0, True) for i in range(10)])
        better = _events([(f"B{i}", 20.0, False) for i in range(10)])
        result = hazard_ratio_log_rank(worse, better)
        self.assertIsNotNone(result)
        self.assertGreater(result.hazard_ratio, 1.0)
        self.assertGreater(result.ci_lower, 1.0)

    def test_no_shared_risk_set_returns_none(self):
        self.assertIsNone(hazard_ratio_log_rank([], []))


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

    def test_plot_runs_without_warnings(self):
        import warnings
        group_a = _events([("A1", 2.0, True), ("A2", 9.0, False)])
        group_b = _events([("B1", 4.0, True), ("B2", 6.0, True)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, ax = plot_kaplan_meier({"Group A": group_a, "Group B": group_b}, show_ci=True)
        self.assertEqual(len(fig.axes), 1)


class TestRiskTableHelpers(unittest.TestCase):
    def test_time_points_are_evenly_spaced_unique_integers_within_range(self):
        group_a = _events([("A1", 2.0, True), ("A2", 20.0, False)])
        points = risk_table_time_points({"Group A": group_a}, n_points=5)
        self.assertEqual(points, sorted(set(points)))
        self.assertTrue(all(p == round(p) for p in points))
        self.assertGreaterEqual(points[0], 0.0)
        self.assertLessEqual(points[-1], 20.0)

    def test_no_events_gives_single_zero_point(self):
        self.assertEqual(risk_table_time_points({"Group A": []}), [0.0])

    def test_counts_decrease_as_animals_leave_risk_set(self):
        group_a = _events([("A1", 2.0, True), ("A2", 4.0, True), ("A3", 9.0, False)])
        counts = n_at_risk_table({"Group A": group_a}, [0.0, 3.0, 5.0, 10.0])
        self.assertEqual(counts["Group A"], [3, 2, 1, 0])

    def test_max_observed_day_is_the_latest_event_or_censor_day(self):
        group_a = _events([("A1", 2.0, True), ("A2", 26.0, False)])
        self.assertEqual(max_observed_day(group_a), 26.0)

    def test_max_observed_day_is_none_when_no_events(self):
        self.assertIsNone(max_observed_day([]))
        self.assertIsNone(max_observed_day([RatSurvivalEvent("X", None, False, "")]))


class TestFormatCalculationSteps(unittest.TestCase):
    def test_matches_ten_patient_textbook_example(self):
        # Стандартный учебный пример: 10 пациентов, 5 лет наблюдения.
        events = _events([
            ("1", 0.5, True), ("2", 1.2, True), ("3", 1.5, False), ("4", 2.0, True),
            ("5", 2.3, False), ("6", 3.0, True), ("7", 3.5, False), ("8", 4.0, True),
            ("9", 4.5, False), ("10", 5.0, False),
        ])
        km = kaplan_meier_estimate(events)
        steps = format_calculation_steps(km)

        self.assertEqual(steps[0], "t=0.5: n=10, d=1 → S(0.5) = 1 × (1 − 1/10) = 0.9000")
        self.assertEqual(steps[1], "t=1.2: n=9, d=1 → S(1.2) = 0.9000 × (1 − 1/9) = 0.8000")
        self.assertEqual(steps[2], "t=1.5: цензурировано 1 → S(1.5) = 0.8000 (без изменений)")
        self.assertAlmostEqual(km.survival[-1], 0.366, places=3)

    def test_empty_result_gives_no_steps(self):
        km = kaplan_meier_estimate([])
        self.assertEqual(format_calculation_steps(km), [])


if __name__ == "__main__":
    unittest.main()
