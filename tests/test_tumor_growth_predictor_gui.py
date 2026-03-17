import unittest

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor_gui import (
    TumorGrowthPredictorWindow,
)


class TumorGrowthPredictorGuiTests(unittest.TestCase):
    def test_build_display_days_uses_whole_days(self) -> None:
        days = TumorGrowthPredictorWindow.build_display_days([0.0, 0.041667, 0.083333, 1.25, 2.75, 30.0])

        self.assertTrue(np.array_equal(days, np.arange(0.0, 31.0, 1.0)))

    def test_build_display_times_daily_uses_whole_days(self) -> None:
        days = TumorGrowthPredictorWindow.build_display_times(
            [0.0, 0.041667, 0.083333, 1.25, 2.75, 30.0],
            "daily",
        )

        self.assertTrue(np.array_equal(days, np.arange(0.0, 31.0, 1.0)))

    def test_build_display_times_raw_preserves_dense_times(self) -> None:
        times = TumorGrowthPredictorWindow.build_display_times(
            [0.0, 0.041667, 0.083333, 1.25, 2.75],
            "raw",
        )

        self.assertTrue(np.array_equal(times, np.array([0.0, 0.041667, 0.083333, 1.25, 2.75])))

    def test_resolve_playback_step_uses_fast_auto_for_raw(self) -> None:
        self.assertEqual(TumorGrowthPredictorWindow.resolve_playback_step("raw", "auto"), 8)
        self.assertEqual(TumorGrowthPredictorWindow.resolve_playback_step("daily", "auto"), 1)

    def test_resolve_playback_step_uses_selected_multiplier(self) -> None:
        self.assertEqual(TumorGrowthPredictorWindow.resolve_playback_step("raw", 4), 4)
        self.assertEqual(TumorGrowthPredictorWindow.resolve_playback_step("daily", 16), 16)

    def test_interpolate_series_samples_dense_curve_at_day(self) -> None:
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
        values = np.array([10.0, 12.0, 14.0, 18.0, 22.0], dtype=float)

        sampled = TumorGrowthPredictorWindow.interpolate_series(times, values, 1.0)

        self.assertAlmostEqual(sampled, 14.0, places=8)


if __name__ == "__main__":
    unittest.main()
