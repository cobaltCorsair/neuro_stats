import unittest

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    AnalysisRunSummary,
    LQFitResult,
)
from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
    IntervalSensitivityReport,
    IntervalSensitivityRow,
    ParameterInfluence,
    ParameterSensitivityReport,
    ParameterSensitivityRow,
    ScenarioComparisonReport,
    ScenarioComparisonRow,
)
from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor_gui import (
    TumorGrowthPredictorWindow,
    build_comparison_table_rows,
    build_family_parameter_overrides,
    build_influence_table_rows,
    build_interval_sensitivity_table_rows,
    build_parameter_sensitivity_table_rows,
    parse_percentage_list,
    parse_positive_float_list,
)
from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
    GrowthModelParameters,
    TreatmentFraction,
)


class TumorGrowthPredictorGuiTests(unittest.TestCase):
    def test_parse_positive_float_list_uses_defaults(self) -> None:
        self.assertEqual(parse_positive_float_list("", (0.5, 1.0, 2.5)), [0.5, 1.0, 2.5])

    def test_parse_percentage_list_converts_to_fractions(self) -> None:
        self.assertEqual(parse_percentage_list("10, 20, 30", (10.0, 20.0)), [0.1, 0.2, 0.3])

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

    def test_format_family_sequence_joins_entries(self) -> None:
        self.assertEqual(
            TumorGrowthPredictorWindow.format_family_sequence(("y", "p_peak", "n")),
            "y -> p_peak -> n",
        )

    def test_interpolate_series_samples_dense_curve_at_day(self) -> None:
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
        values = np.array([10.0, 12.0, 14.0, 18.0, 22.0], dtype=float)

        sampled = TumorGrowthPredictorWindow.interpolate_series(times, values, 1.0)

        self.assertAlmostEqual(sampled, 14.0, places=8)

    def test_build_family_parameter_overrides_uses_matching_family_runs(self) -> None:
        base_parameters = GrowthModelParameters(
            alpha=0.10,
            beta=0.02,
            growth_rate=0.15,
            carrying_capacity=80.0,
            clearance_rate=0.10,
            repair_half_time_hours=1.0,
        )
        proton_fit = LQFitResult(
            alpha=0.25,
            beta=0.03,
            train_count=3,
            train_kind="all",
            family="p_peak",
            sf_mode="absolute",
            repair_half_time_hours=2.5,
        )
        proton_run = AnalysisRunResult(
            summary=AnalysisRunSummary(
                sf_mode="absolute",
                family="p_peak",
                total_count=3,
                single_count=2,
                fractionated_count=1,
                train_count=3,
                validation_count=0,
                status="ok",
            ),
            train=(),
            validation=(),
            train_kind="all",
            validation_kind="none",
            fit_result=proton_fit,
        )

        overrides, missing = build_family_parameter_overrides(
            [proton_run],
            base_parameters,
            [
                TreatmentFraction(day=0.0, dose=2.0, family="y"),
                TreatmentFraction(day=0.5, dose=2.0, family="p_peak"),
                TreatmentFraction(day=1.0, dose=2.0, family="c"),
            ],
            default_family="y",
        )

        self.assertAlmostEqual(overrides["y"].alpha, base_parameters.alpha, places=8)
        self.assertAlmostEqual(overrides["p_peak"].alpha, proton_fit.alpha, places=8)
        self.assertAlmostEqual(overrides["p_peak"].beta, proton_fit.beta, places=8)
        self.assertAlmostEqual(overrides["p_peak"].repair_half_time_hours, 2.5, places=8)
        self.assertEqual(missing, ("c",))

    def test_build_parameter_sensitivity_table_rows_formats_report(self) -> None:
        report = ParameterSensitivityReport(
            baseline_rmse=0.12,
            rows=(
                ParameterSensitivityRow(
                    parameter="alpha",
                    perturbation_fraction=0.1,
                    baseline_value=0.013,
                    varied_value=0.0143,
                    rmse=0.11,
                    delta_rmse=-0.01,
                    rmse_ratio=0.916,
                    status="ok",
                    reason="",
                ),
            ),
            influence=(
                ParameterInfluence(
                    parameter="alpha",
                    max_abs_delta_rmse=0.01,
                    mean_abs_delta_rmse=0.01,
                    tested_cases=1,
                ),
            ),
        )

        parameter_rows = build_parameter_sensitivity_table_rows(report)
        influence_rows = build_influence_table_rows(report)

        self.assertEqual(parameter_rows[0][0], "alpha")
        self.assertEqual(parameter_rows[0][1], "+10.0")
        self.assertEqual(parameter_rows[0][4], "0.110000")
        self.assertEqual(influence_rows[0][1], "0.010000")

    def test_build_interval_and_comparison_rows_format_outputs(self) -> None:
        interval_report = IntervalSensitivityReport(
            baseline_rmse=0.1,
            rows=(
                IntervalSensitivityRow(
                    interval_hours=2.5,
                    rmse=0.08,
                    delta_rmse=-0.02,
                    rmse_ratio=0.8,
                    schedule_days=(0.0, 0.1041667, 0.2083333),
                ),
            ),
        )
        comparison_report = ScenarioComparisonReport(
            rows=(
                ScenarioComparisonRow(
                    scenario="Current",
                    total_physical_dose=40.0,
                    family_sequence=("p_peak", "n"),
                    min_total_volume=1.2,
                    min_total_volume_day=7.0,
                    final_total_volume=3.4,
                    auc_total_volume=55.0,
                ),
            )
        )

        interval_rows = build_interval_sensitivity_table_rows(interval_report)
        comparison_rows = build_comparison_table_rows(comparison_report)

        self.assertEqual(interval_rows[0][0], "2.500")
        self.assertEqual(interval_rows[0][4], "t=2.5 ч./2.5 ч.")
        self.assertEqual(comparison_rows[0][2], "p_peak -> n")
        self.assertEqual(comparison_rows[0][6], "55.000000")

    def test_build_interval_preview_from_days_uses_t_expression(self) -> None:
        preview = TumorGrowthPredictorWindow._build_interval_preview_from_days(
            [0.0, 1.0 / 24.0, 25.0 / 24.0, 26.0 / 24.0],
            label="Intervals",
        )

        self.assertEqual(preview, "Intervals: t=1 ч./1 сут./1 ч.")


if __name__ == "__main__":
    unittest.main()
