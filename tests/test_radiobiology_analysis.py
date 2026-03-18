import unittest

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    AnalysisRunSummary,
    LQFitResult,
)
from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
    analyze_interval_sensitivity,
    analyze_parameter_sensitivity,
    build_rbe_series,
    compare_treatment_scenarios,
    compare_sf_metric_sensitivity,
    compute_rbe,
)
from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
    GeometryReference,
    GrowthModelParameters,
    TreatmentFraction,
    build_schedule_from_intervals,
    simulate_growth,
)


class RadiobiologyAnalysisTests(unittest.TestCase):
    def test_compute_rbe_is_one_for_identical_reference_and_test(self) -> None:
        fit = LQFitResult(
            alpha=0.10,
            beta=0.02,
            train_count=3,
            train_kind="all",
            family="y",
            sf_mode="absolute",
        )

        point = compute_rbe(fit, fit, 2.0)

        self.assertAlmostEqual(point.reference_dose, 2.0, places=7)
        self.assertAlmostEqual(point.rbe, 1.0, places=7)

    def test_build_rbe_series_reports_higher_rbe_for_more_effective_family(self) -> None:
        reference = LQFitResult(
            alpha=0.10,
            beta=0.02,
            train_count=3,
            train_kind="all",
            family="y",
            sf_mode="absolute",
        )
        test_fit = LQFitResult(
            alpha=0.20,
            beta=0.03,
            train_count=3,
            train_kind="all",
            family="p_peak",
            sf_mode="absolute",
        )

        rows = build_rbe_series(reference, {"p_peak": test_fit}, doses=(2.0, 10.0))

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.test_family == "p_peak" for row in rows))
        self.assertTrue(all(row.reference_family == "y" for row in rows))
        self.assertTrue(all(row.rbe > 1.0 for row in rows))

    def test_parameter_sensitivity_reports_nonzero_rmse_for_perturbations(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=3.0, axis_c=4.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.08,
            beta=0.02,
            growth_rate=0.15,
            carrying_capacity=80.0,
            clearance_rate=0.10,
        )
        schedule = [TreatmentFraction(day=0.0, dose=2.0)]
        observed_days = np.array([0.0, 2.0, 4.0, 7.0, 10.0], dtype=float)
        observed = simulate_growth(
            sample_times=observed_days,
            parameters=parameters,
            reference=reference,
            schedule=schedule,
        )

        report = analyze_parameter_sensitivity(
            observed_days=observed_days,
            observed_volume=observed.total_volume,
            reference=reference,
            parameters=parameters,
            schedule=schedule,
            perturbation_fractions=(0.1,),
            vary=("alpha", "dose"),
        )

        self.assertAlmostEqual(report.baseline_rmse, 0.0, places=8)
        alpha_rows = [row for row in report.rows if row.parameter == "alpha" and row.status == "ok"]
        dose_rows = [row for row in report.rows if row.parameter == "dose" and row.status == "ok"]
        self.assertTrue(any((row.rmse or 0.0) > 0.0 for row in alpha_rows))
        self.assertTrue(any((row.rmse or 0.0) > 0.0 for row in dose_rows))
        self.assertTrue(any(item.parameter == "alpha" for item in report.influence))

    def test_interval_sensitivity_prefers_true_interval(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=3.0, axis_c=4.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.0,
            beta=0.04,
            growth_rate=0.0,
            carrying_capacity=80.0,
            clearance_rate=2.0,
            repair_half_time_hours=1.0,
        )
        fractions = [4.0, 4.0, 32.0]
        true_schedule = build_schedule_from_intervals(fractions, [1.0 / 24.0])
        observed_days = np.array(
            [0.0, 0.5 / 24.0, 1.0 / 24.0, 2.0 / 24.0, 0.25, 1.0, 3.0],
            dtype=float,
        )
        observed = simulate_growth(
            sample_times=observed_days,
            parameters=parameters,
            reference=reference,
            schedule=true_schedule,
        )

        report = analyze_interval_sensitivity(
            observed_days=observed_days,
            observed_volume=observed.total_volume,
            reference=reference,
            parameters=parameters,
            fractions=fractions,
            interval_hours=(0.5, 1.0, 2.5, 24.0),
            baseline_schedule=true_schedule,
        )

        self.assertGreaterEqual(len(report.rows), 4)
        self.assertAlmostEqual(report.rows[0].interval_hours, 1.0, places=8)

    def test_interval_sensitivity_accepts_mixed_family_schedule_template(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=3.0, axis_c=4.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.05,
            beta=0.01,
            growth_rate=0.0,
            carrying_capacity=80.0,
            clearance_rate=1.0,
            repair_half_time_hours=1.0,
        )
        family_parameters = {
            "p_peak": GrowthModelParameters(
                alpha=0.20,
                beta=0.02,
                growth_rate=0.0,
                carrying_capacity=80.0,
                clearance_rate=1.0,
                repair_half_time_hours=1.0,
            ),
        }
        schedule = [
            TreatmentFraction(day=0.0, dose=4.0, family="y"),
            TreatmentFraction(day=1.0 / 24.0, dose=4.0, family="p_peak"),
            TreatmentFraction(day=2.0 / 24.0, dose=32.0, family="y"),
        ]
        observed_days = np.array([0.0, 1.0 / 24.0, 2.0 / 24.0, 0.5, 1.0], dtype=float)
        observed = simulate_growth(
            sample_times=observed_days,
            parameters=parameters,
            reference=reference,
            schedule=schedule,
            family_parameters=family_parameters,
        )

        report = analyze_interval_sensitivity(
            observed_days=observed_days,
            observed_volume=observed.total_volume,
            reference=reference,
            parameters=parameters,
            fractions=[4.0, 4.0, 32.0],
            interval_hours=(0.5, 1.0, 2.5),
            baseline_schedule=schedule,
            schedule_template=schedule,
            family_parameters=family_parameters,
        )

        self.assertEqual(len(report.rows), 3)
        self.assertTrue(all(len(row.schedule_days) == 3 for row in report.rows))

    def test_compare_sf_metric_sensitivity_reports_parameter_drift(self) -> None:
        baseline_fit = LQFitResult(
            alpha=0.10,
            beta=0.02,
            train_count=3,
            train_kind="all",
            family="y",
            sf_mode="absolute",
        )
        shifted_fit = LQFitResult(
            alpha=0.12,
            beta=0.025,
            train_count=3,
            train_kind="all",
            family="y",
            sf_mode="absindex:1",
        )

        runs = [
            AnalysisRunResult(
                summary=AnalysisRunSummary(
                    sf_mode="absolute",
                    family="y",
                    total_count=3,
                    single_count=2,
                    fractionated_count=1,
                    train_count=3,
                    validation_count=0,
                    status="ok",
                    response_mode="scalar",
                    model_kind="classic_lq",
                    alpha=baseline_fit.alpha,
                    beta=baseline_fit.beta,
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=baseline_fit,
            ),
            AnalysisRunResult(
                summary=AnalysisRunSummary(
                    sf_mode="absindex:1",
                    family="y",
                    total_count=3,
                    single_count=2,
                    fractionated_count=1,
                    train_count=3,
                    validation_count=0,
                    status="ok",
                    response_mode="scalar",
                    model_kind="classic_lq",
                    alpha=shifted_fit.alpha,
                    beta=shifted_fit.beta,
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=shifted_fit,
            ),
        ]

        rows = compare_sf_metric_sensitivity(runs)

        self.assertEqual(len(rows), 2)
        shifted = next(row for row in rows if row.sf_mode == "absindex:1")
        self.assertGreater(shifted.delta_alpha_pct or 0.0, 0.0)
        self.assertGreater(shifted.delta_beta_pct or 0.0, 0.0)

    def test_compare_treatment_scenarios_ranks_more_effective_schedule_first(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=3.0, axis_c=4.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.05,
            beta=0.01,
            growth_rate=0.10,
            carrying_capacity=80.0,
            clearance_rate=0.40,
        )
        family_parameters = {
            "p_peak": GrowthModelParameters(
                alpha=0.20,
                beta=0.03,
                growth_rate=0.10,
                carrying_capacity=80.0,
                clearance_rate=0.40,
            )
        }
        sample_times = np.array([0.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
        report = compare_treatment_scenarios(
            sample_times=sample_times,
            reference=reference,
            parameters=parameters,
            scenarios={
                "gamma_only": (
                    TreatmentFraction(day=0.0, dose=4.0, family="y"),
                    TreatmentFraction(day=1.0, dose=4.0, family="y"),
                ),
                "mixed_peak": (
                    TreatmentFraction(day=0.0, dose=4.0, family="y"),
                    TreatmentFraction(day=1.0, dose=4.0, family="p_peak"),
                ),
            },
            family_parameters=family_parameters,
        )

        self.assertEqual(len(report.rows), 2)
        self.assertEqual(report.rows[0].scenario, "mixed_peak")
        self.assertEqual(report.rows[0].total_physical_dose, 8.0)
        self.assertLess(report.rows[0].final_total_volume, report.rows[1].final_total_volume)


if __name__ == "__main__":
    unittest.main()
