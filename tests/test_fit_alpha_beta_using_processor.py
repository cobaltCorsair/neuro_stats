import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    Fitter,
    RawTumorSeries,
    AnalysisRunSummary,
    TumorExperiment,
    analyze_fitter,
    infer_radiation_family,
    is_control_file,
    parse_schedule_days,
    parse_sf_modes,
    resolve_requested_families,
)


class FitAlphaBetaUsingProcessorTests(unittest.TestCase):
    def test_fit_recovers_alpha_beta_from_two_regimens(self) -> None:
        expected_alpha = 0.12
        expected_beta = 0.03

        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        fitter.experiments = [
            TumorExperiment(
                path=Path("single_fraction.xlsx"),
                fractions=(10.0,),
                sf=math.exp(-(expected_alpha * 10.0 + expected_beta * 100.0)),
                family="y",
            ),
            TumorExperiment(
                path=Path("split_fraction.xlsx"),
                fractions=(5.0, 5.0),
                sf=math.exp(-(expected_alpha * 10.0 + expected_beta * 50.0)),
                family="y",
            ),
        ]

        result = fitter.fit()

        self.assertAlmostEqual(result.alpha, expected_alpha, places=6)
        self.assertAlmostEqual(result.beta, expected_beta, places=6)

    def test_fit_respects_fixed_alpha(self) -> None:
        expected_alpha = 0.15
        expected_beta = 0.02

        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=expected_alpha,
            verbose=False,
        )
        fitter.experiments = [
            TumorExperiment(
                path=Path("a.xlsx"),
                fractions=(8.0,),
                sf=math.exp(-(expected_alpha * 8.0 + expected_beta * 64.0)),
                family="y",
            ),
            TumorExperiment(
                path=Path("b.xlsx"),
                fractions=(4.0, 4.0),
                sf=math.exp(-(expected_alpha * 8.0 + expected_beta * 32.0)),
                family="y",
            ),
            TumorExperiment(
                path=Path("c.xlsx"),
                fractions=(6.0, 2.0),
                sf=math.exp(-(expected_alpha * 8.0 + expected_beta * 40.0)),
                family="y",
            ),
        ]

        result = fitter.fit()

        self.assertAlmostEqual(result.alpha, expected_alpha, places=6)
        self.assertAlmostEqual(result.beta, expected_beta, places=6)

    def test_build_train_validation_sets_separates_single_and_fractionated(self) -> None:
        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        fitter.experiments = [
            TumorExperiment(
                path=Path("y_single_32.xlsx"),
                fractions=(32.0,),
                sf=0.3,
                family="y",
            ),
            TumorExperiment(
                path=Path("y_single_40.xlsx"),
                fractions=(40.0,),
                sf=0.2,
                family="y",
            ),
            TumorExperiment(
                path=Path("y_fractionated_40.xlsx"),
                fractions=(4.0, 4.0, 32.0),
                sf=0.4,
                family="y",
            ),
            TumorExperiment(
                path=Path("p_single_40.xlsx"),
                fractions=(40.0,),
                sf=0.25,
                family="p",
            ),
        ]

        train, validation = fitter.build_train_validation_sets(
            fit_kind="single",
            validate_kind="all",
            family="y",
        )

        self.assertEqual(
            [experiment.path.name for experiment in train],
            ["y_single_32.xlsx", "y_single_40.xlsx"],
        )
        self.assertEqual(
            [experiment.path.name for experiment in validation],
            ["y_fractionated_40.xlsx"],
        )

    def test_parse_schedule_days_converts_hour_intervals(self) -> None:
        schedule_days, has_explicit_timing = parse_schedule_days(
            ["y = 4 Гр", "y = 4 Гр", "y = 32 Гр", "t = 1 ч"],
            fractions=(4.0, 4.0, 32.0),
        )

        self.assertTrue(has_explicit_timing)
        self.assertEqual(len(schedule_days), 3)
        self.assertAlmostEqual(schedule_days[0], 0.0, places=8)
        self.assertAlmostEqual(schedule_days[1], 1.0 / 24.0, places=8)
        self.assertAlmostEqual(schedule_days[2], 2.0 / 24.0, places=8)

    def test_available_families_and_resolution_for_batch_mode(self) -> None:
        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        fitter.experiments = [
            TumorExperiment(Path("y_32.xlsx"), (32.0,), 0.3, "y"),
            TumorExperiment(Path("p_36.xlsx"), (36.0,), 0.2, "p"),
            TumorExperiment(Path("unknown.xlsx"), (20.0,), 0.5, None),
        ]

        families = fitter.available_families()

        self.assertEqual(families, ["p", "y"])
        self.assertEqual(resolve_requested_families(families, None, by_family=False), [None])
        self.assertEqual(resolve_requested_families(families, None, by_family=True), ["p", "y"])
        self.assertEqual(resolve_requested_families(families, "y", by_family=True), ["y"])

    def test_write_analysis_summaries_csv(self) -> None:
        summaries = [
            AnalysisRunSummary(
                sf_mode="absolute",
                family="e",
                total_count=3,
                single_count=2,
                fractionated_count=1,
                train_count=2,
                validation_count=1,
                status="ok",
                alpha=0.1,
                beta=0.02,
                reason=None,
            ),
            AnalysisRunSummary(
                sf_mode="absolute",
                family="n",
                total_count=1,
                single_count=1,
                fractionated_count=0,
                train_count=1,
                validation_count=0,
                status="skipped",
                alpha=None,
                beta=None,
                reason="not enough training experiments",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.csv"
            Fitter.write_analysis_summaries_csv(output, summaries)
            text = output.read_text(encoding="utf-8")

        self.assertIn("sf_mode,family,status,total_count", text)
        self.assertIn("absolute,e,ok,3,2,1,2,1,0.1,0.02,5.0,", text)
        self.assertIn("absolute,n,skipped,1,1,0,1,0,,,", text)

    def test_aggregate_experiments_merges_repeats(self) -> None:
        experiments = [
            TumorExperiment(Path("y_36_a.xlsx"), (36.0,), 0.20, "y"),
            TumorExperiment(Path("y_36_b.xlsx"), (36.0,), 0.30, "y"),
            TumorExperiment(Path("y_40.xlsx"), (40.0,), 0.10, "y"),
        ]

        aggregated = Fitter.aggregate_experiments(experiments)
        aggregated_by_name = {experiment.path.name: experiment for experiment in aggregated}

        self.assertEqual(len(aggregated), 2)
        merged = aggregated_by_name["y_36_a.xlsx"]
        self.assertEqual(merged.repeat_count, 2)
        self.assertAlmostEqual(merged.sf, 0.25, places=6)
        self.assertAlmostEqual(merged.sf_std, 0.05, places=6)
        self.assertEqual(len(merged.source_paths), 2)

    def test_aggregate_experiments_keeps_same_doses_with_different_timing_separate(self) -> None:
        experiments = [
            TumorExperiment(
                Path("y_split_1h.xlsx"),
                (4.0, 4.0, 32.0),
                0.20,
                "y",
                schedule_days=(0.0, 1.0 / 24.0, 2.0 / 24.0),
                has_explicit_timing=True,
            ),
            TumorExperiment(
                Path("y_split_24h.xlsx"),
                (4.0, 4.0, 32.0),
                0.22,
                "y",
                schedule_days=(0.0, 1.0, 2.0),
                has_explicit_timing=True,
            ),
        ]

        aggregated = Fitter.aggregate_experiments(experiments)

        self.assertEqual(len(aggregated), 2)

    def test_weighted_fit_matches_explicit_repeats(self) -> None:
        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        repeated = [
            TumorExperiment(Path("a1.xlsx"), (10.0,), 0.1, "y"),
            TumorExperiment(Path("a2.xlsx"), (10.0,), 0.1, "y"),
            TumorExperiment(Path("b.xlsx"), (5.0, 5.0), 0.3, "y"),
        ]
        aggregated = Fitter.aggregate_experiments(repeated)

        explicit_result = fitter.fit(experiments=repeated)
        weighted_result = fitter.fit(experiments=aggregated)

        self.assertAlmostEqual(explicit_result.alpha, weighted_result.alpha, places=6)
        self.assertAlmostEqual(explicit_result.beta, weighted_result.beta, places=6)

    def test_fit_downweights_noisy_aggregated_regimen(self) -> None:
        expected_alpha = 0.12
        expected_beta = 0.03

        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        baseline_experiments = [
            TumorExperiment(
                Path("a.xlsx"),
                (8.0,),
                math.exp(-(expected_alpha * 8.0 + expected_beta * 64.0)),
                "y",
            ),
            TumorExperiment(
                Path("b.xlsx"),
                (4.0, 4.0),
                math.exp(-(expected_alpha * 8.0 + expected_beta * 32.0)),
                "y",
            ),
        ]
        noisy_sf = 0.5
        noisy_low_var = TumorExperiment(
            Path("c.xlsx"),
            (6.0,),
            noisy_sf,
            "y",
            repeat_count=5,
            sf_std=0.01,
        )
        noisy_high_var = TumorExperiment(
            Path("c.xlsx"),
            (6.0,),
            noisy_sf,
            "y",
            repeat_count=5,
            sf_std=0.30,
        )

        low_var_result = fitter.fit(experiments=baseline_experiments + [noisy_low_var])
        high_var_result = fitter.fit(experiments=baseline_experiments + [noisy_high_var])

        low_var_error = abs(low_var_result.alpha - expected_alpha) + abs(
            low_var_result.beta - expected_beta
        )
        high_var_error = abs(high_var_result.alpha - expected_alpha) + abs(
            high_var_result.beta - expected_beta
        )

        self.assertLess(high_var_error, low_var_error)

    def test_fit_uses_repair_aware_quadratic_term(self) -> None:
        expected_alpha = 0.11
        expected_beta = 0.025
        repair_half_time_hours = 1.5
        repair_rate_per_day = math.log(2.0) * 24.0 / repair_half_time_hours

        short_gap = TumorExperiment(
            Path("short_gap.xlsx"),
            (4.0, 4.0, 32.0),
            1.0,
            "y",
            schedule_days=(0.0, 0.5 / 24.0, 3.0 / 24.0),
            has_explicit_timing=True,
        )
        long_gap = TumorExperiment(
            Path("long_gap.xlsx"),
            (4.0, 4.0, 32.0),
            1.0,
            "y",
            schedule_days=(0.0, 24.0 / 24.0, 72.0 / 24.0),
            has_explicit_timing=True,
        )
        single = TumorExperiment(
            Path("single.xlsx"),
            (40.0,),
            1.0,
            "y",
        )
        experiments = []
        for template in (single, short_gap, long_gap):
            sf = math.exp(
                -(
                    expected_alpha * template.dose_sum
                    + expected_beta * template.quadratic_term(repair_rate_per_day)
                )
            )
            experiments.append(
                TumorExperiment(
                    path=template.path,
                    fractions=template.fractions,
                    sf=sf,
                    family=template.family,
                    schedule_days=template.schedule_days,
                    has_explicit_timing=template.has_explicit_timing,
                )
            )

        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
            repair_half_time_hours=repair_half_time_hours,
        )
        result = fitter.fit(experiments=experiments)

        self.assertAlmostEqual(result.alpha, expected_alpha, places=6)
        self.assertAlmostEqual(result.beta, expected_beta, places=6)

    def test_compute_timing_diagnostics_warns_for_weak_repair_dataset(self) -> None:
        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
            repair_half_time_hours=1.0,
        )
        experiments = [
            TumorExperiment(Path("single.xlsx"), (40.0,), 0.1, "y"),
            TumorExperiment(
                Path("split.xlsx"),
                (4.0, 4.0, 32.0),
                0.2,
                "y",
                schedule_days=(0.0, 1.0 / 24.0, 2.0 / 24.0),
                has_explicit_timing=True,
            ),
        ]

        diagnostics = fitter.compute_timing_diagnostics(experiments)

        self.assertTrue(diagnostics.repair_model_enabled)
        self.assertEqual(diagnostics.same_fractions_multi_timing_count, 0)
        self.assertGreaterEqual(len(diagnostics.warnings), 1)
        self.assertTrue(
            any("likely unstable" in warning for warning in diagnostics.warnings)
        )

    def test_compute_timing_diagnostics_detects_direct_timing_contrast(self) -> None:
        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
            repair_half_time_hours=1.0,
        )
        experiments = [
            TumorExperiment(Path("single.xlsx"), (40.0,), 0.1, "y"),
            TumorExperiment(
                Path("split_1h.xlsx"),
                (4.0, 4.0, 32.0),
                0.2,
                "y",
                schedule_days=(0.0, 1.0 / 24.0, 2.0 / 24.0),
                has_explicit_timing=True,
            ),
            TumorExperiment(
                Path("split_24h.xlsx"),
                (4.0, 4.0, 32.0),
                0.25,
                "y",
                schedule_days=(0.0, 1.0, 2.0),
                has_explicit_timing=True,
            ),
        ]

        diagnostics = fitter.compute_timing_diagnostics(experiments)

        self.assertEqual(diagnostics.same_fractions_multi_timing_count, 1)
        self.assertEqual(diagnostics.explicit_fractionated_count, 2)
        self.assertGreaterEqual(diagnostics.unique_quadratic_count, 2)
        self.assertFalse(
            any("No matched dose pattern" in warning for warning in diagnostics.warnings)
        )

    def test_parse_sf_modes_supports_many_formats(self) -> None:
        self.assertEqual(parse_sf_modes(None), ["absolute"])
        self.assertEqual(parse_sf_modes(["absolute"]), ["absolute"])
        self.assertEqual(
            parse_sf_modes(["absolute,absindex:2", "absindex:3"]),
            ["absolute", "absindex:2", "absindex:3"],
        )

    def test_bootstrap_fit_over_raw_animals(self) -> None:
        expected_alpha = 0.12
        expected_beta = 0.03
        sf_single = math.exp(-(expected_alpha * 10.0 + expected_beta * 100.0))
        sf_split = math.exp(-(expected_alpha * 10.0 + expected_beta * 50.0))

        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        fitter.controls = {
            Path("control.xlsx"): np.array(
                [
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                ],
                dtype=float,
            )
        }
        fitter.raw_experiments = [
            RawTumorSeries(
                path=Path("single.xlsx"),
                fractions=(10.0,),
                family="y",
                volumes=np.array(
                    [
                        [10.0, 20.0 * sf_single, 36.0],
                        [10.0, 20.0 * sf_single, 36.0],
                    ],
                    dtype=float,
                ),
            ),
            RawTumorSeries(
                path=Path("split.xlsx"),
                fractions=(5.0, 5.0),
                family="y",
                volumes=np.array(
                    [
                        [10.0, 20.0 * sf_split, 36.0],
                        [10.0, 20.0 * sf_split, 36.0],
                    ],
                    dtype=float,
                ),
            ),
        ]

        summary = fitter.bootstrap_fit(
            sf_mode="absolute",
            fit_kind="all",
            family="y",
            repeats=10,
            seed=123,
        )

        self.assertEqual(summary.requested_repeats, 10)
        self.assertEqual(summary.successful_repeats, 10)
        self.assertEqual(summary.failed_repeats, 0)
        self.assertAlmostEqual(summary.alpha.mean, expected_alpha, places=6)
        self.assertAlmostEqual(summary.beta.mean, expected_beta, places=6)
        self.assertAlmostEqual(summary.alpha.q025, expected_alpha, places=6)
        self.assertAlmostEqual(summary.beta.q975, expected_beta, places=6)

    def test_materialize_experiments_can_use_different_controls_per_experiment(self) -> None:
        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        control_a = Path("control_a.xlsx")
        control_b = Path("control_b.xlsx")
        fitter.controls = {
            control_a: np.array(
                [
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                ],
                dtype=float,
            ),
            control_b: np.array(
                [
                    [10.0, 40.0, 60.0],
                    [10.0, 40.0, 60.0],
                ],
                dtype=float,
            ),
        }
        experiment_volumes = np.array(
            [
                [10.0, 10.0, 30.0],
                [10.0, 10.0, 30.0],
            ],
            dtype=float,
        )
        fitter.raw_experiments = [
            RawTumorSeries(
                path=Path("exp_a.xlsx"),
                fractions=(10.0,),
                family="y",
                volumes=experiment_volumes,
                control_path=control_a,
            ),
            RawTumorSeries(
                path=Path("exp_b.xlsx"),
                fractions=(10.0,),
                family="y",
                volumes=experiment_volumes,
                control_path=control_b,
            ),
        ]

        experiments = fitter.materialize_experiments(sf_mode="absolute")
        by_name = {experiment.path.name: experiment for experiment in experiments}

        self.assertAlmostEqual(by_name["exp_a.xlsx"].sf, 0.5, places=6)
        self.assertAlmostEqual(by_name["exp_b.xlsx"].sf, 0.25, places=6)
        self.assertEqual(by_name["exp_a.xlsx"].control_path, control_a)
        self.assertEqual(by_name["exp_b.xlsx"].control_path, control_b)

    def test_infer_radiation_family_from_file_name(self) -> None:
        self.assertEqual(infer_radiation_family(Path("19.03.2025_y_40.xlsx")), "y")
        self.assertEqual(infer_radiation_family(Path("22.10.2025_p16_p16_p16_in_peak.xlsx")), "p")
        self.assertEqual(infer_radiation_family(Path("02.02.2023_n_12.xlsx")), "n")
        self.assertEqual(infer_radiation_family(Path("15.01.2026_e_18.xlsx")), "e")

    def test_is_control_file_uses_file_name(self) -> None:
        self.assertTrue(is_control_file(Path("control_2016.xlsx")))
        self.assertTrue(is_control_file(Path("gamma_CONTROL_series.xlsx")))
        self.assertFalse(is_control_file(Path("19.03.2025_y_40.xlsx")))

    def test_analyze_fitter_returns_structured_result(self) -> None:
        expected_alpha = 0.12
        expected_beta = 0.03

        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        fitter.experiments = [
            TumorExperiment(
                path=Path("y_single_32.xlsx"),
                fractions=(32.0,),
                sf=math.exp(-(expected_alpha * 32.0 + expected_beta * (32.0 ** 2))),
                family="y",
            ),
            TumorExperiment(
                path=Path("y_single_40.xlsx"),
                fractions=(40.0,),
                sf=math.exp(-(expected_alpha * 40.0 + expected_beta * (40.0 ** 2))),
                family="y",
            ),
            TumorExperiment(
                path=Path("y_fractionated_40.xlsx"),
                fractions=(4.0, 4.0, 32.0),
                sf=math.exp(-(expected_alpha * 40.0 + expected_beta * (4.0 ** 2 + 4.0 ** 2 + 32.0 ** 2))),
                family="y",
            ),
        ]

        results = analyze_fitter(
            fitter=fitter,
            sf_modes=["absolute"],
            fit_kind="single",
            validate_kind="fractionated",
            family="y",
            by_family=False,
            bootstrap=0,
            bootstrap_seed=None,
        )

        self.assertEqual(len(results), 1)
        run = results[0]
        self.assertEqual(run.summary.status, "ok")
        self.assertEqual(run.summary.train_count, 2)
        self.assertEqual(run.summary.validation_count, 1)
        self.assertEqual(len(run.train), 2)
        self.assertEqual(len(run.validation), 1)
        self.assertIsNotNone(run.fit_result)
        self.assertIsNotNone(run.validation_summary)
        self.assertIsNotNone(run.timing_diagnostics)
        self.assertFalse(run.timing_diagnostics.repair_model_enabled)
        self.assertAlmostEqual(run.fit_result.alpha, expected_alpha, places=6)
        self.assertAlmostEqual(run.fit_result.beta, expected_beta, places=6)


if __name__ == "__main__":
    unittest.main()
