import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_gui import (
    FitAlphaBetaWindow,
    SUMMARY_HEADERS,
    build_cross_validation_table_rows,
    build_let_alpha_table_rows,
    build_ntcp_source_table_rows,
    build_ntcp_table_rows,
    build_rbe_table_rows,
    build_sf_metric_table_rows,
    build_tcp_table_rows,
    parse_positive_float_csv,
    parse_positive_scalar,
    select_let_run_context,
    select_rbe_run_context,
)
from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    AnalysisRunSummary,
    LQFitResult,
    PredictionRow,
    TumorExperiment,
)
from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
    NTCPFitGroup,
    NTCPPoint,
    RBEPoint,
    SFMetricComparisonRow,
    TCPResult,
)


class FitAlphaBetaGuiHelperTests(unittest.TestCase):
    def test_summary_headers_include_bed_eqd2_and_g(self) -> None:
        self.assertIn("BED", SUMMARY_HEADERS)
        self.assertIn("EQD2", SUMMARY_HEADERS)
        self.assertIn("G", SUMMARY_HEADERS)

    def test_parse_positive_float_csv_uses_defaults_for_empty_input(self) -> None:
        self.assertEqual(parse_positive_float_csv(""), [2.0, 10.0])

    def test_parse_positive_float_csv_parses_csv_values(self) -> None:
        self.assertEqual(parse_positive_float_csv("2, 10, 15.5"), [2.0, 10.0, 15.5])

    def test_parse_positive_scalar_accepts_scientific_notation(self) -> None:
        self.assertEqual(parse_positive_scalar("1e7", label="Cell density"), 1.0e7)

    def test_select_rbe_run_context_uses_same_response_model_and_sf_mode(self) -> None:
        reference_fit = LQFitResult(
            alpha=0.10,
            beta=0.02,
            train_count=3,
            train_kind="all",
            family="y",
            sf_mode="absolute",
            model_kind="classic_lq",
            response_mode="scalar",
        )
        proton_fit = LQFitResult(
            alpha=0.20,
            beta=0.03,
            train_count=3,
            train_kind="all",
            family="p_peak",
            sf_mode="absolute",
            model_kind="classic_lq",
            response_mode="scalar",
        )
        ignored_fit = LQFitResult(
            alpha=0.30,
            beta=0.04,
            train_count=3,
            train_kind="all",
            family="n",
            sf_mode="absindex:1",
            model_kind="classic_lq",
            response_mode="scalar",
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
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=reference_fit,
            ),
            AnalysisRunResult(
                summary=AnalysisRunSummary(
                    sf_mode="absolute",
                    family="p_peak",
                    total_count=3,
                    single_count=2,
                    fractionated_count=1,
                    train_count=3,
                    validation_count=0,
                    status="ok",
                    response_mode="scalar",
                    model_kind="classic_lq",
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=proton_fit,
            ),
            AnalysisRunResult(
                summary=AnalysisRunSummary(
                    sf_mode="absindex:1",
                    family="n",
                    total_count=3,
                    single_count=2,
                    fractionated_count=1,
                    train_count=3,
                    validation_count=0,
                    status="ok",
                    response_mode="scalar",
                    model_kind="classic_lq",
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=ignored_fit,
            ),
        ]

        reference_result, comparison_results, context_label = select_rbe_run_context(
            runs,
            selected_index=1,
            reference_family="y",
        )

        self.assertAlmostEqual(reference_result.alpha, reference_fit.alpha, places=8)
        self.assertEqual(sorted(comparison_results.keys()), ["p_peak"])
        self.assertIn("sf=absolute", context_label)
        self.assertIn("model=classic_lq", context_label)

    def test_build_rbe_table_rows_orders_by_family_then_dose(self) -> None:
        rows = build_rbe_table_rows(
            [
                RBEPoint(
                    reference_family="y",
                    test_family="p_peak",
                    test_dose=10.0,
                    reference_dose=11.0,
                    rbe=1.1,
                    test_alpha_beta_ratio=6.5,
                    reference_alpha_beta_ratio=7.5,
                    test_model_kind="classic_lq",
                    reference_model_kind="classic_lq",
                ),
                RBEPoint(
                    reference_family="y",
                    test_family="c",
                    test_dose=2.0,
                    reference_dose=4.0,
                    rbe=2.0,
                    test_alpha_beta_ratio=3.2,
                    reference_alpha_beta_ratio=7.5,
                    test_model_kind="glq",
                    reference_model_kind="classic_lq",
                ),
            ]
        )

        self.assertEqual(rows[0][1], "c")
        self.assertEqual(rows[0][2], "2.000")
        self.assertEqual(rows[1][1], "p_peak")


class FitAlphaBetaGuiWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_growth_predictor_button_opens_window(self) -> None:
        window = FitAlphaBetaWindow()

        window.open_growth_predictor()

        self.assertIsNotNone(window.growth_predictor_window)
        self.assertEqual(
            window.growth_predictor_window.windowTitle(),
            "Tumor growth predictor",
        )

    def test_build_sf_metric_table_rows_formats_numeric_columns(self) -> None:
        rows = build_sf_metric_table_rows(
            [
                SFMetricComparisonRow(
                    family="y",
                    response_mode="scalar",
                    model_kind="classic_lq",
                    sf_mode="absolute",
                    alpha=0.0123,
                    beta=0.0012,
                    alpha_beta_ratio=10.25,
                    delta_alpha_pct=1.5,
                    delta_beta_pct=-2.5,
                    delta_ratio_pct=3.75,
                    baseline_sf_mode="absolute",
                )
            ]
        )

        self.assertEqual(rows[0][4], "0.012300")
        self.assertEqual(rows[0][6], "10.250")
        self.assertEqual(rows[0][9], "3.75")

    def test_select_let_run_context_prefers_classic_per_family_points(self) -> None:
        let_fit = LQFitResult(
            alpha=0.03,
            beta=0.004,
            train_count=7,
            train_kind="all",
            family=None,
            sf_mode="absolute",
            model_kind="let_dependent",
            response_mode="scalar",
            alpha_0=0.03,
            lambda_alpha=0.0015,
        )
        gamma_fit = LQFitResult(
            alpha=0.010,
            beta=0.001,
            train_count=3,
            train_kind="all",
            family="y",
            sf_mode="absolute",
            model_kind="classic_lq",
            response_mode="scalar",
        )
        proton_classic = LQFitResult(
            alpha=0.020,
            beta=0.0015,
            train_count=3,
            train_kind="all",
            family="p_peak",
            sf_mode="absolute",
            model_kind="classic_lq",
            response_mode="scalar",
        )
        proton_glq = LQFitResult(
            alpha=0.030,
            beta=0.0020,
            train_count=3,
            train_kind="all",
            family="p_peak",
            sf_mode="absolute",
            model_kind="glq",
            response_mode="scalar",
            saturation_dose=20.0,
        )

        runs = [
            AnalysisRunResult(
                summary=AnalysisRunSummary(
                    sf_mode="absolute",
                    family=None,
                    total_count=7,
                    single_count=5,
                    fractionated_count=2,
                    train_count=7,
                    validation_count=0,
                    status="ok",
                    response_mode="scalar",
                    model_kind="let_dependent",
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=let_fit,
            ),
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
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=gamma_fit,
            ),
            AnalysisRunResult(
                summary=AnalysisRunSummary(
                    sf_mode="absolute",
                    family="p_peak",
                    total_count=3,
                    single_count=2,
                    fractionated_count=1,
                    train_count=3,
                    validation_count=0,
                    status="ok",
                    response_mode="scalar",
                    model_kind="glq",
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=proton_glq,
            ),
            AnalysisRunResult(
                summary=AnalysisRunSummary(
                    sf_mode="absolute",
                    family="p_peak",
                    total_count=3,
                    single_count=2,
                    fractionated_count=1,
                    train_count=3,
                    validation_count=0,
                    status="ok",
                    response_mode="scalar",
                    model_kind="classic_lq",
                ),
                train=(),
                validation=(),
                train_kind="all",
                validation_kind="none",
                fit_result=proton_classic,
            ),
        ]

        selected_fit, points, context_label = select_let_run_context(runs, selected_index=0)

        self.assertEqual(selected_fit.model_kind, "let_dependent")
        self.assertIn("let-model=let_dependent", context_label)
        self.assertEqual([point[0] for point in points], ["y", "p_peak"])
        self.assertAlmostEqual(points[1][2], proton_classic.alpha, places=8)

    def test_build_let_alpha_table_rows_formats_numeric_columns(self) -> None:
        rows = build_let_alpha_table_rows(
            [
                ("y", 0.3, 0.0102, "classic_lq"),
                ("c", 100.0, 0.0850, "classic_lq"),
            ]
        )

        self.assertEqual(rows[0][0], "y")
        self.assertEqual(rows[0][1], "0.300")
        self.assertEqual(rows[1][0], "c")
        self.assertEqual(rows[1][2], "0.085000")

    def test_build_ntcp_table_rows_formats_numeric_columns(self) -> None:
        rows = build_ntcp_table_rows(
            [
                NTCPPoint(dose_total=10.0, ntcp=0.0123, td50=50.0, m=0.2),
                NTCPPoint(dose_total=50.0, ntcp=0.5, td50=50.0, m=0.2),
            ]
        )

        self.assertEqual(rows[0][0], "10.000")
        self.assertEqual(rows[0][1], "0.012300")
        self.assertEqual(rows[1][2], "50.000")
        self.assertEqual(rows[1][3], "0.2000")

    def test_build_ntcp_source_table_rows_formats_group_counts(self) -> None:
        rows = build_ntcp_source_table_rows(
            [
                NTCPFitGroup(
                    label="p_40_skin.xlsx",
                    dose_total=40.0,
                    n_subjects=12,
                    n_complications=3,
                    complication_rate=0.25,
                    peak_grade_mean=2.75,
                    threshold_grade=3,
                )
            ]
        )

        self.assertEqual(rows[0][0], "p_40_skin.xlsx")
        self.assertEqual(rows[0][1], "40.000")
        self.assertEqual(rows[0][2], "12")
        self.assertEqual(rows[0][3], "3")
        self.assertEqual(rows[0][4], "0.250000")
        self.assertEqual(rows[0][5], "2.750")

    def test_build_cross_validation_table_rows_formats_prediction_errors(self) -> None:
        experiment = TumorExperiment(
            path=Path("experiment.xlsx"),
            fractions=(10.0,),
            sf=0.200000,
            family="y",
        )
        rows = build_cross_validation_table_rows(
            [
                PredictionRow(
                    experiment=experiment,
                    predicted_sf=0.250000,
                    abs_error=0.050000,
                    rel_error=0.25,
                    log_error=0.223144,
                )
            ]
        )

        self.assertEqual(rows[0][0], "experiment.xlsx")
        self.assertEqual(rows[0][1], "0.200000")
        self.assertEqual(rows[0][2], "0.250000")
        self.assertEqual(rows[0][3], "0.050000")
        self.assertEqual(rows[0][5], "25.00%")

    def test_build_tcp_table_rows_derives_bed_eqd2_and_g(self) -> None:
        fit_result = LQFitResult(
            alpha=0.10,
            beta=0.02,
            train_count=4,
            train_kind="all",
            family="y",
            sf_mode="absolute",
            model_kind="classic_lq",
            response_mode="scalar",
        )
        rows = build_tcp_table_rows(
            fit_result,
            [
                TCPResult(
                    dose_total=10.0,
                    sf=0.301000,
                    n_cells=1.2e7,
                    tcp=0.000123,
                    cell_density=1.0e7,
                    initial_volume_cm3=1.2,
                    family="y",
                    model_kind="classic_lq",
                )
            ],
            n_fractions=5,
            schedule_interval_days=1.0,
        )

        self.assertEqual(rows[0][0], "synthetic_tcp.xlsx")
        self.assertEqual(rows[0][1], "fractionated")
        self.assertEqual(rows[0][2], "10.000")
        self.assertEqual(rows[0][3], "2+2+2+2+2")
        self.assertEqual(rows[0][5], "14.0000")
        self.assertEqual(rows[0][6], "10.0000")
        self.assertEqual(rows[0][7], "0.2000")
        self.assertEqual(rows[0][8], "0.301000")
        self.assertEqual(rows[0][9], "0.000123")


if __name__ == "__main__":
    unittest.main()
