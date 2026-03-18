import unittest

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_gui import (
    build_rbe_table_rows,
    build_sf_metric_table_rows,
    parse_positive_float_csv,
    select_rbe_run_context,
)
from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    AnalysisRunSummary,
    LQFitResult,
)
from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
    RBEPoint,
    SFMetricComparisonRow,
)


class FitAlphaBetaGuiHelperTests(unittest.TestCase):
    def test_parse_positive_float_csv_uses_defaults_for_empty_input(self) -> None:
        self.assertEqual(parse_positive_float_csv(""), [2.0, 10.0])

    def test_parse_positive_float_csv_parses_csv_values(self) -> None:
        self.assertEqual(parse_positive_float_csv("2, 10, 15.5"), [2.0, 10.0, 15.5])

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


if __name__ == "__main__":
    unittest.main()
