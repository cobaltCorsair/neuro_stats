import tempfile
import unittest
from pathlib import Path

try:
    from survival.geant4_pipeline_gui import resolve_radiobiology_source_from_run_results
    from survival.fit_alpha_beta_using_processor import (
        AnalysisRunResult,
        AnalysisRunSummary,
        LQFitResult,
    )
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.geant4_pipeline_gui import (
        resolve_radiobiology_source_from_run_results,
    )
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
        AnalysisRunResult,
        AnalysisRunSummary,
        LQFitResult,
    )


class Geant4PipelineGuiTests(unittest.TestCase):
    def test_resolve_source_exports_family_specific_runs_to_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_path = Path(tmp_dir) / "fit_results.csv"
            run = _build_run_result(family="y", alpha=0.1, beta=0.02)

            source = resolve_radiobiology_source_from_run_results([run], export_path)

            self.assertEqual(source.fit_results_csv, export_path)
            self.assertIsNone(source.manual_alpha_beta)
            self.assertTrue(export_path.exists())
            content = export_path.read_text(encoding="utf-8")
            self.assertIn("family", content)
            self.assertIn("y", content)

    def test_resolve_source_uses_manual_alpha_beta_for_all_family_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_path = Path(tmp_dir) / "fit_results.csv"
            run = _build_run_result(family=None, alpha=0.12, beta=0.03)

            source = resolve_radiobiology_source_from_run_results([run], export_path)

            self.assertEqual(source.manual_alpha_beta, (0.12, 0.03))
            self.assertIsNone(source.fit_results_csv)
            self.assertFalse(export_path.exists())

    def test_resolve_source_uses_let_profile_for_let_dependent_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_path = Path(tmp_dir) / "fit_results.csv"
            run = _build_run_result(
                family=None,
                alpha=0.11,
                beta=0.025,
                model_kind="let_dependent",
                alpha_0=0.08,
                lambda_alpha=0.003,
            )

            source = resolve_radiobiology_source_from_run_results([run], export_path)

            self.assertIsNone(source.fit_results_csv)
            self.assertIsNone(source.manual_alpha_beta)
            self.assertIsNotNone(source.let_params)
            self.assertAlmostEqual(source.let_params.alpha_0, 0.08, places=8)
            self.assertAlmostEqual(source.let_params.lambda_alpha, 0.003, places=8)
            self.assertAlmostEqual(source.let_params.beta_0, 0.025, places=8)


def _build_run_result(
    *,
    family: str | None,
    alpha: float,
    beta: float,
    model_kind: str = "classic_lq",
    alpha_0: float | None = None,
    lambda_alpha: float | None = None,
) -> AnalysisRunResult:
    fit_result = LQFitResult(
        alpha=alpha,
        beta=beta,
        train_count=3,
        train_kind="all",
        family=family,
        sf_mode="absolute",
        model_kind=model_kind,  # type: ignore[arg-type]
        alpha_0=alpha_0,
        lambda_alpha=lambda_alpha,
    )
    return AnalysisRunResult(
        summary=AnalysisRunSummary(
            sf_mode="absolute",
            family=family,
            total_count=3,
            single_count=2,
            fractionated_count=1,
            train_count=3,
            validation_count=0,
            status="ok",
            response_mode="scalar",
            model_kind=model_kind,
            alpha=alpha,
            beta=beta,
        ),
        train=(),
        validation=(),
        train_kind="all",
        validation_kind="none",
        fit_result=fit_result,
    )


if __name__ == "__main__":
    unittest.main()
