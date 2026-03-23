import unittest
from unittest.mock import patch

try:
    from survival.fit_alpha_beta_using_processor import (
        Fitter,
        LQFitResult,
        parse_cli,
        parse_family_let_values,
    )
    from survival.let_parametrization import LETDependentParams, fit_let_dependence
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
        Fitter,
        LQFitResult,
        parse_cli,
        parse_family_let_values,
    )
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
        fit_let_dependence,
    )


class LETParametrizationTests(unittest.TestCase):
    def test_fit_let_dependence_recovers_linear_alpha_beta(self) -> None:
        alpha_0 = 0.08
        lambda_alpha = 0.002
        beta_0 = 0.01
        lambda_beta = 0.00005
        family_lets = {"y": 0.3, "p": 12.0, "n": 45.0, "c": 100.0}
        family_results = {
            family: LQFitResult(
                alpha=alpha_0 + lambda_alpha * let_value,
                beta=beta_0 + lambda_beta * let_value,
                train_count=index + 2,
                train_kind="all",
                family=family,
                sf_mode="absolute",
            )
            for index, (family, let_value) in enumerate(family_lets.items())
        }

        params = fit_let_dependence(family_results, family_lets)

        self.assertAlmostEqual(params.alpha_0, alpha_0, places=8)
        self.assertAlmostEqual(params.lambda_alpha, lambda_alpha, places=8)
        self.assertAlmostEqual(params.beta_0, beta_0, places=8)
        self.assertAlmostEqual(params.lambda_beta, lambda_beta, places=8)
        self.assertEqual(params.family_order, ("y", "p", "n", "c"))
        self.assertAlmostEqual(params.alpha_r_squared or 0.0, 1.0, places=8)
        self.assertAlmostEqual(params.beta_r_squared or 0.0, 1.0, places=8)

    def test_fitter_fit_let_dependence_uses_same_parametrization_logic(self) -> None:
        fitter = Fitter(
            sf_mode="absolute",
            min_sf=1.0,
            alpha_fixed=None,
            verbose=False,
        )
        family_lets = {"y": 0.3, "p": 12.0, "n": 45.0}
        family_results = {
            family: LQFitResult(
                alpha=0.1 + 0.001 * let_value,
                beta=0.02,
                train_count=3,
                train_kind="all",
                family=family,
                sf_mode="absolute",
            )
            for family, let_value in family_lets.items()
        }

        params = fitter.fit_let_dependence(family_results, family_lets)

        self.assertAlmostEqual(params.alpha(12.0), 0.112, places=8)
        self.assertAlmostEqual(params.beta(45.0), 0.02, places=8)
        self.assertAlmostEqual(params.alpha_beta_ratio(12.0), 0.112 / 0.02, places=8)

    def test_let_dependent_params_support_alpha_saturation(self) -> None:
        params = LETDependentParams(
            alpha_0=0.1,
            lambda_alpha=0.01,
            beta_0=0.02,
            lambda_beta=0.0,
            let_max=10.0,
        )

        self.assertAlmostEqual(params.alpha(5.0), 0.15, places=8)
        self.assertAlmostEqual(params.alpha(50.0), 0.2, places=8)
        self.assertAlmostEqual(params.beta(50.0), 0.02, places=8)

    def test_parse_family_let_values_and_cli_accept_let_flags(self) -> None:
        parsed = parse_family_let_values("y=0.5,p=11.0")

        self.assertAlmostEqual(parsed["y"], 0.5, places=8)
        self.assertAlmostEqual(parsed["p"], 11.0, places=8)
        self.assertIn("n", parsed)

        with patch(
            "sys.argv",
            [
                "fit_alpha_beta_using_processor.py",
                "--let-fit",
                "--let-values",
                "y=0.5,p=11.0",
            ],
        ):
            args = parse_cli()

        self.assertTrue(args.let_fit)
        self.assertEqual(args.let_values, "y=0.5,p=11.0")


if __name__ == "__main__":
    unittest.main()
