import math
import unittest

try:
    from survival.let_parametrization import LETDependentParams
    from survival.mixed_field_model import (
        FieldComponent,
        compute_mixed_field_sf,
    )
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
    )
    from work_with_prepared_data.radiobioligy_project.survival.mixed_field_model import (
        FieldComponent,
        compute_mixed_field_sf,
    )


class MixedFieldModelTests(unittest.TestCase):
    def test_pure_field_gives_same_sf_for_zaider_rossi_and_tdra(self) -> None:
        let_params = LETDependentParams(
            alpha_0=0.1,
            lambda_alpha=0.0,
            beta_0=0.02,
            lambda_beta=0.0,
        )
        result = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=4.0, mean_let_kev_um=0.3)],
            let_params,
            method="zaider_rossi",
        )

        expected_sf = math.exp(-(0.1 * 4.0 + 0.02 * 16.0))
        self.assertAlmostEqual(result.sf_zaider_rossi, expected_sf, places=8)
        self.assertAlmostEqual(result.sf_tdra, expected_sf, places=8)
        self.assertAlmostEqual(result.effective_alpha, 0.1, places=8)
        self.assertAlmostEqual(result.effective_beta, 0.02, places=8)

    def test_mixed_field_is_more_lethal_than_low_let_only_case(self) -> None:
        let_params = LETDependentParams(
            alpha_0=0.08,
            lambda_alpha=0.002,
            beta_0=0.002,
            lambda_beta=0.0,
        )
        low_let_only = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=4.0, mean_let_kev_um=1.0)],
            let_params,
        )
        mixed = compute_mixed_field_sf(
            [
                FieldComponent(family="y", dose_fraction_gy=2.0, mean_let_kev_um=1.0),
                FieldComponent(family="n", dose_fraction_gy=2.0, mean_let_kev_um=40.0),
            ],
            let_params,
        )

        self.assertLess(mixed.sf_zaider_rossi, low_let_only.sf_zaider_rossi)
        self.assertLess(mixed.sf_tdra, low_let_only.sf_tdra)

    def test_tdra_and_zaider_rossi_are_reasonably_close_for_small_beta(self) -> None:
        let_params = LETDependentParams(
            alpha_0=0.1,
            lambda_alpha=0.001,
            beta_0=0.0002,
            lambda_beta=0.0,
        )
        result = compute_mixed_field_sf(
            [
                FieldComponent(family="p", dose_fraction_gy=2.5, mean_let_kev_um=8.0),
                FieldComponent(family="n", dose_fraction_gy=1.5, mean_let_kev_um=30.0),
            ],
            let_params,
            method="tdra",
        )

        relative_difference = abs(result.sf_tdra - result.sf_zaider_rossi) / result.sf_zaider_rossi
        self.assertLess(relative_difference, 0.15)
        expected_tdra_sf = math.exp(
            -(result.effective_alpha * result.total_dose_gy + result.effective_beta * result.total_dose_gy ** 2)
        )
        self.assertAlmostEqual(result.sf_tdra, expected_tdra_sf, places=8)


if __name__ == "__main__":
    unittest.main()
