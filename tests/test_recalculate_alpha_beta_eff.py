from pathlib import Path
import csv
import unittest

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.recalculate_alpha_beta_eff import (
    ExperimentRecord,
    analysis_radiation_family,
    fit_model,
    normalized_curve_and_sf,
    parse_dose_entries,
)

# Path to the engine output produced by run_4_2_final.py
_EFF_CSV = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.2_Подгонка_alpha_beta\alpha_beta_eff_results\experiment_sf_eff.csv"
)


def _read_sf_eff_csv():
    """Read experiment_sf_eff.csv; return list of dicts.  Returns [] if file missing."""
    if not _EFF_CSV.exists():
        return []
    with open(_EFF_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


class RecalculateAlphaBetaEffTests(unittest.TestCase):
    def test_protons_without_explicit_peak_are_classified_as_through_beam(self) -> None:
        self.assertEqual(
            analysis_radiation_family(Path("04.05.2017_p32.xlsx")),
            "p_through",
        )
        self.assertEqual(
            analysis_radiation_family(Path("22.10.2025_p32_in_peak.xlsx")),
            "p_peak",
        )

    def test_carbon_isotope_mass_is_not_parsed_as_a_fraction(self) -> None:
        entries = parse_dose_entries(["C12 = 34 Гр.", "Date=10.12.2014"])

        self.assertEqual(entries, (("c12", 34.0),))

    def test_text_after_dose_unit_is_not_parsed_as_dose(self) -> None:
        entries = parse_dose_entries(["e = 47 Гр, тубус 6 см"])

        self.assertEqual(entries, (("e", 47.0),))

    def test_sf_eff_uses_baseline_normalized_control_ratio_and_time_alignment(self) -> None:
        sf_eff, minimum_day, point_count = normalized_curve_and_sf(
            np.asarray([0.0, 2.0, 4.0]),
            np.asarray([100.0, 150.0, 180.0]),
            np.asarray([0.0, 1.0, 3.0, 5.0]),
            np.asarray([100.0, 150.0, 250.0, 350.0]),
        )

        # Interpolated control is 200 at day 2 and 300 at day 4.
        self.assertTrue(np.isclose(sf_eff, 0.6))
        self.assertEqual(minimum_day, 4.0)
        self.assertEqual(point_count, 3)

    def test_weighted_fit_respects_nonnegative_parameter_bounds(self) -> None:
        records = []
        for dose, sf in ((10.0, 0.85), (20.0, 0.72), (30.0, 0.64), (40.0, 0.55)):
            records.append(
                ExperimentRecord(
                    path=Path(f"dose_{dose:g}.xlsx"),
                    relative_path=f"dose_{dose:g}.xlsx",
                    fractions=(dose,),
                    sf_eff=sf,
                    sf_se=0.03,
                )
            )

        fit = fit_model(records, "lq", sigma_log_floor=0.05)

        self.assertGreaterEqual(fit.alpha, 0.0)
        self.assertGreaterEqual(fit.beta, 0.0)
        self.assertTrue(np.isfinite(fit.aicc))


    # ------------------------------------------------------------------
    # Integration tests — require engine output CSV to be present
    # ------------------------------------------------------------------

    @unittest.skipUnless(_EFF_CSV.exists(), "engine output CSV not found — run run_4_2_final.py first")
    def test_y32_duplicate_resolved_correctly(self) -> None:
        """16.05.2018_y32 must be included; 03.05.2018_y32 must be excluded."""
        rows = _read_sf_eff_csv()
        included  = {r["relative_path"]: r["included_analysis"] for r in rows}

        kept    = next((v for k, v in included.items() if "16.05.2018_y32" in k), None)
        dropped = next((v for k, v in included.items() if "03.05.2018_y32" in k), None)

        self.assertIsNotNone(kept,    "16.05.2018_y32 not in experiment_sf_eff.csv")
        self.assertIsNotNone(dropped, "03.05.2018_y32 not in experiment_sf_eff.csv")
        self.assertEqual(kept,    "1", "16.05.2018_y32 should be included (included_analysis=1)")
        self.assertEqual(dropped, "0", "03.05.2018_y32 should be excluded (included_analysis=0)")

    @unittest.skipUnless(_EFF_CSV.exists(), "engine output CSV not found — run run_4_2_final.py first")
    def test_control_2019_assigned_to_2019_experiments(self) -> None:
        """Experiments from 2019 must use control_15.10.2019.xlsx, not the 2016 default."""
        rows = _read_sf_eff_csv()
        year_2019 = [r for r in rows if r.get("date", "").startswith("2019-")]
        self.assertTrue(year_2019, "No 2019 experiments found in experiment_sf_eff.csv")
        for r in year_2019:
            ctrl = r.get("control_file", "")
            self.assertIn(
                "15.10.2019",
                ctrl,
                f"Expected control_15.10.2019 for {r['relative_path']}, got '{ctrl}'",
            )

    # ------------------------------------------------------------------
    # New unit tests
    # ------------------------------------------------------------------

    def test_carbon_family_classified_correctly(self) -> None:
        # Family name is "c" (isotope number stripped); family code used in model_fits.csv
        self.assertEqual(analysis_radiation_family(Path("14.12.2015_c12_34.xlsx")), "c")

    def test_electron_family_classified_correctly(self) -> None:
        self.assertEqual(analysis_radiation_family(Path("15.10.2019_e32.xlsx")), "e")

    def test_multiple_single_dose_levels_respect_nonnegative_lq_bounds(self) -> None:
        """Several single-dose levels can identify curvature, subject to dose contrast."""
        records = []
        for dose, sf in ((10.0, 0.90), (20.0, 0.80), (30.0, 0.70)):
            records.append(
                ExperimentRecord(
                    path=Path(f"mono_{dose:g}.xlsx"),
                    relative_path=f"mono_{dose:g}.xlsx",
                    fractions=(dose,),
                    sf_eff=sf,
                    sf_se=0.02,
                )
            )
        fit_lq = fit_model(records, "lq", sigma_log_floor=0.05)
        # D and D² are distinct columns across multiple dose levels. This test only
        # verifies the constrained fit; identifiability depends on the actual dose
        # contrast and is not ruled out merely because every regimen is one fraction.
        self.assertGreaterEqual(fit_lq.beta, 0.0)
        self.assertGreaterEqual(fit_lq.alpha, 0.0)


if __name__ == "__main__":
    unittest.main()
