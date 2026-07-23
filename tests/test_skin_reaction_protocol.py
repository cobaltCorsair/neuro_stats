from pathlib import Path
import ast
import unittest

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.skin_reaction_protocol import (
    DAY_CORRECTIONS,
    PEAK_PROTON_RBE_2026,
    apply_day_correction,
    auc_without_extrapolation,
    build_experiment_metadata,
    convert_scores_to_rtog,
    duration_above_threshold,
    holm_adjust,
    parse_recorded_components,
    peak_with_day,
    remove_patent_time_component,
    time_to_sustained_normalisation,
)


class SkinReactionProtocolTests(unittest.TestCase):
    def test_parser_reads_all_components_from_one_header_cell(self) -> None:
        components = parse_recorded_components(
            ["p = 2.36 Гр / y = 18.8 Гр / p = 2.36 Гр / y = 18.8 Гр (in peak)"]
        )
        self.assertEqual(components, (("p", 2.36), ("y", 18.8), ("p", 2.36), ("y", 18.8)))

    def test_2026_peak_proton_recorded_dose_is_converted_to_physical(self) -> None:
        metadata = build_experiment_metadata(
            ["p = 38 Гр", "Date=25.5.2026"],
            Path("25.05.2026_p38_in_peak_skin_reactions.xlsx"),
        )
        self.assertEqual(metadata.family, "p_peak")
        self.assertAlmostEqual(metadata.recorded_total_dose, 38.0)
        self.assertAlmostEqual(metadata.physical_total_dose, 38.0 / PEAK_PROTON_RBE_2026)
        self.assertNotEqual(metadata.dose_basis, "physical")

    def test_carbon_geometry_uses_task_4_2_audit(self) -> None:
        metadata = build_experiment_metadata(
            ["c12 = 25 Гр", "Date=11.4.2016"],
            Path("2_11.04.2016_с_12_skin_reactions.xlsx"),
        )
        self.assertEqual(metadata.family, "c12_through")

    def test_converter_is_the_program_converter(self) -> None:
        converted = convert_scores_to_rtog([0, 1, 200, 201, 600, 601, np.nan])
        np.testing.assert_equal(converted[:6], np.asarray([0, 1, 1, 2, 3, 4], dtype=float))
        self.assertTrue(np.isnan(converted[-1]))

    def test_day_correction_matches_program_table_and_leaves_day_zero(self) -> None:
        corrected = apply_day_correction([0, 2, 18], [50, 200, 300])
        np.testing.assert_allclose(corrected, [50, 200 - DAY_CORRECTIONS[2], 300 - DAY_CORRECTIONS[18]])

    def test_day_correction_table_matches_existing_analyzer(self) -> None:
        source_path = Path(
            "work_with_prepared_data/radiobioligy_project/stats_methods/from_our_scale_to_rtog.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        table = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "day_corrections" for target in node.targets
            ):
                table = ast.literal_eval(node.value)
                break
        self.assertEqual(DAY_CORRECTIONS, table)

    def test_exact_patent_time_component_does_not_inherit_day_17_typo(self) -> None:
        corrected = remove_patent_time_component([17, 18], [500, 500])
        np.testing.assert_allclose(corrected, [500 - 57.5, 500 - 55.0])
        self.assertNotEqual(corrected[0], 500 - DAY_CORRECTIONS[17])

    def test_patent_peak_returns_first_day_of_plateau(self) -> None:
        peak, day = peak_with_day([2, 5, 7, 9], [100, 300, 300, 200])
        self.assertEqual(peak, 300)
        self.assertEqual(day, 5)

    def test_duration_above_threshold_interpolates_both_crossings(self) -> None:
        duration, censored = duration_above_threshold(
            [0, 10, 20], [0, 500, 0], 250, lower=0, upper=20
        )
        self.assertAlmostEqual(duration, 10.0)
        self.assertFalse(censored)

    def test_normalisation_uses_last_downward_crossing(self) -> None:
        day, censored = time_to_sustained_normalisation(
            [0, 5, 10, 15], [0, 200, 50, 50], threshold=100
        )
        self.assertAlmostEqual(day, 5 + (100 - 200) / (50 - 200) * 5)
        self.assertFalse(censored)

    def test_auc_never_extrapolates_outside_observed_support(self) -> None:
        auc, normalised, duration, first, last = auc_without_extrapolation(
            [3, 10, 25], [1, 2, 3], lower=0, upper=24
        )
        self.assertEqual(first, 3)
        self.assertEqual(last, 24)
        self.assertEqual(duration, 21)
        self.assertAlmostEqual(normalised, auc / duration)

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03, np.nan])
        np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.06])
        self.assertTrue(np.isnan(adjusted[3]))


if __name__ == "__main__":
    unittest.main()
