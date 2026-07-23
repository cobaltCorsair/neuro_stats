import math
import unittest

import numpy as np

from work_with_prepared_data.radiobioligy_project.tomography.task2_tomography import (
    concordance_correlation,
    parse_date_cell,
    parse_lwh,
    shoelace_area_xy,
)


class Task2TomographyTests(unittest.TestCase):
    def test_parse_lwh_accepts_decimal_commas(self) -> None:
        self.assertEqual(parse_lwh("1,2 - 2,3 - 3,4"), (1.2, 2.3, 3.4))

    def test_parse_lwh_converts_mm(self) -> None:
        self.assertEqual(parse_lwh("12-23-34", unit="mm"), (1.2, 2.3, 3.4))

    def test_parse_date_from_compound_label(self) -> None:
        self.assertEqual(parse_date_cell("9 сут - 08.08").isoformat(), "2025-08-08")

    def test_concordance_is_one_for_identical_values(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        self.assertAlmostEqual(concordance_correlation(values, values), 1.0)

    def test_concordance_penalizes_constant_bias(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0, 8.0])
        self.assertLess(concordance_correlation(values, values + 2.0), 1.0)

    def test_shoelace_area(self) -> None:
        square = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
        )
        self.assertTrue(math.isclose(shoelace_area_xy(square), 4.0))


if __name__ == "__main__":
    unittest.main()
