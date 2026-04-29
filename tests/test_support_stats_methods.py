import math
import unittest

from work_with_prepared_data.radiobioligy_project.stats_methods.support_stats_methods import (
    SupportingFunctions,
)


class SupportingFunctionsTests(unittest.TestCase):
    def test_to_float_list_returns_converted_values(self) -> None:
        values = SupportingFunctions.to_float_list(["1", "2,5", "", None, "nan", "bad"])

        self.assertEqual(values[:2], [1.0, 2.5])
        self.assertTrue(math.isnan(values[2]))
        self.assertTrue(math.isnan(values[3]))
        self.assertTrue(math.isnan(values[4]))
        self.assertTrue(math.isnan(values[5]))


if __name__ == "__main__":
    unittest.main()
