import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from work_with_prepared_data.radiobioligy_project.tumor_3d_viewer import (  # noqa: E402
    format_interpolated_day_label,
    interpolate_scalar,
)


class Tumor3DViewerHelpersTests(unittest.TestCase):
    def test_interpolate_scalar_returns_midpoint(self) -> None:
        self.assertAlmostEqual(interpolate_scalar(2.0, 6.0, 0.25), 3.0, places=6)

    def test_format_interpolated_day_label_numeric(self) -> None:
        self.assertEqual(format_interpolated_day_label("2", "5", 0.5), "3.5")

    def test_format_interpolated_day_label_text(self) -> None:
        self.assertEqual(
            format_interpolated_day_label("baseline", "followup", 0.25),
            "baseline -> followup (25%)",
        )


if __name__ == "__main__":
    unittest.main()
