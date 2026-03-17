import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from work_with_prepared_data.radiobioligy_project.data_processing.tumor_geometry_processor import (
    ellipsoid_volume_from_diameters,
    equivalent_sphere_diameter,
    process_tumor_geometry_excel,
)


class TumorGeometryProcessorTests(unittest.TestCase):
    def test_time_labels_accept_numeric_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "geometry.xlsx"
            df = pd.DataFrame(
                [
                    ["Experiment", "y 40 Gy", None],
                    [None, 0.0, 1.5],
                    ["rat-1", "2-4-6", "8"],
                ]
            )
            df.to_excel(path, header=False, index=False)

            dataset = process_tumor_geometry_excel(path)

        self.assertEqual(dataset.time_data, ("0", "1.5"))

    def test_process_tumor_geometry_excel_preserves_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "geometry.xlsx"
            df = pd.DataFrame(
                [
                    ["Experiment", "y 40 Gy", "t = 1 ч"],
                    [None, "0", "1"],
                    ["rat-1", "2-4-6", "8"],
                    ["rat-2", "3-3-3", "1-1-1"],
                ]
            )
            df.to_excel(path, header=False, index=False)

            dataset = process_tumor_geometry_excel(path)

        self.assertEqual(dataset.rat_labels, ("rat-1", "rat-2"))
        self.assertEqual(dataset.time_data, ("0", "1"))
        self.assertAlmostEqual(dataset.axis_a[0, 0], 2.0, places=6)
        self.assertAlmostEqual(dataset.axis_b[0, 0], 4.0, places=6)
        self.assertAlmostEqual(dataset.axis_c[0, 0], 6.0, places=6)
        self.assertAlmostEqual(
            dataset.volumes[0, 0],
            ellipsoid_volume_from_diameters(2.0, 4.0, 6.0),
            places=6,
        )
        self.assertTrue(dataset.explicit_axes_mask[0, 0])

        expected_sphere = equivalent_sphere_diameter(8.0)
        self.assertAlmostEqual(dataset.axis_a[0, 1], expected_sphere, places=6)
        self.assertAlmostEqual(dataset.axis_b[0, 1], expected_sphere, places=6)
        self.assertAlmostEqual(dataset.axis_c[0, 1], expected_sphere, places=6)
        self.assertFalse(dataset.explicit_axes_mask[0, 1])
        self.assertIn("Irradiation Time=t = 1 ч", dataset.experiment_params)

    def test_mean_geometry_returns_per_day_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "geometry.xlsx"
            df = pd.DataFrame(
                [
                    ["Experiment", "y 40 Gy", None],
                    [None, "0", "1"],
                    ["rat-1", "2-2-2", "4-4-4"],
                    ["rat-2", "6-6-6", "8-8-8"],
                ]
            )
            df.to_excel(path, header=False, index=False)
            dataset = process_tumor_geometry_excel(path)

        mean_a, mean_b, mean_c, mean_volume = dataset.mean_geometry()

        self.assertEqual(mean_a.shape, (2,))
        self.assertTrue(np.allclose(mean_a, [4.0, 6.0]))
        self.assertTrue(np.allclose(mean_b, [4.0, 6.0]))
        self.assertTrue(np.allclose(mean_c, [4.0, 6.0]))
        self.assertEqual(mean_volume.shape, (2,))


if __name__ == "__main__":
    unittest.main()
