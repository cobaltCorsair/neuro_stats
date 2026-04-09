import math
import unittest

import numpy as np

try:
    from survival.tumor_growth_predictor import (
        GeometryReference,
        geometry_volume_consistency,
        scale_axes_from_volume,
    )
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
        GeometryReference,
        geometry_volume_consistency,
        scale_axes_from_volume,
    )


class TumorGrowthPredictorCoreTests(unittest.TestCase):
    def test_fixed_geometry_scaling_reconstructs_exact_total_volume(self) -> None:
        reference = GeometryReference(
            axis_a=2.0,
            axis_b=3.0,
            axis_c=4.0,
            volume=(math.pi * 2.0 * 3.0 * 4.0) / 6.0,
        )
        total_volume = np.asarray(
            [reference.volume, reference.volume * 2.5, reference.volume * 6.0],
            dtype=float,
        )

        axis_a, axis_b, axis_c = scale_axes_from_volume(total_volume, reference)
        ellipsoid, delta, relative = geometry_volume_consistency(
            total_volume,
            axis_a,
            axis_b,
            axis_c,
        )

        np.testing.assert_allclose(ellipsoid, total_volume, rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(delta, np.zeros_like(total_volume), rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(relative, np.zeros_like(total_volume), rtol=1.0e-10, atol=1.0e-10)

    def test_geometry_volume_consistency_detects_mismatch(self) -> None:
        total_volume = np.asarray([1.0, 8.0], dtype=float)
        axis_a = np.asarray([1.0, 2.0], dtype=float)
        axis_b = np.asarray([1.0, 2.0], dtype=float)
        axis_c = np.asarray([1.0, 2.0], dtype=float)

        ellipsoid, delta, relative = geometry_volume_consistency(
            total_volume,
            axis_a,
            axis_b,
            axis_c,
        )

        self.assertAlmostEqual(float(ellipsoid[0]), math.pi / 6.0, places=10)
        self.assertGreater(abs(float(delta[0])), 0.4)
        self.assertGreater(abs(float(relative[0])), 0.4)


if __name__ == "__main__":
    unittest.main()
