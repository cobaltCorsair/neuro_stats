import io
import os
import unittest
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib.pyplot as plt

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


class MatplotlibConfiguratorTests(unittest.TestCase):
    def test_custom_font_fallback_renders_dose_subscripts(self) -> None:
        configurator = MatplotlibConfigurator()
        configurator.apply_custom_styles()

        try:
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.plot([0, 1], [0, 1], label="Dₙ = 22 Гр, Dₚ = 36 Гр")
            ax.legend()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig.canvas.draw()
                fig.savefig(io.BytesIO(), format="png")

            missing_subscript_warnings = [
                warning
                for warning in caught
                if "LATIN SUBSCRIPT SMALL LETTER" in str(warning.message)
            ]
            self.assertEqual([], missing_subscript_warnings)
        finally:
            plt.close("all")
            configurator.restore_original_styles()


if __name__ == "__main__":
    unittest.main()
