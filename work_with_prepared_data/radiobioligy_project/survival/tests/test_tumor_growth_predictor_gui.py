import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QGridLayout

try:
    from survival.tumor_growth_predictor_gui import TumorGrowthPredictorWindow
    from survival.tumor_growth_predictor import GrowthSimulationResult, TreatmentFraction
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor_gui import (
        TumorGrowthPredictorWindow,
    )
    from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
        GrowthSimulationResult,
        TreatmentFraction,
    )


class TumorGrowthPredictorGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_model_parameters_use_single_field_column(self) -> None:
        window = TumorGrowthPredictorWindow()
        try:
            layout = window.parameter_group.layout()

            self.assertIsInstance(layout, QGridLayout)
            self.assertIsNotNone(layout.itemAtPosition(0, 1))
            self.assertIsNotNone(layout.itemAtPosition(9, 1))
            for row in range(10):
                self.assertIsNone(layout.itemAtPosition(row, 2))
                self.assertIsNone(layout.itemAtPosition(row, 3))
        finally:
            window.close()
            self.app.processEvents()

    def test_comparison_plot_collapses_identical_curves_into_one_legend_entry(self) -> None:
        window = TumorGrowthPredictorWindow()
        try:
            sample_times = np.asarray([0.0, 5.0, 10.0], dtype=float)
            values = np.asarray([1.0, 0.8, 0.6], dtype=float)
            window.comparison_curves = {
                "Current": values,
                "Alternative": values.copy(),
            }

            window.refresh_comparison_plot(sample_times)
            self.app.processEvents()

            axis = window.comparison_figure.axes[0]
            legend = axis.get_legend()

            self.assertIsNotNone(legend)
            labels = [text.get_text() for text in legend.get_texts()]
            self.assertEqual(labels, ["Current = Alternative"])
        finally:
            window.close()
            self.app.processEvents()

    def test_schedules_equivalent_detects_identical_rows(self) -> None:
        first = [
            TreatmentFraction(day=0.0, dose=4.0, family="y"),
            TreatmentFraction(day=1.0 / 24.0, dose=4.0, family="y"),
            TreatmentFraction(day=2.0 / 24.0, dose=32.0, family="y"),
        ]
        second = [
            TreatmentFraction(day=0.0, dose=4.0, family="y"),
            TreatmentFraction(day=1.0 / 24.0, dose=4.0, family="y"),
            TreatmentFraction(day=2.0 / 24.0, dose=32.0, family="y"),
        ]
        third = [
            TreatmentFraction(day=0.0, dose=4.0, family="y"),
            TreatmentFraction(day=1.0 / 24.0, dose=8.0, family="y"),
            TreatmentFraction(day=2.0 / 24.0, dose=28.0, family="y"),
        ]

        self.assertTrue(TumorGrowthPredictorWindow.schedules_equivalent(first, second))
        self.assertFalse(TumorGrowthPredictorWindow.schedules_equivalent(first, third))

    def test_geometry_consistency_report_lists_each_time_point(self) -> None:
        window = TumorGrowthPredictorWindow()
        try:
            result = GrowthSimulationResult(
                times=np.asarray([0.0, 2.0], dtype=float),
                live_volume=np.asarray([1.0, 1.5], dtype=float),
                dead_volume=np.asarray([0.0, 0.1], dtype=float),
                total_volume=np.asarray([1.0, 1.6], dtype=float),
                axis_a=np.asarray([1.0, 1.2], dtype=float),
                axis_b=np.asarray([1.0, 1.1], dtype=float),
                axis_c=np.asarray([1.0, 1.05], dtype=float),
            )

            lines = window.build_geometry_consistency_report_lines(result)

            self.assertIn("Per-time check:", lines)
            self.assertEqual(len([line for line in lines if line.startswith("t=")]), 2)
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
