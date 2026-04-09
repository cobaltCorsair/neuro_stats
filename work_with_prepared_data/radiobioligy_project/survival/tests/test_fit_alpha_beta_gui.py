import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

try:
    from survival.fit_alpha_beta_gui import FitAlphaBetaWindow
    from survival.radiobiology_analysis import NTCPFitGroup
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_gui import (
        FitAlphaBetaWindow,
    )
    from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
        NTCPFitGroup,
    )


class FitAlphaBetaGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_ntcp_panel_uses_splitter_sections_without_overlap(self) -> None:
        window = FitAlphaBetaWindow()
        try:
            window.resize(1260, 980)
            window.show()
            self._activate_detail_tab(window, "NTCP")
            self.app.processEvents()

            controls_rect = window.ntcp_controls_group.geometry()
            splitter_rect = window.ntcp_content_splitter.geometry()

            self.assertLess(controls_rect.bottom(), splitter_rect.top())
            self.assertEqual(window.ntcp_content_splitter.count(), 4)
            self.assertGreaterEqual(window.ntcp_content_splitter.handleWidth(), 4)
            self.assertGreater(window.ntcp_text.maximumHeight(), 1000)
            self.assertEqual(window.ntcp_source_table.minimumHeight(), 0)
            self.assertEqual(window.ntcp_source_group.minimumHeight(), 0)
            self.assertTrue(window.ntcp_content_splitter.isCollapsible(0))
        finally:
            window.close()
            self.app.processEvents()

    def test_tcp_panel_uses_splitter_sections_without_overlap(self) -> None:
        window = FitAlphaBetaWindow()
        try:
            window.resize(1260, 980)
            window.show()
            self._activate_detail_tab(window, "TCP")
            self.app.processEvents()

            controls_rect = window.tcp_controls_group.geometry()
            splitter_rect = window.tcp_content_splitter.geometry()

            self.assertLess(controls_rect.bottom(), splitter_rect.top())
            self.assertEqual(window.tcp_content_splitter.count(), 3)
            self.assertGreaterEqual(window.tcp_content_splitter.handleWidth(), 4)
            self.assertGreater(window.tcp_text.maximumHeight(), 1000)
        finally:
            window.close()
            self.app.processEvents()

    def test_summary_panel_uses_smaller_minimum_height_and_shrinkable_selector(self) -> None:
        window = FitAlphaBetaWindow()
        try:
            self.assertEqual(window.summary_table.minimumHeight(), 120)
            self.assertEqual(window.results_splitter.handleWidth(), 4)
            self.assertLessEqual(window.run_selector.minimumWidth(), 220)
        finally:
            window.close()
            self.app.processEvents()

    def test_ntcp_plot_shows_curve_and_observed_rate_legend_entries(self) -> None:
        window = FitAlphaBetaWindow()
        try:
            window.ntcp_fit_groups = [
                NTCPFitGroup(
                    label="skin_reactions_y4_y4_y32_11.04.2025.xlsx",
                    dose_total=40.0,
                    n_subjects=5,
                    n_complications=4,
                    complication_rate=0.8,
                    peak_grade_mean=3.2,
                    threshold_grade=3,
                    path=Path("skin_reactions_y4_y4_y32_11.04.2025.xlsx"),
                ),
                NTCPFitGroup(
                    label="skin_reactions_y_40_19.03.2025.xlsx",
                    dose_total=50.0,
                    n_subjects=8,
                    n_complications=6,
                    complication_rate=0.75,
                    peak_grade_mean=3.375,
                    threshold_grade=3,
                    path=Path("skin_reactions_y_40_19.03.2025.xlsx"),
                ),
            ]

            window.refresh_ntcp_view()
            self.app.processEvents()

            axis = window.ntcp_figure.axes[0]
            legend = axis.get_legend()

            self.assertIsNotNone(legend)
            labels = [text.get_text() for text in legend.get_texts()]
            self.assertEqual(labels, ["NTCP", "Observed complication rate"])
        finally:
            window.close()
            self.app.processEvents()

    @staticmethod
    def _activate_detail_tab(window: FitAlphaBetaWindow, title: str) -> None:
        for index in range(window.detail_tabs.count()):
            if window.detail_tabs.tabText(index) == title:
                window.detail_tabs.setCurrentIndex(index)
                return
        raise AssertionError(f"{title} tab not found")


if __name__ == "__main__":
    unittest.main()
