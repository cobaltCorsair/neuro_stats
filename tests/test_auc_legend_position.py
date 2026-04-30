import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADIOBIOLOGY_ROOT = PROJECT_ROOT / "work_with_prepared_data" / "radiobioligy_project"
if str(RADIOBIOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(RADIOBIOLOGY_ROOT))

from draw_base_graphs import TumorDataVisualizer
from work_with_prepared_data.radiobioligy_project.gui import graph_manager
from work_with_prepared_data.radiobioligy_project.gui.legend_window import LegendManager


class AucLegendPositionTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")
        graph_manager.update_legend_position("best")

    def test_tumor_auc_comparison_uses_selected_legend_position(self) -> None:
        data_root = PROJECT_ROOT / "work_with_prepared_data" / "datas" / "control"
        file_paths = [
            str(data_root / "16.03.2023_n_22.xlsx"),
            str(data_root / "02.02.2023_n_12.xlsx"),
            str(data_root / "02.02.2023_n_18.xlsx"),
        ]

        graph_manager.update_legend_position("center left")
        TumorDataVisualizer.plot_auc_comparison(file_paths)

        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(6, legend._loc)

    def test_separate_legend_keeps_shapiro_off_plot_but_extractable(self) -> None:
        data_root = PROJECT_ROOT / "work_with_prepared_data" / "datas" / "control"
        file_paths = [
            str(data_root / "16.03.2023_n_22.xlsx"),
            str(data_root / "02.02.2023_n_12.xlsx"),
            str(data_root / "02.02.2023_n_18.xlsx"),
        ]

        TumorDataVisualizer.plot_auc_comparison(
            file_paths,
            show_separate_legend=True,
            use_shapiro=True,
        )

        legends = []
        for artist in plt.gca().artists:
            if type(artist).__name__ == "Legend":
                legends.append(artist)
        current_legend = plt.gca().get_legend()
        if current_legend is not None and current_legend not in legends:
            legends.append(current_legend)

        self.assertTrue(legends)
        self.assertFalse(any(legend.get_visible() for legend in legends))

        labels = [
            text.get_text()
            for legend in legends
            for text in legend.get_texts()
        ]
        self.assertTrue(any("Дата:" in label for label in labels))
        self.assertTrue(any("кр. Шапиро-Уилка" in label for label in labels))

        legend_data = LegendManager().extract_legend_from_figure(plt.gcf())
        extracted_labels = [item["label"] for item in legend_data]
        self.assertTrue(any("Дата:" in label for label in extracted_labels))
        self.assertTrue(any("кр. Шапиро-Уилка" in label for label in extracted_labels))


if __name__ == "__main__":
    unittest.main()
