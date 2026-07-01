import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Длинный путь — тот же самый, что используют plotting_helpers.py/draw_abs_rel_graph_compare.py/
# skin_reactions_base_grapf.py внутри себя; короткий "from gui import graph_manager" дал бы
# ВТОРОЙ независимый экземпляр модуля с отдельным глобальным состоянием.
from work_with_prepared_data.radiobioligy_project.gui import graph_manager
from utils.plotting_helpers import format_experiment_params
from draw_abs_rel_graph_compare import TumorDataComparatorAdvanced
from skin_reactions_base_grapf import SkinReactionsVisualizer


class TestFormatExperimentParamsDateToggle(unittest.TestCase):
    """Показ "Дата: ..." в подписях экспериментов — отдельная опциональная функция
    (checkBox_show_date в интерфейсе), а не всегда включённое поведение."""

    def tearDown(self):
        graph_manager.set_show_date_in_legend(True)

    def test_date_shown_by_default(self):
        label = format_experiment_params(["p=23 Гр", "Date=2.12.2025"])
        self.assertIn("Дата: 2.12.2025", label)

    def test_date_hidden_when_disabled(self):
        graph_manager.set_show_date_in_legend(False)
        label = format_experiment_params(["p=23 Гр", "Date=2.12.2025"])
        self.assertNotIn("Дата", label)
        self.assertNotIn("2.12.2025", label)

    def test_other_parts_unaffected_by_date_toggle(self):
        graph_manager.set_show_date_in_legend(False)
        label = format_experiment_params(["p=23 Гр", "Irradiation Time=6 ч", "Date=2.12.2025"])
        self.assertIn("Интервал: 6 ч", label)
        self.assertNotIn("Дата", label)


class TestBuildSignificanceTestLegendLabel(unittest.TestCase):
    def tearDown(self):
        graph_manager.set_holm_correction_enabled(True)

    def test_label_mentions_test_name_and_holm_enabled(self):
        graph_manager.set_holm_correction_enabled(True)
        lines = graph_manager.build_significance_test_legend_label("Манна-Уитни")
        joined = " ".join(lines)
        self.assertIn("Манна-Уитни", joined)
        self.assertIn("с поправкой Холма", joined)
        # При включённой поправке Холма на графике встречаются оба маркера ('*' и '(*)'),
        # значит легенда обязана объяснять оба, а не только основной.
        self.assertTrue(any(line.startswith("*") for line in lines))
        self.assertTrue(any(line.startswith("(*)") for line in lines))

    def test_label_mentions_no_holm_when_disabled(self):
        graph_manager.set_holm_correction_enabled(False)
        lines = graph_manager.build_significance_test_legend_label("Стьюдента")
        joined = " ".join(lines)
        self.assertIn("Стьюдента", joined)
        self.assertIn("без поправки Холма", joined)
        # Без поправки Холма маркер '(*)' не используется вообще (только '*'),
        # поэтому его объяснение в легенде не нужно.
        self.assertFalse(any(line.startswith("(*)") for line in lines))


class TestTumorVolumeSignificanceLegend(unittest.TestCase):
    """_add_significance_test_legend_if_active — отдельная легенда НА графике (через
    GraphVisualizer.add_legend), а не подпись под всей фигурой."""

    def tearDown(self):
        graph_manager.set_holm_correction_enabled(True)

    def test_no_legend_when_no_test_active(self):
        comparator = TumorDataComparatorAdvanced()
        mock_drawgraph = MagicMock()
        comparator._add_significance_test_legend_if_active(mock_drawgraph)
        mock_drawgraph.add_legend.assert_not_called()

    def test_legend_added_for_mann_whitney(self):
        comparator = TumorDataComparatorAdvanced()
        comparator.perform_stat_test = True
        mock_drawgraph = MagicMock()
        comparator._add_significance_test_legend_if_active(mock_drawgraph)
        mock_drawgraph.add_legend.assert_called_once()
        label = mock_drawgraph.add_legend.call_args[0][0]
        self.assertIn("Манна-Уитни", " ".join(label))
        kwargs = mock_drawgraph.add_legend.call_args.kwargs
        self.assertEqual(kwargs.get("loc"), "lower right")
        self.assertFalse(kwargs.get("display_marker"))

    def test_legend_added_for_student(self):
        comparator = TumorDataComparatorAdvanced()
        comparator.use_ttest = True
        mock_drawgraph = MagicMock()
        comparator._add_significance_test_legend_if_active(mock_drawgraph)
        label = mock_drawgraph.add_legend.call_args[0][0]
        self.assertIn("Стьюдента", " ".join(label))


class _StubSkinDataProcessor:
    def get_mean_skin_reactions(self):
        return None, None, None


def _make_skin_visualizer(name, time_data, skin_reactions):
    vis = SkinReactionsVisualizer.__new__(SkinReactionsVisualizer)
    vis.file_path = name
    vis.experiment_params = [name]
    vis.time_data = time_data
    vis.skin_reactions = skin_reactions
    vis.data_processor = _StubSkinDataProcessor()
    return vis


class TestSkinReactionsSignificanceLegendOnChart(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
        graph_manager.set_holm_correction_enabled(True)

    @patch("skin_reactions_base_grapf.format_experiment_params", side_effect=lambda params: params[0])
    def test_legend_appears_on_axes_when_test_active(self, _format_params):
        vis_a = _make_skin_visualizer("expA", [0, 1, 2], [[0, 50, 100], [0, 60, 110]])
        vis_b = _make_skin_visualizer("expB", [0, 1, 2], [[0, 40, 90], [0, 45, 95]])

        SkinReactionsVisualizer.plot_multiple_experiments_from_visualizers(
            [vis_a, vis_b], use_AUC=False, apply_statistical_test=True
        )

        ax = plt.gcf().get_axes()[0]
        legends = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
        titles = [leg.get_title().get_text() for leg in legends]
        self.assertIn("Критерий значимости", titles)

        test_legend = next(leg for leg in legends if leg.get_title().get_text() == "Критерий значимости")
        legend_text = " ".join(t.get_text() for t in test_legend.get_texts())
        self.assertIn("Манна-Уитни", legend_text)

    @patch("skin_reactions_base_grapf.format_experiment_params", side_effect=lambda params: params[0])
    def test_no_significance_legend_when_test_not_requested(self, _format_params):
        vis_a = _make_skin_visualizer("expA", [0, 1, 2], [[0, 50, 100], [0, 60, 110]])
        vis_b = _make_skin_visualizer("expB", [0, 1, 2], [[0, 40, 90], [0, 45, 95]])

        SkinReactionsVisualizer.plot_multiple_experiments_from_visualizers(
            [vis_a, vis_b], use_AUC=False, apply_statistical_test=False
        )

        ax = plt.gcf().get_axes()[0]
        legends = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
        titles = [leg.get_title().get_text() for leg in legends]
        self.assertNotIn("Критерий значимости", titles)


if __name__ == "__main__":
    unittest.main()
