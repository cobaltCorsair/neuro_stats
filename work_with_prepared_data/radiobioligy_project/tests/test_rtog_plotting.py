import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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

from skin_reactions_base_grapf import SkinReactionsVisualizer, RTOG_YLIM


class _StubSkinDataProcessor:
    def __init__(self, mean_reactions):
        self._mean = mean_reactions

    def get_mean_skin_reactions(self):
        n = len(self._mean)
        return self._mean, [0.0] * n, [0.0] * n


def _make_visualizer(time_data, skin_reactions, mean_reactions):
    vis = SkinReactionsVisualizer.__new__(SkinReactionsVisualizer)
    vis.file_path = "stub.xlsx"
    vis.experiment_params = ["stub"]
    vis.time_data = time_data
    vis.skin_reactions = skin_reactions
    vis.data_processor = _StubSkinDataProcessor(mean_reactions)
    return vis


class TestPlotMeanSkinReactionsRtog(unittest.TestCase):
    """show_rtog ЗАМЕНЯЕТ кривую сырых баллов на степень RTOG (а не накладывает поверх
    через вторую ось) — пользователь явно попросил не плодить двойную ось, это путает."""

    def tearDown(self):
        plt.close('all')

    def test_show_rtog_false_plots_raw_mean_reactions(self):
        vis = _make_visualizer([0, 1, 2], [[10, 20, 30]], [10, 20, 30])
        with patch("skin_reactions_base_grapf.SupportingFunctions.aggregate_rtog_grades") as mock_agg:
            vis.plot_mean_skin_reactions(show_rtog=False)
        mock_agg.assert_not_called()

        axes = plt.gcf().get_axes()
        self.assertEqual(len(axes), 1)
        line = axes[0].get_lines()[0]
        self.assertEqual(list(line.get_ydata()), [10, 20, 30])

    def test_show_rtog_true_plots_rtog_grades_not_raw_on_single_axis(self):
        vis = _make_visualizer([0, 1, 2], [[10, 20, 30]], [10, 20, 30])
        with patch("skin_reactions_base_grapf.SupportingFunctions.aggregate_rtog_grades",
                   return_value=[0, 1, 2]) as mock_agg:
            vis.plot_mean_skin_reactions(show_rtog=True)
        mock_agg.assert_called_once_with(vis.skin_reactions)

        axes = plt.gcf().get_axes()
        self.assertEqual(len(axes), 1, "show_rtog не должен создавать вторую ось — он заменяет кривую")
        # Небольшой отступ от (0,4), а не точная граница — иначе засечки доверительного
        # интервала, лежащие ровно на 0 или 4 (частый случай для дискретной шкалы RTOG),
        # визуально обрезаются рамкой осей matplotlib (см. RTOG_YLIM).
        self.assertEqual(axes[0].get_ylim(), RTOG_YLIM)
        line = axes[0].get_lines()[0]
        self.assertEqual(list(line.get_ydata()), [0, 1, 2])

    def test_ylim_has_padding_beyond_data_range_to_avoid_frame_clipping(self):
        """
        Регрессия: ylim, зафиксированный РОВНО на (0,4), приводит к тому, что засечки
        доверительного интервала (custom_fill_between), лежащие точно на границе диапазона
        (а степень 4 — частый случай, это максимум шкалы RTOG), визуально сливаются с рамкой
        осей matplotlib и выглядят "обрезанными". Должен быть зазор по обе стороны.
        """
        vis = _make_visualizer([0, 1, 2], [[10, 20, 30]], [10, 20, 30])
        with patch("skin_reactions_base_grapf.SupportingFunctions.aggregate_rtog_grades",
                   return_value=[0, 1, 4]):
            vis.plot_mean_skin_reactions(show_rtog=True)

        bottom, top = plt.gcf().get_axes()[0].get_ylim()
        self.assertLess(bottom, 0.0)
        self.assertGreater(top, 4.0)


class TestPlotMultipleExperimentsRtog(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    @patch("skin_reactions_base_grapf.format_experiment_params", side_effect=lambda params: params[0])
    def test_show_rtog_true_plots_rtog_grades_for_each_experiment_on_single_axis(self, _format_params):
        vis_a = _make_visualizer([0, 1, 2], [[0, 50, 100], [0, 60, 110]], None)
        vis_a.experiment_params = ["expA"]
        vis_b = _make_visualizer([0, 1, 2], [[0, 40, 90], [0, 45, 95]], None)
        vis_b.experiment_params = ["expB"]

        with patch("skin_reactions_base_grapf.SupportingFunctions.aggregate_rtog_grades",
                   side_effect=[[0, 1, 2], [0, 1, 3]]) as mock_agg:
            SkinReactionsVisualizer.plot_multiple_experiments_from_visualizers(
                [vis_a, vis_b], use_AUC=False, apply_statistical_test=False, show_rtog=True
            )

        self.assertEqual(mock_agg.call_count, 2)

        axes = plt.gcf().get_axes()
        self.assertEqual(len(axes), 1, "show_rtog не должен создавать вторую ось — он заменяет кривые")
        # Небольшой отступ от (0,4), а не точная граница — иначе засечки доверительного
        # интервала, лежащие ровно на 0 или 4 (частый случай для дискретной шкалы RTOG),
        # визуально обрезаются рамкой осей matplotlib (см. RTOG_YLIM).
        self.assertEqual(axes[0].get_ylim(), RTOG_YLIM)

        # Основные кривые опознаём по подписи (label) — custom_fill_between (IQR-полоса)
        # добавляет дополнительные Line2D без label на каждую временную точку.
        main_lines = [line for line in axes[0].get_lines() if line.get_label() in ("expA", "expB")]
        self.assertEqual(len(main_lines), 2)
        plotted_ydata = sorted(tuple(line.get_ydata()) for line in main_lines)
        self.assertEqual(plotted_ydata, [(0, 1, 2), (0, 1, 3)])

        # IQR-полоса добавляет дополнительные линии (засечки доверительного интервала)
        self.assertGreater(len(axes[0].get_lines()), len(main_lines))

    @patch("skin_reactions_base_grapf.format_experiment_params", side_effect=lambda params: params[0])
    def test_show_rtog_false_plots_raw_mean_reactions_not_clipped(self, _format_params):
        vis_a = _make_visualizer([0, 1, 2], [[0, 50, 100], [0, 60, 110]], None)
        vis_a.experiment_params = ["expA"]
        vis_b = _make_visualizer([0, 1, 2], [[0, 40, 90], [0, 45, 95]], None)
        vis_b.experiment_params = ["expB"]

        SkinReactionsVisualizer.plot_multiple_experiments_from_visualizers(
            [vis_a, vis_b], use_AUC=False, apply_statistical_test=False, show_rtog=False
        )

        axes = plt.gcf().get_axes()
        self.assertEqual(len(axes), 1)
        # Реальные баллы (до ~110) не должны попасть под диапазон 0-4
        self.assertGreater(axes[0].get_ylim()[1], 4.0)


if __name__ == "__main__":
    unittest.main()
