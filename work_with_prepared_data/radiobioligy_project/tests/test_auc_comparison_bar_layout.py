import math
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

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

from draw_base_graphs import TumorDataVisualizer


def _make_stub_visualizer(dose, tumor_volumes):
    vis = MagicMock()
    vis.tumor_volumes = tumor_volumes
    vis.time_data = [0, 1, 2]
    vis.experiment_params = [f"p={dose} Гр"]
    return vis


class TestPlotAucComparisonBarLayout(unittest.TestCase):
    """
    Регрессия: фиксированная ширина столбца (2.5) не учитывала реальное расстояние между
    дозами — при близко расположенных дозах столбцы налезали друг на друга. Плюс SEM для
    группы из одного животного — NaN (std с ddof=1 делит на 0), а matplotlib молча не рисует
    ни error bar, ни текст с NaN-координатой — из-за этого пропадали и интервал, и подпись AUC.
    """

    def tearDown(self):
        plt.close('all')

    @patch("draw_base_graphs.format_experiment_params", side_effect=lambda params: ", ".join(params))
    @patch("draw_base_graphs.TumorDataVisualizer")
    def test_bars_at_close_doses_do_not_overlap(self, mock_cls, _format_params):
        # Зазор между дозами всего 1 Гр — старая фиксированная ширина 2.5 гарантированно
        # вызвала бы перекрытие соседних столбцов.
        stub_map = {
            "a.xlsx": _make_stub_visualizer(23, [[10, 20, 30], [12, 22, 32]]),
            "b.xlsx": _make_stub_visualizer(24, [[11, 21, 31], [13, 23, 33]]),
            "c.xlsx": _make_stub_visualizer(25, [[9, 19, 29], [10, 20, 30]]),
        }
        mock_cls.side_effect = lambda file_path: stub_map[file_path]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TumorDataVisualizer.plot_auc_comparison(list(stub_map.keys()))

        ax = plt.gca()
        # Каждой дозе соответствует ровно один прямоугольник (группы без стека)
        spans = sorted((p.get_x(), p.get_x() + p.get_width()) for p in ax.patches)
        self.assertEqual(len(spans), 3)
        for (left1, right1), (left2, right2) in zip(spans, spans[1:]):
            self.assertLessEqual(right1, left2 + 1e-9, "соседние столбцы перекрываются")

    @patch("draw_base_graphs.format_experiment_params", side_effect=lambda params: ", ".join(params))
    @patch("draw_base_graphs.TumorDataVisualizer")
    def test_bar_width_bounded_by_min_dose_gap(self, mock_cls, _format_params):
        stub_map = {
            "a.xlsx": _make_stub_visualizer(10, [[10, 20, 30], [12, 22, 32]]),
            "b.xlsx": _make_stub_visualizer(11, [[11, 21, 31], [13, 23, 33]]),
        }
        mock_cls.side_effect = lambda file_path: stub_map[file_path]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TumorDataVisualizer.plot_auc_comparison(list(stub_map.keys()))

        ax = plt.gca()
        widths = {round(p.get_width(), 6) for p in ax.patches}
        self.assertEqual(len(widths), 1)
        # Зазор = 1 Гр -> ширина не должна превышать 0.6 (60% от зазора)
        self.assertLessEqual(next(iter(widths)), 0.6 + 1e-9)

    @patch("draw_base_graphs.format_experiment_params", side_effect=lambda params: ", ".join(params))
    @patch("draw_base_graphs.TumorDataVisualizer")
    def test_single_animal_group_still_gets_value_label(self, mock_cls, _format_params):
        """Группа с одним животным -> NaN SEM -> подпись AUC не должна пропадать."""
        stub_map = {
            "a.xlsx": _make_stub_visualizer(30, [[10, 20, 30]]),  # одно животное
        }
        mock_cls.side_effect = lambda file_path: stub_map[file_path]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TumorDataVisualizer.plot_auc_comparison(list(stub_map.keys()))

        ax = plt.gca()
        self.assertEqual(len(ax.texts), 1)
        text = ax.texts[0]
        x, y = text.get_position()
        self.assertFalse(math.isnan(y), "подпись AUC потерялась из-за NaN-координаты")

    @patch("draw_base_graphs.format_experiment_params", side_effect=lambda params: ", ".join(params))
    @patch("draw_base_graphs.TumorDataVisualizer")
    def test_single_animal_group_in_stack_still_gets_value_label(self, mock_cls, _format_params):
        """То же самое, но для столбца внутри стека (несколько файлов на одной дозе) —
        ветка кода с bottom= отдельная и нуждается в той же защите от NaN."""
        stub_map = {
            "a.xlsx": _make_stub_visualizer(30, [[10, 20, 30], [12, 22, 32]]),
            "b.xlsx": _make_stub_visualizer(30, [[8, 18, 28]]),  # одно животное, та же доза
        }
        mock_cls.side_effect = lambda file_path: stub_map[file_path]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TumorDataVisualizer.plot_auc_comparison(list(stub_map.keys()))

        ax = plt.gca()
        self.assertEqual(len(ax.texts), 2)
        for text in ax.texts:
            _, y = text.get_position()
            self.assertFalse(math.isnan(y), "подпись AUC потерялась из-за NaN-координаты")

    @patch("draw_base_graphs.format_experiment_params", side_effect=lambda params: ", ".join(params))
    @patch("draw_base_graphs.TumorDataVisualizer")
    def test_value_label_stays_within_ylim_with_margin(self, mock_cls, _format_params):
        """
        Регрессия: автомасштаб оси Y учитывает только столбцы и error bar, но не текст
        (plt.text не влияет на dataLim) — при небольшом среднем AUC и широком доверительном
        интервале подпись оказывалась ровно на границе области построения или за ней.
        Разброс индивидуальных AUC подобран так, чтобы top error bar был намного выше
        среднего (имитирует "маленький столбец с большим разбросом" со скриншота).
        """
        stub_map = {
            "a.xlsx": _make_stub_visualizer(38, [[5, 8, 9], [6, 9, 10]]),
            "b.xlsx": _make_stub_visualizer(46, [[5, 20, 45], [5, 30, 15]]),
            "c.xlsx": _make_stub_visualizer(48, [[3, 9, 11]]),
        }
        mock_cls.side_effect = lambda file_path: stub_map[file_path]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TumorDataVisualizer.plot_auc_comparison(list(stub_map.keys()))

        ax = plt.gca()
        ylim_top = ax.get_ylim()[1]
        for text in ax.texts:
            _, label_y = text.get_position()
            self.assertGreater(
                ylim_top, label_y,
                f"подпись на y={label_y} выходит за верхнюю границу оси ({ylim_top})",
            )

    @patch("draw_base_graphs.format_experiment_params", side_effect=lambda params: ", ".join(params))
    @patch("draw_base_graphs.TumorDataVisualizer")
    def test_significance_symbol_stays_within_ylim_with_margin(self, mock_cls, _format_params):
        """Символ значимости (звёздочка) рисуется даже выше подписи AUC (свой y_offset) —
        должен точно так же попадать в auto-extended ylim, а не обрезаться по границе."""
        # Полностью разделённые группы по 4 животных -> двусторонний Манна-Уитни
        # гарантированно даёт минимально возможное p (~0.029) для такого n, что <0.05.
        stub_map = {
            "control.xlsx": _make_stub_visualizer(0, [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]]),
            "exp.xlsx": _make_stub_visualizer(20, [[9, 18, 27], [9, 19, 26], [9, 17, 28], [9, 20, 25]]),
        }
        mock_cls.side_effect = lambda file_path: stub_map[file_path]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TumorDataVisualizer.plot_auc_comparison(
                list(stub_map.keys()), perform_stat_test=True, control_index=0,
            )

        ax = plt.gca()
        star_texts = [t for t in ax.texts if t.get_text().strip() == '*']
        self.assertTrue(star_texts, "символ значимости не найден — тест сам по себе не сработал")
        ylim_top = ax.get_ylim()[1]
        for text in star_texts:
            _, symbol_y = text.get_position()
            self.assertGreater(
                ylim_top, symbol_y,
                f"символ значимости на y={symbol_y} выходит за верхнюю границу оси ({ylim_top})",
            )


if __name__ == "__main__":
    unittest.main()
