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
import matplotlib.patches as mpatches

# Длинный путь — тот же самый, что используют plotting_helpers.py/draw_base_graphs.py/
# skin_reactions_base_grapf.py внутри себя; короткий "from gui import graph_manager" дал бы
# ВТОРОЙ независимый экземпляр модуля с отдельным глобальным состоянием.
from work_with_prepared_data.radiobioligy_project.gui import graph_manager
from utils.plotting_helpers import add_legend_below_chart
from skin_reactions_base_grapf import SkinReactionsVisualizer


def _make_stub_visualizer(reactions, params):
    vis = MagicMock()
    vis.skin_reactions = reactions
    vis.time_data = list(range(len(reactions[0])))
    vis.experiment_params = params
    return vis


class TestAddLegendBelowChart(unittest.TestCase):
    """
    Легенды AUC-графиков (сравнение по дозам) содержат длинные подписи экспериментов и
    почти при любом расположении внутри осей (в т.ч. "best") налезают на столбцы —
    поэтому такие легенды всегда выносятся под график через этот общий хелпер.
    """

    def setUp(self):
        fig, ax = plt.subplots()
        self.fig, self.ax = fig, ax
        self.handles = [mpatches.Patch(color="red", label="A"), mpatches.Patch(color="blue", label="B")]

    def tearDown(self):
        plt.close("all")
        graph_manager.update_legend_position("best")

    def test_returns_none_when_position_is_none(self):
        graph_manager.update_legend_position(None)
        self.assertIsNone(add_legend_below_chart(self.handles))

    def test_returns_none_when_no_handles(self):
        graph_manager.update_legend_position("best")
        self.assertIsNone(add_legend_below_chart([]))

    def test_places_legend_below_axes_regardless_of_selected_position(self):
        for position in ("best", "upper right", "lower left", "center"):
            with self.subTest(position=position):
                graph_manager.update_legend_position(position)
                legend = add_legend_below_chart(self.handles)
                self.assertIsNotNone(legend)
                self.assertEqual(9, legend._loc)  # 9 == "upper center"
                self.assertLess(legend._bbox_to_anchor._bbox.y0, 0)
                legend.remove()

    def test_default_ncol_is_single_column(self):
        """
        Проверено эмпирически: при длинных подписях эксперимента ncol>1 переполняет
        фигуру по ширине (tight_layout не может подобрать поля), поэтому под графиком
        по умолчанию должен использоваться список в столбик.
        """
        graph_manager.update_legend_position("best")
        legend = add_legend_below_chart(self.handles)
        self.assertEqual(1, legend._ncols)

    def test_custom_ncol_is_respected(self):
        graph_manager.update_legend_position("best")
        legend = add_legend_below_chart(self.handles, ncol=2)
        self.assertEqual(2, legend._ncols)


class TestSkinReactionAucLegendBelowChart(unittest.TestCase):
    """Оба варианта AUC-графика кожных реакций используют тот же общий хелпер, что и
    AUC объёмов опухоли — легенда всегда под графиком, а не там, куда попадёт 'best'."""

    def tearDown(self):
        plt.close("all")
        graph_manager.update_legend_position("best")

    def test_plot_auc_comparison_from_visualizers_places_legend_below(self):
        graph_manager.update_legend_position("best")
        visualizers = [
            _make_stub_visualizer([[0, 1, 2, 1]], ["p=23 Гр", "Date=1.1.2025"]),
            _make_stub_visualizer([[0, 2, 3, 2]], ["p=16 Гр", "Date=2.1.2025"]),
        ]

        SkinReactionsVisualizer.plot_auc_comparison_from_visualizers(visualizers)

        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(9, legend._loc)
        self.assertLess(legend._bbox_to_anchor._bbox.y0, 0)

    @patch("skin_reactions_base_grapf.SkinReactionsVisualizer")
    def test_plot_auc_comparison_places_legend_below(self, mock_cls):
        stub_map = {
            "a.xlsx": _make_stub_visualizer([[0, 1, 2, 1]], ["p=23 Гр", "Date=1.1.2025"]),
            "b.xlsx": _make_stub_visualizer([[0, 2, 3, 2]], ["p=16 Гр", "Date=2.1.2025"]),
        }
        mock_cls.side_effect = lambda file_path: stub_map[file_path]
        graph_manager.update_legend_position("best")

        SkinReactionsVisualizer.plot_auc_comparison(list(stub_map.keys()))

        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(9, legend._loc)
        self.assertLess(legend._bbox_to_anchor._bbox.y0, 0)

    def test_none_position_hides_legend(self):
        graph_manager.update_legend_position(None)
        visualizers = [
            _make_stub_visualizer([[0, 1, 2, 1]], ["p=23 Гр", "Date=1.1.2025"]),
            _make_stub_visualizer([[0, 2, 3, 2]], ["p=16 Гр", "Date=2.1.2025"]),
        ]

        SkinReactionsVisualizer.plot_auc_comparison_from_visualizers(visualizers)

        self.assertIsNone(plt.gca().get_legend())


class TestSkinReactionAucSymbolYlimMargin(unittest.TestCase):
    """
    Автомасштаб оси Y учитывает только столбцы и error bar, но не текст поверх них —
    символ значимости (звёздочка), рисуемый выше подписи AUC своим y_offset, мог оказаться
    ровно на границе области построения или за ней. Та же регрессия и тот же фикс, что и
    для draw_base_graphs.TumorDataVisualizer.plot_auc_comparison
    (см. test_significance_symbol_stays_within_ylim_with_margin в
    test_auc_comparison_bar_layout.py), только в дублирующем коде для кожных реакций.
    """

    def tearDown(self):
        plt.close("all")
        graph_manager.update_legend_position("best")

    @staticmethod
    def _separated_groups():
        # Полностью разделённые группы по 4 животных -> двусторонний Манна-Уитни даёт
        # минимально возможное p (~0.029) для такого n, что <0.05.
        control = _make_stub_visualizer(
            [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]], ["p=0 Гр", "Date=1.1.2025"]
        )
        exp = _make_stub_visualizer(
            [[9, 18, 27], [9, 19, 26], [9, 17, 28], [9, 20, 25]], ["p=20 Гр", "Date=2.1.2025"]
        )
        return control, exp

    def test_plot_auc_comparison_from_visualizers_symbol_within_ylim(self):
        control, exp = self._separated_groups()
        graph_manager.update_legend_position("best")

        SkinReactionsVisualizer.plot_auc_comparison_from_visualizers(
            [control, exp], perform_stat_test=True, control_index=0,
        )

        ax = plt.gca()
        star_texts = [t for t in ax.texts if t.get_text().strip() == '*']
        self.assertTrue(star_texts, "символ значимости не найден — тест сам по себе не сработал")
        ylim_top = ax.get_ylim()[1]
        for text in star_texts:
            _, symbol_y = text.get_position()
            self.assertGreater(ylim_top, symbol_y)

    @patch("skin_reactions_base_grapf.SkinReactionsVisualizer")
    def test_plot_auc_comparison_symbol_within_ylim(self, mock_cls):
        control, exp = self._separated_groups()
        stub_map = {"control.xlsx": control, "exp.xlsx": exp}
        mock_cls.side_effect = lambda file_path: stub_map[file_path]
        graph_manager.update_legend_position("best")

        SkinReactionsVisualizer.plot_auc_comparison(
            list(stub_map.keys()), perform_stat_test=True, control_index=0,
        )

        ax = plt.gca()
        star_texts = [t for t in ax.texts if t.get_text().strip() == '*']
        self.assertTrue(star_texts, "символ значимости не найден — тест сам по себе не сработал")
        ylim_top = ax.get_ylim()[1]
        for text in star_texts:
            _, symbol_y = text.get_position()
            self.assertGreater(ylim_top, symbol_y)


if __name__ == "__main__":
    unittest.main()
