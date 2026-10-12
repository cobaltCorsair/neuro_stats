import math
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
import numpy as np
from numpy.testing import assert_allclose

from data_processing.data_processing import TumorDataProcessor
from draw_abs_rel_graph_compare import TumorDataComparatorAdvanced
from draw_base_graphs import TumorDataVisualizer
from utils.visualizer import GraphVisualizer

NAN = float("nan")
SQRT3 = math.sqrt(3)
TIME_DATA = [0, 5, 10]

# mean V(0) = 2: тут смешение абсолютной SD с относительным средним даёт заметно неверную полосу.
# mean V = [2, 4, 6] -> отн. кривая [1, 2, 3]; SD(V(t)) = [1, 2, 2], n = 3.
GROUP_V0_MEAN_2 = [
    [1.0, 2.0, 4.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, 8.0],
]
# SE(V(t)) / mean V(0) = [1/sqrt3, 2/sqrt3, 2/sqrt3] / 2
EXPECTED_REL_SE = [1 / (2 * SQRT3), 1 / SQRT3, 1 / SQRT3]
EXPECTED_ABS_SE = [1 / SQRT3, 2 / SQRT3, 2 / SQRT3]
# Старая формула: SD абсолютных объемов вокруг относительного среднего [1, 2, 3], / sqrt(3)
OLD_BUGGY_REL_SE = [math.sqrt(5 / 2) / SQRT3, math.sqrt(20 / 2) / SQRT3, math.sqrt(35 / 2) / SQRT3]

# Та же группа, деленная на 2 (mean V(0) = 1): отн. кривая и её SEM обязаны совпасть с GROUP_V0_MEAN_2.
GROUP_V0_MEAN_1 = [[v / 2 for v in row] for row in GROUP_V0_MEAN_2]

# Третье животное выбыло к t=10: в t=10 V = [4, 6], mean 5, SD sqrt2, n_t = 2 -> SE = 1, отн. SE = 1 / 2.
GROUP_WITH_DROPOUT = [
    [1.0, 2.0, 4.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, NAN],
]


class StubVisualizer:
    # Хешируемый (prepare_and_add_data_to_graph использует визуализаторы как ключи словаря)
    def __init__(self, tumor_volumes, name="exp"):
        self.time_data = list(TIME_DATA)
        self.experiment_params = [name]
        self.tumor_volumes = tumor_volumes
        self.data_processor = TumorDataProcessor(tumor_volumes)



def _make_tumor_data_visualizer(tumor_volumes):
    # Минуя __init__, который читает Excel
    vis = TumorDataVisualizer.__new__(TumorDataVisualizer)
    vis.file_path = "stub.xlsx"
    vis.time_data = list(TIME_DATA)
    vis.experiment_params = ["exp"]
    vis.rat_labels = [str(i) for i in range(len(tumor_volumes))]
    vis.tumor_volumes = tumor_volumes
    vis.data_processor = TumorDataProcessor(tumor_volumes)
    return vis


def _plotted_error_margins(graph_visualizer):
    """error_margin, переданный в каждый вызов add_plot (5-й позиционный аргумент)."""
    return [np.asarray(c.args[4], dtype=float) for c in graph_visualizer.add_plot.call_args_list]


class TestTumorDataProcessorErrors(unittest.TestCase):
    def test_relative_se_is_absolute_se_over_mean_v0(self):
        processor = TumorDataProcessor(GROUP_V0_MEAN_2)
        assert_allclose(processor.get_mean_relative_tumor_volumes(), [1.0, 2.0, 3.0])
        assert_allclose(processor.get_mean_tumor_volume_errors(), EXPECTED_ABS_SE)
        assert_allclose(processor.get_mean_relative_tumor_volume_errors(), EXPECTED_REL_SE)

    def test_relative_se_does_not_depend_on_volume_scale(self):
        rel_se_v0_2 = TumorDataProcessor(GROUP_V0_MEAN_2).get_mean_relative_tumor_volume_errors()
        rel_se_v0_1 = TumorDataProcessor(GROUP_V0_MEAN_1).get_mean_relative_tumor_volume_errors()
        assert_allclose(rel_se_v0_2, rel_se_v0_1)

    def test_n_is_number_of_animals_measured_at_t(self):
        processor = TumorDataProcessor(GROUP_WITH_DROPOUT)
        rel_se = processor.get_mean_relative_tumor_volume_errors()
        self.assertAlmostEqual(rel_se[2], 0.5, places=12)  # n_t = 2, а не 3


class TestPrepareAndAddDataToGraph(unittest.TestCase):
    def setUp(self):
        patcher = patch("utils.visualizer.format_experiment_params", side_effect=lambda params: params[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_relative_error_extractor_is_used_for_band(self):
        graph = MagicMock()
        GraphVisualizer.prepare_and_add_data_to_graph(
            [StubVisualizer(GROUP_V0_MEAN_2)],
            lambda v: v.data_processor.get_mean_relative_tumor_volumes(),
            graph,
            "",
            error_extractor_func=lambda v: v.data_processor.get_mean_relative_tumor_volume_errors(),
        )
        (error_margin,) = _plotted_error_margins(graph)
        assert_allclose(error_margin, EXPECTED_REL_SE)
        self.assertFalse(np.allclose(error_margin, OLD_BUGGY_REL_SE))

    def test_relative_bands_match_for_groups_differing_only_in_scale(self):
        graph = MagicMock()
        GraphVisualizer.prepare_and_add_data_to_graph(
            [StubVisualizer(GROUP_V0_MEAN_2, "v0=2"), StubVisualizer(GROUP_V0_MEAN_1, "v0=1")],
            lambda v: v.data_processor.get_mean_relative_tumor_volumes(),
            graph,
            "",
            error_extractor_func=lambda v: v.data_processor.get_mean_relative_tumor_volume_errors(),
        )
        band_v0_2, band_v0_1 = _plotted_error_margins(graph)
        assert_allclose(band_v0_2, band_v0_1)

    def test_absolute_plot_without_error_extractor_keeps_absolute_se(self):
        graph = MagicMock()
        GraphVisualizer.prepare_and_add_data_to_graph(
            [StubVisualizer(GROUP_V0_MEAN_2)],
            lambda v: v.data_processor.get_mean_tumor_volumes(),
            graph,
            "M/V абс.: ",
        )
        (error_margin,) = _plotted_error_margins(graph)
        assert_allclose(error_margin, EXPECTED_ABS_SE)
        assert_allclose(error_margin, TumorDataProcessor(GROUP_V0_MEAN_2).get_mean_tumor_volume_errors())


class TestComparatorRelativePlots(unittest.TestCase):
    """Сравнительные относительные графики должны передавать в полосу SEM в отн. ед."""

    def setUp(self):
        patcher = patch("utils.visualizer.format_experiment_params", side_effect=lambda params: params[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        # GraphVisualizer подменяем, чтобы не рисовать фигуру, но статический
        # prepare_and_add_data_to_graph оставляем настоящим.
        self.graph_cls = MagicMock()
        self.graph_cls.prepare_and_add_data_to_graph.side_effect = GraphVisualizer.prepare_and_add_data_to_graph
        patcher = patch("draw_abs_rel_graph_compare.GraphVisualizer", self.graph_cls)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_compare_relative_volumes(self):
        comparator = TumorDataComparatorAdvanced(
            StubVisualizer(GROUP_V0_MEAN_2, "v0=2"), StubVisualizer(GROUP_V0_MEAN_1, "v0=1"))
        comparator.compare_relative_volumes()

        bands = _plotted_error_margins(self.graph_cls.return_value)
        self.assertEqual(len(bands), 2)
        for band in bands:
            assert_allclose(band, EXPECTED_REL_SE)

    def test_compare_control_and_experiment(self):
        comparator = TumorDataComparatorAdvanced(
            StubVisualizer(GROUP_V0_MEAN_2, "exp1"), StubVisualizer(GROUP_WITH_DROPOUT, "exp2"))
        comparator.compare_control_and_experiment([StubVisualizer(GROUP_V0_MEAN_2, "control")])

        control_band, exp1_band, exp2_band = _plotted_error_margins(self.graph_cls.return_value)
        assert_allclose(control_band, EXPECTED_REL_SE)
        assert_allclose(exp1_band, EXPECTED_REL_SE)
        assert_allclose(exp2_band, TumorDataProcessor(GROUP_WITH_DROPOUT).get_mean_relative_tumor_volume_errors())


class TestSingleGroupRelativePlots(unittest.TestCase):
    def setUp(self):
        patcher = patch("draw_base_graphs.format_experiment_params", side_effect=lambda params: params[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("draw_base_graphs.GraphVisualizer")
        self.graph_cls = patcher.start()
        self.addCleanup(patcher.stop)

    def _plotted(self):
        (call,) = self.graph_cls.return_value.add_plot.call_args_list
        return np.asarray(call.args[1], dtype=float), np.asarray(call.args[4], dtype=float)

    def test_plot_mean_relative_mean_tumor_volume_uses_pointwise_relative_se(self):
        _make_tumor_data_visualizer(GROUP_V0_MEAN_2).plot_mean_relative_mean_tumor_volume()

        values, error_margin = self._plotted()
        assert_allclose(values, [1.0, 2.0, 3.0])
        # Раньше — одна константа: SD точек кривой [1, 2, 3] по времени / sqrt(3) = 1/sqrt3 для всех t
        self.assertEqual(error_margin.shape, (3,))
        assert_allclose(error_margin, EXPECTED_REL_SE)

    def test_plot_average_relative_tumor_volume_uses_relative_se_of_plotted_curve(self):
        _make_tumor_data_visualizer(GROUP_V0_MEAN_2).plot_average_relative_tumor_volume()

        values, error_margin = self._plotted()
        assert_allclose(values, [1.0, 2.0, 3.0])
        assert_allclose(error_margin, EXPECTED_REL_SE)


if __name__ == "__main__":
    unittest.main()
