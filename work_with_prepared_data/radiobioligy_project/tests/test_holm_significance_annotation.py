import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from stats_methods.support_stats_methods import SupportingFunctions as SF
from utils.visualizer import GraphVisualizer
# Длинный путь — тот же самый, что используют support_stats_methods.py и utils/visualizer.py
# внутри себя; короткий "from gui import graph_manager" дал бы ВТОРОЙ независимый экземпляр
# модуля с отдельным глобальным состоянием (см. test_rtog_checkbox_activation.py).
from work_with_prepared_data.radiobioligy_project.gui import graph_manager


class _FakeVisualizer:
    def __init__(self, time_data):
        self.time_data = time_data


class TestApplyMannWhitneyHolmIntegration(unittest.TestCase):
    def tearDown(self):
        # graph_manager — общий для процесса модуль; не даём состоянию утечь в другие тесты.
        graph_manager.set_holm_correction_enabled(True)

    def test_uses_holm_correction_and_distinguishes_markers(self):
        # 2 эксперимента -> 1 пара, 3 общих временных точки -> одно "семейство" из 3 p-значений
        all_reactions = [
            {'reactions': [[1.0, 1.0, 1.0], [1.2, 1.2, 1.2]]},
            {'reactions': [[5.0, 5.0, 5.0], [5.5, 5.5, 5.5]]},
        ]
        common_timepoints = [0, 1, 2]
        upper_bounds_by_time = {0: 10.0, 1: 10.0, 2: 10.0}

        # m=3, пороги Холма: 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05
        # t0 p=0.001 -> отвергается (rank1); t1 p=0.04 -> не проходит rank2 (0.025) -> '(*)';
        # t2 p=0.5 -> не значимо вовсе, аннотация не рисуется
        p_sequence = [0.001, 0.04, 0.5]

        with patch('stats_methods.support_stats_methods.mannwhitneyu') as mock_mw, \
                patch('stats_methods.support_stats_methods.plt') as mock_plt:
            mock_mw.side_effect = [(None, p) for p in p_sequence]
            mock_plt.ylim.return_value = (0.0, 10.0)
            SF.apply_mann_whitney_test(all_reactions, common_timepoints, upper_bounds_by_time)

        texts = [call.args[2] for call in mock_plt.text.call_args_list]
        self.assertEqual(texts.count('*'), 1)
        self.assertEqual(texts.count('(*)'), 1)
        self.assertEqual(len(texts), 2)  # t2 (p=0.5) не рисуется вовсе

    def test_disabled_holm_falls_back_to_raw_significance_no_parenthesized_marker(self):
        """Поправка Холма — отдельная опциональная функция (checkBox_holm в интерфейсе).
        При выключении: '*' = значимо по сырому p<0.05, '(*)' никогда не рисуется."""
        graph_manager.set_holm_correction_enabled(False)

        all_reactions = [
            {'reactions': [[1.0, 1.0, 1.0], [1.2, 1.2, 1.2]]},
            {'reactions': [[5.0, 5.0, 5.0], [5.5, 5.5, 5.5]]},
        ]
        common_timepoints = [0, 1, 2]
        upper_bounds_by_time = {0: 10.0, 1: 10.0, 2: 10.0}
        # Те же p-значения, что и в тесте с включённой поправкой: t0 и t1 проходят сырой
        # p<0.05, t2 — нет. Без Холма оба значимых отмечаются простым '*'.
        p_sequence = [0.001, 0.04, 0.5]

        with patch('stats_methods.support_stats_methods.mannwhitneyu') as mock_mw, \
                patch('stats_methods.support_stats_methods.plt') as mock_plt:
            mock_mw.side_effect = [(None, p) for p in p_sequence]
            mock_plt.ylim.return_value = (0.0, 10.0)
            SF.apply_mann_whitney_test(all_reactions, common_timepoints, upper_bounds_by_time)

        texts = [call.args[2] for call in mock_plt.text.call_args_list]
        self.assertEqual(texts, ['*', '*'])  # оба значимых по сырому p<0.05, без '(*)'


class TestAddSignificanceAnnotationHolmIntegration(unittest.TestCase):
    def tearDown(self):
        graph_manager.set_holm_correction_enabled(True)

    def test_marks_holm_and_raw_only_differently(self):
        vis = _FakeVisualizer([0, 1, 2])
        upper_bounds_dict = {vis: [10.0, 10.0, 10.0]}
        # m=3: тот же набор, что и выше -> t0 '*', t1 '(*)', t2 без аннотации
        p_values = [0.001, 0.04, 0.5]
        x_positions = [0, 1, 2]

        with patch('utils.visualizer.plt') as mock_plt:
            mock_plt.ylim.return_value = (0.0, 10.0)
            GraphVisualizer.add_significance_annotation(
                [vis], p_values, x_positions, upper_bounds_dict
            )

        texts = [call.args[2] for call in mock_plt.text.call_args_list]
        self.assertEqual(texts.count('*'), 1)
        self.assertEqual(texts.count('(*)'), 1)
        self.assertEqual(len(texts), 2)

    def test_disabled_holm_falls_back_to_raw_significance_no_parenthesized_marker(self):
        graph_manager.set_holm_correction_enabled(False)

        vis = _FakeVisualizer([0, 1, 2])
        upper_bounds_dict = {vis: [10.0, 10.0, 10.0]}
        p_values = [0.001, 0.04, 0.5]
        x_positions = [0, 1, 2]

        with patch('utils.visualizer.plt') as mock_plt:
            mock_plt.ylim.return_value = (0.0, 10.0)
            GraphVisualizer.add_significance_annotation(
                [vis], p_values, x_positions, upper_bounds_dict
            )

        texts = [call.args[2] for call in mock_plt.text.call_args_list]
        self.assertEqual(texts, ['*', '*'])

    def test_none_p_values_skipped_and_excluded_from_holm_family(self):
        vis = _FakeVisualizer([0, 1, 2])
        upper_bounds_dict = {vis: [10.0, 10.0, 10.0]}
        p_values = [0.01, None, 0.02]
        x_positions = [0, 1, 2]

        with patch('utils.visualizer.plt') as mock_plt:
            mock_plt.ylim.return_value = (0.0, 10.0)
            GraphVisualizer.add_significance_annotation(
                [vis], p_values, x_positions, upper_bounds_dict
            )

        # m=2 (None исключён): оба 0.01 и 0.02 проходят свои пороги (0.025, 0.05) -> оба '*'
        texts = [call.args[2] for call in mock_plt.text.call_args_list]
        self.assertEqual(texts, ['*', '*'])


if __name__ == "__main__":
    unittest.main()
