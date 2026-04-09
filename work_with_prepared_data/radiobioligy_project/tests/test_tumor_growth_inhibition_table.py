import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
from PyQt6.QtWidgets import QApplication

import gui.main_window as main_window_module
from draw_abs_rel_graph_compare import TumorDataComparatorAdvanced
from gui.main_window import MainWindow
from gui.tgi_table_window import TumorGrowthInhibitionTableWindow


TIME_COLUMN = "Время (сут)"
PAIR_COLUMN = "Пара экспериментов"
ABS_DIFF_COLUMN = "Средняя абсолютная разница ТРО, %"
REL_DIFF_COLUMN = "Средняя относительная разница ТРО, %"
COUNT_COLUMN = "Число временных точек"


class StubProcessor:
    def __init__(self, mean_tumor_volumes):
        self._mean_tumor_volumes = list(mean_tumor_volumes)

    def get_mean_tumor_volumes(self):
        return list(self._mean_tumor_volumes)


class StubVisualizer:
    def __init__(self, time_data, mean_tumor_volumes, experiment_name):
        self.time_data = list(time_data)
        self.data_processor = StubProcessor(mean_tumor_volumes)
        self.experiment_params = [experiment_name]


class TumorGrowthInhibitionTableTests(unittest.TestCase):
    @staticmethod
    def _build_control(time_data, mean_tumor_volumes):
        return StubVisualizer(time_data, mean_tumor_volumes, "control")

    @staticmethod
    def _build_experiment(time_data, mean_tumor_volumes, experiment_name):
        return StubVisualizer(time_data, mean_tumor_volumes, experiment_name)

    @patch("draw_abs_rel_graph_compare.format_experiment_params", side_effect=lambda params: params[0])
    def test_single_experiment_returns_main_table_without_pairwise_summary(self, _format_params):
        control = self._build_control([0, 5, 10, 15], [100, 100, 100, 100])
        experiment = self._build_experiment([0, 5, 10, 15], [80, 70, 60, 50], "exp1")

        comparator = TumorDataComparatorAdvanced(experiment)
        tgi_df, pairwise_summary_df = comparator.create_tumor_growth_inhibition_table(control, [experiment])

        self.assertEqual(list(tgi_df.columns), [TIME_COLUMN, "exp1"])
        self.assertIsNone(pairwise_summary_df)
        self.assertEqual(list(tgi_df[TIME_COLUMN]), [0.0, 5.0, 10.0, 15.0])
        self.assertEqual(list(tgi_df["exp1"]), [20.0, 30.0, 40.0, 50.0])

    @patch("draw_abs_rel_graph_compare.format_experiment_params", side_effect=lambda params: params[0])
    def test_two_experiments_return_single_pairwise_summary_row(self, _format_params):
        control = self._build_control([0, 5, 10, 15], [100, 100, 100, 100])
        experiment_1 = self._build_experiment([0, 5, 10, 15], [80, 70, 60, 50], "exp1")
        experiment_2 = self._build_experiment([0, 5, 10, 15], [90, 80, 70, 60], "exp2")

        comparator = TumorDataComparatorAdvanced(experiment_1, experiment_2)
        tgi_df, pairwise_summary_df = comparator.create_tumor_growth_inhibition_table(
            control,
            [experiment_1, experiment_2]
        )

        self.assertEqual(list(tgi_df.columns), [TIME_COLUMN, "exp1", "exp2"])
        self.assertIsNotNone(pairwise_summary_df)
        self.assertEqual(len(pairwise_summary_df), 1)

        summary_row = pairwise_summary_df.iloc[0]
        self.assertEqual(summary_row[PAIR_COLUMN], "exp1 vs exp2")
        self.assertAlmostEqual(summary_row[ABS_DIFF_COLUMN], 10.0)
        self.assertAlmostEqual(summary_row[REL_DIFF_COLUMN], 22.5)
        self.assertEqual(summary_row[COUNT_COLUMN], 2)

    @patch("draw_abs_rel_graph_compare.format_experiment_params", side_effect=lambda params: params[0])
    def test_three_experiments_return_three_pairwise_summary_rows_in_selection_order(self, _format_params):
        control = self._build_control([0, 5, 10, 15], [100, 100, 100, 100])
        experiment_1 = self._build_experiment([0, 5, 10, 15], [80, 70, 60, 50], "exp1")
        experiment_2 = self._build_experiment([0, 5, 10, 15], [90, 80, 70, 60], "exp2")
        experiment_3 = self._build_experiment([0, 5, 10, 15], [95, 85, 75, 65], "exp3")

        comparator = TumorDataComparatorAdvanced(experiment_1, experiment_2, experiment_3)
        tgi_df, pairwise_summary_df = comparator.create_tumor_growth_inhibition_table(
            control,
            [experiment_1, experiment_2, experiment_3]
        )

        self.assertEqual(list(tgi_df.columns), [TIME_COLUMN, "exp1", "exp2", "exp3"])
        self.assertIsNotNone(pairwise_summary_df)
        self.assertEqual(len(pairwise_summary_df), 3)
        self.assertEqual(
            list(pairwise_summary_df[PAIR_COLUMN]),
            ["exp1 vs exp2", "exp1 vs exp3", "exp2 vs exp3"]
        )

    @patch("draw_abs_rel_graph_compare.format_experiment_params", side_effect=lambda params: params[0])
    def test_pairwise_summary_uses_only_timepoints_from_day_nine(self, _format_params):
        control = self._build_control([0, 8, 10], [100, 100, 100])
        experiment_1 = self._build_experiment([0, 8, 10], [0, 0, 90], "exp1")
        experiment_2 = self._build_experiment([0, 8, 10], [100, 100, 80], "exp2")

        comparator = TumorDataComparatorAdvanced(experiment_1, experiment_2)
        _, pairwise_summary_df = comparator.create_tumor_growth_inhibition_table(
            control,
            [experiment_1, experiment_2]
        )

        summary_row = pairwise_summary_df.iloc[0]
        self.assertAlmostEqual(summary_row[ABS_DIFF_COLUMN], 10.0)
        self.assertEqual(summary_row[COUNT_COLUMN], 1)

    @patch("draw_abs_rel_graph_compare.format_experiment_params", side_effect=lambda params: params[0])
    def test_default_table_uses_control_days_and_interpolates_experiments(self, _format_params):
        control = self._build_control([0, 7, 14, 21], [100, 100, 100, 100])
        experiment_1 = self._build_experiment([0, 10, 20], [80, 60, 40], "exp1")
        experiment_2 = self._build_experiment([0, 14, 21], [90, 50, 30], "exp2")

        comparator = TumorDataComparatorAdvanced(experiment_1, experiment_2)
        tgi_df, pairwise_summary_df = comparator.create_tumor_growth_inhibition_table(
            control,
            [experiment_1, experiment_2]
        )

        self.assertEqual(list(tgi_df[TIME_COLUMN]), [0.0, 7.0, 14.0, 21.0])
        self.assertFalse(tgi_df[["exp1", "exp2"]].isna().any().any())
        self.assertAlmostEqual(tgi_df.loc[tgi_df[TIME_COLUMN] == 7.0, "exp1"].iloc[0], 34.0)
        self.assertAlmostEqual(tgi_df.loc[tgi_df[TIME_COLUMN] == 7.0, "exp2"].iloc[0], 30.0)
        self.assertAlmostEqual(tgi_df.loc[tgi_df[TIME_COLUMN] == 21.0, "exp1"].iloc[0], 60.0)

        summary_row = pairwise_summary_df.iloc[0]
        self.assertEqual(summary_row[COUNT_COLUMN], 2)

    @patch("draw_abs_rel_graph_compare.format_experiment_params", side_effect=lambda params: params[0])
    def test_daily_interpolation_mode_builds_integer_day_grid_without_gaps(self, _format_params):
        control = self._build_control([0, 10, 20], [100, 100, 100])
        experiment_1 = self._build_experiment([0, 10, 20], [80, 60, 40], "exp1")
        experiment_2 = self._build_experiment([2, 15, 18], [95, 50, 35], "exp2")

        comparator = TumorDataComparatorAdvanced(experiment_1, experiment_2)
        tgi_df, pairwise_summary_df = comparator.create_tumor_growth_inhibition_table(
            control,
            [experiment_1, experiment_2],
            time_grid_mode=TumorDataComparatorAdvanced.TGI_TIME_GRID_DAILY_INTERPOLATION
        )

        self.assertEqual(list(tgi_df[TIME_COLUMN]), list(range(0, 21)))
        self.assertFalse(tgi_df[["exp1", "exp2"]].isna().any().any())
        self.assertAlmostEqual(tgi_df.loc[tgi_df[TIME_COLUMN] == 1.0, "exp2"].iloc[0], 5.0)
        self.assertAlmostEqual(tgi_df.loc[tgi_df[TIME_COLUMN] == 16.0, "exp2"].iloc[0], 55.0, places=6)

        summary_row = pairwise_summary_df.iloc[0]
        self.assertEqual(summary_row[COUNT_COLUMN], 12)


class TumorGrowthInhibitionTableGuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @staticmethod
    def _build_window_stub(selected_paths):
        window = types.SimpleNamespace()
        window.control_path = "control.xlsx"
        window.tgi_table_window = None
        window.get_selected_experiments = lambda: list(selected_paths)
        return window

    def test_handle_tumor_growth_inhibition_table_opens_separate_window(self):
        class FakeComparator:
            def __init__(self, *visualizers):
                self.visualizers = visualizers

            def create_tumor_growth_inhibition_tables(self, control_visualizer, experiment_visualizers):
                experiment_count = len(experiment_visualizers)

                def build_table(timepoints):
                    tgi_data = {TIME_COLUMN: list(timepoints)}
                    for index in range(experiment_count):
                        tgi_data[f"exp{index + 1}"] = [10 + index + step for step, _ in enumerate(timepoints)]
                    tgi_df = pd.DataFrame(tgi_data)

                    if experiment_count < 2:
                        return tgi_df, None

                    pairwise_rows = []
                    for left_index in range(experiment_count - 1):
                        for right_index in range(left_index + 1, experiment_count):
                            pairwise_rows.append({
                                PAIR_COLUMN: f"exp{left_index + 1} vs exp{right_index + 1}",
                                ABS_DIFF_COLUMN: 1.0,
                                REL_DIFF_COLUMN: 2.0,
                                COUNT_COLUMN: 1,
                            })
                    return tgi_df, pd.DataFrame(pairwise_rows)

                return {
                    TumorDataComparatorAdvanced.TGI_TIME_GRID_CONTROL_DAYS: build_table([0, 10]),
                    TumorDataComparatorAdvanced.TGI_TIME_GRID_DAILY_INTERPOLATION: build_table([0, 1, 2, 3]),
                }

        class FakeTableWindow:
            def __init__(self, parent=None):
                self.parent = parent
                self.tables_calls = []
                self.show_called = False
                self.raise_called = False
                self.activate_called = False

            def set_tables(self, tables_by_mode):
                self.tables_calls.append(tables_by_mode)

            def show(self):
                self.show_called = True

            def raise_(self):
                self.raise_called = True

            def activateWindow(self):
                self.activate_called = True

        for experiment_count in (1, 2, 3):
            selected_paths = [f"exp_{index}.xlsx" for index in range(experiment_count)]
            window = self._build_window_stub(selected_paths)

            with self.subTest(experiment_count=experiment_count):
                with patch.object(main_window_module, "TumorDataComparatorAdvanced", FakeComparator), \
                        patch.object(main_window_module, "TumorDataVisualizer", side_effect=lambda path: path), \
                        patch.object(main_window_module, "ControlGroupVisualizer", side_effect=lambda path: path), \
                        patch.object(main_window_module, "TumorGrowthInhibitionTableWindow", FakeTableWindow):
                    MainWindow.handle_tumor_growth_inhibition_table(window)

                self.assertIsInstance(window.tgi_table_window, FakeTableWindow)
                self.assertEqual(len(window.tgi_table_window.tables_calls), 1)
                self.assertTrue(window.tgi_table_window.show_called)
                self.assertTrue(window.tgi_table_window.raise_called)
                self.assertTrue(window.tgi_table_window.activate_called)

                tables_by_mode = window.tgi_table_window.tables_calls[0]
                self.assertIn(TumorDataComparatorAdvanced.TGI_TIME_GRID_CONTROL_DAYS, tables_by_mode)
                self.assertIn(TumorDataComparatorAdvanced.TGI_TIME_GRID_DAILY_INTERPOLATION, tables_by_mode)

    def test_tgi_table_window_formats_headers_and_switches_modes(self):
        table_window = TumorGrowthInhibitionTableWindow()

        tables_by_mode = {
            TumorGrowthInhibitionTableWindow.MODE_CONTROL_DAYS: (
                pd.DataFrame({
                    TIME_COLUMN: [0, 7, 14],
                    "Очень длинное название эксперимента 1": [31.95, 45.0, 69.58],
                    "Очень длинное название эксперимента 2": [40.306, 56.925, 60.0],
                }),
                pd.DataFrame({
                    PAIR_COLUMN: ["Очень длинное название эксперимента 1 vs Очень длинное название эксперимента 2"],
                    ABS_DIFF_COLUMN: [1.133],
                    REL_DIFF_COLUMN: [1.184],
                    COUNT_COLUMN: [7],
                })
            ),
            TumorGrowthInhibitionTableWindow.MODE_DAILY_INTERPOLATION: (
                pd.DataFrame({
                    TIME_COLUMN: [0, 1, 2],
                    "Очень длинное название эксперимента 1": [31.95, 40.0, 69.58],
                    "Очень длинное название эксперимента 2": [40.306, 50.0, 60.0],
                }),
                pd.DataFrame({
                    PAIR_COLUMN: ["Очень длинное название эксперимента 1 vs Очень длинное название эксперимента 2"],
                    ABS_DIFF_COLUMN: [2.5],
                    REL_DIFF_COLUMN: [3.5],
                    COUNT_COLUMN: [3],
                })
            ),
        }

        table_window.set_tables(tables_by_mode)

        self.assertEqual(table_window.mode_selector.count(), 2)
        self.assertEqual(
            table_window.mode_selector.currentData(),
            TumorGrowthInhibitionTableWindow.MODE_CONTROL_DAYS
        )
        self.assertIn("сутки контрольной группы", table_window.legend_note_label.text())

        self.assertEqual(table_window.legend_table.rowCount(), 2)
        self.assertEqual(table_window.legend_table.item(0, 0).text(), "Э1")
        self.assertEqual(table_window.legend_table.item(1, 0).text(), "Э2")
        self.assertEqual(table_window.legend_table.item(0, 1).text(), "Очень длинное название эксперимента 1")

        self.assertEqual(table_window.main_table.horizontalHeaderItem(0).text(), TIME_COLUMN)
        self.assertEqual(table_window.main_table.horizontalHeaderItem(1).text(), "Э1")
        self.assertEqual(table_window.main_table.horizontalHeaderItem(2).text(), "Э2")
        self.assertEqual(
            table_window.main_table.horizontalHeaderItem(1).toolTip(),
            "Очень длинное название эксперимента 1"
        )
        self.assertEqual(table_window.main_table.item(1, 0).text(), "7")

        daily_mode_index = table_window.mode_selector.findData(TumorGrowthInhibitionTableWindow.MODE_DAILY_INTERPOLATION)
        table_window.mode_selector.setCurrentIndex(daily_mode_index)

        self.assertIn("ежедневной сетке", table_window.legend_note_label.text())
        self.assertEqual(table_window.main_table.item(1, 0).text(), "1")
        self.assertEqual(table_window.summary_table.item(0, 0).text(), "Э1 vs Э2")
        self.assertNotEqual(table_window.tabs.indexOf(table_window.summary_tab), -1)
        self.assertIn("QComboBox QAbstractItemView::item:hover", table_window.mode_selector.styleSheet())
        self.assertIn("selection-background-color: #dbe8f6", table_window.mode_selector.styleSheet())
