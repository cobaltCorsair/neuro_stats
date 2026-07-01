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

import pandas as pd
from PyQt6.QtWidgets import QApplication

import gui.main_window as main_window_module
from skin_reactions_base_grapf import SkinReactionsVisualizer
from gui.skin_reaction_summary_window import SkinReactionSummaryWindow


class _StubSkinDataProcessor:
    def __init__(self, mean_reactions):
        self._mean = mean_reactions

    def get_mean_skin_reactions(self):
        n = len(self._mean)
        return self._mean, [0.0] * n, [0.0] * n


def _make_visualizer(time_data, mean_reactions, experiment_name):
    vis = SkinReactionsVisualizer.__new__(SkinReactionsVisualizer)
    vis.file_path = "stub.xlsx"
    vis.experiment_params = [experiment_name]
    vis.time_data = list(time_data)
    vis.skin_reactions = [list(mean_reactions)]
    vis.data_processor = _StubSkinDataProcessor(mean_reactions)
    return vis


class TestBuildSummaryTable(unittest.TestCase):
    @patch("skin_reactions_base_grapf.format_experiment_params", side_effect=lambda params: params[0])
    def test_basic_two_groups(self, _format_params):
        # group A: поднимается до 10 (день2), спадает до 0 (день4) — порог 5
        vis_a = _make_visualizer([0, 1, 2, 3, 4], [0, 5, 10, 5, 0], "groupA")
        # group B: всё время ниже порога 5 — нет реакции выше порога вовсе
        vis_b = _make_visualizer([0, 1, 2, 3, 4], [0, 1, 2, 1, 0], "groupB")

        summary_df = SkinReactionsVisualizer.build_summary_table([vis_a, vis_b], threshold=5.0)

        self.assertEqual(
            list(summary_df.columns),
            ["Группа", "Пик, балл", "День пика", "Длительность >= порога, сут",
             "Цензурировано (длительность)", "День нормализации", "Цензурировано (нормализация)"]
        )
        self.assertEqual(len(summary_df), 2)

        row_a = summary_df.iloc[0]
        self.assertEqual(row_a["Группа"], "groupA")
        self.assertEqual(row_a["Пик, балл"], 10)
        self.assertEqual(row_a["День пика"], 2)
        self.assertGreater(row_a["Длительность >= порога, сут"], 0)
        self.assertFalse(row_a["Цензурировано (длительность)"])

        row_b = summary_df.iloc[1]
        self.assertEqual(row_b["Группа"], "groupB")
        self.assertEqual(row_b["Длительность >= порога, сут"], 0.0)

    @patch("skin_reactions_base_grapf.format_experiment_params", side_effect=lambda params: params[0])
    def test_string_time_data_from_excel_does_not_crash(self, _format_params):
        """
        Регрессия: SkinReactionsVisualizer.time_data приходит из Excel как сырые СТРОКИ
        (например, "2 сут. - 21.06" или просто "2"), а не float. Без преобразования
        арифметика внутри calculate_skin_reaction_* падает с
        TypeError: unsupported operand type(s) for -: 'str' and 'str'.
        """
        vis = _make_visualizer(["0", "1", "2", "3", "4"], [0, 5, 10, 5, 0], "groupA")
        summary_df = SkinReactionsVisualizer.build_summary_table([vis], threshold=5.0)
        self.assertEqual(summary_df.iloc[0]["Пик, балл"], 10)

    @patch("skin_reactions_base_grapf.format_experiment_params", side_effect=lambda params: params[0])
    def test_normalization_grade_defaults_to_threshold(self, _format_params):
        vis = _make_visualizer([0, 1, 2], [10, 10, 10], "groupA")  # никогда не падает ниже порога
        summary_df = SkinReactionsVisualizer.build_summary_table([vis], threshold=5.0)
        row = summary_df.iloc[0]
        # normalization_grade по умолчанию = threshold = 5 -> кривая (всегда 10) никогда не
        # достигает порога -> цензурировано
        self.assertTrue(row["Цензурировано (нормализация)"])


class TestSkinReactionSummaryWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_set_summary_table_populates_widget(self):
        window = SkinReactionSummaryWindow()
        df = pd.DataFrame({
            "Группа": ["groupA", "groupB"],
            "Пик, балл": [10.0, 8.0],
            "День пика": [2.0, 3.0],
        })
        window.set_summary_table(df)

        self.assertEqual(window.table.columnCount(), 3)
        self.assertEqual(window.table.rowCount(), 2)
        self.assertEqual(window.table.horizontalHeaderItem(0).text(), "Группа")
        self.assertEqual(window.table.item(0, 0).text(), "groupA")
        self.assertEqual(window.table.item(0, 1).text(), "10")

    def test_table_header_uses_dark_app_style_not_default_qt_light(self):
        """
        Регрессия: MainWindow._apply_stylesheet применяется через self.setStyleSheet(...) на
        самом MainWindow и не каскадируется на отдельные top-level окна (SkinReactionSummaryWindow
        — самостоятельный QDialog), поэтому заголовок таблицы по умолчанию рендерился светлым Qt
        стилем вместо тёмного навигационного, как у остальных таблиц приложения.
        """
        window = SkinReactionSummaryWindow()
        self.assertIn("QHeaderView::section", window.table.styleSheet())

    def test_table_item_text_has_explicit_dark_color_not_low_contrast_default(self):
        """
        Регрессия: QTableView::item без явного color: полагается на дефолтный цвет текста
        ячеек Qt, который в этой теме рендерится низкоконтрастным (почти неразличимым) на
        светлом фоне таблицы — пользователь явно попросил сделать темнее. Явный color: в
        QTableView::item гарантирует контраст независимо от платформенной темы Qt.
        """
        window = SkinReactionSummaryWindow()
        self.assertIn("QTableView::item", window.table.styleSheet())
        self.assertIn("color: #1C2733", window.table.styleSheet())

    def test_set_summary_table_handles_none(self):
        window = SkinReactionSummaryWindow()
        window.set_summary_table(None)
        self.assertEqual(window.table.rowCount(), 0)


class TestHandleSkinReactionSummaryTableWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_cancel_dialog_does_not_open_window(self):
        import types
        window = types.SimpleNamespace()
        window.skin_reaction_summary_window = None
        window.get_selected_experiments = lambda: ["exp1_skin_reactions.xlsx"]

        with patch.object(main_window_module, "QInputDialog") as mock_dialog:
            mock_dialog.getDouble.return_value = (0.0, False)  # пользователь нажал "Отмена"
            main_window_module.MainWindow.handle_skin_reaction_summary_table(window)

        self.assertIsNone(window.skin_reaction_summary_window)

    def test_rejects_files_without_skin_reactions_in_path_with_warning(self):
        """
        Регрессия: пункт меню "Сводка кожных реакций" не проверял тип выбранных файлов (в
        отличие от pushButton_4/pushButton_8, см. update_fourth_button_state) — файлы объёмов
        опухоли молча читались как баллы кожной реакции и выдавали бессмысленные числа вместо
        явной ошибки.
        """
        import types
        window = types.SimpleNamespace()
        window.skin_reaction_summary_window = None
        window.get_selected_experiments = lambda: ["exp1_skin_reactions.xlsx", "exp2_in_peak.xlsx"]

        with patch.object(main_window_module, "QMessageBox") as mock_msgbox, \
                patch.object(main_window_module, "QInputDialog") as mock_dialog:
            main_window_module.MainWindow.handle_skin_reaction_summary_table(window)

        mock_msgbox.warning.assert_called_once()
        mock_dialog.getDouble.assert_not_called()
        self.assertIsNone(window.skin_reaction_summary_window)

    def test_confirmed_dialog_opens_window_with_summary(self):
        import types

        class FakeVisualizer:
            def __init__(self, path):
                self.path = path

        class FakeSummaryWindow:
            def __init__(self, parent=None):
                self.parent = parent
                self.summary_calls = []
                self.show_called = False

            def set_summary_table(self, df):
                self.summary_calls.append(df)

            def show(self):
                self.show_called = True

            def raise_(self):
                pass

            def activateWindow(self):
                pass

        window = types.SimpleNamespace()
        window.skin_reaction_summary_window = None
        window.get_selected_experiments = lambda: ["exp1_skin_reactions.xlsx"]

        fake_df = pd.DataFrame({"Группа": ["exp1"]})

        with patch.object(main_window_module, "QInputDialog") as mock_dialog, \
                patch.object(main_window_module, "SkinReactionsVisualizer") as mock_vis_cls, \
                patch.object(main_window_module, "SkinReactionSummaryWindow", FakeSummaryWindow):
            mock_dialog.getDouble.return_value = (200.0, True)
            mock_vis_cls.side_effect = FakeVisualizer
            mock_vis_cls.build_summary_table.return_value = fake_df

            main_window_module.MainWindow.handle_skin_reaction_summary_table(window)

        self.assertIsInstance(window.skin_reaction_summary_window, FakeSummaryWindow)
        self.assertEqual(len(window.skin_reaction_summary_window.summary_calls), 1)
        self.assertTrue(window.skin_reaction_summary_window.summary_calls[0] is fake_df)
        self.assertTrue(window.skin_reaction_summary_window.show_called)
        mock_vis_cls.build_summary_table.assert_called_once()


if __name__ == "__main__":
    unittest.main()
