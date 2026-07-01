import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox, QTableWidgetItem, QWidget

from data_processing.excel_data_processor import RatSurvivalEvent
from gui.kaplan_meier_calculator_window import KaplanMeierCalculatorWindow

# Стандартный учебный пример: 10 пациентов, рак, 5 лет наблюдения. Эталонные значения
# (S(5)=0.366, число в риске на 1..5 = 9,7,5,3,1) проверены в test_kaplan_meier.py
# напрямую на уровне формул; здесь проверяется именно путь "ввод в таблицу -> расчёт".
TEN_PATIENT_EXAMPLE = [
    ("1", "0.5", "Смерть"), ("2", "1.2", "Смерть"), ("3", "1.5", "Цензурирован"),
    ("4", "2.0", "Смерть"), ("5", "2.3", "Цензурирован"), ("6", "3.0", "Смерть"),
    ("7", "3.5", "Цензурирован"), ("8", "4.0", "Смерть"), ("9", "4.5", "Цензурирован"),
    ("10", "5.0", "Цензурирован"),
]


class TestKaplanMeierCalculatorWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _fill_rows(self, win, rows):
        while win.input_table.rowCount() < len(rows):
            win._add_input_row()
        for row, (label, time, event) in enumerate(rows):
            win.input_table.setItem(row, 0, QTableWidgetItem(label))
            win.input_table.setItem(row, 1, QTableWidgetItem(time))
            win.input_table.cellWidget(row, 2).setCurrentText(event)

    def test_starts_with_three_empty_rows(self):
        win = KaplanMeierCalculatorWindow()
        self.assertEqual(win.input_table.rowCount(), 3)

    def test_add_and_remove_row(self):
        win = KaplanMeierCalculatorWindow()
        win._add_input_row()
        self.assertEqual(win.input_table.rowCount(), 4)
        win.input_table.setCurrentCell(0, 0)
        win._remove_selected_row()
        self.assertEqual(win.input_table.rowCount(), 3)

    def test_ten_patient_example_matches_textbook_calculation(self):
        win = KaplanMeierCalculatorWindow()
        self._fill_rows(win, TEN_PATIENT_EXAMPLE)
        win._handle_calculate()

        steps = win.steps_text.toPlainText().splitlines()
        self.assertEqual(steps[0], "t=0.5: n=10, d=1 → S(0.5) = 1 × (1 − 1/10) = 0.9000")
        self.assertEqual(steps[2], "t=1.5: цензурировано 1 → S(1.5) = 0.8000 (без изменений)")

        headers = [win.risk_table.horizontalHeaderItem(c).text() for c in range(win.risk_table.columnCount())]
        values = [win.risk_table.item(0, c).text() for c in range(win.risk_table.columnCount())]
        at_points_1_to_5 = dict(zip(headers, values))
        self.assertEqual(
            [at_points_1_to_5[h] for h in ("1", "2", "3", "4", "5")],
            ["9", "7", "5", "3", "1"],
        )

        self.assertIn("S(5) = 0.3657", win.summary_label.text())
        self.assertFalse(win.plot_label.pixmap().isNull())

    def test_risk_table_columns_stretch_to_fill_width(self):
        # Регрессия: без setSectionResizeMode столбцы оставались фиксированной ширины,
        # и при растяжении окна справа от последнего столбца была пустая серая область.
        from PyQt6.QtWidgets import QHeaderView

        win = KaplanMeierCalculatorWindow()
        self._fill_rows(win, TEN_PATIENT_EXAMPLE)
        win._handle_calculate()

        header = win.risk_table.horizontalHeader()
        for col in range(win.risk_table.columnCount()):
            self.assertEqual(header.sectionResizeMode(col), QHeaderView.ResizeMode.Stretch)

    def test_blank_rows_are_skipped_not_treated_as_errors(self):
        win = KaplanMeierCalculatorWindow()
        self._fill_rows(win, [("1", "2.0", "Смерть"), ("2", "", "Смерть"), ("3", "4.0", "Цензурирован")])
        win._handle_calculate()  # не должно показывать предупреждение об ошибке ввода
        self.assertIn("N = 2", win.summary_label.text())

    def test_invalid_time_text_shows_warning_and_does_not_crash(self):
        win = KaplanMeierCalculatorWindow()
        self._fill_rows(win, [("1", "не число", "Смерть")])
        from unittest.mock import patch
        with patch.object(QMessageBox, "warning") as mock_warning:
            win._handle_calculate()
        mock_warning.assert_called_once()

    def test_all_blank_rows_shows_information_message(self):
        win = KaplanMeierCalculatorWindow()
        from unittest.mock import patch
        with patch.object(QMessageBox, "information") as mock_info:
            win._handle_calculate()
        mock_info.assert_called_once()

    def test_custom_label_is_preserved(self):
        win = KaplanMeierCalculatorWindow()
        self._fill_rows(win, [("Крыса А", "3.0", "Смерть")])
        events = win._read_input_events()
        self.assertEqual(events[0].label, "Крыса А")


class TestLoadFromMainWindow(unittest.TestCase):
    """
    Кнопка «Загрузить из выбранных файлов» переиспользует
    MainWindow.build_kaplan_meier_groups_from_selection — те же файлы/группы A-B,
    что показывает основное окно сравнения групп (KaplanMeierWindow).
    """

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @staticmethod
    def _build_main_window_stub(groups):
        stub = QWidget()
        stub.build_kaplan_meier_groups_from_selection = lambda: groups
        return stub

    def test_single_group_loads_directly_without_prompting(self):
        events = [RatSurvivalEvent("r1", 5.0, True, "death"), RatSurvivalEvent("r2", 10.0, False, "")]
        stub = self._build_main_window_stub({"control": events})
        win = KaplanMeierCalculatorWindow(stub)

        from unittest.mock import patch
        with patch.object(QInputDialog, "getItem") as mock_dialog:
            win._handle_load_from_main_window()
        mock_dialog.assert_not_called()

        self.assertEqual(win.input_table.rowCount(), 2)
        self.assertEqual(win.input_table.item(0, 0).text(), "r1")
        self.assertEqual(win.input_table.item(0, 1).text(), "5")
        self.assertEqual(win.input_table.cellWidget(0, 2).currentText(), "Смерть")
        self.assertEqual(win.input_table.cellWidget(1, 2).currentText(), "Цензурирован")

    def test_multiple_groups_prompts_and_loads_chosen_one(self):
        groups = {
            "Группа A": [RatSurvivalEvent("a1", 5.0, True, "death")],
            "Группа B": [RatSurvivalEvent("b1", 10.0, False, "")],
        }
        stub = self._build_main_window_stub(groups)
        win = KaplanMeierCalculatorWindow(stub)

        from unittest.mock import patch
        with patch.object(QInputDialog, "getItem", return_value=("Группа B", True)):
            win._handle_load_from_main_window()

        self.assertEqual(win.input_table.rowCount(), 1)
        self.assertEqual(win.input_table.item(0, 0).text(), "b1")

    def test_cancelling_group_dialog_leaves_table_unchanged(self):
        groups = {
            "Группа A": [RatSurvivalEvent("a1", 5.0, True, "death")],
            "Группа B": [RatSurvivalEvent("b1", 10.0, False, "")],
        }
        stub = self._build_main_window_stub(groups)
        win = KaplanMeierCalculatorWindow(stub)
        rows_before = win.input_table.rowCount()

        from unittest.mock import patch
        with patch.object(QInputDialog, "getItem", return_value=("", False)):
            win._handle_load_from_main_window()

        self.assertEqual(win.input_table.rowCount(), rows_before)

    def test_no_selection_in_main_window_shows_message_and_keeps_table(self):
        stub = self._build_main_window_stub(None)
        win = KaplanMeierCalculatorWindow(stub)
        rows_before = win.input_table.rowCount()

        from unittest.mock import patch
        with patch.object(QMessageBox, "information") as mock_info:
            win._handle_load_from_main_window()

        mock_info.assert_called_once()
        self.assertEqual(win.input_table.rowCount(), rows_before)

    def test_no_parent_shows_message(self):
        win = KaplanMeierCalculatorWindow(None)
        from unittest.mock import patch
        with patch.object(QMessageBox, "information") as mock_info:
            win._handle_load_from_main_window()
        mock_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
