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

import pandas as pd
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

import gui.main_window as main_window_module
from gui.main_window import MainWindow
# handle_kaplan_meier импортирует KaplanMeierWindow по полному пути (work_with_prepared_data...) —
# используем тот же путь здесь, иначе isinstance видит "два разных класса" из-за двойного импорта модуля.
from work_with_prepared_data.radiobioligy_project.gui.kaplan_meier_window import KaplanMeierWindow


def _write_xlsx(tmp_path: Path, name: str, header_row, time_labels, data_rows) -> str:
    width = max(len(header_row), len(time_labels), max((len(r) for r in data_rows), default=0))

    def pad(row):
        return list(row) + [None] * (width - len(row))

    rows = [pad(header_row), pad(time_labels)] + [pad(r) for r in data_rows]
    df = pd.DataFrame(rows)
    file_path = tmp_path / name
    df.to_excel(file_path, header=False, index=False, engine="openpyxl")
    return str(file_path)


class TestHandleKaplanMeier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    @staticmethod
    def _build_window_stub(selected_paths, group_assignment):
        # QWidget (not SimpleNamespace) — KaplanMeierWindow(self) requires a real QWidget as parent.
        window = QWidget()
        window.kaplan_meier_window = None
        window.get_selected_experiments = lambda: list(selected_paths)
        window.get_group_assignment = lambda: dict(group_assignment)
        window._show_tool_open_error = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected error path"))
        return window

    def _make_file(self, name, death_day_marker, value_at_marker_col):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", value_at_marker_col]]
        return _write_xlsx(self.tmp_path, name, header, labels, rows)

    def test_two_groups_a_b_produce_two_curve_window_with_log_rank(self):
        path_a = self._make_file("a.xlsx", None, "⊗ 29.03")  # death
        path_b = self._make_file("b.xlsx", None, "1.2-1.2-1.2")  # alive

        window = self._build_window_stub(
            selected_paths=[path_a, path_b],
            group_assignment={path_a: "A", path_b: "B"},
        )

        MainWindow.handle_kaplan_meier(window)

        self.assertIsInstance(window.kaplan_meier_window, KaplanMeierWindow)
        table = window.kaplan_meier_window.summary_table
        self.assertEqual(table.rowCount(), 2)
        group_names = {table.item(row, 0).text() for row in range(table.rowCount())}
        self.assertEqual(group_names, {"Группа A", "Группа B"})
        self.assertFalse(window.kaplan_meier_window.plot_label.pixmap().isNull())
        self.assertIn("Лог-ранговый тест", window.kaplan_meier_window.log_rank_label.text())

    def test_no_group_assignment_produces_single_curve(self):
        path_a = self._make_file("solo.xlsx", None, "1.2-1.2-1.2")
        window = self._build_window_stub(selected_paths=[path_a], group_assignment={})

        MainWindow.handle_kaplan_meier(window)

        table = window.kaplan_meier_window.summary_table
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 0).text(), "Выбранные эксперименты")

    def test_reuses_existing_window_instance_on_second_call(self):
        path_a = self._make_file("reuse.xlsx", None, "1.2-1.2-1.2")
        window = self._build_window_stub(selected_paths=[path_a], group_assignment={})

        MainWindow.handle_kaplan_meier(window)
        first_instance = window.kaplan_meier_window
        MainWindow.handle_kaplan_meier(window)

        self.assertIs(window.kaplan_meier_window, first_instance)

    def test_empty_selection_shows_message_box_and_does_not_create_window(self):
        window = self._build_window_stub(selected_paths=[], group_assignment={})
        from unittest.mock import patch
        with patch.object(QMessageBox, "information") as mock_info:
            MainWindow.handle_kaplan_meier(window)
        mock_info.assert_called_once()
        self.assertIsNone(window.kaplan_meier_window)


if __name__ == "__main__":
    unittest.main()
