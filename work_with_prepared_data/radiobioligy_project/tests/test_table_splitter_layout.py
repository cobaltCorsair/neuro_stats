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

from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtTest import QTest

from gui.main_window import MainWindow


class TableSplitterAutoFitTests(unittest.TestCase):
    """Регрессия: tableView.setMinimumHeight(desired), выставленный без учёта реального
    минимума нижней панели (frame.minimumHeight()), заставлял таблицу физически перекрывать
    график/кнопки, когда сплиттеру не хватало высоты — вместо скролла в таблице. Проявлялось
    при добавлении достаточного числа экспериментов и/или после ручного ужатия окна."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        QTest.qWait(50)

    def _add_fake_files(self, n, prefix="file"):
        fake_files = [fr"V:\fake\{prefix}_{i}.xlsx" for i in range(n)]
        with patch.object(QFileDialog, "getOpenFileNames", return_value=(fake_files, "")):
            self.window.open_files()
        QTest.qWait(150)

    def _assert_no_overlap(self):
        min_bottom = self.window.frame.minimumHeight()
        total = self.window.splitter_2.height()
        self.assertGreater(total, 0)
        self.assertLessEqual(
            self.window.tableView.minimumHeight() + min_bottom, total,
            "таблица форсирует высоту больше, чем сплиттер способен выдать без сжатия "
            "графика ниже его минимума — визуально перекрывает нижнюю панель"
        )

    def test_no_overlap_at_default_size_with_many_rows(self):
        self._add_fake_files(8)
        self._assert_no_overlap()

    def test_window_grows_when_rows_dont_fit_at_current_size(self):
        height_before = self.window.height()
        self._add_fake_files(8)
        self.assertGreaterEqual(self.window.height(), height_before)

    def test_manual_shrink_after_adding_rows_reclamps_table_without_overlap(self):
        self._add_fake_files(8)
        # Пользователь вручную ужимает окно до минимума приложения — раньше
        # tableView.setMinimumHeight, выставленный при добавлении файлов, оставался
        # завышенным и не пересчитывался, что и приводило к перекрытию.
        self.window.resize(1150, 650)
        QTest.qWait(50)
        self._assert_no_overlap()

    def test_frame_never_squeezed_below_its_own_minimum(self):
        self._add_fake_files(8)
        self.window.resize(1150, 650)
        QTest.qWait(50)
        self.assertGreaterEqual(self.window.frame.height(), self.window.frame.minimumHeight())


if __name__ == "__main__":
    unittest.main()
