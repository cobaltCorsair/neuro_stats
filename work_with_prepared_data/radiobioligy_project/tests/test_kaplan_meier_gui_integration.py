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
from work_with_prepared_data.radiobioligy_project.gui.scaling_image_label import (
    ScalingImageLabel,
    fit_plot_frame_to_available_space,
    measure_other_content_height,
)


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
        # вызывает настоящую реализацию из MainWindow, привязанную к этой заглушке как self —
        # handle_kaplan_meier делегирует построение групп туда же, что и калькулятор
        window.build_kaplan_meier_groups_from_selection = (
            lambda: MainWindow.build_kaplan_meier_groups_from_selection(window)
        )
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

    def test_single_file_without_group_is_named_by_its_filename(self):
        path_a = self._make_file("solo.xlsx", None, "1.2-1.2-1.2")
        window = self._build_window_stub(selected_paths=[path_a], group_assignment={})

        MainWindow.handle_kaplan_meier(window)

        table = window.kaplan_meier_window.summary_table
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 0).text(), "solo")

    def test_two_files_without_group_become_two_separate_curves(self):
        # Регрессия: ранее несколько выбранных файлов БЕЗ явной группы A/B сливались
        # в одну общую кривую "Выбранные эксперименты" — пользователь ожидал, что
        # просто отмеченные чекбоксом файлы дадут отдельные кривые для сравнения.
        path_a = self._make_file("first.xlsx", None, "⊗ 29.03")
        path_b = self._make_file("second.xlsx", None, "1.2-1.2-1.2")
        window = self._build_window_stub(selected_paths=[path_a, path_b], group_assignment={})

        MainWindow.handle_kaplan_meier(window)

        table = window.kaplan_meier_window.summary_table
        self.assertEqual(table.rowCount(), 2)
        names = {table.item(row, 0).text() for row in range(table.rowCount())}
        self.assertEqual(names, {"first", "second"})
        # ровно 2 группы -> лог-ранг должен посчитаться, а не сказать "нужно ровно 2"
        self.assertIn("Лог-ранговый тест", window.kaplan_meier_window.log_rank_label.text())

    def test_one_grouped_file_and_one_ungrouped_file_mix_correctly(self):
        path_a = self._make_file("grouped.xlsx", None, "⊗ 29.03")
        path_b = self._make_file("standalone.xlsx", None, "1.2-1.2-1.2")
        window = self._build_window_stub(
            selected_paths=[path_a, path_b], group_assignment={path_a: "A"})

        MainWindow.handle_kaplan_meier(window)

        table = window.kaplan_meier_window.summary_table
        names = {table.item(row, 0).text() for row in range(table.rowCount())}
        self.assertEqual(names, {"Группа A", "standalone"})

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


class TestRiskTableDashBeyondGroupRange(unittest.TestCase):
    """
    Группы из разных файлов могут иметь разный реальный срок наблюдения. Таблица
    «число в риске» использует общую сетку времени (нужно для сравнения по столбцам),
    но за пределами собственного максимума группы должна показывать «—», а не «0» —
    иначе выглядит так, будто обе группы наблюдались одинаково долго.
    """

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_shorter_group_shows_dash_past_its_own_last_day(self):
        from data_processing.excel_data_processor import RatSurvivalEvent

        long_group = [RatSurvivalEvent("L1", 30.0, False, "")]
        short_group = [RatSurvivalEvent("S1", 6.0, True, "death")]

        win = KaplanMeierWindow()
        win.set_groups({"Long": long_group, "Short": short_group})

        headers = [win.risk_table.horizontalHeaderItem(c).text() for c in range(win.risk_table.columnCount())]
        self.assertEqual(headers[-1], "30")

        rows = {
            win.risk_table.item(r, 0).text(): [win.risk_table.item(r, c).text() for c in range(1, win.risk_table.columnCount())]
            for r in range(win.risk_table.rowCount())
        }
        self.assertEqual(rows["Long"][-1], "1")
        self.assertEqual(rows["Short"][-1], "—")


class TestScalingImageLabel(unittest.TestCase):
    """
    setFixedSize() на лейбле графика не реагирует на последующий resize окна — при
    сужении картинка обрезалась вместо уменьшения. ScalingImageLabel хранит исходный
    pixmap и пересчитывает масштаб (с сохранением пропорций) на каждый resizeEvent.
    """

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_shrinking_the_label_scales_pixmap_down_without_cropping(self):
        from PyQt6.QtGui import QPixmap

        label = ScalingImageLabel()
        original = QPixmap(1000, 600)
        original.fill()
        label.set_original_pixmap(original)
        label.show()
        label.resize(400, 300)
        self._app.processEvents()

        displayed = label.pixmap()
        self.assertLessEqual(displayed.width(), 400)
        self.assertLessEqual(displayed.height(), 300)
        # пропорции исходного изображения сохраняются (10:6 = 5:3)
        self.assertAlmostEqual(displayed.width() / displayed.height(), 1000 / 600, places=1)

    def test_growing_the_label_scales_pixmap_up(self):
        from PyQt6.QtGui import QPixmap

        label = ScalingImageLabel()
        original = QPixmap(400, 300)
        original.fill()
        label.set_original_pixmap(original)
        label.show()
        label.resize(1200, 900)
        self._app.processEvents()

        displayed = label.pixmap()
        self.assertGreater(displayed.width(), 400)

    def test_size_hint_matches_original_pixmap_for_initial_window_sizing(self):
        from PyQt6.QtGui import QPixmap

        label = ScalingImageLabel()
        original = QPixmap(1000, 600)
        original.fill()
        label.set_original_pixmap(original)

        self.assertEqual(label.sizeHint(), original.size())

    def test_clearing_pixmap_does_not_reappear_after_resize(self):
        from PyQt6.QtGui import QPixmap

        label = ScalingImageLabel()
        original = QPixmap(1000, 600)
        original.fill()
        label.set_original_pixmap(original)
        label.set_original_pixmap(None)
        label.show()
        label.resize(500, 400)
        self._app.processEvents()

        self.assertTrue(label.pixmap() is None or label.pixmap().isNull())


class TestFitPlotFrameToAvailableSpace(unittest.TestCase):
    """
    Рамка с stretch=1 растягивается на всю ширину диалога независимо от пропорций
    картинки внутри — при сужении окна по высоте (но не по ширине) это оставляло
    пустые поля внутри рамки вокруг уменьшенной (с сохранением пропорций) картинки.
    fit_plot_frame_to_available_space пересчитывает размер рамки так, чтобы она
    точно облегала текущий масштаб картинки, без пустых полей.
    """

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @staticmethod
    def _build_dialog(other_content_height=604):
        from PyQt6.QtWidgets import QDialog, QFrame, QLabel as QLabelWidget, QVBoxLayout

        dialog = QDialog()
        layout = QVBoxLayout(dialog)
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(12, 12, 12, 12)
        label = ScalingImageLabel()
        frame_layout.addWidget(label)
        layout.addWidget(frame, stretch=1)
        other = QLabelWidget()
        other.setFixedHeight(other_content_height)
        layout.addWidget(other)
        return dialog, frame, label

    @staticmethod
    def _waste(frame, label):
        margin = 24
        pixmap = label.pixmap()
        return (
            frame.width() - margin - pixmap.width(),
            frame.height() - margin - pixmap.height(),
        )

    def test_frame_tightly_wraps_image_when_window_is_wide_and_short(self):
        from PyQt6.QtGui import QPixmap

        dialog, frame, label = self._build_dialog()
        label.set_original_pixmap(QPixmap(1000, 600))
        other_h = measure_other_content_height(dialog, frame)
        dialog.show()

        dialog.resize(1310, 1100)  # шире и при этом ниже "естественного" размера
        self._app.processEvents()
        fit_plot_frame_to_available_space(dialog, frame, label, other_h)
        self._app.processEvents()

        waste_w, waste_h = self._waste(frame, label)
        self.assertLessEqual(abs(waste_w), 5)
        self.assertLessEqual(abs(waste_h), 5)

    def test_frame_tightly_wraps_image_when_window_is_tall_and_narrow(self):
        from PyQt6.QtGui import QPixmap

        dialog, frame, label = self._build_dialog()
        label.set_original_pixmap(QPixmap(1000, 600))
        other_h = measure_other_content_height(dialog, frame)
        dialog.show()

        dialog.resize(700, 1300)
        self._app.processEvents()
        fit_plot_frame_to_available_space(dialog, frame, label, other_h)
        self._app.processEvents()

        waste_w, waste_h = self._waste(frame, label)
        self.assertLessEqual(abs(waste_w), 5)
        self.assertLessEqual(abs(waste_h), 5)

    def test_frame_never_shrinks_below_labels_own_minimum(self):
        """
        Регрессия: при экстремальном сжатии расчётный размер рамки уходил НИЖЕ
        minimumSize самого label, и Qt всё равно не давал label сжаться — картинка
        оказывалась больше рамки (вылезала за границу).
        """
        from PyQt6.QtGui import QPixmap

        dialog, frame, label = self._build_dialog()
        label.set_original_pixmap(QPixmap(1000, 600))
        other_h = measure_other_content_height(dialog, frame)
        dialog.show()

        dialog.resize(550, 500)  # ниже минимума диалога - будет clamped, всё равно проверяем
        self._app.processEvents()
        fit_plot_frame_to_available_space(dialog, frame, label, other_h)
        self._app.processEvents()

        margin = 24
        label_min = label.minimumSize()
        self.assertGreaterEqual(frame.width() - margin, label_min.width())
        self.assertGreaterEqual(frame.height() - margin, label_min.height())

        pixmap = label.pixmap()
        self.assertLessEqual(pixmap.width(), frame.width() - margin)
        self.assertLessEqual(pixmap.height(), frame.height() - margin)

    def test_measure_other_content_height_matches_fixed_sibling_height(self):
        from PyQt6.QtGui import QPixmap

        dialog, frame, label = self._build_dialog(other_content_height=400)
        label.set_original_pixmap(QPixmap(1000, 600))
        dialog.show()

        other_h = measure_other_content_height(dialog, frame)
        # допускаем небольшой запас на отступы/рамки самого layout'а диалога
        self.assertAlmostEqual(other_h, 400, delta=40)

    def test_does_nothing_when_other_content_height_is_none(self):
        from PyQt6.QtGui import QPixmap

        dialog, frame, label = self._build_dialog()
        label.set_original_pixmap(QPixmap(1000, 600))
        dialog.show()
        before = frame.size()

        fit_plot_frame_to_available_space(dialog, frame, label, None)

        self.assertEqual(frame.size(), before)


if __name__ == "__main__":
    unittest.main()
