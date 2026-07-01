import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
# ВАЖНО: main_window.py импортирует graph_manager длинным путём
# (work_with_prepared_data.radiobioligy_project.gui) — короткий "from gui import graph_manager"
# дал бы ВТОРОЙ, независимый экземпляр модуля с собственным глобальным состоянием (Python
# кеширует модули по полному импортируемому пути в sys.modules), и тест проверял бы не тот
# флаг, который реально читает main_window.py.
from work_with_prepared_data.radiobioligy_project.gui import graph_manager


class TestRtogCheckboxActivation(unittest.TestCase):
    """checkBox_rtog должен включаться/выключаться по тем же принципам, что и остальные
    опции (AUC, критерии), завязанные на set_state_of_auc_and_tests_checkbox — то есть
    только когда реально доступен просмотр УСРЕДНЁННЫХ кожных реакций (одна группа или
    сравнение групп через pushButton_2/pushButton_4), а не для объёмов опухоли и не для
    индивидуальных кривых ("общие"). Тесты гоняют РЕАЛЬНЫЙ каскад сигналов чекбоксов
    (on_checkbox_pair_changed -> update_*_button_state -> set_state_of_auc_and_tests_checkbox),
    а не вызывают set_state_of_auc_and_tests_checkbox в отрыве от состояния кнопок — иначе
    промежуточные срабатывания сигналов перезаписывают состояние кнопок по реальному
    (пустому в тесте) списку выбранных файлов."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self._selected_paths = []
        self.window.get_selected_experiments = lambda: list(self._selected_paths)

    def test_enabled_for_single_skin_reaction_group_with_mean_view(self):
        self._selected_paths = ["exp1_skin_reactions.xlsx"]
        self.window.checkBox_3.setChecked(True)  # "абс.ед"
        self.window.checkBox_6.setChecked(True)  # "средние"

        self.assertTrue(self.window.pushButton_2.isEnabled())
        self.assertTrue(self.window.checkBox_rtog.isEnabled())

    def test_enabled_for_skin_reaction_comparison_with_mean_view(self):
        self._selected_paths = ["exp1_skin_reactions.xlsx", "exp2_skin_reactions.xlsx"]
        self.window.checkBox_3.setChecked(True)
        self.window.checkBox_6.setChecked(True)

        self.assertTrue(self.window.pushButton_4.isEnabled())
        self.assertTrue(self.window.checkBox_rtog.isEnabled())

    def test_disabled_for_tumor_volume_views(self):
        self._selected_paths = ["exp1_tumor.xlsx", "exp2_tumor.xlsx"]
        self.window.checkBox_3.setChecked(True)
        self.window.checkBox_6.setChecked(True)

        self.assertTrue(self.window.pushButton_3.isEnabled())
        self.assertFalse(self.window.checkBox_rtog.isEnabled())

    def test_disabled_for_individual_curves_view_even_for_skin_reactions(self):
        self._selected_paths = ["exp1_skin_reactions.xlsx", "exp2_skin_reactions.xlsx"]
        self.window.checkBox_3.setChecked(True)
        self.window.checkBox_5.setChecked(True)  # "общие" (индивидуальные кривые), не "средние"

        # pushButton_4 включается уже по "абс.ед" + ("средние" ИЛИ "общие")...
        self.assertTrue(self.window.pushButton_4.isEnabled())
        # ...но RTOG привязан только к усреднённой кривой (checkBox_6), поэтому остаётся выключен
        self.assertFalse(self.window.checkBox_rtog.isEnabled())

    def test_becoming_disabled_also_unchecks_it(self):
        self._selected_paths = ["exp1_skin_reactions.xlsx"]
        self.window.checkBox_3.setChecked(True)
        self.window.checkBox_6.setChecked(True)
        self.assertTrue(self.window.checkBox_rtog.isEnabled())

        self.window.checkBox_rtog.setChecked(True)
        self.assertTrue(self.window.show_rtog)

        # checkBox_4 ("отн.ед") снимает checkBox_3 (пара) -> pushButton_2 требует именно
        # checkBox_3, значит выключается -> каскад должен выключить и снять checkBox_rtog
        self.window.checkBox_4.setChecked(True)

        self.assertFalse(self.window.pushButton_2.isEnabled())
        self.assertFalse(self.window.checkBox_rtog.isEnabled())
        self.assertFalse(self.window.checkBox_rtog.isChecked())
        self.assertFalse(self.window.show_rtog)


class TestHolmCorrectionCheckbox(unittest.TestCase):
    """
    Поправка Холма — отдельная опциональная функция (chekBox_holm), не встроена молча в
    checkBox_7/checkBox: пользователь должен явно видеть и контролировать её через свой
    собственный чекбокс, включённый по умолчанию (сохраняет прежнее поведение)."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        # graph_manager — модуль с глобальным состоянием; сбрасываем перед каждым тестом,
        # чтобы тесты не зависели от порядка выполнения.
        graph_manager.set_holm_correction_enabled(True)

    def test_mann_whitney_and_student_labels_are_unchanged(self):
        self.assertEqual(self.window.checkBox_7.text(), "Критерий Манна-Уитни")
        self.assertEqual(self.window.checkBox.text(), "Критерий Стьюдента")

    def test_holm_checkbox_label_and_default_checked(self):
        self.assertEqual(self.window.checkBox_holm.text(), "Поправка Холма")
        self.assertTrue(self.window.checkBox_holm.isChecked())

    def test_holm_checkbox_tooltip_explains_markers(self):
        tooltip = self.window.checkBox_holm.toolTip()
        self.assertIn("Холма", tooltip)
        self.assertIn("(*)", tooltip)

    def test_toggling_checkbox_updates_graph_manager_flag(self):
        self.window.checkBox_holm.setChecked(False)
        self.assertFalse(graph_manager.is_holm_correction_enabled())

        self.window.checkBox_holm.setChecked(True)
        self.assertTrue(graph_manager.is_holm_correction_enabled())

    def test_disabled_by_default_before_any_selection(self):
        self.assertFalse(self.window.checkBox_holm.isEnabled())

    def test_enabled_when_mann_whitney_active(self):
        self._selected_paths = ["exp1_tumor.xlsx", "exp2_tumor.xlsx"]
        self.window.get_selected_experiments = lambda: list(self._selected_paths)
        self.window.checkBox_3.setChecked(True)
        self.window.checkBox_6.setChecked(True)
        self.assertTrue(self.window.pushButton_3.isEnabled())

        self.window.checkBox_7.setChecked(True)

        self.assertTrue(self.window.checkBox_holm.isEnabled())

    def test_checked_state_preserved_when_temporarily_disabled(self):
        """В отличие от checkBox_rtog, checkBox_holm не сбрасывается в False при выключении —
        это глобальный флаг, который должен сохранять значение между переключениями режимов
        просмотра, а не обнуляться каждый раз, когда критерии временно недоступны."""
        self._selected_paths = ["exp1_tumor.xlsx", "exp2_tumor.xlsx"]
        self.window.get_selected_experiments = lambda: list(self._selected_paths)
        self.window.checkBox_3.setChecked(True)
        self.window.checkBox_6.setChecked(True)
        self.window.checkBox_7.setChecked(True)
        self.assertTrue(self.window.checkBox_holm.isEnabled())
        self.assertTrue(self.window.checkBox_holm.isChecked())

        # Переключаемся в режим, где критерии недоступны (объёмы одной группы)
        self._selected_paths = ["exp1_tumor.xlsx"]
        self.window.checkBox_6.setChecked(False)
        self.window.checkBox_6.setChecked(True)  # перетриггерить каскад с новым selected_paths

        self.assertFalse(self.window.checkBox_holm.isEnabled())
        self.assertTrue(self.window.checkBox_holm.isChecked())  # не сброшен


class TestShowDateInLegendCheckbox(unittest.TestCase):
    """Показ даты в легенде — отдельная опциональная функция (checkBox_show_date),
    включена по умолчанию (сохраняет прежнее поведение)."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        graph_manager.set_show_date_in_legend(True)

    def test_label_and_default_checked(self):
        self.assertEqual(self.window.checkBox_show_date.text(), "Дата в легенде")
        self.assertTrue(self.window.checkBox_show_date.isChecked())

    def test_toggling_checkbox_updates_graph_manager_flag(self):
        self.window.checkBox_show_date.setChecked(False)
        self.assertFalse(graph_manager.is_show_date_in_legend())

        self.window.checkBox_show_date.setChecked(True)
        self.assertTrue(graph_manager.is_show_date_in_legend())

    def test_always_enabled_unlike_holm_and_rtog(self):
        # В отличие от checkBox_holm/checkBox_rtog, показ даты не зависит от типа графика
        # или активных критериев — должен быть доступен всегда.
        self.assertTrue(self.window.checkBox_show_date.isEnabled())


if __name__ == "__main__":
    unittest.main()
