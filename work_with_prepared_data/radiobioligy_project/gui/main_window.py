import io
import numbers
import subprocess
import sys
import os
from pathlib import Path

import pandas as pd

# Добавляем директорию radiobioligy_project/ в sys.path, чтобы модули
# с голыми импортами (controls.py, draw_base_graphs.py и др.) находили друг друга
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from PyQt6.QtCore import QFileInfo, Qt, QUrl, QTimer
from PyQt6.QtGui import QAction, QDesktopServices, QStandardItemModel, QStandardItem, QPixmap, QPalette, QColor, QBrush
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView, QSizePolicy, QVBoxLayout, QLabel, \
    QTableWidget, QTableWidgetItem, QMessageBox, QButtonGroup, QComboBox, QListView, QStyledItemDelegate, \
    QStyle, QStyleOptionViewItem, QInputDialog

# Импорт сгенерированного класса из gui.py
from work_with_prepared_data.radiobioligy_project.gui.gui import Ui_MainWindow
from work_with_prepared_data.radiobioligy_project.controls import ControlGroupVisualizer
from work_with_prepared_data.radiobioligy_project.data_processing.rat_manager import register_rat_labels, \
    get_rat_labels, clear_rat_labels, rat_labels_with_indices
from work_with_prepared_data.radiobioligy_project.draw_abs_rel_graph_compare import TumorDataComparatorAdvanced
from work_with_prepared_data.radiobioligy_project.draw_base_graphs import TumorDataVisualizer
from work_with_prepared_data.radiobioligy_project.draw_base_graphs_compare import TumorDataComparator
from work_with_prepared_data.radiobioligy_project.gui import graph_manager
from work_with_prepared_data.radiobioligy_project.skin_reactions_base_grapf import SkinReactionsVisualizer
from work_with_prepared_data.radiobioligy_project.stats_methods.support_stats_methods import ExtractOutliers
from work_with_prepared_data.radiobioligy_project.gui.checkable_combobox import CheckableComboBox
from work_with_prepared_data.radiobioligy_project.gui.legend_window import LegendManager
from work_with_prepared_data.radiobioligy_project.gui.tgi_table_window import TumorGrowthInhibitionTableWindow
from work_with_prepared_data.radiobioligy_project.gui.skin_reaction_summary_window import SkinReactionSummaryWindow

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')  # Установка бэкенда до импорта pyplot (автоматически выберет Qt5 или Qt6).


class DataProcessor:
    def __init__(self):
        pass

    def process_for_single_experiment(self, selected_path, checkboxes_state):
        # Определение функции визуализации на основе состояния чекбоксов
        if checkboxes_state == (True, False, True, False):
            plotting_func = TumorDataVisualizer.plot_tumor_volumes_single_graph
        elif checkboxes_state == (False, True, True, False):
            plotting_func = TumorDataVisualizer.plot_relative_tumor_volumes_single_graph
        elif checkboxes_state == (True, False, False, True):
            plotting_func = TumorDataVisualizer.plot_mean_tumor_volume
        elif checkboxes_state == (False, True, False, True):
            plotting_func = TumorDataVisualizer.plot_average_relative_tumor_volume
        else:
            raise ValueError("Invalid checkbox state")
        return plotting_func, selected_path

    def process_skin_reactions(self, selected_path, checkboxes_state):
        # Проверяем тип: строка (один файл) или список (несколько файлов)
        is_single_file = isinstance(selected_path, str)
        is_skin_reactions = False

        if is_single_file:
            is_skin_reactions = "skin_reactions" in selected_path
            # Преобразуем в список для единообразия
            path_list = [selected_path]
        else:
            is_skin_reactions = all("skin_reactions" in path for path in selected_path)
            path_list = selected_path

        if not is_skin_reactions:
            raise ValueError("Invalid checkbox state or name")

        # "абс.ед" + "индивидуальные" -> индивидуальные кривые (для любого количества файлов)
        if checkboxes_state == (True, False, True, False):
            plotting_func = SkinReactionsVisualizer.plot_all_individual_curves
            return plotting_func, path_list

        # "абс.ед" + "средние" -> усреднённая кривая (один файл) или сравнение экспериментов (несколько)
        elif checkboxes_state == (True, False, False, True):
            if is_single_file:
                plotting_func = SkinReactionsVisualizer.plot_mean_skin_reactions
                return plotting_func, selected_path  # Для одного файла возвращаем строку
            else:
                plotting_func = SkinReactionsVisualizer.plot_multiple_experiments
                return plotting_func, path_list
        else:
            raise ValueError("Invalid checkbox state or name")

    def process_for_comparison(self, selected_paths, checkboxes_state):
        if checkboxes_state == (True, False, True, False):
            plotting_func = TumorDataComparatorAdvanced.compare_mean_volumes
        elif checkboxes_state == (False, True, True, False):
            plotting_func = TumorDataComparatorAdvanced.compare_relative_volumes
        elif checkboxes_state == (True, False, False, True):
            plotting_func = TumorDataComparator.compare_tumor_volumes
        elif checkboxes_state == (False, True, False, True):
            plotting_func = TumorDataComparator.compare_relative_tumor_volumes
        else:
            raise ValueError("Invalid checkbox state")
        return plotting_func, selected_paths

    def process_variability(self, checkboxes_state):
        """
        Возвращает функцию визуализации межособевой вариабельности.

        Args:
            checkboxes_state (tuple): (CB3, CB4, CB5, CB6) — состояния чекбоксов.

        Returns:
            callable: Метод TumorDataVisualizer для построения графика.
        """
        _, _, all_curves, mean_curves = checkboxes_state
        # checkBox_5 = «общие» → расхождение на крысу со средним и горизонтальной линией
        # checkBox_6 = «средние» → CV(t) по группе
        if all_curves:
            return TumorDataVisualizer.plot_relative_divergence_per_rat
        elif mean_curves:
            return TumorDataVisualizer.plot_cv
        else:
            raise ValueError("Выберите режим отображения: «общие» или «средние» (CV)")

    def process_for_control_comparison(self, selected_paths, control_path, checkboxes_state):
        if checkboxes_state == (True, True):
            plotting_func = TumorDataComparatorAdvanced.compare_control_and_experiment
            control_visualizer = ControlGroupVisualizer(control_path)
        elif checkboxes_state == (False, True):
            plotting_func = TumorDataComparatorAdvanced.compare_tumor_growth_inhibition_with_multiple_experiments
            control_visualizer = ControlGroupVisualizer(control_path)
        else:
            raise ValueError("Invalid checkbox state")

        return plotting_func, selected_paths, control_visualizer


class ComboPopupItemDelegate(QStyledItemDelegate):
    """РёСЃРєР»СЋС‡Р°РµС‚ РЅР°С‚РёРІРЅСѓСЋ С‡С‘СЂРЅСѓСЋ РїРѕРґСЃРІРµС‚РєСѓ РІ popup-СЃРїРёСЃРєР°С… QComboBox."""

    _base_color = QColor('#F0F5FA')
    _text_color = QColor('#243040')
    _highlight_color = QColor('#C5D9EE')
    _highlight_text_color = QColor('#1A3050')

    def paint(self, painter, option, index):
        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)

        is_highlighted = bool(
            item_option.state & QStyle.StateFlag.State_MouseOver
            or item_option.state & QStyle.StateFlag.State_Selected
        )
        background = self._highlight_color if is_highlighted else self._base_color
        foreground = self._highlight_text_color if is_highlighted else self._text_color

        painter.fillRect(item_option.rect, background)
        item_option.backgroundBrush = QBrush(background)
        item_option.palette.setColor(QPalette.ColorRole.Base, background)
        item_option.palette.setColor(QPalette.ColorRole.Window, background)
        item_option.palette.setColor(QPalette.ColorRole.Text, foreground)
        item_option.palette.setColor(QPalette.ColorRole.WindowText, foreground)
        item_option.state &= ~QStyle.StateFlag.State_Selected
        item_option.state &= ~QStyle.StateFlag.State_MouseOver
        item_option.state &= ~QStyle.StateFlag.State_HasFocus
        super().paint(painter, item_option, index)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height() + 8, 30))
        return size


class MainWindow(QMainWindow, Ui_MainWindow):
    """Главное окно приложения.

    Attributes:
        ax: Объект осей для рисования графиков matplotlib.
        canvas: Холст matplotlib для интеграции графика в приложение Qt.
        figure: Объект фигуры matplotlib для рисования.
    """

    def __init__(self, parent=None):
        """Инициализация главного окна приложения."""
        super(MainWindow, self).__init__(parent)
        self.control_path = None
        self.control_groups = {}  # Словарь {file_path: control_type} для хранения типов контрольных групп
        self.ax = None
        self.canvas = None
        self.figure = None
        self.current_visualizer = None
        self.current_plotting_func = None
        self.current_selected_paths = []
        self.current_control = None
        self.selected_outlier_method = None
        self.current_plot_type = None
        self._paths_group_a = []
        self._paths_group_b = []
        self.saved_checked_items = []
        self.perform_stat_test = False
        self.use_ttest = False
        self.use_shapiro = False
        self.use_AUC = False
        self.show_rtog = False
        self.annotation_multiplier = 0
        self.data_processor = DataProcessor()
        self.legend_manager = LegendManager()
        self.fit_alpha_beta_window = None
        self.growth_predictor_window = None
        self.geant4_pipeline_window = None
        self.tumor_3d_viewer_window = None
        self.tgi_table_window = None
        self.skin_reaction_summary_window = None
        self.kaplan_meier_window = None
        self.kaplan_meier_calculator_window = None
        self.cached_visualizer = None  # Кеш для модифицированного визуализатора
        self.cache_key = None  # Ключ для проверки актуальности кеша
        self.show_legend_separately = False  # Флаг для отображения легенды отдельно
        self.setupUi(self)

        # Добавляем чекбокс "Нарисовать легенду отдельно"
        from PyQt6.QtWidgets import QCheckBox
        self.checkBox_separate_legend = QCheckBox("Легенда отдельно", self.centralwidget)
        self.checkBox_separate_legend.setObjectName("checkBox_separate_legend")
        # Вставляем чекбокс перед label_6 (положение основной легенды)
        label_index = self.horizontalLayout_7.indexOf(self.label_6)
        self.horizontalLayout_7.insertWidget(label_index, self.checkBox_separate_legend)

        # Чекбокс "Дата в легенде" — опционально скрывает "Дата: ..." из подписей
        # экспериментов в легенде графика (format_experiment_params). Включён по умолчанию
        # (сохраняет прежнее поведение).
        self.checkBox_show_date = QCheckBox("Дата в легенде", self.centralwidget)
        self.checkBox_show_date.setObjectName("checkBox_show_date")
        self.checkBox_show_date.setChecked(True)
        self.horizontalLayout_7.insertWidget(label_index + 1, self.checkBox_show_date)
        self.checkBox_show_date.stateChanged.connect(self.on_show_date_changed)

        # Чекбокс "Шкала RTOG" — рядом с остальными опциями графика кожных реакций
        # (checkBox_5/checkBox_6 в этом же layout).
        self.checkBox_rtog = QCheckBox("Шкала RTOG", self.centralwidget)
        self.checkBox_rtog.setObjectName("checkBox_rtog")
        self.horizontalLayout_10.addWidget(self.checkBox_rtog)
        self.checkBox_rtog.stateChanged.connect(self.on_rtog_changed)

        # Чекбокс "Поправка Холма" — отдельная опция, а не встроенное поведение. Включена по
        # умолчанию (сохраняет прежнее поведение для тех, кто ничего не трогает), но её можно
        # выключить, чтобы увидеть значимость по сырому p<0.05 без коррекции на множественность
        # поточечных сравнений. Вставляем сразу после checkBox (Критерий Стьюдента), перед
        # checkBox_shapiro — она относится только к Манна-Уитни/Стьюденту, не к Шапиро-Уилку.
        self.checkBox_holm = QCheckBox("Поправка Холма", self.centralwidget)
        self.checkBox_holm.setObjectName("checkBox_holm")
        self.checkBox_holm.setChecked(True)
        self.checkBox_holm.setToolTip(
            "Пошаговая поправка Холма на множественность поточечных сравнений по нескольким "
            "временным точкам в одном сравнении (контроль FWER).\n"
            "Включена: '*' — значимо после поправки Холма; "
            "'(*)' — значимо только по сырому p<0.05, поправку не прошло.\n"
            "Выключена: '*' — значимо по сырому p<0.05, без коррекции."
        )
        student_index = self.horizontalLayout.indexOf(self.checkBox)
        self.horizontalLayout.insertWidget(student_index + 1, self.checkBox_holm)
        self.checkBox_holm.stateChanged.connect(self.on_holm_correction_changed)

        self.tools_menu = self.menubar.addMenu("Инструменты")
        self.action_open_survival_fitter = QAction("LQ fitter и радиобиология", self)
        self.action_open_survival_fitter.triggered.connect(self.open_survival_fitter)
        self.tools_menu.addAction(self.action_open_survival_fitter)
        self.action_open_growth_predictor = QAction("Предсказание роста опухоли", self)
        self.action_open_growth_predictor.triggered.connect(self.open_growth_predictor)
        self.tools_menu.addAction(self.action_open_growth_predictor)
        self.action_open_geant4_pipeline = QAction("GEANT4 / RT Dose pipeline", self)
        self.action_open_geant4_pipeline.triggered.connect(self.open_geant4_pipeline)
        self.tools_menu.addAction(self.action_open_geant4_pipeline)
        self.action_open_tumor_3d_viewer = QAction("3D геометрия опухоли", self)
        self.action_open_tumor_3d_viewer.triggered.connect(self.open_tumor_3d_viewer)
        self.tools_menu.addAction(self.action_open_tumor_3d_viewer)
        self.action_open_kaplan_meier = QAction("Каплан-Майер (выживаемость)", self)
        self.action_open_kaplan_meier.triggered.connect(self.handle_kaplan_meier)
        self.tools_menu.addAction(self.action_open_kaplan_meier)
        self.action_open_kaplan_meier_calculator = QAction("Калькулятор Каплана-Майера (ручной ввод)", self)
        self.action_open_kaplan_meier_calculator.triggered.connect(self.open_kaplan_meier_calculator)
        self.tools_menu.addAction(self.action_open_kaplan_meier_calculator)
        self.action_skin_reaction_summary = QAction("Сводка кожных реакций (пик/длительность/нормализация)", self)
        self.action_skin_reaction_summary.triggered.connect(self.handle_skin_reaction_summary_table)
        self.tools_menu.addAction(self.action_skin_reaction_summary)
        self.action_about_docs = QAction("О программе", self)
        self.action_about_docs.triggered.connect(self.open_project_documentation)
        self.menubar.addAction(self.action_about_docs)
        self.action.triggered.connect(self.open_files)
        self.action_2.triggered.connect(self.save_graph)

        # Блокируем неиспользуемые кнопки меню
        self.action_4.setEnabled(False)    # Сохранить таблицу
        self.menu_2.setEnabled(False)      # Редактировать (весь выпадающий список)
        self.menu_3.setEnabled(False)      # Распознать (весь выпадающий список)

        # Настраиваем модель для 4 столбцов
        self.model = QStandardItemModel(0, 4, self)
        # Заменяем стандартный comboBox_4 на кастомный комбобокс с чекбоксами
        self.replace_combobox_4()
        self.change_table()
        # Кнопки по умолчанию неактивны
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_10.setEnabled(False)
        self.checkBox_2.setDisabled(True)
        self.checkBox_7.setDisabled(True)
        self.checkBox.setDisabled(True)
        self.checkBox_shapiro.setDisabled(True)
        self.checkBox_rtog.setDisabled(True)
        self.checkBox_holm.setDisabled(True)
        self.pushButton_4.setCheckable(False)
        self.pushButton_8.setCheckable(False)
        # Биндинг кнопок
        self.pushButton.clicked.connect(self.handle_all_of_rats)
        self.pushButton_2.clicked.connect(self.handle_skin_reactions)
        self.pushButton_3.clicked.connect(self.handle_compare_rats)
        # Связь сигнала изменения состояния кнопки с проверкой состояния чекбокса
        self.pushButton_3.clicked.connect(self.set_state_of_auc_and_tests_checkbox)
        self.pushButton_4.clicked.connect(self.handle_compare_skin_reactions)
        self.pushButton_5.clicked.connect(self.handle_compare_tumor_growth_inhibition)
        self.pushButton_6.clicked.connect(self.handle_tumor_growth_inhibition_table)
        self.pushButton_7.clicked.connect(self.handle_compare_with_control)
        self.pushButton_4.clicked.connect(self.handle_pushButton_4)
        self.pushButton_8.clicked.connect(self.handle_pushButton_8)
        self.pushButton_10.clicked.connect(self.handle_divergence_per_rat)
        # Подключение сигнала изменения выбора комбобокса к обработчику
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)
        # Подключаем сигналы изменения состояния чекбоксов
        self.checkBox_3.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_3, self.checkBox_4))
        self.checkBox_4.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_4, self.checkBox_3))
        self.checkBox_5.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_5, self.checkBox_6))
        self.checkBox_6.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_6, self.checkBox_5))
        self.checkBox_7.stateChanged.connect(lambda: self.on_checkbox_tests_changed(self.checkBox_7, self.checkBox))
        self.checkBox.stateChanged.connect(lambda: self.on_checkbox_tests_changed(self.checkBox, self.checkBox_7))
        self.checkBox_shapiro.stateChanged.connect(self.on_shapiro_changed)
        self.checkBox_2.stateChanged.connect(self.set_auc_checkbox)
        self.model.itemChanged.connect(self.update_first_button_state)
        self.model.itemChanged.connect(self.update_second_button_state)
        self.model.itemChanged.connect(self.update_third_button_state)
        self.model.itemChanged.connect(self.update_fourth_button_state)
        self.model.itemChanged.connect(self.update_fifth_button_state)
        self.model.itemChanged.connect(self.update_seventh_button_state)
        self.model.itemChanged.connect(self.update_tenth_button_state)
        self.model.itemChanged.connect(self.on_control_checkbox_changed)
        self.comboBox_2.currentTextChanged.connect(self.update_control_path)
        self.doubleSpinBox.valueChanged.connect(self.update_annotation_multiplier)
        self.comboBox_3.currentIndexChanged.connect(self.on_legend_position_changed)
        self.comboBox.currentIndexChanged.connect(self.update_doubleSpinBox_value)
        # Подключение новых кнопок для управления легендой
        self.pushButton_legend_window.clicked.connect(self.show_legend_preview)

        # Удаляем кнопку "Сохранить легенду" - она больше не нужна
        self.pushButton_save_legend.setVisible(False)

        # Подключение чекбокса "Легенда отдельно"
        self.checkBox_separate_legend.stateChanged.connect(self.on_separate_legend_changed)

        # Подключаем сигналы изменения модели таблицы к слоту
        self.model.rowsInserted.connect(self.on_table_data_changed)
        self.model.rowsRemoved.connect(self.on_table_data_changed)
        self.model.itemChanged.connect(self.on_table_data_changed)

        self.action_3.triggered.connect(self.edit_experiment_files)

        self._apply_stylesheet()

    @staticmethod
    def _fix_combo_palette(combo: QComboBox):
        """Устанавливает правильные цвета выпадающего списка через палитру (QSS не работает для hover)."""
        combo.setView(QListView(combo))
        view = combo.view()
        view.setMouseTracking(True)
        view.setSpacing(0)
        view.setUniformItemSizes(True)
        view.setAutoFillBackground(True)
        view.viewport().setAutoFillBackground(True)
        view.viewport().setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        view.setItemDelegate(ComboPopupItemDelegate(view))
        palette = view.palette()
        combo_colors = {
            QPalette.ColorRole.Base: QColor('#F0F5FA'),
            QPalette.ColorRole.AlternateBase: QColor('#F0F5FA'),
            QPalette.ColorRole.Window: QColor('#F0F5FA'),
            QPalette.ColorRole.Text: QColor('#243040'),
            QPalette.ColorRole.WindowText: QColor('#243040'),
            QPalette.ColorRole.ButtonText: QColor('#243040'),
            QPalette.ColorRole.Highlight: QColor('#C5D9EE'),
            QPalette.ColorRole.HighlightedText: QColor('#1A3050'),
        }
        for color_group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            for color_role, color in combo_colors.items():
                palette.setColor(color_group, color_role, color)
        view.setPalette(palette)
        view.viewport().setPalette(palette)
        view.setStyleSheet("""
            QAbstractItemView {
                background-color: #F0F5FA;
                color: #243040;
                border: 1px solid #AABBCC;
                outline: 0;
                selection-background-color: #C5D9EE;
                selection-color: #1A3050;
            }
            QAbstractItemView::item {
                background-color: #F0F5FA;
                color: #243040;
                padding: 4px 8px;
                min-height: 30px;
            }
            QAbstractItemView::item:hover,
            QAbstractItemView::item:selected,
            QAbstractItemView::item:selected:active,
            QAbstractItemView::item:selected:!active {
                background-color: #C5D9EE;
                color: #1A3050;
            }
        """)

    def _apply_stylesheet(self):
        """Применяет строгий стеклянный стиль ко всему главному окну."""
        self.setStyleSheet("""
            /* ── Фон окна — холодный нейтральный ── */
            QMainWindow {
                background-color: #CDD5DF;
            }
            QWidget {
                background-color: #CDD5DF;
                font-family: "Segoe UI", "Arial", sans-serif;
                font-size: 13px;
                color: #1C2733;
            }

            /* ── Менюбар — тёмное стекло ── */
            QMenuBar {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2C3E52, stop:1 #243344
                );
                color: #C8D6E5;
                padding: 2px 4px;
                spacing: 0px;
                border-bottom: 1px solid #1A2634;
            }
            QMenuBar::item {
                background: transparent;
                padding: 5px 16px;
                letter-spacing: 0.3px;
            }
            QMenuBar::item:selected {
                background: rgba(255, 255, 255, 0.12);
                color: #FFFFFF;
            }
            QMenu {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(245,248,252,0.98), stop:1 rgba(232,238,246,0.98)
                );
                border: 1px solid #B0BDC8;
                border-radius: 4px;
                padding: 3px 0;
            }
            QMenu::item {
                padding: 6px 24px 6px 16px;
                color: #1C2733;
            }
            QMenu::item:selected {
                background: rgba(74, 115, 160, 0.15);
                color: #1A3550;
            }
            QMenu::item:disabled {
                color: #9AAAB8;
            }
            QMenu::separator {
                height: 1px;
                background: #CBD5DF;
                margin: 3px 8px;
            }

            /* ── Кнопки — матовое стекло ── */
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.82),
                    stop:1 rgba(220,230,242,0.75)
                );
                border: 1px solid rgba(160,180,200,0.70);
                border-bottom: 1px solid rgba(130,155,180,0.80);
                border-radius: 5px;
                padding: 6px 12px;
                color: #243040;
                font-weight: 500;
                text-align: left;
            }
            QPushButton:hover:enabled {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.95),
                    stop:1 rgba(210,228,248,0.90)
                );
                border-color: rgba(80,130,190,0.75);
                color: #1A3050;
            }
            QPushButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(195,215,238,0.90),
                    stop:1 rgba(215,230,248,0.85)
                );
                border-color: rgba(70,115,170,0.80);
            }
            QPushButton:disabled {
                background: rgba(200,210,220,0.40);
                border-color: rgba(160,175,190,0.40);
                color: #8A9BAB;
            }

            /* ── Чекбоксы ── */
            QCheckBox {
                spacing: 6px;
                color: #243040;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid rgba(120,150,180,0.80);
                border-radius: 3px;
                background: rgba(255,255,255,0.75);
            }
            QCheckBox::indicator:hover {
                border-color: rgba(70,115,170,0.90);
                background: rgba(255,255,255,0.90);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5A8FC0, stop:1 #3A6A9A
                );
                border-color: #2E5A88;
            }
            QCheckBox:disabled {
                color: #8A9BAB;
            }
            QCheckBox::indicator:disabled {
                background: rgba(190,200,210,0.45);
                border-color: rgba(150,165,180,0.45);
            }

            /* ── Комбобоксы — матовое стекло ── */
            QComboBox {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.82),
                    stop:1 rgba(220,230,242,0.75)
                );
                border: 1px solid rgba(150,170,195,0.70);
                border-radius: 4px;
                padding: 3px 8px;
                color: #243040;
                min-height: 24px;
            }
            QComboBox:hover {
                border-color: rgba(70,115,170,0.80);
                background: rgba(255,255,255,0.92);
            }
            QComboBox:disabled {
                background: rgba(200,210,220,0.40);
                color: #8A9BAB;
                border-color: rgba(160,175,190,0.40);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #F0F5FA;
                border: 1px solid #AABBCC;
                outline: 0;
                selection-background-color: #C5D9EE;
                selection-color: #1A3050;
            }

            /* ── Спинбоксы ── */
            QDoubleSpinBox {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.82),
                    stop:1 rgba(220,230,242,0.75)
                );
                border: 1px solid rgba(150,170,195,0.70);
                border-radius: 4px;
                padding: 3px 6px;
                color: #243040;
                min-height: 24px;
            }
            QDoubleSpinBox:hover {
                border-color: rgba(70,115,170,0.80);
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                border: none;
                background: transparent;
                width: 14px;
            }

            /* ── Метки ── */
            QLabel {
                color: #364656;
                background: transparent;
            }

            /* ── Таблица ── */
            QTableView {
                background: rgba(255,255,255,0.78);
                alternate-background-color: rgba(235,242,250,0.65);
                border: 1px solid rgba(150,170,195,0.60);
                border-radius: 6px;
                gridline-color: rgba(180,200,220,0.45);
                selection-background-color: rgba(74,115,160,0.20);
                selection-color: #1A2D40;
                outline: 0;
            }
            QTableView::item {
                padding: 3px 8px;
                border: none;
                color: #1C2733;
            }
            QHeaderView::section {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3A5068, stop:1 #2C3E52
                );
                color: #C8D8E8;
                padding: 6px 8px;
                border: none;
                border-right: 1px solid rgba(255,255,255,0.08);
                font-weight: 600;
                font-size: 12px;
                letter-spacing: 0.2px;
            }

            /* ── Фрейм с графиком ── */
            QFrame#frame {
                background: rgba(255,255,255,0.80);
                border: 1px solid rgba(150,170,195,0.55);
                border-radius: 6px;
            }

            /* ── Сплиттер ── */
            QSplitter::handle {
                background: rgba(130,155,180,0.35);
            }
            QSplitter::handle:horizontal { width: 2px; }
            QSplitter::handle:vertical   { height: 2px; }

            /* ── Статусбар ── */
            QStatusBar {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2C3E52, stop:1 #243344
                );
                color: #7A9AB8;
                font-size: 11px;
                border-top: 1px solid #1A2634;
            }

            /* ── Тонкие скроллбары ── */
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(100,130,160,0.45);
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(80,115,155,0.70);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

            QScrollBar:horizontal {
                background: transparent;
                height: 6px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: rgba(100,130,160,0.45);
                border-radius: 3px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(80,115,155,0.70);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

            /* ── Разделители VLine/HLine ── */
            QFrame[frameShape="5"], QFrame[frameShape="6"] {
                color: rgba(130,155,180,0.50);
            }

            /* ── Чекбоксы внутри таблицы (QStandardItem) ── */
            QAbstractItemView::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid rgba(110,140,175,0.75);
                border-radius: 3px;
                background: rgba(255,255,255,0.80);
            }
            QAbstractItemView::indicator:unchecked {
                background: rgba(255,255,255,0.80);
                border-color: rgba(110,140,175,0.75);
            }
            QAbstractItemView::indicator:checked {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5A8FC0, stop:1 #3A6A9A
                );
                border-color: #2E5A88;
            }
            QAbstractItemView::indicator:hover {
                border-color: rgba(70,115,170,0.90);
            }

            /* ── Пункты в выпадающем меню ComboBox ── */
            QComboBox QAbstractItemView::item {
                padding: 4px 8px;
                color: #243040;
                min-height: 22px;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #C5D9EE;
                color: #1A3050;
            }

            /* ── Группы в панели управления ── */
            QGroupBox {
                background: rgba(255,255,255,0.35);
                border: 1px solid rgba(150,170,195,0.55);
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px 8px 6px 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: -1px;
                padding: 1px 8px;
                color: #2C3E52;
                font-size: 11px;
                font-weight: 600;
                background: rgba(222,232,242,0.95);
                border: 1px solid rgba(150,170,195,0.55);
                border-radius: 4px;
            }
        """)

        self.setWindowTitle("Радиобиология — анализ опухолей")
        self.label_2.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #2C3E52; "
            "letter-spacing: 0.2px; padding: 2px 0; background: transparent;"
        )

        # Фиксируем палитру для всех статичных комбобоксов
        for cb in (self.comboBox, self.comboBox_2, self.comboBox_3, self.comboBox_4):
            self._fix_combo_palette(cb)

    def change_table(self):
        """
        Настраивает внешний вид и поведение таблицы для отображения списка файлов экспериментов.

        Устанавливает названия столбцов, регулирует их ширину и определяет внешний вид таблицы.
        Первый столбец содержит чекбоксы для выбора файлов, второй столбец отображает путь к файлу эксперимента.
        Ширина первого столбца фиксирована, второй столбец растягивается, чтобы занять все доступное пространство.

        Args:
            Нет аргументов.

        Returns:
            None.
        """
        self.model.setHorizontalHeaderLabels(['Выбор файла', 'Путь к файлу эксперимента', 'Пометить как контрольный', 'Группа', 'Контроль для статистики'])
        self.tableView.setModel(self.model)
        # Настройка ширины столбцов
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 150)
        header.resizeSection(2, 185)   # «Пометить как контрольный» — достаточно для текста
        header.resizeSection(3, 65)    # «Группа»
        header.resizeSection(4, 175)   # «Контроль для статистики»
        # Настройка внешнего вида таблицы
        self.tableView.setShowGrid(True)  # Показать сетку
        # Устанавливаем размеры политики для таблицы, чтобы она заполняла все доступное пространство
        self.tableView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Клик по ячейке столбца «Группа» — циклически переключает метку A / B / —
        self.tableView.clicked.connect(self._on_table_cell_clicked)

    def on_combobox_changed(self):
        self.selected_outlier_method = self.comboBox.currentIndex()
        # Сбрасываем кеш при смене метода исключения
        self.cached_visualizer = None
        self.cache_key = None

    def set_auc_checkbox(self):
        if self.checkBox_2.isChecked():
            self.use_AUC = True
        else:
            self.use_AUC = False

    def replace_combobox_4(self):
        index = self.horizontalLayout_groups.indexOf(self.comboBox_4)
        if self.comboBox_4 is not None:
            self.horizontalLayout_groups.removeWidget(self.comboBox_4)
            # deleteLater() only schedules destruction for the next event-loop pass —
            # until then the widget stays a visible child at its old (layout-assigned)
            # geometry, which collapses to (0, 0) once it's no longer layout-managed.
            # Hide it immediately so it doesn't render on top of the group box title.
            self.comboBox_4.hide()
            self.comboBox_4.deleteLater()

        self.comboBox_4 = CheckableComboBox(self.groupBox_groups)
        self.horizontalLayout_groups.insertWidget(index, self.comboBox_4)

    def update_combobox_with_labels(self):
        # Сохраняем индексы выбранных элементов перед обновлением
        self.saved_checked_items = self.comboBox_4.save_checked_indices()

        rat_labels = get_rat_labels()  # Получаем метки с информацией о наборе данных
        if rat_labels:
            self.comboBox_4.clear()  # Очищаем существующие элементы комбобокса
            
            # Удаляем дубликаты, сохраняя порядок
            seen = set()
            unique_labels = []
            for label, file_name in rat_labels:
                key = (label, file_name)
                if key not in seen:
                    seen.add(key)
                    unique_labels.append((label, file_name))
            
            # Добавляем метки с форматом "метка крысы (имя файла)"
            display_labels = [f"{label} ({file_name})" for label, file_name in unique_labels]
            self.comboBox_4.add_checkable_items(display_labels)

    def get_selected_rat_labels_with_index(self):
        """
        Возвращает список меток крыс с индексами наборов данных из выбранных элементов CheckableComboBox.
        """
        selected_items = self.comboBox_4.checked_items()
        if not selected_items:
            return []

        # Сбрасываем кеш при изменении выбора крыс для исключения
        self.cached_visualizer = None
        self.cache_key = None

        # Получаем все зарегистрированные метки и удаляем дубликаты
        all_rat_labels = get_rat_labels()
        seen = set()
        unique_rat_labels = []
        for label, file_name in all_rat_labels:
            key = (label, file_name)
            if key not in seen:
                seen.add(key)
                unique_rat_labels.append((label, file_name))

        # Извлекаем метки и имена файлов из выбранных элементов
        selected_with_files = []
        for item in selected_items:
            try:
                parts = item.rsplit(" (", 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    file_name = parts[1].rstrip(")")
                    selected_with_files.append((label, file_name))
            except Exception:
                pass

        # Находим соответствующие метки в unique_rat_labels
        result = []
        for sel_label, sel_file in selected_with_files:
            for label, file_name in unique_rat_labels:
                if label == sel_label and file_name == sel_file:
                    result.append((label, file_name))
                    break

        return result

    def set_state_of_auc_and_tests_checkbox(self):
        # checkBox_rtog осмысленен только там, где реально строится усреднённая кривая кожных
        # реакций (pushButton_2 — одна группа, pushButton_4 — сравнение групп; именно эти ветки
        # вызывают plot_mean_skin_reactions/plot_multiple_experiments(_from_visualizers), куда
        # подключён RTOG) — для объёмов опухоли (pushButton/pushButton_3) он неприменим.
        is_skin_reactions_view = (
            (self.pushButton_2.isEnabled() or self.pushButton_4.isEnabled())
            and self.checkBox_6.isChecked()
        )
        self.checkBox_rtog.setEnabled(is_skin_reactions_view)
        if not is_skin_reactions_view:
            self.checkBox_rtog.setChecked(False)

        if self.pushButton_3.isEnabled() and self.checkBox_6.isChecked():
            self.checkBox_2.setEnabled(True)
            self.checkBox_7.setEnabled(True)
            self.checkBox.setEnabled(True)
            self.checkBox_shapiro.setEnabled(True)
        elif self.pushButton_4.isEnabled() and self.checkBox_6.isChecked():
            self.checkBox_2.setEnabled(True)
            self.checkBox_7.setEnabled(True)
            #self.checkBox.setEnabled(True)
            self.checkBox_shapiro.setEnabled(True)
        elif self.pushButton.isEnabled() and self.checkBox_6.isChecked():
            # Для одной группы опухолей также можно вычислить AUC
            self.checkBox_2.setEnabled(True)
            # Статистические тесты сравнения не имеют смысла для одной группы,
            # но Шапиро–Уилк можно применить к одной группе
            self.checkBox_7.setEnabled(False)
            self.checkBox_7.setChecked(False)
            self.checkBox.setEnabled(False)
            self.checkBox.setChecked(False)
            self.checkBox_shapiro.setEnabled(True)
        elif self.pushButton_2.isEnabled() and self.checkBox_6.isChecked():
            # Для одной группы кожных реакций также можно вычислить AUC
            self.checkBox_2.setEnabled(True)
            # Статистические тесты сравнения не имеют смысла для одной группы,
            # но Шапиро–Уилк можно применить к одной группе
            self.checkBox_7.setEnabled(False)
            self.checkBox_7.setChecked(False)
            self.checkBox.setEnabled(False)
            self.checkBox.setChecked(False)
            self.checkBox_shapiro.setEnabled(True)
        else:
            self.checkBox_2.setEnabled(False)
            self.checkBox_2.setChecked(False)

            self.checkBox_7.setEnabled(False)
            self.checkBox_7.setChecked(False)

            self.checkBox.setEnabled(False)
            self.checkBox.setChecked(False)

            self.checkBox_shapiro.setEnabled(False)
            self.checkBox_shapiro.setChecked(False)

        # Поправка Холма относится только к поточечным сравнениям Манна-Уитни/Стьюдента —
        # включаем доступность чекбокса, только когда один из этих критериев реально активен
        # (учитываем уже скорректированное выше состояние checkBox_7/checkBox в этом вызове).
        # Сохраняем CHECKED-состояние при отключении (не сбрасываем, в отличие от checkBox_rtog):
        # это глобальный флаг (graph_manager), который читается только в момент реального
        # построения теста, а не привязан к текущему режиму просмотра — сброс по умолчанию ON
        # должен сохраняться между переключениями режимов, пока пользователь сам его не снимет.
        mann_whitney_or_student_active = (
            (self.checkBox_7.isEnabled() and self.checkBox_7.isChecked())
            or (self.checkBox.isEnabled() and self.checkBox.isChecked())
        )
        self.checkBox_holm.setEnabled(mann_whitney_or_student_active)

    def on_legend_position_changed(self):
        selected_position = self.comboBox_3.currentText()
        # Если выбрано "None", передаем None вместо строки
        if selected_position == "None":
            selected_position = None
        graph_manager.update_legend_position(selected_position)

        # Автоматически перестраиваем график, если он есть
        if self.figure is not None:
            self.create_graphic()

    def on_separate_legend_changed(self):
        """
        Обработчик изменения состояния чекбокса "Легенда отдельно".
        Блокирует/разблокирует выбор положения легенды и автоматически открывает окно легенды.
        """
        is_checked = self.checkBox_separate_legend.isChecked()
        self.show_legend_separately = is_checked

        # Блокируем/разблокируем comboBox_3 (положение легенды)
        self.comboBox_3.setEnabled(not is_checked)

        if is_checked:
            # Устанавливаем положение легенды в None (скрываем на графике)
            self.comboBox_3.setCurrentText("None")
            graph_manager.update_legend_position(None)

            # Если есть текущий график, обновляем его
            if self.figure is not None:
                self.create_graphic()
        else:
            # Восстанавливаем положение легенды на "best"
            if self.comboBox_3.currentText() == "None":
                self.comboBox_3.setCurrentText("best")
                graph_manager.update_legend_position("best")

            # Если есть текущий график, обновляем его
            if self.figure is not None:
                self.create_graphic()

    def update_doubleSpinBox_value(self):
        """
        Обновляет значение в doubleSpinBox_2 на основе выбранного метода исключения выбросов.
        """
        # Получаем выбранный индекс из выпадающего списка
        selected_method = self.comboBox.currentIndex()

        # В зависимости от выбранного метода подставляем оптимальное значение
        if selected_method == 1:  # Метод с Z-score
            self.doubleSpinBox_2.setValue(2.0)  # Оптимальный порог для Z-score
        elif selected_method == 2:  # Метод с IQR
            self.doubleSpinBox_2.setValue(1.5)  # Оптимальное значение для k в IQR
        elif selected_method == 3:  # Elliptic Envelope
            self.doubleSpinBox_2.setValue(0.1)  # Оптимальная доля выбросов для Elliptic Envelope
        elif selected_method == 4:  # Isolation Forest
            self.doubleSpinBox_2.setValue(0.1)  # Оптимальная доля выбросов для Isolation Forest
        elif selected_method == 5:  # Mahalanobis Distance
            # TODO: Добавить метод ручного выброса под номером 6 и переместить его на 2
            self.doubleSpinBox_2.setValue(0.01)  # Оптимальное значение для alpha
        elif selected_method == 7:  # Euclidean Distance
            self.doubleSpinBox_2.setValue(90.0)  # Оптимальный процентиль для Euclidean Distance
        elif selected_method == 8:  # KL Divergence
            self.doubleSpinBox_2.setValue(0.5)  # Оптимальная ширина полосы для KL Divergence

    def apply_selected_outlier_method(self, visualizer_instances):
        # Если visualizer_instances не список, оборачиваем его в список
        if not isinstance(visualizer_instances, list):
            visualizer_instances = [visualizer_instances]

        # Получаем значение из doubleSpinBox_2
        coefficient = self.doubleSpinBox_2.value()

        updated_instances = []
        for visualizer_instance in visualizer_instances:
            # Создаем экземпляр ExtractOutliers для каждого визуализатора в списке
            outlier_extractor = ExtractOutliers(visualizer_instance)

            # Применяем выбранный метод исключения выбросов, используя значение из doubleSpinBox_2 как коэффициент
            if self.selected_outlier_method == 1:
                outlier_extractor.remove_outliers(threshold=coefficient)  # Используем threshold
            elif self.selected_outlier_method == 2:
                outlier_extractor.remove_outliers_iqr(k=coefficient)  # Используем k
            elif self.selected_outlier_method == 3:
                outlier_extractor.remove_outliers_elliptic_envelope(
                    contamination=coefficient)  # Используем contamination
            elif self.selected_outlier_method == 4:
                outlier_extractor.remove_outliers_isolation_forest(
                    contamination=coefficient)  # Используем contamination
            elif self.selected_outlier_method == 5:
                # TODO: Добавить метод ручного выброса под номером 6 и переместить его на 2
                outlier_extractor.remove_outliers_mahalanobis(alpha=0.01)
            elif self.selected_outlier_method == 6:  # Метод ручного исключения
                selected_rats_with_indices = self.get_selected_rat_labels_with_index()
                
                # Извлекаем только метки крыс, если список не пуст
                excluded_rats = [label for label, index in selected_rats_with_indices] if selected_rats_with_indices else []

                # Если список исключаемых крыс ПУСТ, то ничего не делаем (эквивалентно "Без исключения")
                if not excluded_rats:
                    print("Ручное исключение выбрано, но ни одна крыса не отмечена. Исключение не применяется.")
                    # Пропускаем вызов outlier_extractor.exclude_rats, 
                    # визуализатор будет добавлен без изменений в updated_instances ниже
                else:
                    # Если список НЕ пуст, продолжаем логику ручного исключения
                    # Проверяем, не пытается ли пользователь исключить всех крыс
                    if len(excluded_rats) >= len(visualizer_instance.rat_labels):
                        QMessageBox.warning(
                            self,
                            "Предупреждение",
                            "Нельзя исключить всех крыс из эксперимента!"
                        )
                        # Пропускаем применение exclude_rats для этого визуализатора,
                        # но он все равно будет добавлен без изменений
                    else:
                        # Определяем правильный атрибут данных в зависимости от типа визуализатора
                        data_attr = None
                        if hasattr(visualizer_instance, 'skin_reactions'):
                            data_attr = 'skin_reactions'
                        elif hasattr(visualizer_instance, 'tumor_volumes'):
                            data_attr = 'tumor_volumes'

                        # Вызываем метод исключения крыс с явным указанием атрибута
                        outlier_extractor.exclude_rats(excluded_rats, data_attr)
            elif self.selected_outlier_method == 7:  # Метод для Евклидова расстояния
                outlier_extractor.remove_outliers_by_euclidean(
                    percentile_threshold=coefficient)  # Используем percentile_threshold
            elif self.selected_outlier_method == 8:  # Метод для KL-дивергенции
                outlier_extractor.remove_outliers_kl_divergence(bandwidth=coefficient,
                                                                percentile_threshold=90)  # Используем bandwidth

            # Добавляем обновленный визуализатор в список обновленных экземпляров
            updated_instances.append(outlier_extractor.base_class)

        # Если изначально был передан один экземпляр, возвращаем один экземпляр, а не список
        if len(updated_instances) == 1:
            return updated_instances[0]
        else:
            return updated_instances

    def open_files(self):
        """
        Открывает диалоговое окно для выбора файлов экспериментов и обновляет модель QTableView.

        Этот метод позволяет пользователю выбрать один или несколько файлов экспериментов через стандартное диалоговое окно.
        Выбранные файлы добавляются в модель QTableView, при этом исключаются файлы, которые уже были добавлены ранее.
        Для каждого файла создаётся элемент модели с встроенным чекбоксом и именем файла, а также отдельный элемент
        для хранения полного пути к файлу. После добавления файлов в модель обновляется высота строк в таблице.

        Args:
            Нет аргументов.

        Returns:
            None.
        """
        files, _ = QFileDialog.getOpenFileNames(self, "Открыть файлы эксперимента")

        # Получаем уже добавленные пути
        existing_files = [self.model.item(row, 1).text() for row in range(self.model.rowCount())]

        # Обновляем модель для QTableView
        for file_path in files:
            if file_path in existing_files:
                continue  # Пропускаем файлы, которые уже были добавлены

            file_name = QFileInfo(file_path).fileName()  # Получаем только имя файла

            # Создаем элемент со встроенным чекбоксом и именем файла
            check_and_name_item = QStandardItem(file_name)
            check_and_name_item.setCheckable(True)
            check_and_name_item.setEditable(True)
            file_path_item = QStandardItem(file_path)

            # Создаем чекбокс "Пометить как контрольный"
            control_checkbox_item = QStandardItem()
            control_checkbox_item.setCheckable(True)
            control_checkbox_item.setEditable(False)

            # Элемент для пути к файлу
            file_path_item = QStandardItem(file_path)

            # Элемент «Группа» (col 3) — кликабельный текст: «—» → «A» → «B» → «—»
            group_item = QStandardItem("—")
            group_item.setEditable(False)
            group_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Пустой элемент для ComboBox (col 4)
            control_type_item = QStandardItem()
            control_type_item.setEditable(False)

            # Добавление строки в модель
            self.model.appendRow([check_and_name_item, file_path_item, control_checkbox_item, group_item, control_type_item])

            # Создаем ComboBox для выбора типа контрольной группы (col 4)
            combo = QComboBox()
            combo.addItems(['Не контроль', 'Контроль 1', 'Контроль 2', 'Контроль 3'])
            combo.setCurrentIndex(0)
            self._fix_combo_palette(combo)

            # Сохраняем путь к файлу в данных ComboBox для последующего использования
            combo.setProperty('file_path', file_path)
            combo.currentIndexChanged.connect(self.on_control_type_changed)

            # Устанавливаем ComboBox в ячейку col 4
            row_index = self.model.rowCount() - 1
            self.tableView.setIndexWidget(self.model.index(row_index, 4), combo)

            # Высота строки с запасом под ComboBox
            for row in range(self.model.rowCount()):
                self.tableView.setRowHeight(row, 36)

        # Подстраиваем сплиттер после завершения отрисовки
        QTimer.singleShot(50, self._fit_splitter_to_table)

    def _fit_splitter_to_table(self):
        """
        Расширяет верхнюю панель сплиттера, чтобы таблица вмещала все строки без скролла.

        Если строк много и текущей высоты окна не хватает даже с учётом минимума графика
        снизу (frame.minimumHeight()) — сначала пробуем вырастить само окно (в пределах
        доступной области экрана), а не сжимать нижнюю панель ниже её минимума: явный
        tableView.setMinimumHeight(desired) больше, чем реально способен выдать сплиттер,
        не приводит к ошибке — QSplitter не может дать виджету меньше его minimumHeight,
        поэтому виджет просто визуально перекрывает соседнюю панель, а не скроллится.

        Вызывается только при изменении списка экспериментов (после open_files). При обычном
        изменении размера окна пользователем растить окно в ответ было бы навязчиво — для
        этого случая пересчёт без попытки роста делает _clamp_table_height_to_splitter,
        вызываемая из resizeEvent.
        """
        row_height = 36
        header_h = self.tableView.horizontalHeader().height()
        desired = header_h + self.model.rowCount() * row_height + 6
        min_bottom = self.frame.minimumHeight()

        total = self.splitter_2.height()
        if total <= 0:
            # Виджет ещё не отрисован — повторим чуть позже
            QTimer.singleShot(100, self._fit_splitter_to_table)
            return

        shortfall = (desired + min_bottom) - total
        if shortfall > 0:
            screen = self.screen()
            max_height = screen.availableGeometry().height() if screen else self.height()
            new_height = min(self.height() + shortfall, max_height)
            if new_height > self.height():
                self.resize(self.width(), new_height)
                # Геометрия сплиттера после resize() обновится не сразу — пересчитываем
                # на следующем проходе событийного цикла, а не на непрогретых размерах.
                QTimer.singleShot(0, self._fit_splitter_to_table)
                return

        self._clamp_table_height_to_splitter()

    def _clamp_table_height_to_splitter(self):
        """
        Пересчитывает минимальную высоту таблицы под ТЕКУЩИЙ размер сплиттера, не пытаясь
        вырастить окно — вызывается на каждый resizeEvent (в т.ч. когда пользователь сам
        вручную ужимает окно), поэтому не должна конкурировать с его же намерением сделать
        окно меньше. Без этого пересчёта после ручного ужатия окна старое (большее)
        tableView.setMinimumHeight, выставленное при последнем добавлении файлов, не даёт
        сплиттеру honestly распределить пространство — таблица перекрывает график/кнопки.
        """
        if self.model.rowCount() == 0:
            return

        row_height = 36
        header_h = self.tableView.horizontalHeader().height()
        desired = header_h + self.model.rowCount() * row_height + 6
        min_bottom = self.frame.minimumHeight()

        total = self.splitter_2.height()
        if total <= 0:
            return

        table_height = min(desired, max(total - min_bottom, 0))
        self.tableView.setMinimumHeight(table_height)
        bottom = max(total - table_height, min_bottom)
        self.splitter_2.setSizes([table_height, bottom])

    def on_table_data_changed(self, *args):
        """
        Этот слот вызывается при изменении данных в таблице.
        Он отвечает за сброс состояния всех чекбоксов в comboBox_4 при изменении файлов.
        """
        # Сбрасываем кеш визуализатора
        self.cached_visualizer = None
        self.cache_key = None
        
        self.comboBox_4.clear_all_checkboxes()
        # НЕ очищаем метки здесь, они очистятся при следующем построении графика
        self.update_combobox_with_labels()

    def update_control_path(self, text):
        self.control_path = text

    def on_control_checkbox_changed(self, item):
        if item.column() == 2:  # Проверяем, что изменение произошло в столбце "Пометить как контрольный"
            if item.checkState() == Qt.CheckState.Checked:
                # Проходимся по всем чекбоксам и снимаем отметку, кроме текущего
                for row in range(self.model.rowCount()):
                    otherItem = self.model.item(row, 2)
                    if otherItem != item:
                        otherItem.setCheckState(Qt.CheckState.Unchecked)
                file_path = self.model.item(item.row(), 1).text()
                self.comboBox_2.clear()
                self.comboBox_2.addItem(file_path)
                self.control_path = file_path  # Обновляем контрольный путь
            else:
                self.comboBox_2.clear()
                self.control_path = None
            self.update_seventh_button_state()
            self.update_fifth_button_state()
            self.update_third_button_state()

    def on_control_type_changed(self, index):
        """
        Обработчик изменения типа контрольной группы в ComboBox.
        Сохраняет выбранный тип в словарь control_groups.
        Проверяет уникальность - один тип контроля может быть назначен только одному файлу.
        """
        sender = self.sender()  # Получаем ComboBox, который отправил сигнал
        if sender:
            file_path = sender.property('file_path')
            if index == 0:  # "Не контроль"
                if file_path in self.control_groups:
                    del self.control_groups[file_path]
            else:  # Контроль 1, 2 или 3
                # Проверяем, не назначен ли уже этот тип контроля другому файлу
                for existing_path, existing_type in list(self.control_groups.items()):
                    if existing_type == index and existing_path != file_path:
                        # Сбрасываем ComboBox у другого файла
                        for row in range(self.model.rowCount()):
                            path_item = self.model.item(row, 1)
                            if path_item and path_item.text() == existing_path:
                                other_combo = self.tableView.indexWidget(self.model.index(row, 3))
                                if other_combo:
                                    other_combo.blockSignals(True)  # Блокируем сигналы, чтобы избежать рекурсии
                                    other_combo.setCurrentIndex(0)  # Сбрасываем на "Не контроль"
                                    other_combo.blockSignals(False)
                                break
                        # Удаляем старую запись
                        del self.control_groups[existing_path]
                        break

                self.control_groups[file_path] = index  # index: 1=Контроль1, 2=Контроль2, 3=Контроль3

    def get_selected_experiments(self):
        """
        Собирает пути к файлам выбранных экспериментов из таблицы.

        Проходит по всем строкам модели таблицы, проверяет состояние чекбокса в первой колонке.
        Если чекбокс отмечен, добавляет путь файла из второй колонки в список выбранных экспериментов.

        Returns:
            list: Список строк, содержащий пути к файлам выбранных экспериментов.
        """
        selected_paths = []
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item and item.isCheckable() and item.checkState() == Qt.CheckState.Checked:
                path = self.model.item(row, 1).text()
                selected_paths.append(path)

        return selected_paths

    def _on_table_cell_clicked(self, index):
        """
        Обрабатывает клик по ячейке таблицы.
        Для столбца «Группа» (3) циклически переключает значение: «—» → «A» → «B» → «—».
        """
        if index.column() != 3:
            return
        item = self.model.itemFromIndex(index)
        if item is None:
            return
        current = item.text()
        cycle = {"—": "A", "A": "B", "B": "—"}
        next_val = cycle.get(current, "—")
        item.setText(next_val)
        # Цветовая подсветка
        from PyQt6.QtGui import QColor, QBrush
        color_map = {"A": QColor(173, 216, 230), "B": QColor(255, 200, 150), "—": QColor(255, 255, 255)}
        item.setBackground(QBrush(color_map[next_val]))
        self.update_tenth_button_state()

    def get_group_assignment(self):
        """
        Возвращает словарь {path: 'A' | 'B' | None} для всех строк таблицы.
        Строки с «—» получают None.
        """
        result = {}
        for row in range(self.model.rowCount()):
            path_item = self.model.item(row, 1)
            group_item = self.model.item(row, 3)
            if path_item and group_item:
                val = group_item.text()
                result[path_item.text()] = None if val == "—" else val
        return result

    def _find_control_index(self):
        """
        Находит индекс контрольного файла в списке current_selected_paths.
        (Оставлено для обратной совместимости)

        Returns:
            int: Индекс контрольного файла в списке, или 0 если контроль не найден.
        """
        if not self.control_path or not self.current_selected_paths:
            return 0

        # Нормализуем пути для корректного сравнения
        import os
        normalized_control = os.path.normpath(self.control_path)

        for i, path in enumerate(self.current_selected_paths):
            normalized_path = os.path.normpath(path)
            if normalized_path == normalized_control:
                return i

        # Если контрольный файл не найден, возвращаем 0
        return 0

    def _get_control_groups_info(self):
        """
        Возвращает информацию о всех контрольных группах.

        Returns:
            dict: Словарь {control_type: [indices]}, где control_type - номер контроля (1, 2, 3),
                  а indices - список индексов в current_selected_paths
        """
        import os
        if not self.current_selected_paths:
            return {}

        control_info = {}  # {control_type: [indices]}

        for i, path in enumerate(self.current_selected_paths):
            normalized_path = os.path.normpath(path)
            # Проверяем, является ли этот файл контрольным
            for control_path, control_type in self.control_groups.items():
                normalized_control = os.path.normpath(control_path)
                if normalized_path == normalized_control:
                    if control_type not in control_info:
                        control_info[control_type] = []
                    control_info[control_type].append(i)
                    break

        return control_info

    def update_first_button_state(self):
        """
        Обновляет состояние кнопки в зависимости от выбранных экспериментов и чекбоксов.

        Этот метод проверяет, выбран ли ровно один эксперимент и отмечен ли хотя бы один
        из двух наборов чекбоксов (checkBox_3 или checkBox_4, и checkBox_5 или checkBox_6).
        Если оба условия удовлетворены, кнопка становится активной. В противном случае
        кнопка деактивируется.

        Args:
            Нет аргументов.

        Returns:
            Ничего не возвращает, но изменяет состояние активности pushButton.
        """
        selected_paths = self.get_selected_experiments()
        oneExperimentSelected = len(selected_paths) == 1
        anyCheckboxChecked = ((self.checkBox_3.isChecked() or self.checkBox_4.isChecked()) and
                              (self.checkBox_5.isChecked() or self.checkBox_6.isChecked()))
        fileName = "skin_reactions" not in selected_paths[0] if oneExperimentSelected else False
        self.pushButton.setEnabled(oneExperimentSelected and anyCheckboxChecked and fileName)

    def update_second_button_state(self):
        """
        Обновляет состояние кнопки в зависимости от выбранных экспериментов и чекбоксов.

        Этот метод проверяет, выбран ли ровно один эксперимент с кожными реакциями
        и отмечены ли чекбоксы "абс.ед" + ("средние" ИЛИ "индивидуальные").
        Если все условия удовлетворены, кнопка становится активной. В противном случае
        кнопка деактивируется.

        Args:
            Нет аргументов.

        Returns:
            Ничего не возвращает, но изменяет состояние активности pushButton.
        """
        selected_paths = self.get_selected_experiments()
        oneExperimentSelected = len(selected_paths) == 1
        # Кнопка активна при "абс.ед" + ("средние" ИЛИ "индивидуальные")
        anyCheckboxChecked = self.checkBox_3.isChecked() and (self.checkBox_6.isChecked() or self.checkBox_5.isChecked())
        fileName = "skin_reactions" in selected_paths[0] if oneExperimentSelected else False
        self.pushButton_2.setEnabled(oneExperimentSelected and anyCheckboxChecked and fileName)

    def update_third_button_state(self):
        """
        Обновляет состояние кнопки в зависимости от выбранных экспериментов и чекбоксов.

        Этот метод проверяет, выбран ли ровно один эксперимент и отмечен ли хотя бы один
        из двух наборов чекбоксов (checkBox_3 или checkBox_4, и checkBox_6).
        Если оба условия удовлетворены, кнопка становится активной. В противном случае
        кнопка деактивируется.

        Args:
            Нет аргументов.

        Returns:
            Ничего не возвращает, но изменяет состояние активности pushButton.
        """
        selected_paths = self.get_selected_experiments()
        oneExperimentSelected = len(selected_paths) >= 2
        anyCheckboxChecked = ((self.checkBox_3.isChecked() or self.checkBox_4.isChecked()) and
                              (self.checkBox_5.isChecked() or self.checkBox_6.isChecked()))
        # Проверяем, что во всех выбранных путях отсутствует "skin_reactions"
        allPathsValid = all("skin_reactions" not in path for path in selected_paths)
        controlChecked = self.comboBox_2.count() == 0
        self.pushButton_3.setEnabled(oneExperimentSelected and
                                     anyCheckboxChecked and allPathsValid and controlChecked)

    def update_fourth_button_state(self):
        selected_paths = self.get_selected_experiments()
        twoOrMoreSelected = len(selected_paths) >= 2
        # Кнопка активна при "абс.ед" + ("средние" ИЛИ "индивидуальные") для ДВУХ или более экспериментов
        anyCheckboxChecked = self.checkBox_3.isChecked() and (self.checkBox_6.isChecked() or self.checkBox_5.isChecked())
        all_skin = all("skin_reactions" in path for path in selected_paths) if selected_paths else False
        all_tumor = all("skin_reactions" not in path for path in selected_paths) if selected_paths else False

        # pushButton_4 активна для 2+ экспериментов
        self.pushButton_4.setEnabled(twoOrMoreSelected and anyCheckboxChecked and all_skin)

        # Для AUC требуется минимум 2 эксперимента и только "средние"
        auc_checkbox_checked = self.checkBox_3.isChecked() and self.checkBox_6.isChecked()
        self.pushButton_8.setEnabled(twoOrMoreSelected and auc_checkbox_checked and (all_skin or all_tumor))

    def update_fifth_button_state(self):
        """
        Обновляет состояние кнопки в зависимости от выбранных экспериментов и чекбоксов.

        Этот метод проверяет, выбран ли ровно один эксперимент и отмечен ли хотя бы один
        из двух наборов чекбоксов (checkBox_3, и checkBox_6).
        Если оба условия удовлетворены, кнопка становится активной. В противном случае
        кнопка деактивируется.

        Args:
            Нет аргументов.

        Returns:
            Ничего не возвращает, но изменяет состояние активности pushButton.
        """
        selected_paths = self.get_selected_experiments()
        oneExperimentSelected = len(selected_paths) >= 1
        anyCheckboxChecked = self.checkBox_3.isChecked() and self.checkBox_6.isChecked()
        controlChecked = self.comboBox_2.count() > 0
        self.pushButton_5.setEnabled(oneExperimentSelected and anyCheckboxChecked and controlChecked)
        self.pushButton_6.setEnabled(oneExperimentSelected and anyCheckboxChecked and controlChecked)

    def update_seventh_button_state(self):
        """
        Обновляет состояние кнопки в зависимости от выбранных экспериментов и чекбоксов.

        Этот метод проверяет, выбран ли ровно один эксперимент и отмечен ли хотя бы один
        из двух наборов чекбоксов (checkBox_4, и checkBox_6).
        Если оба условия удовлетворены, кнопка становится активной. В противном случае
        кнопка деактивируется.

        Args:
            Нет аргументов.

        Returns:
            Ничего не возвращает, но изменяет состояние активности pushButton.
        """
        selected_paths = self.get_selected_experiments()
        oneExperimentSelected = len(selected_paths) >= 1
        anyCheckboxChecked = self.checkBox_4.isChecked() and self.checkBox_6.isChecked()
        controlChecked = self.comboBox_2.count() > 0
        # Проверяем, что во всех выбранных путях отсутствует "skin_reactions"
        allPathsValid = all("skin_reactions" not in path for path in selected_paths)
        self.pushButton_7.setEnabled(oneExperimentSelected and anyCheckboxChecked and controlChecked and allPathsValid)

    def update_tenth_button_state(self):
        """
        Обновляет состояние кнопки «Расхождение по крысам».

        Кнопка активна, если:
        - Отмечены «отн. ед.» (checkBox_4) И «общие» (checkBox_5);
        - Нет файлов skin_reactions;
        - Режим A vs B: |A| == |B| ≥ 1, нет файлов без группы («—»);
          ИЛИ режим без групп: выбрано ≥2 файлов (у всех группа «—»).
        """
        selected_paths = self.get_selected_experiments()
        at_least_one = len(selected_paths) >= 1
        not_skin = all("skin_reactions" not in p for p in selected_paths) if at_least_one else False
        mode_ok = self.checkBox_4.isChecked() and self.checkBox_5.isChecked()

        if at_least_one:
            groups = self.get_group_assignment()
            selected_groups = {p: groups.get(p) for p in selected_paths}
            paths_a = [p for p, g in selected_groups.items() if g == 'A']
            paths_b = [p for p, g in selected_groups.items() if g == 'B']
            paths_no_group = [p for p, g in selected_groups.items() if g is None]
            if paths_a and paths_b:
                # Режим A vs B: нет файлов без группы И |A| == |B|
                enough = len(paths_no_group) == 0 and len(paths_a) == len(paths_b)
            else:
                enough = len(selected_paths) >= 2   # режим без групп — нужно ≥2
        else:
            enough = False

        self.pushButton_10.setEnabled(at_least_one and not_skin and mode_ok and enough)

    def handle_divergence_per_rat(self):
        """
        Обрабатывает нажатие кнопки «Расхождение по крысам».

        Если среди выбранных файлов есть группы A и B — строит попарный сравнительный
        график «расхождение замеров» (одна крыса × два метода измерения).
        Иначе — стандартный межособевой комбинированный график.
        """
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 1:
            print("Для построения графика выберите хотя бы один файл")
            return

        groups = self.get_group_assignment()
        selected_groups = {p: groups.get(p) for p in selected_paths}
        paths_a = [p for p, g in selected_groups.items() if g == 'A']
        paths_b = [p for p, g in selected_groups.items() if g == 'B']

        if paths_a and paths_b:
            # Режим сравнения замеров: группа A vs группа B
            self.current_plot_type = 'measurement_comparison'
            self.current_selected_paths = selected_paths
            self._paths_group_a = paths_a
            self._paths_group_b = paths_b
            self.current_visualizer = TumorDataVisualizer
            self.current_plotting_func = TumorDataVisualizer.plot_relative_divergence_per_rat
            self.current_control = None
            self.create_graphic()
        else:
            # Обычный режим: межособевое расхождение
            self.current_plot_type = 'divergence_per_rat'
            self.draw_graphic(selected_paths, TumorDataVisualizer,
                              TumorDataVisualizer.plot_relative_divergence_per_rat)

    def build_kaplan_meier_groups_from_selection(self):
        """
        Возвращает {имя группы: List[RatSurvivalEvent]} по текущему выбору в главной
        таблице, либо None, если ничего не выбрано.

        Файлы, явно отмеченные группой A или B (столбец «Группа» в таблице),
        объединяются в одну кривую на группу — так можно слить крыс из нескольких
        файлов одного режима в одну выборку. Любой выбранный файл БЕЗ группы
        становится отдельной кривой (по умолчанию — каждый выбранный файл это
        своё сравнение, а не один общий пул).

        Используется и окном сравнения групп (handle_kaplan_meier), и калькулятором
        (кнопка «Загрузить из выбранных файлов») — единая точка построения групп.
        """
        selected_paths = self.get_selected_experiments()
        if not selected_paths:
            return None

        from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (
            extract_survival_events,
        )

        groups_assignment = self.get_group_assignment()
        paths_a = [p for p in selected_paths if groups_assignment.get(p) == 'A']
        paths_b = [p for p in selected_paths if groups_assignment.get(p) == 'B']
        paths_other = [p for p in selected_paths if groups_assignment.get(p) is None]

        groups = {}
        if paths_a:
            groups["Группа A"] = [e for p in paths_a for e in extract_survival_events(p)]
        if paths_b:
            groups["Группа B"] = [e for p in paths_b for e in extract_survival_events(p)]
        for path in paths_other:
            groups[Path(path).stem] = extract_survival_events(path)
        return groups

    def handle_kaplan_meier(self):
        """Строит кривые Каплана-Майера по крысам из выбранных файлов (см. build_kaplan_meier_groups_from_selection)."""
        try:
            groups = self.build_kaplan_meier_groups_from_selection()
        except Exception as error:
            self._show_tool_open_error("Каплан-Майер", error)
            return

        if groups is None:
            QMessageBox.information(self, "Каплан-Майер", "Выберите хотя бы один файл.")
            return

        if self.kaplan_meier_window is None:
            from work_with_prepared_data.radiobioligy_project.gui.kaplan_meier_window import KaplanMeierWindow
            self.kaplan_meier_window = KaplanMeierWindow(self)

        self.kaplan_meier_window.set_groups(groups)
        self.kaplan_meier_window.show()
        self.kaplan_meier_window.raise_()
        self.kaplan_meier_window.activateWindow()

    def on_checkbox_pair_changed(self, thisCheckbox, pairedCheckbox):
        """Обработка изменения состояния пары взаимоисключающих чекбоксов.

        Args:
            this_checkbox: Чекбокс, состояние которого изменилось.
            other_checkbox: Второй чекбокс в паре, состояние которого необходимо обновить.
        """
        # Если активирован один чекбокс, деактивируем связанный с ним
        if thisCheckbox.isChecked():
            pairedCheckbox.setChecked(False)
        self.update_first_button_state()
        self.update_second_button_state()
        self.update_third_button_state()
        self.update_fourth_button_state()
        self.update_fifth_button_state()
        self.update_seventh_button_state()
        self.update_tenth_button_state()
        self.set_state_of_auc_and_tests_checkbox()

    def on_checkbox_tests_changed(self, thisCheckbox, pairedCheckbox):
        """
        Обрабатывает изменение состояния пары чекбоксов и обновляет соответствующие переменные.
        Убирает выделение с другого чекбокса, если текущий активирован.
        """
        if thisCheckbox.isChecked():
            pairedCheckbox.setChecked(False)  # Отключаем другой чекбокс

        # Обновляем состояние переменных
        self.perform_stat_test = self.checkBox_7.isChecked()
        self.use_ttest = self.checkBox.isChecked()
        # checkBox_holm должен реагировать на CHECKED-состояние самих критериев, а не только на
        # пересчёт их доступности (тот идёт через on_checkbox_pair_changed для checkBox_3/4/5/6,
        # которая не запускается при переключении checkBox_7/checkBox).
        self.set_state_of_auc_and_tests_checkbox()

    def on_shapiro_changed(self):
        """Обновляет флаг теста Шапиро–Уилка."""
        self.use_shapiro = self.checkBox_shapiro.isChecked()

    def on_rtog_changed(self):
        """Обновляет флаг отображения шкалы RTOG на графиках кожных реакций."""
        self.show_rtog = self.checkBox_rtog.isChecked()

    def on_holm_correction_changed(self):
        """Включает/выключает поправку Холма для поточечных сравнений Манна-Уитни/Стьюдента."""
        graph_manager.set_holm_correction_enabled(self.checkBox_holm.isChecked())

    def on_show_date_changed(self):
        """Включает/выключает показ "Дата: ..." в подписях экспериментов в легенде графика."""
        graph_manager.set_show_date_in_legend(self.checkBox_show_date.isChecked())
        if self.figure is not None:
            self.create_graphic()

    def handle_all_of_rats(self):
        """
        Обрабатывает запрос на создание графика на основе выбранных экспериментов и условий выбора чекбоксов.

        Выбирает один из предопределённых методов визуализации на основе активированных чекбоксов.
        Выводит сообщение и прекращает выполнение в случаях, когда выбрано недопустимое количество экспериментов
        или не выбран тип графика.

        Args:
            None

        Returns:
            None
        """
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 1:
            print("Выберите хотя бы один эксперимент")
            return
        elif len(selected_paths) > 1:
            print("Для построения данного графика нужен лишь один эксперимент")
            return

        checkboxes_state = (self.checkBox_3.isChecked(), self.checkBox_4.isChecked(),
                            self.checkBox_5.isChecked(), self.checkBox_6.isChecked())
        try:
            plotting_func, selected_path = self.data_processor.process_for_single_experiment(selected_paths[0],
                                                                                             checkboxes_state)
            self.current_plot_type = None
            self.draw_graphic([selected_path], TumorDataVisualizer, plotting_func)
        except ValueError as e:
            print(e)

    def handle_compare_rats(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 2:
            print("Необходимо выбрать два или более экспериментов")
            return

        checkboxes_state = (self.checkBox_3.isChecked(), self.checkBox_4.isChecked(), self.checkBox_6.isChecked(),
                            self.checkBox_5.isChecked())
        try:
            plotting_func, selected_paths = self.data_processor.process_for_comparison(selected_paths, checkboxes_state)
            self.current_plot_type = None
            self.draw_graphic(selected_paths, TumorDataComparatorAdvanced, plotting_func)
        except ValueError as e:
            print(e)

    def handle_compare_with_control(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 1:
            print("Необходимо выбрать хотя бы один или более экспериментов")
            return

        # Предположим, что контрольный путь уже сохранен в атрибуте класса
        control_path = self.control_path

        # Проверка наличия контрольного пути
        if not control_path:
            print("Необходимо указать путь к контрольной группе")
            return

        checkboxes_state = (self.checkBox_4.isChecked(), self.checkBox_6.isChecked())
        try:
            plotting_func, selected_paths, control_visualizer = self.data_processor.process_for_control_comparison(
                selected_paths, control_path, checkboxes_state)
            self.draw_graphic(selected_paths, TumorDataComparatorAdvanced, plotting_func, control_visualizer)
        except ValueError as e:
            print(e)

    def handle_skin_reactions(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 1:
            print("Выберите хотя бы один эксперимент")
            return
        elif len(selected_paths) > 1:
            print("Для построения данного графика нужен лишь один эксперимент")
            return

        checkboxes_state = (self.checkBox_3.isChecked(), self.checkBox_4.isChecked(),
                            self.checkBox_5.isChecked(), self.checkBox_6.isChecked())
        try:
            plotting_func, selected_path = self.data_processor.process_skin_reactions(selected_paths[0],
                                                                                      checkboxes_state)

            # Устанавливаем тип графика в зависимости от выбранных чекбоксов
            if self.checkBox_5.isChecked():  # "индивидуальные"
                self.current_plot_type = 'all_individual_curves'
            else:  # "средние"
                self.current_plot_type = None

            # selected_path может быть строкой или списком в зависимости от типа графика
            # Не оборачиваем в список, если это уже список
            if isinstance(selected_path, list):
                self.draw_graphic(selected_path, SkinReactionsVisualizer, plotting_func)
            else:
                self.draw_graphic([selected_path], SkinReactionsVisualizer, plotting_func)
        except ValueError as e:
            print(e)

    def handle_compare_skin_reactions(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 2:
            print("Необходимо выбрать два или более экспериментов")
            return

        checkboxes_state = (self.checkBox_3.isChecked(), self.checkBox_4.isChecked(),
                            self.checkBox_5.isChecked(), self.checkBox_6.isChecked())
        try:
            # Для нескольких файлов передаём список
            plotting_func, selected_path = self.data_processor.process_skin_reactions(selected_paths,
                                                                                      checkboxes_state)
            self.draw_graphic(selected_path if isinstance(selected_path, list) else [selected_path],
                            SkinReactionsVisualizer, plotting_func)
        except ValueError as e:
            print(e)

    def handle_compare_tumor_growth_inhibition(self):
        # Получаем выбранные пути
        selected_paths = self.get_selected_experiments()

        # Проверяем наличие контрольного пути и наличие выбранных экспериментов
        if not self.control_path or len(selected_paths) < 1:
            print("Необходимо выбрать контрольную группу и хотя бы один эксперимент")
            return

        checkboxes_state = (self.checkBox_4.isChecked(), self.checkBox_6.isChecked())
        try:
            plotting_func, selected_paths, control_visualizer = self.data_processor.process_for_control_comparison(
                selected_paths, self.control_path, checkboxes_state)
            self.draw_graphic(selected_paths, TumorDataComparatorAdvanced, plotting_func, control_visualizer)
        except ValueError as e:
            print(e)

    def handle_tumor_growth_inhibition_table(self):
        selected_paths = self.get_selected_experiments()

        # Проверяем наличие контрольного пути и наличие выбранных экспериментов
        if not self.control_path or len(selected_paths) < 1:
            print("нужен контроль и хотя бы один эксперимент")
            return

        # Получение контрольного и экспериментальных визуализаторов
        control_visualizer = ControlGroupVisualizer(self.control_path)
        experiment_visualizers = [TumorDataVisualizer(path) for path in selected_paths]
        visualizer = TumorDataComparatorAdvanced(*experiment_visualizers)

        # Предполагаем, что функция модифицирована для возврата DataFrame
        tables_by_mode = visualizer.create_tumor_growth_inhibition_tables(
            control_visualizer,
            experiment_visualizers
        )
        # create_tumor_growth_inhibition_tables уже нормализовала time_data визуализаторов
        # (normalize_time_data_min внутри _build_tumor_growth_inhibition_series) —
        # повторная нормализация здесь не нужна.
        tgd_df = visualizer.create_tgd_table(
            control_visualizer,
            experiment_visualizers,
            normalize_time=False
        )

        # Очистка layout перед добавлением нового содержимого
        if self.tgi_table_window is None:
            self.tgi_table_window = TumorGrowthInhibitionTableWindow(self)

        # Проверка, существует ли layout. Если нет, создаем новый.
        self.tgi_table_window.set_tables(tables_by_mode)
        self.tgi_table_window.set_tgd_table(tgd_df)
        self.tgi_table_window.show()
        self.tgi_table_window.raise_()
        self.tgi_table_window.activateWindow()

        # Создание QTableWidget и заполнение его данными из DataFrame

    def handle_skin_reaction_summary_table(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 1:
            print("Выберите хотя бы один эксперимент")
            return

        # Единое пороговое значение задаётся исследователем (раздел "Конечные точки
        # исследования": "продолжительность реакции выше ЗАДАННОГО порогового значения") —
        # используется и для длительности превышения, и для нормализации (возврат ниже него).
        threshold, ok = QInputDialog.getDouble(
            self,
            "Порог кожной реакции",
            "Пороговое значение балла (используется для длительности реакции и нормализации):",
            decimals=1
        )
        if not ok:
            return

        visualizers = [SkinReactionsVisualizer(path) for path in selected_paths]
        summary_df = SkinReactionsVisualizer.build_summary_table(visualizers, threshold=threshold)

        if self.skin_reaction_summary_window is None:
            self.skin_reaction_summary_window = SkinReactionSummaryWindow(self)

        self.skin_reaction_summary_window.set_summary_table(summary_df)
        self.skin_reaction_summary_window.show()
        self.skin_reaction_summary_window.raise_()
        self.skin_reaction_summary_window.activateWindow()

    def handle_pushButton_4(self):
        # Определяем тип графика в зависимости от выбранных чекбоксов
        if self.checkBox_5.isChecked():  # Если выбран "индивидуальные"
            self.current_plot_type = 'all_individual_curves'
        else:  # Если выбран "средние"
            self.current_plot_type = 'multiple_experiments'
        self.create_graphic()

    def handle_pushButton_8(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 2:
            print("Необходимо выбрать два или более экспериментов")
            return

        # Если включен критерий Манна-Уитни, но контроли не выбраны, автоматически помечаем первый файл
        if self.perform_stat_test and not self.control_groups:
            # Находим первый выбранный файл в таблице и помечаем его как "Контроль 1"
            for row in range(self.model.rowCount()):
                item = self.model.item(row, 0)
                if item and item.isCheckable() and item.checkState() == Qt.CheckState.Checked:
                    combo = self.tableView.indexWidget(self.model.index(row, 3))
                    if combo:
                        combo.setCurrentIndex(1)  # Устанавливаем "Контроль 1"
                    break

        all_skin = all("skin_reactions" in p for p in selected_paths)
        all_tumor = all("skin_reactions" not in p for p in selected_paths)

        if all_skin:
            self.current_visualizer = SkinReactionsVisualizer
            self.current_plot_type = 'auc_comparison'
        elif all_tumor:
            self.current_visualizer = TumorDataVisualizer
            self.current_plot_type = 'tumor_auc_comparison'
        else:
            print("Выберите данные одного типа: skin_reactions или опухоли")
            return

        self.current_selected_paths = selected_paths
        self.current_plotting_func = None
        self.current_control = None
        self.create_graphic()

    def draw_graphic(self, selected_paths, visualizer, plotting_func, current_control=None):
        self.current_selected_paths = selected_paths
        self.current_visualizer = visualizer
        self.current_plotting_func = plotting_func
        self.current_control = current_control
        self.create_graphic()

    def draw_figure_to_pixmap(self, visualizer, plotting_func):
        """
        Сохраняет визуализацию, созданную функцией отрисовки, в QPixmap объект.

        Этот метод использует переданную функцию отрисовки для генерации графика, который затем
        сохраняется в объект QPixmap. Это позволяет отображать график в интерфейсе приложения PyQt,
        не открывая отдельное окно Matplotlib.

        Args:
            visualizer: Экземпляр класса, отвечающего за визуализацию данных.
            plotting_func: Функция, которая будет использована для создания графика.
                           Должна принимать экземпляр `visualizer` в качестве аргумента.

        Returns:
        """
        # Перенаправляем вывод графика в объект BytesIO вместо отображения в окне
        with io.BytesIO() as buf:
            # Режим сравнения замеров A vs B
            if self.current_plot_type == 'measurement_comparison':
                paths_a = getattr(self, '_paths_group_a', [])
                paths_b = getattr(self, '_paths_group_b', [])
                TumorDataVisualizer.plot_measurement_divergence(paths_a, paths_b)
                if self.figure is not None:
                    plt.close(self.figure)
                self.figure = plt.gcf()
                plt.savefig(buf, format='png')
                buf.seek(0)
                pixmap = QPixmap()
                pixmap.loadFromData(buf.getvalue())
                return pixmap

            # Проверяем, является ли visualizer списком визуализаторов для множественных экспериментов
            if isinstance(visualizer, list) and len(visualizer) > 0 and isinstance(visualizer[0], SkinReactionsVisualizer):
                # Используем модифицированные экземпляры визуализаторов
                if self.current_plot_type == 'multiple_experiments':
                    SkinReactionsVisualizer.plot_multiple_experiments_from_visualizers(visualizer, self.use_AUC, self.perform_stat_test, self.show_rtog)
                elif self.current_plot_type == 'auc_comparison':
                    # Получить информацию о контрольных группах
                    control_groups_info = self._get_control_groups_info()
                    control_idx = self._find_control_index()  # для обратной совместимости
                    SkinReactionsVisualizer.plot_auc_comparison_from_visualizers(visualizer, perform_stat_test=self.perform_stat_test, control_index=control_idx, control_groups_info=control_groups_info)
                elif self.current_plot_type == 'all_individual_curves':
                    SkinReactionsVisualizer.plot_all_individual_curves_from_visualizers(visualizer)
            elif isinstance(visualizer, SkinReactionsVisualizer) and self.current_plot_type == 'all_individual_curves':
                # Для одного эксперимента с индивидуальными кривыми
                SkinReactionsVisualizer.plot_all_individual_curves_from_visualizers([visualizer])
            elif isinstance(visualizer, SkinReactionsVisualizer) and len(self.current_selected_paths) > 1:
                # Вызов статического метода для рисования графика из путей (для обратной совместимости)
                if self.current_plot_type == 'multiple_experiments':
                    SkinReactionsVisualizer.plot_multiple_experiments(self.current_selected_paths, self.use_AUC, self.perform_stat_test, self.show_rtog)
                elif self.current_plot_type == 'auc_comparison':
                    # Получить информацию о контрольных группах
                    control_groups_info = self._get_control_groups_info()
                    control_idx = self._find_control_index()  # для обратной совместимости
                    SkinReactionsVisualizer.plot_auc_comparison(self.current_selected_paths, perform_stat_test=self.perform_stat_test, control_index=control_idx, control_groups_info=control_groups_info)
                elif self.current_plot_type == 'all_individual_curves':
                    SkinReactionsVisualizer.plot_all_individual_curves(self.current_selected_paths)
            elif isinstance(visualizer, TumorDataVisualizer) and len(self.current_selected_paths) > 1 and self.current_plot_type == 'tumor_auc_comparison':
                # Получить информацию о контрольных группах
                control_groups_info = self._get_control_groups_info()
                control_idx = self._find_control_index()  # для обратной совместимости
                TumorDataVisualizer.plot_auc_comparison(self.current_selected_paths, perform_stat_test=self.perform_stat_test, control_index=control_idx, control_groups_info=control_groups_info, show_separate_legend=self.show_legend_separately, use_shapiro=self.use_shapiro)
            else:
                # Для других случаев, когда используется один файл или другие типы визуализаторов
                if self.current_control is not None and plotting_func == TumorDataComparatorAdvanced.compare_control_and_experiment:
                    plotting_func(visualizer, [self.current_control])
                elif self.current_control is not None and plotting_func == TumorDataComparatorAdvanced.compare_tumor_growth_inhibition_with_multiple_experiments:
                    experiment_visualizers = [TumorDataVisualizer(path) for path in self.current_selected_paths]
                    plotting_func(visualizer, self.current_control, experiment_visualizers)
                elif plotting_func == SkinReactionsVisualizer.plot_mean_skin_reactions:
                    plotting_func(visualizer, show_rtog=self.show_rtog)
                else:
                    plotting_func(visualizer)
                    # Шапиро–Уилк для одиночного эксперимента
                    if isinstance(visualizer, TumorDataVisualizer) and getattr(visualizer, 'use_shapiro', False):
                        from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer
                        GraphVisualizer._add_shapiro_annotation([visualizer])

            # Очищаем предыдущую фигуру, если она есть
            if self.figure is not None:
                plt.close(self.figure)
            
            # Сохраняем текущую фигуру для возможного извлечения легенды
            current_figure = plt.gcf()
            self.figure = current_figure
            
            # Сохраняем генерируемый график в буфер
            # После генерации графика нужно сохранить текущий рисунок в buf
            plt.savefig(buf, format='png')
            
            # НЕ закрываем эту фигуру, чтобы можно было извлечь данные легенды
            # plt.close()  # Закомментировано для работы с легендой
            
            # Обновляем комбобокс ПОСЛЕ построения графика
            # (Убрано отсюда - будет вызвано в create_graphic после draw_figure_to_pixmap)
            
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            return pixmap

    def create_graphic(self):
        """
        Создает и отображает графики для выбранных экспериментов.

        Этот метод генерирует графики для каждого пути в selected_paths, используя
        указанную функцию отрисовки из класса visualizer. Графики отображаются внутри
        виджета frame текущего интерфейса.

        Args:
            selected_paths (List[str]): Список путей к файлам экспериментов.
            visualizer (Visualizer): Класс визуализатора, который используется для генерации графиков.
            plotting_func (function): Функция визуализатора для генерации графика.

        """
        # Для сравнения AUC (кожные реакции или объёмы опухоли) график строится
        # через статический метод, поэтому self.current_plotting_func может быть
        # None. В остальных случаях эта функция должна быть задана.
        if not self.current_visualizer or (
            self.current_plotting_func is None
            and self.current_plot_type not in [
                'auc_comparison',
                'tumor_auc_comparison',
                'all_individual_curves',
            ]
        ):
            return  # Ничего не делаем, если параметры не заданы

        # Очищаем старые метки крыс перед построением графика
        clear_rat_labels()

        # Очищаем layout, если он уже существует
        if self.frame.layout() is not None:
            # Удаляем все виджеты из layout
            while self.frame.layout().count():
                child = self.frame.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            # Если layout еще не был установлен, создаем его
            layout = QVBoxLayout(self.frame)
            self.frame.setLayout(layout)

        if self.current_plot_type == 'measurement_comparison':
            # Сравнение замеров: группа A vs группа B попарно
            pixmap = self.draw_figure_to_pixmap(None, None)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            self.frame.layout().addWidget(label)
            self.update_combobox_with_labels()
            self.comboBox_4.restore_checked_indices(self.saved_checked_items)
            if self.show_legend_separately and self.figure is not None:
                self.show_legend_window()
            return

        if self.current_plot_type in ('variability', 'divergence_per_rat'):
            # Объединяем крыс из всех выбранных файлов в один псевдо-визуализатор.
            # Это позволяет считать d(t)/CV как между крысами внутри одного файла,
            # так и между крысами из разных файлов (по одной крысе на файл).
            sub_visualizers = [TumorDataVisualizer(p) for p in self.current_selected_paths]
            if self.selected_outlier_method is not None:
                sub_visualizers = self.apply_selected_outlier_method(sub_visualizers)
                if not isinstance(sub_visualizers, list):
                    sub_visualizers = [sub_visualizers]

            # Берём первый визуализатор как базу и дополняем его данными из остальных.
            # Метки крыс делаем уникальными: если метка уже встречалась, добавляем
            # суффикс из имени файла (без расширения), чтобы избежать одинаковых
            # меток и путаницы в легенде и кэше стилей.
            from work_with_prepared_data.radiobioligy_project.data_processing.data_processing import TumorDataProcessor
            import os as _os
            seen_labels: set = set()

            def _unique_labels(labels, file_path):
                basename = _os.path.splitext(_os.path.basename(file_path))[0]
                parts = basename.split('_')
                suffix = '_'.join(parts[:2]) if len(parts) >= 2 else basename
                result = []
                for lbl in labels:
                    if lbl in seen_labels:
                        unique = f"{lbl} ({suffix})"
                    else:
                        unique = lbl
                    seen_labels.add(unique)
                    result.append(unique)
                return result

            visualizer_instance = sub_visualizers[0]
            # Обрабатываем метки первого файла
            visualizer_instance.rat_labels = _unique_labels(
                visualizer_instance.rat_labels, self.current_selected_paths[0]
            )
            for idx, other in enumerate(sub_visualizers[1:], start=1):
                unique = _unique_labels(other.rat_labels, self.current_selected_paths[idx])
                visualizer_instance.rat_labels = visualizer_instance.rat_labels + unique
                visualizer_instance.tumor_volumes = visualizer_instance.tumor_volumes + other.tumor_volumes
                # Обновляем data_processor с объединёнными данными
                visualizer_instance.data_processor = TumorDataProcessor(visualizer_instance.tumor_volumes)

        elif self.current_visualizer is TumorDataVisualizer:
            # Случай для одного эксперимента
            visualizer_instance = self.current_visualizer(self.current_selected_paths[0])
            if self.selected_outlier_method is not None:
                visualizer_instance = self.apply_selected_outlier_method(visualizer_instance)
            # Устанавливаем параметры для одного эксперимента
            visualizer_instance.use_AUC = self.use_AUC
            visualizer_instance.use_shapiro = self.use_shapiro
        elif self.current_visualizer is TumorDataComparatorAdvanced and self.current_control is None:
            # Случай для сравнения нескольких экспериментов
            visualizer_instances = [TumorDataVisualizer(path) for path in self.current_selected_paths]
            if self.selected_outlier_method is not None:
                visualizer_instances = self.apply_selected_outlier_method(visualizer_instances)
            visualizer_instance = self.current_visualizer(*visualizer_instances)
            visualizer_instance.perform_stat_test = self.perform_stat_test
            visualizer_instance.annotation_multiplier = self.annotation_multiplier
            visualizer_instance.use_ttest = self.use_ttest
            visualizer_instance.use_shapiro = self.use_shapiro
            visualizer_instance.use_AUC = self.use_AUC
        elif self.current_visualizer is TumorDataComparatorAdvanced and self.current_control is not None:
            # Случай для сравнения нескольких экспериментов с контрольной группой
            visualizer_instances = [TumorDataVisualizer(path) for path in self.current_selected_paths]
            if self.selected_outlier_method is not None:
                visualizer_instances = self.apply_selected_outlier_method(visualizer_instances)
            visualizer_instance = self.current_visualizer(*visualizer_instances)
            visualizer_instance.perform_stat_test = self.perform_stat_test
            visualizer_instance.annotation_multiplier = self.annotation_multiplier
            visualizer_instance.use_ttest = self.use_ttest
            visualizer_instance.use_shapiro = self.use_shapiro
            visualizer_instance.use_AUC = self.use_AUC
        elif self.current_visualizer is SkinReactionsVisualizer:
            # ВАЖНО: Для кожных реакций НЕ используем кеш при методе ручного исключения
            # так как данные должны обновляться при каждом изменении выбора крыс
            if self.selected_outlier_method == 6:
                # Метод ручного исключения - всегда создаём новый визуализатор
                # Для множественных экспериментов создаём список модифицированных визуализаторов
                if len(self.current_selected_paths) > 1:
                    visualizer_instances = []
                    for path in self.current_selected_paths:
                        vis = self.current_visualizer(path)
                        if self.selected_outlier_method is not None:
                            vis = self.apply_selected_outlier_method(vis)
                        visualizer_instances.append(vis)
                    visualizer_instance = visualizer_instances
                else:
                    visualizer_instance = self.current_visualizer(self.current_selected_paths[0])
                    if self.selected_outlier_method is not None:
                        visualizer_instance = self.apply_selected_outlier_method(visualizer_instance)
                    # Устанавливаем параметры для одного эксперимента
                    visualizer_instance.use_AUC = self.use_AUC

                # НЕ сохраняем в кеш для метода ручного исключения
                self.cached_visualizer = None
                self.cache_key = None
            else:
                # Для других методов используем кеш
                current_cache_key = (
                    tuple(self.current_selected_paths),
                    self.current_visualizer,
                    self.selected_outlier_method
                )

                # Проверяем, можем ли использовать кешированный визуализатор
                if self.cache_key == current_cache_key and self.cached_visualizer is not None:
                    visualizer_instance = self.cached_visualizer
                else:
                    # Для множественных экспериментов создаём список визуализаторов
                    if len(self.current_selected_paths) > 1:
                        visualizer_instances = []
                        for path in self.current_selected_paths:
                            vis = self.current_visualizer(path)
                            if self.selected_outlier_method is not None:
                                vis = self.apply_selected_outlier_method(vis)
                            visualizer_instances.append(vis)
                        visualizer_instance = visualizer_instances
                    else:
                        visualizer_instance = self.current_visualizer(self.current_selected_paths[0])
                        if self.selected_outlier_method is not None:
                            visualizer_instance = self.apply_selected_outlier_method(visualizer_instance)
                        # Устанавливаем параметры для одного эксперимента
                        visualizer_instance.use_AUC = self.use_AUC

                    # Сохраняем в кеш
                    self.cached_visualizer = visualizer_instance
                    self.cache_key = current_cache_key

        pixmap = self.draw_figure_to_pixmap(visualizer_instance, self.current_plotting_func)

        # Отображаем созданный pixmap
        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        self.frame.layout().addWidget(label)

        # Обновляем комбобокс с метками крыс после построения графика
        self.update_combobox_with_labels()

        # Восстанавливаем состояние выбранных элементов
        self.comboBox_4.restore_checked_indices(self.saved_checked_items)

        # Автоматически открываем окно легенды, если включена соответствующая опция
        if self.show_legend_separately and self.figure is not None:
            self.show_legend_window()

    def dataframe_to_qtablewidget(self, df):
        table_widget = QTableWidget()
        table_widget.setRowCount(df.shape[0])
        table_widget.setColumnCount(df.shape[1])
        table_widget.setHorizontalHeaderLabels([str(column) for column in df.columns])

        for i, (index, row) in enumerate(df.iterrows()):
            for j, value in enumerate(row):
                if pd.isna(value):
                    display_value = ""
                elif isinstance(value, numbers.Integral):
                    display_value = str(int(value))
                elif isinstance(value, numbers.Real):
                    numeric_value = float(value)
                    if j == 0 and numeric_value.is_integer():
                        display_value = str(int(numeric_value))
                    else:
                        display_value = f"{numeric_value:.3f}"
                else:
                    display_value = str(value)
                item = QTableWidgetItem(display_value)
                table_widget.setItem(i, j, item)

        table_widget.resizeColumnsToContents()
        return table_widget

    def resizeEvent(self, event):
        """Вызывается при изменении размера окна."""
        super(MainWindow, self).resizeEvent(event)
        if hasattr(self, 'model'):
            self._clamp_table_height_to_splitter()
        self.create_graphic()

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def update_annotation_multiplier(self, value):
        self.annotation_multiplier = self.doubleSpinBox.value()
        # print(f"Сдвиг равен {self.annotation_multiplier}")

    def edit_experiment_files(self):
        """
        Открывает диалоговое окно для выбора файлов экспериментов и открывает выбранные файлы в Excel.
        """
        # Разрешаем пользователю выбрать один или несколько файлов
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выбрать файл для редактирования",
            "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        )

        if not files:
            return  # Пользователь отменил диалог

        for file_path in files:
            try:
                if sys.platform.startswith('darwin'):
                    subprocess.call(['open', file_path])
                elif os.name == 'nt':  # Для Windows
                    os.startfile(file_path)
                elif os.name == 'posix':  # Для Linux
                    subprocess.call(['xdg-open', file_path])
                else:
                    raise OSError("Unsupported operating system.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл {file_path}.\n{str(e)}")

    def show_legend_preview(self):
        """
        Показывает окно предпросмотра легенды как изображения с возможностью сохранения
        """
        try:
            # Используем фигуру из canvas, если она есть
            current_figure = self.figure

            if current_figure is None:
                # Если нет фигуры в canvas, пытаемся получить текущую фигуру matplotlib
                current_figure = plt.gcf() if plt.get_fignums() else None

            if current_figure:
                # Показываем окно предпросмотра легенды
                self.legend_manager.show_legend_preview(current_figure, self)
            else:
                self.show_message("Нет активного графика для отображения легенды", "Информация")
        except Exception as e:
            self.show_message(f"Ошибка при отображении легенды: {str(e)}", "Ошибка")

    def show_legend_window(self):
        """
        Показывает легенду в отдельном окне (используется для автоматического открытия)
        """
        # Используем тот же метод для автоматического открытия
        self.show_legend_preview()

    def show_message(self, message, title="Информация"):
        """
        Показывает сообщение пользователю
        """
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, title, message)

    def _show_child_window(self, attr_name, window_factory):
        window = getattr(self, attr_name)
        if window is None:
            window = window_factory()
            setattr(self, attr_name, window)
        window.show()
        window.raise_()
        window.activateWindow()

    def _show_tool_open_error(self, tool_name, error):
        QMessageBox.critical(
            self,
            "Ошибка",
            f"Не удалось открыть окно «{tool_name}».\n{error}",
        )

    def open_survival_fitter(self):
        try:
            from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_gui import (
                FitAlphaBetaWindow,
            )
            self._show_child_window("fit_alpha_beta_window", FitAlphaBetaWindow)
        except Exception as error:
            self._show_tool_open_error("LQ fitter и радиобиология", error)

    def open_growth_predictor(self):
        try:
            from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor_gui import (
                TumorGrowthPredictorWindow,
            )
            self._show_child_window("growth_predictor_window", TumorGrowthPredictorWindow)
        except Exception as error:
            self._show_tool_open_error("Предсказание роста опухоли", error)

    def open_geant4_pipeline(self):
        try:
            from work_with_prepared_data.radiobioligy_project.survival.geant4_pipeline_gui import (
                Geant4PipelineWindow,
            )
            self._show_child_window("geant4_pipeline_window", Geant4PipelineWindow)
        except Exception as error:
            self._show_tool_open_error("GEANT4 / RT Dose pipeline", error)

    def open_tumor_3d_viewer(self):
        try:
            from work_with_prepared_data.radiobioligy_project.tumor_3d_viewer import (
                Tumor3DViewerWindow,
            )
            self._show_child_window("tumor_3d_viewer_window", Tumor3DViewerWindow)
        except Exception as error:
            self._show_tool_open_error("3D геометрия опухоли", error)

    def open_kaplan_meier_calculator(self):
        try:
            from work_with_prepared_data.radiobioligy_project.gui.kaplan_meier_calculator_window import (
                KaplanMeierCalculatorWindow,
            )
            # передаём self как parent, чтобы калькулятор мог подтянуть выбор файлов
            # из главного окна (кнопка «Загрузить из выбранных файлов»)
            self._show_child_window("kaplan_meier_calculator_window", lambda: KaplanMeierCalculatorWindow(self))
        except Exception as error:
            self._show_tool_open_error("Калькулятор Каплана-Майера", error)

    def open_project_documentation(self):
        """Open the project overview HTML page in the default browser."""
        docs_path = Path(__file__).resolve().parent.parent / "docs" / "index.html"
        if not docs_path.exists():
            self.show_message(
                f"Не найден файл документации:\n{docs_path}",
                "Ошибка",
            )
            return

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs_path))):
            self.show_message(
                "Не удалось открыть документацию в браузере по умолчанию.",
                "Ошибка",
            )

    def save_graph(self):
        """
        Сохраняет текущий график в файл
        """
        if self.figure is None:
            self.show_message("Нет графика для сохранения. Сначала постройте график.", "Информация")
            return

        try:
            # Предлагаем сохранить в папку Downloads
            from PyQt6.QtCore import QStandardPaths
            default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить график",
                f"{default_path}/graph.png",
                "PNG файлы (*.png);;JPEG файлы (*.jpg);;PDF файлы (*.pdf);;SVG файлы (*.svg);;Все файлы (*)"
            )

            if file_path:
                # Сохраняем текущую фигуру matplotlib
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight',
                                   facecolor='white', edgecolor='none')
                self.show_message("График успешно сохранён!", "Успех")

        except Exception as e:
            self.show_message(f"Ошибка при сохранении графика: {str(e)}", "Ошибка")

def excepthook(type, value, traceback):
    app = QApplication.instance()
    error_msg = f"{type.__name__}: {value}"
    QMessageBox.critical(None, "Необработанная ошибка", error_msg)
    # Не завершаем приложение, чтобы оно продолжало работать

sys.excepthook = excepthook


def _apply_application_palette(app: QApplication):
    """Фиксирует светлые цвета выделения для Fusion и popup-списков."""
    palette = app.palette()
    for color_group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        palette.setColor(color_group, QPalette.ColorRole.Highlight, QColor('#C5D9EE'))
        palette.setColor(color_group, QPalette.ColorRole.HighlightedText, QColor('#1A3050'))
    app.setPalette(palette)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    _apply_application_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
