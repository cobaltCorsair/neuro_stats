import io
from PyQt6.QtCore import QFileInfo, Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView, QSizePolicy, QVBoxLayout, QLabel, \
    QTableWidget, QTableWidgetItem, QMessageBox, QButtonGroup, QComboBox
import subprocess
import sys
import os

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
        self.saved_checked_items = []
        self.perform_stat_test = False
        self.use_ttest = False
        self.use_AUC = False
        self.annotation_multiplier = 0
        self.data_processor = DataProcessor()
        self.legend_manager = LegendManager()
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
        self.action.triggered.connect(self.open_files)
        self.action_2.triggered.connect(self.save_graph)

        # Блокируем неиспользуемые кнопки меню
        self.action_4.setEnabled(False)    # Сохранить таблицу
        self.menu_2.setEnabled(False)      # Редактировать (весь выпадающий список)
        self.menu_3.setEnabled(False)      # Распознать (весь выпадающий список)

        # Настраиваем модель для 2 столбцов
        self.model = QStandardItemModel(0, 3, self)
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
        self.checkBox_2.setDisabled(True)
        self.checkBox_7.setDisabled(True)
        self.checkBox.setDisabled(True)
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
        # Подключение сигнала изменения выбора комбобокса к обработчику
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)
        # Подключаем сигналы изменения состояния чекбоксов
        self.checkBox_3.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_3, self.checkBox_4))
        self.checkBox_4.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_4, self.checkBox_3))
        self.checkBox_5.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_5, self.checkBox_6))
        self.checkBox_6.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_6, self.checkBox_5))
        self.checkBox_7.stateChanged.connect(lambda: self.on_checkbox_tests_changed(self.checkBox_7, self.checkBox))
        self.checkBox.stateChanged.connect(lambda: self.on_checkbox_tests_changed(self.checkBox, self.checkBox_7))
        self.checkBox_2.stateChanged.connect(self.set_auc_checkbox)
        self.model.itemChanged.connect(self.update_first_button_state)
        self.model.itemChanged.connect(self.update_second_button_state)
        self.model.itemChanged.connect(self.update_third_button_state)
        self.model.itemChanged.connect(self.update_fourth_button_state)
        self.model.itemChanged.connect(self.update_fifth_button_state)
        self.model.itemChanged.connect(self.update_seventh_button_state)
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
        self.model.setHorizontalHeaderLabels(['Выбор файла', 'Путь к файлу эксперимента', 'Пометить как контрольный', 'Тип группы'])
        self.tableView.setModel(self.model)
        # Настройка ширины столбцов
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        # Устанавливаем фиксированную ширину для столбца с чекбоксами и именем файла
        header.resizeSection(0, 150)  # Подстраиваем под нужный размер
        header.resizeSection(1, 250)  # Подстраиваем под нужный размер
        header.resizeSection(2, 170)  # Подстраиваем под нужный размер
        header.resizeSection(3, 150)  # Ширина для выпадающего списка
        # Настройка внешнего вида таблицы
        self.tableView.setShowGrid(True)  # Показать сетку
        # Устанавливаем размеры политики для таблицы, чтобы она заполняла все доступное пространство
        self.tableView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        index = self.horizontalLayout_7.indexOf(self.comboBox_4)
        if self.comboBox_4 is not None:
            self.horizontalLayout_7.removeWidget(self.comboBox_4)
            self.comboBox_4.deleteLater()

        self.comboBox_4 = CheckableComboBox(self)
        self.horizontalLayout_7.insertWidget(index, self.comboBox_4)

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
        if self.pushButton_3.isEnabled() and self.checkBox_6.isChecked():
            self.checkBox_2.setEnabled(True)
            self.checkBox_7.setEnabled(True)
            self.checkBox.setEnabled(True)
        elif self.pushButton_4.isEnabled() and self.checkBox_6.isChecked():
            self.checkBox_2.setEnabled(True)
            self.checkBox_7.setEnabled(True)
            #self.checkBox.setEnabled(True)
        elif self.pushButton.isEnabled() and self.checkBox_6.isChecked():
            # Для одной группы опухолей также можно вычислить AUC
            self.checkBox_2.setEnabled(True)
            # Но статистические тесты не имеют смысла для одной группы
            self.checkBox_7.setEnabled(False)
            self.checkBox_7.setChecked(False)
            self.checkBox.setEnabled(False)
            self.checkBox.setChecked(False)
        elif self.pushButton_2.isEnabled() and self.checkBox_6.isChecked():
            # Для одной группы кожных реакций также можно вычислить AUC
            self.checkBox_2.setEnabled(True)
            # Но статистические тесты не имеют смысла для одной группы
            self.checkBox_7.setEnabled(False)
            self.checkBox_7.setChecked(False)
            self.checkBox.setEnabled(False)
            self.checkBox.setChecked(False)
        else:
            self.checkBox_2.setEnabled(False)
            self.checkBox_2.setChecked(False)

            self.checkBox_7.setEnabled(False)
            self.checkBox_7.setChecked(False)

            self.checkBox.setEnabled(False)
            self.checkBox.setChecked(False)

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

            # Создаем пустой элемент для ComboBox в 4-м столбце
            control_type_item = QStandardItem()
            control_type_item.setEditable(False)

            # Добавление строки в модель
            self.model.appendRow([check_and_name_item, file_path_item, control_checkbox_item, control_type_item])

            # Создаем ComboBox для выбора типа группы
            combo = QComboBox()
            combo.addItems(['Не контроль', 'Контроль 1', 'Контроль 2', 'Контроль 3'])
            combo.setCurrentIndex(0)

            # Сохраняем путь к файлу в данных ComboBox для последующего использования
            combo.setProperty('file_path', file_path)
            combo.currentIndexChanged.connect(self.on_control_type_changed)

            # Устанавливаем ComboBox в ячейку
            row_index = self.model.rowCount() - 1
            self.tableView.setIndexWidget(self.model.index(row_index, 3), combo)

            # Устанавливаем высоту строк
            for row in range(self.model.rowCount()):
                self.tableView.setRowHeight(row, 20)  # Задаем желаемую высоту строки

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
        """
        sender = self.sender()  # Получаем ComboBox, который отправил сигнал
        if sender:
            file_path = sender.property('file_path')
            if index == 0:  # "Не контроль"
                if file_path in self.control_groups:
                    del self.control_groups[file_path]
            else:  # Контроль 1, 2 или 3
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
        if not self.control_path or len(selected_paths) != 2:
            print("Необходимо выбрать контрольную группу и два эксперимента")
            return

        # Получение контрольного и экспериментальных визуализаторов
        visualizer = TumorDataComparatorAdvanced(*[TumorDataVisualizer(path) for path in selected_paths])
        control_visualizer = ControlGroupVisualizer(self.control_path)
        experiment_visualizers = [TumorDataVisualizer(path) for path in selected_paths]

        # Предполагаем, что функция модифицирована для возврата DataFrame
        df = (TumorDataComparatorAdvanced.
              create_tumor_growth_inhibition_table(visualizer, control_visualizer, experiment_visualizers))

        # Очистка layout перед добавлением нового содержимого
        self.clear_layout(self.frame.layout())

        # Проверка, существует ли layout. Если нет, создаем новый.
        if self.frame.layout() is None:
            layout = QVBoxLayout(self.frame)
            self.frame.setLayout(layout)
        else:
            layout = self.frame.layout()

        # Создание QTableWidget и заполнение его данными из DataFrame
        table = self.dataframe_to_qtablewidget(df)
        layout.addWidget(table)

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
            # Проверяем, является ли visualizer списком визуализаторов для множественных экспериментов
            if isinstance(visualizer, list) and len(visualizer) > 0 and isinstance(visualizer[0], SkinReactionsVisualizer):
                # Используем модифицированные экземпляры визуализаторов
                if self.current_plot_type == 'multiple_experiments':
                    SkinReactionsVisualizer.plot_multiple_experiments_from_visualizers(visualizer, self.use_AUC, self.perform_stat_test)
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
                    SkinReactionsVisualizer.plot_multiple_experiments(self.current_selected_paths, self.use_AUC, self.perform_stat_test)
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
                TumorDataVisualizer.plot_auc_comparison(self.current_selected_paths, perform_stat_test=self.perform_stat_test, control_index=control_idx, control_groups_info=control_groups_info)
            else:
                # Для других случаев, когда используется один файл или другие типы визуализаторов
                if self.current_control is not None and plotting_func == TumorDataComparatorAdvanced.compare_control_and_experiment:
                    plotting_func(visualizer, [self.current_control])
                elif self.current_control is not None and plotting_func == TumorDataComparatorAdvanced.compare_tumor_growth_inhibition_with_multiple_experiments:
                    experiment_visualizers = [TumorDataVisualizer(path) for path in self.current_selected_paths]
                    plotting_func(visualizer, self.current_control, experiment_visualizers)
                else:
                    plotting_func(visualizer)

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

        if self.current_visualizer is TumorDataVisualizer:
            # Случай для одного эксперимента
            visualizer_instance = self.current_visualizer(self.current_selected_paths[0])
            if self.selected_outlier_method is not None:
                visualizer_instance = self.apply_selected_outlier_method(visualizer_instance)
            # Устанавливаем параметры для одного эксперимента
            visualizer_instance.use_AUC = self.use_AUC
        elif self.current_visualizer is TumorDataComparatorAdvanced and self.current_control is None:
            # Случай для сравнения нескольких экспериментов
            visualizer_instances = [TumorDataVisualizer(path) for path in self.current_selected_paths]
            if self.selected_outlier_method is not None:
                visualizer_instances = self.apply_selected_outlier_method(visualizer_instances)
            visualizer_instance = self.current_visualizer(*visualizer_instances)
            visualizer_instance.perform_stat_test = self.perform_stat_test
            visualizer_instance.annotation_multiplier = self.annotation_multiplier
            visualizer_instance.use_ttest = self.use_ttest
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
        table_widget.setHorizontalHeaderLabels(df.columns)

        for i, (index, row) in enumerate(df.iterrows()):
            for j, value in enumerate(row):
                if j == 0:
                    item = QTableWidgetItem(str(int(value)))
                else:
                    # Остальные значения округляем до трех знаков после запятой
                    item = QTableWidgetItem(f"{value:.3f}")
                table_widget.setItem(i, j, item)

        table_widget.resizeColumnsToContents()
        return table_widget

    def resizeEvent(self, event):
        """Вызывается при изменении размера окна."""
        super(MainWindow, self).resizeEvent(event)
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

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
