import io
from PyQt6.QtCore import QFileInfo, Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView, QSizePolicy, QVBoxLayout, QLabel, \
    QTableWidget, QTableWidgetItem
import sys

# Импорт сгенерированного класса из gui.py
from gui import Ui_MainWindow
from work_with_prepared_data.radiobioligy_project.controls import ControlGroupVisualizer
from work_with_prepared_data.radiobioligy_project.draw_abs_rel_graph_compare import TumorDataComparatorAdvanced
from work_with_prepared_data.radiobioligy_project.draw_base_graphs import TumorDataVisualizer
from work_with_prepared_data.radiobioligy_project.draw_base_graphs_compare import TumorDataComparator
from work_with_prepared_data.radiobioligy_project.skin_reactions_base_grapf import SkinReactionsVisualizer

import matplotlib

from work_with_prepared_data.radiobioligy_project.stats_methods.support_stats_methods import ExtractOutliers

matplotlib.use('QT5Agg')  # Установка бэкенда до импорта pyplot.
import matplotlib.pyplot as plt


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
        if checkboxes_state == (True, False, True, False) and "skin_reactions" in selected_path:
            plotting_func = SkinReactionsVisualizer.plot_skin_reactions
        elif checkboxes_state == (False, True, False, True) and "skin_reactions" in selected_path:
            plotting_func = SkinReactionsVisualizer.plot_mean_skin_reactions
        elif checkboxes_state == (True, False, False, True) and all("skin_reactions" in path for path in selected_path):
            plotting_func = SkinReactionsVisualizer.plot_multiple_experiments
        else:
            raise ValueError("Invalid checkbox state or name")
        return plotting_func, selected_path

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
        self.ax = None
        self.canvas = None
        self.figure = None
        self.current_visualizer = None
        self.current_plotting_func = None
        self.current_selected_paths = []
        self.current_control = None
        self.selected_outlier_method = None
        self.data_processor = DataProcessor()
        self.setupUi(self)
        self.action.triggered.connect(self.open_files)
        # Настраиваем модель для 2 столбцов
        self.model = QStandardItemModel(0, 3, self)
        self.change_table()
        # Кнопки по умолчанию неактивны
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        # Биндинг кнопок
        self.pushButton.clicked.connect(self.handle_all_of_rats)
        self.pushButton_2.clicked.connect(self.handle_skin_reactions)
        self.pushButton_3.clicked.connect(self.handle_compare_rats)
        self.pushButton_4.clicked.connect(self.handle_compare_skin_reactions)
        self.pushButton_5.clicked.connect(self.handle_compare_tumor_growth_inhibition)
        self.pushButton_6.clicked.connect(self.handle_tumor_growth_inhibition_table)
        self.pushButton_7.clicked.connect(self.handle_compare_with_control)
        # Подключение сигнала изменения выбора комбобокса к обработчику
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)
        # Подключаем сигналы изменения состояния чекбоксов
        self.checkBox_3.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_3, self.checkBox_4))
        self.checkBox_4.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_4, self.checkBox_3))
        self.checkBox_5.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_5, self.checkBox_6))
        self.checkBox_6.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_6, self.checkBox_5))
        self.model.itemChanged.connect(self.update_first_button_state)
        self.model.itemChanged.connect(self.update_second_button_state)
        self.model.itemChanged.connect(self.update_third_button_state)
        self.model.itemChanged.connect(self.update_fourth_button_state)
        self.model.itemChanged.connect(self.update_fifth_button_state)
        self.model.itemChanged.connect(self.update_seventh_button_state)
        self.model.itemChanged.connect(self.on_control_checkbox_changed)
        self.comboBox_2.currentTextChanged.connect(self.update_control_path)

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
        self.model.setHorizontalHeaderLabels(['Выбор файла', 'Путь к файлу эксперимента', 'Пометить как контрольный'])
        self.tableView.setModel(self.model)
        # Настройка ширины столбцов
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        # Устанавливаем фиксированную ширину для столбца с чекбоксами и именем файла
        header.resizeSection(0, 150)  # Подстраиваем под нужный размер
        header.resizeSection(1, 250)  # Подстраиваем под нужный размер
        header.resizeSection(2, 170)  # Подстраиваем под нужный размер
        # Настройка внешнего вида таблицы
        self.tableView.setShowGrid(True)  # Показать сетку
        # Устанавливаем размеры политики для таблицы, чтобы она заполняла все доступное пространство
        self.tableView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def on_combobox_changed(self):
        self.selected_outlier_method = self.comboBox.currentIndex()

    def apply_selected_outlier_method(self, visualizer_instances):
        # Если visualizer_instances не список, оборачиваем его в список
        if not isinstance(visualizer_instances, list):
            visualizer_instances = [visualizer_instances]

        updated_instances = []
        for visualizer_instance in visualizer_instances:
            # Создаем экземпляр ExtractOutliers для каждого визуализатора в списке
            outlier_extractor = ExtractOutliers(visualizer_instance)

            # Применяем выбранный метод исключения выбросов
            if self.selected_outlier_method == 1:
                outlier_extractor.remove_outliers()
            elif self.selected_outlier_method == 2:
                outlier_extractor.remove_outliers_iqr(k=1.5)
            elif self.selected_outlier_method == 3:
                outlier_extractor.remove_outliers_elliptic_envelope(contamination=0.1)
            elif self.selected_outlier_method == 4:
                outlier_extractor.remove_outliers_isolation_forest(contamination=0.1)
            elif self.selected_outlier_method == 5:
                outlier_extractor.remove_outliers_mahalanobis(alpha=0.01)

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
            check_and_name_item.setEditable(False)
            file_path_item = QStandardItem(file_path)

            # Создаем чекбокс "Пометить как контрольный"
            control_checkbox_item = QStandardItem()
            control_checkbox_item.setCheckable(True)
            control_checkbox_item.setEditable(False)

            # Элемент для пути к файлу
            file_path_item = QStandardItem(file_path)

            # Добавление строки в модель
            self.model.appendRow([check_and_name_item, file_path_item, control_checkbox_item])

            # Устанавливаем высоту строк
            for row in range(self.model.rowCount()):
                self.tableView.setRowHeight(row, 20)  # Задаем желаемую высоту строки

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
        anyCheckboxChecked = ((self.checkBox_3.isChecked() or self.checkBox_6.isChecked()) and
                              (self.checkBox_4.isChecked() or self.checkBox_5.isChecked()))
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
        oneExperimentSelected = len(selected_paths) >= 2
        anyCheckboxChecked = self.checkBox_3.isChecked() and self.checkBox_6.isChecked()
        # Проверяем, что во всех выбранных путях отсутствует "skin_reactions"
        allPathsValid = all("skin_reactions" in path for path in selected_paths)
        self.pushButton_4.setEnabled(oneExperimentSelected and anyCheckboxChecked and allPathsValid)

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
            plotting_func, selected_path = self.data_processor.process_skin_reactions(selected_paths,
                                                                                      checkboxes_state)
            self.draw_graphic(selected_path, SkinReactionsVisualizer, plotting_func)
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
            QPixmap: Объект QPixmap, содержащий изображение сгенерированного графика.

        """
        # Перенаправляем вывод графика в объект BytesIO вместо отображения в окне
        with io.BytesIO() as buf:
            if isinstance(visualizer, SkinReactionsVisualizer) and len(self.current_selected_paths) > 1:
                # Вызов статического метода для рисования графика
                SkinReactionsVisualizer.plot_multiple_experiments(self.current_selected_paths)
            else:
                # Для других случаев, когда используется один файл или другие типы визуализаторов
                if self.current_control is not None and plotting_func == TumorDataComparatorAdvanced.compare_control_and_experiment:
                    plotting_func(visualizer, [self.current_control])
                elif self.current_control is not None and plotting_func == TumorDataComparatorAdvanced.compare_tumor_growth_inhibition_with_multiple_experiments:
                    experiment_visualizers = [TumorDataVisualizer(path) for path in self.current_selected_paths]
                    plotting_func(visualizer, self.current_control, experiment_visualizers)
                else:
                    plotting_func(visualizer)
            # Сохраняем генерируемый график в буфер
            # После генерации графика нужно сохранить текущий рисунок в buf
            plt.savefig(buf, format='png')
            plt.close()  # Закрываем текущее окно plt, чтобы оно не отображалось
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
        if not self.current_visualizer or not self.current_plotting_func:
            return  # Ничего не делаем, если параметры не заданы

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
        elif self.current_visualizer is TumorDataComparatorAdvanced and self.current_control is None:
            # Случай для сравнения нескольких экспериментов
            visualizer_instances = [TumorDataVisualizer(path) for path in self.current_selected_paths]
            if self.selected_outlier_method is not None:
                visualizer_instances = self.apply_selected_outlier_method(visualizer_instances)
            visualizer_instance = self.current_visualizer(*visualizer_instances)
        elif self.current_visualizer is TumorDataComparatorAdvanced and self.current_control is not None:
            # Случай для сравнения нескольких экспериментов с контрольной группой
            visualizer_instances = [TumorDataVisualizer(path) for path in self.current_selected_paths]
            if self.selected_outlier_method is not None:
                visualizer_instances = self.apply_selected_outlier_method(visualizer_instances)
            visualizer_instance = self.current_visualizer(*visualizer_instances)
        elif self.current_visualizer is SkinReactionsVisualizer:
            if len(self.current_selected_paths) == 1:
                visualizer_instance = self.current_visualizer(self.current_selected_paths[0])
            else:
                visualizer_instance = self.current_visualizer(self.current_selected_paths[0])
            # TODO: Метод требует правки  для работы с кожными реакциями
            # if self.selected_outlier_method is not None:
            #     visualizer_instance = self.apply_selected_outlier_method(visualizer_instance)

        pixmap = self.draw_figure_to_pixmap(visualizer_instance, self.current_plotting_func)

        # Отображаем созданный pixmap
        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        self.frame.layout().addWidget(label)

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


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
