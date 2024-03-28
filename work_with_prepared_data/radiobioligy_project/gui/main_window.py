import io
from PyQt6.QtCore import QFileInfo, Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView, QSizePolicy, QVBoxLayout, QLabel
import sys

# Импорт сгенерированного класса из gui.py
from gui import Ui_MainWindow
from work_with_prepared_data.radiobioligy_project.draw_abs_rel_graph_compare import TumorDataComparatorAdvanced
from work_with_prepared_data.radiobioligy_project.draw_base_graphs import TumorDataVisualizer
import matplotlib

matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt


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
        self.ax = None
        self.canvas = None
        self.figure = None
        self.current_visualizer = None
        self.current_plotting_func = None
        self.current_selected_paths = []
        self.setupUi(self)
        self.action.triggered.connect(self.open_files)
        # Настраиваем модель для 2 столбцов
        self.model = QStandardItemModel(0, 2, self)
        self.change_table()
        # Кнопки по умолчанию неактивны
        self.pushButton.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        # Биндинг кнопок
        self.pushButton.clicked.connect(self.handle_all_of_rats)
        self.pushButton_3.clicked.connect(self.handle_compare_rats)
        # Подключаем сигналы изменения состояния чекбоксов
        self.checkBox_3.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_3, self.checkBox_4))
        self.checkBox_4.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_4, self.checkBox_3))
        self.checkBox_5.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_5, self.checkBox_6))
        self.checkBox_6.stateChanged.connect(lambda: self.on_checkbox_pair_changed(self.checkBox_6, self.checkBox_5))
        self.model.itemChanged.connect(self.update_first_button_state)
        self.model.itemChanged.connect(self.update_third_button_state)

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
        self.model.setHorizontalHeaderLabels(['Выбор файла', 'Путь к файлу эксперимента'])
        self.tableView.setModel(self.model)
        # Настройка ширины столбцов
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Устанавливаем фиксированную ширину для столбца с чекбоксами и именем файла
        header.resizeSection(0, 300)  # Подстраиваем под нужный размер
        # Настройка внешнего вида таблицы
        self.tableView.setShowGrid(True)  # Показать сетку
        # Устанавливаем размеры политики для таблицы, чтобы она заполняла все доступное пространство
        self.tableView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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

            # Элемент для пути к файлу
            file_path_item = QStandardItem(file_path)

            # Добавление строки в модель
            self.model.appendRow([check_and_name_item, file_path_item])

            # Устанавливаем высоту строк
            for row in range(self.model.rowCount()):
                self.tableView.setRowHeight(row, 20)  # Задаем желаемую высоту строки

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
        self.pushButton.setEnabled(oneExperimentSelected and anyCheckboxChecked)

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
        self.update_third_button_state()

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

        # Подготовка состояний чекбоксов
        state = (self.checkBox_3.isChecked(), self.checkBox_4.isChecked(),
                 self.checkBox_5.isChecked(), self.checkBox_6.isChecked())

        # Использование match-case для определения функции визуализации
        match state:
            case (True, False, True, False):
                plotting_func = TumorDataVisualizer.plot_tumor_volumes_single_graph
            case (False, True, True, False):
                plotting_func = TumorDataVisualizer.plot_relative_tumor_volumes_single_graph
            case (True, False, False, True):
                plotting_func = TumorDataVisualizer.plot_mean_tumor_volume
            case (False, True, False, True):
                plotting_func = TumorDataVisualizer.plot_average_relative_tumor_volume
            case _:
                print("Необходимо выбрать тип графика")
                return

        self.draw_graphic(selected_paths, TumorDataVisualizer, plotting_func)

    def handle_compare_rats(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 2:
            print("Необходимо выбрать два или более экспериментов")
            return

        # Подготовка состояний чекбоксов
        state = (self.checkBox_3.isChecked(), self.checkBox_4.isChecked(), self.checkBox_6.isChecked())
        # Использование match-case для определения функции визуализации
        match state:
            case (True, False, True):
                plotting_func = TumorDataComparatorAdvanced.compare_mean_volumes
            case (False, True, True):
                plotting_func = TumorDataComparatorAdvanced.compare_relative_volumes
            case _:
                print("Необходимо выбрать тип графика")
                return

        self.draw_graphic(selected_paths, TumorDataComparatorAdvanced, plotting_func)

    def handle_compare_with_control(self):
        selected_paths = self.get_selected_experiments()
        if len(selected_paths) < 2:
            print("Необходимо выбрать два или более экспериментов")
            return
    # TODO: Необходимо доделать

    def draw_graphic(self, selected_paths, visualizer, plotting_func):
        self.current_selected_paths = selected_paths
        self.current_visualizer = visualizer
        self.current_plotting_func = plotting_func
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
            plotting_func(visualizer)
            # После генерации графика нужно сохранить текущий рисунок в buf
            plt.savefig(buf, format='png')
            plt.close()  # Закрываем текущее окно plt, чтобы оно не отображалось
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            return pixmap

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
        anyCheckboxChecked = (
                                         self.checkBox_3.isChecked() or self.checkBox_4.isChecked()) and self.checkBox_6.isChecked()
        self.pushButton_3.setEnabled(oneExperimentSelected and anyCheckboxChecked)

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
        elif self.current_visualizer is TumorDataComparatorAdvanced:
            # Случай для сравнения нескольких экспериментов
            visualizer_instances = [TumorDataVisualizer(path) for path in self.current_selected_paths]
            visualizer_instance = self.current_visualizer(*visualizer_instances)

        pixmap = self.draw_figure_to_pixmap(visualizer_instance, self.current_plotting_func)

        # Отображаем созданный pixmap
        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        self.frame.layout().addWidget(label)

    def resizeEvent(self, event):
        """Вызывается при изменении размера окна."""
        super(MainWindow, self).resizeEvent(event)
        self.create_graphic()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
