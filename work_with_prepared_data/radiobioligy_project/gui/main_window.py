import io

from PyQt6.QtCore import QFileInfo, Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView, QSizePolicy, QVBoxLayout, QMessageBox, \
    QLabel
import sys

# Импорт сгенерированного класса из gui.py
from gui import Ui_MainWindow
from work_with_prepared_data.radiobioligy_project.draw_base_graphs import TumorDataVisualizer
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ax = None
        self.canvas = None
        self.figure = None
        self.setupUi(self)
        self.action.triggered.connect(self.open_files)
        # Настраиваем модель для 2 столбцов
        self.model = QStandardItemModel(0, 2, self)
        self.change_table()
        # Биндинг кнопок
        self.pushButton.clicked.connect(self.handleAllOfRats)

    def change_table(self):
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
        # Диалоговое окно для выбора файлов экспериментов
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

    def getSelectedExperiments(self):
        """Собираем пути к выбранным экспериментам."""
        selected_paths = []
        for row in range(self.model.rowCount()):
            if self.model.item(row, 0).checkState() == Qt.CheckState.Checked:
                path = self.model.item(row, 1).text()  # Измените индекс, если путь хранится в другой колонке
                selected_paths.append(path)
        return selected_paths

    def handleAllOfRats(self):
        selected_paths = self.getSelectedExperiments()
        if len(selected_paths) < 1:
            print("Выберите хотя бы один эксперимент")
            return

        visualizer = TumorDataVisualizer(selected_paths[0])
        pixmap = self.draw_figure_to_pixmap(visualizer)

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

        # Создаем QLabel для отображения графика
        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # Указываем, что содержимое label должно масштабироваться с его размером

        # Добавляем label в layout frame
        self.frame.layout().addWidget(label)

        # Можно также настроить размер label, чтобы он соответствовал размеру frame, если нужно
        label.setMinimumSize(self.frame.size())

    def draw_figure_to_pixmap(self, visualizer):
        # Перенаправляем вывод графика в объект BytesIO вместо отображения в окне
        with io.BytesIO() as buf:
            visualizer.plot_tumor_volumes_single_graph()
            # После генерации графика нужно сохранить текущий рисунок в buf
            plt.savefig(buf, format='png')
            plt.close()  # Закрываем текущее окно plt, чтобы оно не отображалось
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            return pixmap


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
