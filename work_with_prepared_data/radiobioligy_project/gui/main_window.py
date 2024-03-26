from PyQt6.QtCore import QFileInfo, Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView, QSizePolicy
import sys

# Импорт сгенерированного класса из gui.py
from gui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.action.triggered.connect(self.open_files)
        # Настраиваем модель для 2 столбцов
        self.model = QStandardItemModel(0, 2, self)
        self.change_table()

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

    def adjust_table_view_height(self):
        # Получаем количество строк и высоту одной строки
        row_count = self.model.rowCount()
        row_height = self.tableView.rowHeight(0) if self.model.rowCount() > 0 else 20

        # Рассчитываем идеальную высоту таблицы, добавляем немного для заголовка
        ideal_height = row_count * row_height + self.tableView.horizontalHeader().height()

        # Применяем вычисленную высоту в пределах допустимого диапазона
        self.tableView.setFixedHeight(ideal_height)
        self.tableView.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Выключаем вертикальный скролл

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

        # Обновляем высоту таблицы
        self.adjust_table_view_height()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
