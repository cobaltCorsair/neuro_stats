from PyQt6.QtCore import QFileInfo
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableView, QVBoxLayout, QFileDialog, QHeaderView
import sys

# Импорт сгенерированного класса из gui.py
from gui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.action.triggered.connect(self.open_files)

        # Настраиваем модель для 3 столбцов
        self.model = QStandardItemModel(0, 3, self)
        self.model.setHorizontalHeaderLabels(['Выбор', 'Имя файла', 'Путь к файлу эксперимента'])
        self.tableView.setModel(self.model)

        # Настройка ширины столбцов
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        # Устанавливаем фиксированную ширину для столбца чекбоксов
        header.resizeSection(0, 50)  # Можете настроить это значение по вашему усмотрению
        header.resizeSection(1, 300)  # Можете настроить это значение по вашему усмотрению
        # Дополнительные настройки внешнего вида для похожести на Excel
        self.tableView.setShowGrid(True)  # Если нужна сетка как в Excel

    def open_files(self):
        # Диалоговое окно для выбора файлов экспериментов
        files, _ = QFileDialog.getOpenFileNames(self, "Открыть файлы эксперимента")

        # Обновляем модель для QTableView
        for file_path in files:
            file_name = QFileInfo(file_path).fileName()  # Получаем только имя файла
            file_dir = QFileInfo(file_path).path()  # Получаем путь к файлу без имени файла

            # Создаем чекбокс и ассоциируем его с файлом
            check_item = QStandardItem()
            check_item.setCheckable(True)
            check_item.setEditable(False)

            # Элементы для имени файла и пути
            file_name_item = QStandardItem(file_name)
            file_path_item = QStandardItem(file_dir+file_name)

            # Добавление строки в модель
            self.model.appendRow([check_item, file_name_item, file_path_item])

        # Расширение таблицы по мере добавления экспериментов
        self.tableView.resizeRowsToContents()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
