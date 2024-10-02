from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QListView
from PyQt6.QtCore import Qt


class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)

        # Устанавливаем вид для элементов (чтобы можно было использовать чекбоксы)
        self.setView(QListView(self))

        # Модель для работы с элементами
        self.model = QStandardItemModel(self)
        self.setModel(self.model)

        # Добавляем примерные элементы с чекбоксами (или можно добавить позже)
        # self.add_checkable_items(['Option 1', 'Option 2', 'Option 3'])

    def add_checkable_items(self, items):
        """
        Добавляет элементы с чекбоксами в комбобокс
        """
        for item_text in items:
            item = QStandardItem(item_text)
            # Используем правильные флаги для PyQt6
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)  # По умолчанию чекбокс не отмечен
            self.model.appendRow(item)

    def checked_items(self):
        """
        Возвращает список отмеченных элементов
        """
        checked_items = []
        for index in range(self.model.rowCount()):
            item = self.model.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                checked_items.append(item.text())
        return checked_items

    def item_state_changed(self, index):
        """
        Обработчик изменений состояния чекбокса (при необходимости)
        """
        item = self.model.item(index)
        state = item.checkState()
        print(f'Item "{item.text()}" is now {"checked" if state == Qt.CheckState.Checked else "unchecked"}')
