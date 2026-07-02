import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QLabel, QTableWidget, QVBoxLayout

try:
    from gui.dataframe_table_widget import DataFrameTableMixin
except ImportError:
    from work_with_prepared_data.radiobioligy_project.gui.dataframe_table_widget import DataFrameTableMixin

SUMMARY_COLUMNS = [
    "Группа",
    "Пик, балл",
    "День пика",
    "Длительность >= порога, сут",
    "Цензурировано (длительность)",
    "День нормализации",
    "Цензурировано (нормализация)",
]


class SkinReactionSummaryWindow(QDialog, DataFrameTableMixin):
    """Отдельное окно со сводкой производных показателей кожной реакции по группам:
    пиковый балл/день, длительность реакции выше порога, время нормализации.
    В отличие от TumorGrowthInhibitionTableWindow не имеет выбора сетки времени — это
    скалярные показатели на группу, а не временной ряд."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Сводка кожных реакций")
        self.setMinimumSize(900, 500)

        self.table = QTableWidget()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title_label = QLabel("Производные показатели кожной реакции по группам")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title_label)

        self._configure_data_table(self.table, stretch_last_section=True)
        layout.addWidget(self.table)

    def set_summary_table(self, summary_df):
        if summary_df is None:
            summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        self._populate_table(self.table, summary_df, {}, show_missing_as_dash=True)
