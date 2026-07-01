import numbers

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


class DataFrameTableMixin:
    """Общая логика отображения pandas.DataFrame в QTableWidget — вынесена из
    TumorGrowthInhibitionTableWindow, чтобы не дублироваться в SkinReactionSummaryWindow."""

    def _populate_table(self, table_widget, df, column_tooltips, show_missing_as_dash):
        table_widget.clearContents()
        table_widget.setRowCount(df.shape[0])
        table_widget.setColumnCount(df.shape[1])
        table_widget.setHorizontalHeaderLabels([str(column) for column in df.columns])

        for column_index, column_name in enumerate(df.columns):
            header_item = table_widget.horizontalHeaderItem(column_index)
            if header_item is None:
                header_item = QTableWidgetItem(str(column_name))
                table_widget.setHorizontalHeaderItem(column_index, header_item)
            header_item.setToolTip(column_tooltips.get(column_name, str(column_name)))

        for row_index, (_, row) in enumerate(df.iterrows()):
            for column_index, value in enumerate(row):
                display_value = self._format_cell_value(value, column_index, show_missing_as_dash)
                item = QTableWidgetItem(display_value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(display_value)
                if display_value == "—":
                    item.setForeground(table_widget.palette().mid())
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif isinstance(value, numbers.Real) and not pd.isna(value):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                table_widget.setItem(row_index, column_index, item)

        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()

    @staticmethod
    def _format_cell_value(value, column_index, show_missing_as_dash):
        if pd.isna(value):
            return "—" if show_missing_as_dash else ""
        if isinstance(value, bool):
            return "Да" if value else "Нет"
        if isinstance(value, numbers.Integral):
            return str(int(value))
        if isinstance(value, numbers.Real):
            numeric_value = float(value)
            if column_index == 0 and numeric_value.is_integer():
                return str(int(numeric_value))
            if numeric_value.is_integer():
                return str(int(numeric_value))
            return f"{numeric_value:.3f}"
        return str(value)

    # Тот же тёмный заголовок таблицы, что задан в MainWindow._apply_stylesheet — там стиль
    # применён через self.setStyleSheet(...) на самом MainWindow, а отдельные диалоги
    # (TumorGrowthInhibitionTableWindow, SkinReactionSummaryWindow) — самостоятельные
    # top-level окна, и каскадом этот стиль на них не доходит, поэтому дублируем здесь.
    _TABLE_STYLESHEET = """
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
    """

    @classmethod
    def _configure_data_table(cls, table_widget, stretch_last_section):
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        table_widget.setWordWrap(False)
        table_widget.setAlternatingRowColors(True)
        table_widget.verticalHeader().setVisible(False)
        table_widget.horizontalHeader().setStretchLastSection(stretch_last_section)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table_widget.setSortingEnabled(False)
        table_widget.setStyleSheet(cls._TABLE_STYLESHEET)
