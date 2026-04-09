import numbers

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDialog, QHeaderView, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, \
    QTabWidget, QVBoxLayout, QWidget


class TumorGrowthInhibitionTableWindow(QDialog):
    """Отдельное окно для отображения основной таблицы ТРО и попарной сводки."""

    MODE_CONTROL_DAYS = "control_days"
    MODE_DAILY_INTERPOLATION = "daily_interpolation"
    MODE_TITLES = {
        MODE_CONTROL_DAYS: "По суткам контроля",
        MODE_DAILY_INTERPOLATION: "Ежедневная интерполяция",
    }
    MODE_NOTES = {
        MODE_CONTROL_DAYS: (
            "Основная таблица использует сутки контрольной группы. Значения ТРО для экспериментов "
            "линейно интерполируются на эти дни; на границах ряда используется ближайшее доступное значение."
        ),
        MODE_DAILY_INTERPOLATION: (
            "Основная таблица строится по ежедневной сетке от первого до последнего дня выбранных экспериментов. "
            "Значения ТРО линейно интерполируются между замерами; на границах ряда используется ближайшее "
            "доступное значение."
        ),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Таблицы ТРО")
        self.setMinimumSize(1100, 720)

        self.legend_table = None
        self.legend_note_label = None
        self.mode_selector = None
        self.tabs = None
        self.main_table = None
        self.summary_table = None
        self.summary_tab = None

        self._tables_by_mode = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title_label = QLabel("Таблицы торможения роста опухоли")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title_label)

        subtitle_label = QLabel(
            "Для компактности в таблицах используются коды экспериментов. "
            "Полные названия приведены ниже."
        )
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Сетка времени:")
        self.mode_selector = QComboBox()
        self._configure_mode_selector(self.mode_selector)
        self.mode_selector.currentIndexChanged.connect(self._handle_mode_changed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_selector, stretch=1)
        layout.addLayout(mode_layout)

        self.legend_table = QTableWidget()
        self._configure_legend_table(self.legend_table)
        layout.addWidget(self.legend_table)

        self.legend_note_label = QLabel()
        self.legend_note_label.setWordWrap(True)
        layout.addWidget(self.legend_note_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)
        self.main_table = QTableWidget()
        self._configure_data_table(self.main_table, stretch_last_section=False)
        main_tab_layout.addWidget(self.main_table)
        self.tabs.addTab(main_tab, "ТРО по времени")

        self.summary_tab = QWidget()
        summary_tab_layout = QVBoxLayout(self.summary_tab)
        self.summary_table = QTableWidget()
        self._configure_data_table(self.summary_table, stretch_last_section=True)
        summary_tab_layout.addWidget(self.summary_table)

    def set_tables(self, tables_or_tgi_df, pairwise_summary_df=None):
        tables_by_mode = self._normalize_tables_by_mode(tables_or_tgi_df, pairwise_summary_df)
        if not tables_by_mode:
            tables_by_mode = {
                self.MODE_CONTROL_DAYS: {
                    "tgi_df": pd.DataFrame(columns=["Время (сут)"]),
                    "pairwise_summary_df": None,
                }
            }

        previous_mode = self.mode_selector.currentData()
        self._tables_by_mode = tables_by_mode
        self._populate_mode_selector(tables_by_mode)

        selected_mode = previous_mode if previous_mode in tables_by_mode else self._preferred_mode(tables_by_mode)
        self._set_mode(selected_mode)

    @classmethod
    def _normalize_tables_by_mode(cls, tables_or_tgi_df, pairwise_summary_df):
        if isinstance(tables_or_tgi_df, dict):
            normalized = {}
            for mode_key, mode_value in tables_or_tgi_df.items():
                if isinstance(mode_value, dict):
                    tgi_df = mode_value.get("tgi_df")
                    mode_pairwise_summary_df = mode_value.get("pairwise_summary_df")
                else:
                    tgi_df, mode_pairwise_summary_df = mode_value
                normalized[mode_key] = {
                    "tgi_df": tgi_df if tgi_df is not None else pd.DataFrame(columns=["Время (сут)"]),
                    "pairwise_summary_df": mode_pairwise_summary_df,
                }
            return normalized

        return {
            cls.MODE_CONTROL_DAYS: {
                "tgi_df": tables_or_tgi_df if tables_or_tgi_df is not None else pd.DataFrame(columns=["Время (сут)"]),
                "pairwise_summary_df": pairwise_summary_df,
            }
        }

    @classmethod
    def _preferred_mode(cls, tables_by_mode):
        if cls.MODE_CONTROL_DAYS in tables_by_mode:
            return cls.MODE_CONTROL_DAYS
        return next(iter(tables_by_mode))

    def _populate_mode_selector(self, tables_by_mode):
        self.mode_selector.blockSignals(True)
        self.mode_selector.clear()

        for mode_key in self.MODE_TITLES:
            if mode_key in tables_by_mode:
                self.mode_selector.addItem(self.MODE_TITLES[mode_key], mode_key)

        for mode_key in tables_by_mode:
            if self.mode_selector.findData(mode_key) == -1:
                self.mode_selector.addItem(str(mode_key), mode_key)

        self.mode_selector.blockSignals(False)

    def _handle_mode_changed(self, _index):
        mode_key = self.mode_selector.currentData()
        if mode_key in self._tables_by_mode:
            self._apply_mode(mode_key)

    def _set_mode(self, mode_key):
        index = self.mode_selector.findData(mode_key)
        if index == -1:
            return

        self.mode_selector.blockSignals(True)
        self.mode_selector.setCurrentIndex(index)
        self.mode_selector.blockSignals(False)
        self._apply_mode(mode_key)

    def _apply_mode(self, mode_key):
        mode_data = self._tables_by_mode[mode_key]
        tgi_df = mode_data["tgi_df"]
        pairwise_summary_df = mode_data["pairwise_summary_df"]

        display_tgi_df, experiment_code_map = self._build_display_tgi_df(tgi_df)
        self._populate_legend_table(experiment_code_map)
        column_tooltips = {code: full_name for full_name, code in experiment_code_map.items()}
        self._populate_table(self.main_table, display_tgi_df, column_tooltips, show_missing_as_dash=True)

        if pairwise_summary_df is None:
            self._remove_summary_tab()
        else:
            display_pairwise_df = self._build_display_pairwise_df(pairwise_summary_df, experiment_code_map)
            self._ensure_summary_tab()
            self._populate_table(self.summary_table, display_pairwise_df, {}, show_missing_as_dash=True)

        self.legend_note_label.setText(
            f"{self.MODE_NOTES.get(mode_key, '')} В попарной сводке учитываются только точки, начиная с 9-х суток."
        )
        self.tabs.setCurrentIndex(0)

    @staticmethod
    def _build_display_tgi_df(tgi_df):
        experiment_columns = list(tgi_df.columns[1:])
        experiment_code_map = {
            experiment_name: f"Э{index}"
            for index, experiment_name in enumerate(experiment_columns, start=1)
        }

        renamed_columns = {"Время (сут)": "Время (сут)"}
        renamed_columns.update(experiment_code_map)
        return tgi_df.rename(columns=renamed_columns), experiment_code_map

    @staticmethod
    def _build_display_pairwise_df(pairwise_summary_df, experiment_code_map):
        display_pairwise_df = pairwise_summary_df.copy()

        def _format_pair(pair_label):
            if not isinstance(pair_label, str):
                return pair_label
            left_name, separator, right_name = pair_label.partition(" vs ")
            if not separator:
                return pair_label
            return f"{experiment_code_map.get(left_name, left_name)} vs {experiment_code_map.get(right_name, right_name)}"

        display_pairwise_df["Пара экспериментов"] = display_pairwise_df["Пара экспериментов"].map(_format_pair)
        return display_pairwise_df

    def _populate_legend_table(self, experiment_code_map):
        legend_df = pd.DataFrame({
            "Код": list(experiment_code_map.values()),
            "Эксперимент": list(experiment_code_map.keys()),
        })

        self.legend_table.setRowCount(legend_df.shape[0])
        self.legend_table.setColumnCount(legend_df.shape[1])
        self.legend_table.setHorizontalHeaderLabels([str(column) for column in legend_df.columns])
        header = self.legend_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        if legend_df.shape[1] > 0:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        if legend_df.shape[1] > 1:
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        for row_index, row in enumerate(legend_df.itertuples(index=False)):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(str(value))
                self.legend_table.setItem(row_index, column_index, item)

        self.legend_table.resizeRowsToContents()

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

    @staticmethod
    def _configure_legend_table(table_widget):
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_widget.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table_widget.setWordWrap(True)
        table_widget.setAlternatingRowColors(True)
        table_widget.verticalHeader().setVisible(False)
        table_widget.horizontalHeader().setStretchLastSection(True)
        table_widget.setMinimumHeight(140)

    @staticmethod
    def _configure_mode_selector(combo_box):
        combo_box.setStyleSheet(
            """
            QComboBox QAbstractItemView {
                background-color: palette(base);
                color: palette(text);
                selection-background-color: #dbe8f6;
                selection-color: #1f2d3d;
                outline: 0;
            }
            QComboBox QAbstractItemView::item {
                min-height: 24px;
                padding: 4px 8px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #e9f2fb;
                color: #1f2d3d;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #dbe8f6;
                color: #1f2d3d;
            }
            """
        )

    @staticmethod
    def _configure_data_table(table_widget, stretch_last_section):
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        table_widget.setWordWrap(False)
        table_widget.setAlternatingRowColors(True)
        table_widget.verticalHeader().setVisible(False)
        table_widget.horizontalHeader().setStretchLastSection(stretch_last_section)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table_widget.setSortingEnabled(False)

    def _ensure_summary_tab(self):
        if self.tabs.indexOf(self.summary_tab) == -1:
            self.tabs.addTab(self.summary_tab, "Попарная сводка")

    def _remove_summary_tab(self):
        summary_tab_index = self.tabs.indexOf(self.summary_tab)
        if summary_tab_index != -1:
            self.tabs.removeTab(summary_tab_index)
