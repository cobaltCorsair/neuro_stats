import io
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout

from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import RatSurvivalEvent
from work_with_prepared_data.radiobioligy_project.stats_methods.kaplan_meier import (
    kaplan_meier_estimate,
    log_rank_test,
    plot_kaplan_meier,
)


class KaplanMeierWindow(QDialog):
    """Отдельное окно для кривых выживаемости (Каплан-Майер) по группам крыс."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Каплан-Майер: выживаемость")
        self.setMinimumSize(900, 700)

        self.plot_label = None
        self.summary_table = None
        self.log_rank_label = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title_label = QLabel("Кривые выживаемости по группам")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title_label)

        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setMinimumHeight(420)
        layout.addWidget(self.plot_label, stretch=1)

        self.summary_table = QTableWidget()
        self._configure_table(self.summary_table)
        layout.addWidget(self.summary_table)

        bottom_layout = QHBoxLayout()
        self.log_rank_label = QLabel()
        self.log_rank_label.setWordWrap(True)
        bottom_layout.addWidget(self.log_rank_label)
        layout.addLayout(bottom_layout)

    def set_groups(self, groups: Dict[str, Sequence[RatSurvivalEvent]]):
        """
        Строит график и сводную таблицу по группам.

        Args:
            groups: {имя группы: список RatSurvivalEvent}. Пустые группы игнорируются.
        """
        groups = {name: events for name, events in groups.items() if events}
        if not groups:
            self.plot_label.clear()
            self.plot_label.setText("Нет данных для построения (пустой список событий).")
            self.summary_table.clearContents()
            self.summary_table.setRowCount(0)
            self.log_rank_label.clear()
            return

        fig, _ax = plot_kaplan_meier(groups, show_ci=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.plot_label.setPixmap(pixmap.scaled(
            self.plot_label.width() or 800, self.plot_label.height() or 420,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        ))

        self._populate_summary_table(groups)
        self._update_log_rank_label(groups)

    def _populate_summary_table(self, groups: Dict[str, Sequence[RatSurvivalEvent]]):
        headers = ["Группа", "N", "Смертей", "Цензурировано", "Медиана выживаемости, сут."]
        self.summary_table.clear()
        self.summary_table.setColumnCount(len(headers))
        self.summary_table.setHorizontalHeaderLabels(headers)
        self.summary_table.setRowCount(len(groups))

        for row, (name, events) in enumerate(groups.items()):
            km = kaplan_meier_estimate(events)
            n_total = sum(1 for e in events if e.day is not None)
            n_events = sum(1 for e in events if e.event_observed)
            n_censored = n_total - n_events
            median = "—" if km.median_survival is None else f"{km.median_survival:.0f}"

            values = [name, str(n_total), str(n_events), str(n_censored), median]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.summary_table.setItem(row, col, item)

        self.summary_table.resizeColumnsToContents()
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    def _update_log_rank_label(self, groups: Dict[str, Sequence[RatSurvivalEvent]]):
        if len(groups) != 2:
            self.log_rank_label.setText(
                "Лог-ранговый тест доступен только для сравнения ровно двух групп "
                f"(сейчас групп: {len(groups)})."
            )
            return

        names = list(groups.keys())
        chi2_stat, p_value = log_rank_test(groups[names[0]], groups[names[1]])
        significance = " (p < 0.05, различие значимо)" if p_value < 0.05 else " (различие не значимо)"
        self.log_rank_label.setText(
            f"Лог-ранговый тест «{names[0]}» vs «{names[1]}»: "
            f"χ² = {chi2_stat:.3f}, p = {p_value:.4f}{significance}"
        )

    @staticmethod
    def _configure_table(table_widget):
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_widget.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table_widget.setAlternatingRowColors(True)
        table_widget.verticalHeader().setVisible(False)
        table_widget.setMaximumHeight(160)
