import io
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
from PyQt6.QtCore import QStandardPaths, Qt
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QMenuBar, QMessageBox,
    QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import RatSurvivalEvent
from work_with_prepared_data.radiobioligy_project.stats_methods.kaplan_meier import (
    hazard_ratio_log_rank,
    kaplan_meier_estimate,
    log_rank_test,
    max_observed_day,
    median_survival_ci,
    n_at_risk_table,
    plot_kaplan_meier,
    restricted_mean_survival_time,
    risk_table_time_points,
)


class _ScalingImageLabel(QLabel):
    """
    QLabel, хранящий исходный pixmap в полном размере и пересчитывающий масштаб
    (с сохранением пропорций) при каждом изменении размера — иначе при сужении
    окна картинка не уменьшается, а обрезается (setFixedSize/обычный setPixmap
    не реагируют на последующий resize виджета).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_pixmap: Optional[QPixmap] = None
        self.setMinimumSize(300, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_original_pixmap(self, pixmap: Optional[QPixmap]):
        self._original_pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.clear()
        else:
            self._rescale()

    def sizeHint(self):
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            return self._original_pixmap.size()
        return super().sizeHint()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class KaplanMeierWindow(QDialog):
    """Отдельное окно для кривых выживаемости (Каплан-Майер) по группам крыс."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Каплан-Майер: выживаемость")
        self.setMinimumSize(550, 500)

        self.plot_label = None
        self.risk_table = None
        self.summary_table = None
        self.log_rank_label = None
        self.action_save_graph = None
        self._current_figure = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setMenuBar(self._build_menu_bar())

        title_label = QLabel("Кривые выживаемости по группам")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title_label)

        plot_frame = QFrame()
        plot_frame.setObjectName("kmPlotFrame")
        plot_frame.setStyleSheet(
            "#kmPlotFrame { border: 1px solid #c0c0c0; border-radius: 4px; background-color: white; }"
        )
        plot_frame_layout = QVBoxLayout(plot_frame)
        plot_frame_layout.setContentsMargins(12, 12, 12, 12)
        self.plot_label = _ScalingImageLabel()
        plot_frame_layout.addWidget(self.plot_label)
        layout.addWidget(plot_frame, stretch=1)

        risk_table_label = QLabel("Число в риске")
        risk_table_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(risk_table_label)

        self.risk_table = QTableWidget()
        self._configure_table(self.risk_table)
        layout.addWidget(self.risk_table)

        self.summary_table = QTableWidget()
        self._configure_table(self.summary_table)
        layout.addWidget(self.summary_table)

        bottom_layout = QHBoxLayout()
        self.log_rank_label = QLabel()
        self.log_rank_label.setWordWrap(True)
        bottom_layout.addWidget(self.log_rank_label)
        layout.addLayout(bottom_layout)

    def _build_menu_bar(self) -> QMenuBar:
        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("Файл")
        self.action_save_graph = QAction("Сохранить график...", self)
        self.action_save_graph.triggered.connect(self._handle_save_graph)
        self.action_save_graph.setEnabled(False)
        file_menu.addAction(self.action_save_graph)
        return menu_bar

    def set_groups(self, groups: Dict[str, Sequence[RatSurvivalEvent]]):
        """
        Строит график и сводную таблицу по группам.

        Args:
            groups: {имя группы: список RatSurvivalEvent}. Пустые группы игнорируются.
        """
        groups = {name: events for name, events in groups.items() if events}
        if not groups:
            self.plot_label.set_original_pixmap(None)
            self.plot_label.setText("Нет данных для построения (пустой список событий).")
            self.risk_table.clearContents()
            self.risk_table.setRowCount(0)
            self._fit_table_height(self.risk_table)
            self.summary_table.clearContents()
            self.summary_table.setRowCount(0)
            self._fit_table_height(self.summary_table)
            self.log_rank_label.clear()
            if self._current_figure is not None:
                plt.close(self._current_figure)
                self._current_figure = None
            self.action_save_graph.setEnabled(False)
            return

        if self._current_figure is not None:
            plt.close(self._current_figure)

        fig, _ax = plot_kaplan_meier(groups, show_ci=True)
        self._current_figure = fig
        self.action_save_graph.setEnabled(True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.plot_label.set_original_pixmap(pixmap)

        self._populate_risk_table(groups)
        self._populate_summary_table(groups)
        self._update_log_rank_label(groups)
        self.resize(self.sizeHint())

    def _handle_save_graph(self):
        """Сохраняет текущий график в файл — повторяет save_graph() главного окна."""
        if self._current_figure is None:
            QMessageBox.information(self, "Каплан-Майер", "Нет графика для сохранения.")
            return

        try:
            default_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить график",
                f"{default_path}/kaplan_meier.png",
                "PNG файлы (*.png);;JPEG файлы (*.jpg);;PDF файлы (*.pdf);;SVG файлы (*.svg);;Все файлы (*)",
            )
            if file_path:
                self._current_figure.savefig(file_path, dpi=300, bbox_inches='tight',
                                              facecolor='white', edgecolor='none')
                QMessageBox.information(self, "Каплан-Майер", "График успешно сохранён!")
        except Exception as error:
            QMessageBox.critical(self, "Каплан-Майер", f"Ошибка при сохранении графика: {error}")

    def _populate_risk_table(self, groups: Dict[str, Sequence[RatSurvivalEvent]]):
        # Сетка времени общая для всех групп (иначе таблицу нельзя сравнивать по столбцам),
        # но у каждой группы свой реальный срок наблюдения (файлы независимы и не обязаны
        # совпадать по длительности). Точки за пределами СОБСТВЕННОГО максимума группы
        # помечаются «—», а не «0» — иначе выглядит так, будто все группы наблюдались
        # одинаково долго, тогда как кто-то завершился раньше.
        time_points = risk_table_time_points(groups)
        counts = n_at_risk_table(groups, time_points)
        group_max_day = {name: max_observed_day(events) for name, events in groups.items()}

        headers = ["Группа"] + [f"{t:.0f}" for t in time_points]
        self.risk_table.clear()
        self.risk_table.setColumnCount(len(headers))
        self.risk_table.setHorizontalHeaderLabels(headers)
        self.risk_table.setRowCount(len(groups))

        for row, name in enumerate(groups.keys()):
            max_day = group_max_day[name]
            cells = [
                "—" if max_day is None or t > max_day else str(n)
                for t, n in zip(time_points, counts[name])
            ]
            values = [name] + cells
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.risk_table.setItem(row, col, item)

        self.risk_table.resizeColumnsToContents()
        self.risk_table.resizeRowsToContents()
        self.risk_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._fit_table_height(self.risk_table)

    def _populate_summary_table(self, groups: Dict[str, Sequence[RatSurvivalEvent]]):
        headers = [
            "Группа", "N", "Смертей", "Цензурировано",
            "Медиана, сут.", "95% ДИ медианы", "RMST, сут.",
        ]
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

            ci_low, ci_high = median_survival_ci(km)
            if ci_low is None and ci_high is None:
                median_ci = "—"
            else:
                low_text = f"{ci_low:.0f}" if ci_low is not None else "?"
                high_text = f"{ci_high:.0f}" if ci_high is not None else "не достигнута"
                median_ci = f"[{low_text}, {high_text}]"

            rmst = restricted_mean_survival_time(km)

            values = [name, str(n_total), str(n_events), str(n_censored), median, median_ci, f"{rmst:.1f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.summary_table.setItem(row, col, item)

        self.summary_table.resizeColumnsToContents()
        self.summary_table.resizeRowsToContents()
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._fit_table_height(self.summary_table)

    def _update_log_rank_label(self, groups: Dict[str, Sequence[RatSurvivalEvent]]):
        if len(groups) != 2:
            self.log_rank_label.setText(
                "Лог-ранговый тест и hazard ratio доступны только для сравнения ровно двух "
                f"групп (сейчас групп: {len(groups)})."
            )
            return

        names = list(groups.keys())
        events_a, events_b = groups[names[0]], groups[names[1]]
        chi2_stat, p_value = log_rank_test(events_a, events_b)
        significance = "p < 0.05, различие значимо" if p_value < 0.05 else "различие не значимо"
        text = (
            f"Лог-ранговый тест «{names[0]}» vs «{names[1]}»: "
            f"χ² = {chi2_stat:.3f}, p = {p_value:.4f} ({significance})."
        )

        hr_result = hazard_ratio_log_rank(events_a, events_b)
        if hr_result is None:
            text += " HR не вычислен (нет общих интервалов риска с событиями)."
        else:
            text += (
                f"\nHazard ratio «{names[0]}» относительно «{names[1]}»: "
                f"HR = {hr_result.hazard_ratio:.2f}, 95% ДИ [{hr_result.ci_lower:.2f}, {hr_result.ci_upper:.2f}] "
                f"(HR > 1 — выше риск смерти в «{names[0]}»)."
            )
        self.log_rank_label.setText(text)

    @staticmethod
    def _configure_table(table_widget):
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_widget.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table_widget.setAlternatingRowColors(True)
        table_widget.verticalHeader().setVisible(False)
        table_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table_widget.setSizePolicy(table_widget.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed)

    @staticmethod
    def _fit_table_height(table_widget):
        """
        Подгоняет высоту таблицы точно под её содержимое (заголовок + все строки),
        без обрезки и без лишней пустой области — высота зависит от числа групп,
        а не от заранее угаданной константы.
        """
        header_height = table_widget.horizontalHeader().height()
        rows_height = sum(table_widget.rowHeight(row) for row in range(table_widget.rowCount()))
        frame = 2 * table_widget.frameWidth()
        total = header_height + rows_height + frame + 4
        table_widget.setFixedHeight(max(total, header_height + frame + 4))
