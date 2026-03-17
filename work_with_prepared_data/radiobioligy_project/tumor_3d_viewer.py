# coding: utf-8
"""Standalone PyQt6 viewer for 3D tumor geometry reconstructed from a-b-c axes."""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from work_with_prepared_data.radiobioligy_project.data_processing.tumor_geometry_processor import (
    TumorGeometryDataset,
    ellipsoid_volume_from_diameters,
    process_tumor_geometry_excel,
)

SLIDER_SUBSTEPS_PER_DAY = 12
PLAYBACK_INTERVAL_MS = 45


def interpolate_scalar(start: float, end: float, fraction: float) -> float:
    return float((1.0 - fraction) * start + fraction * end)


def format_interpolated_day_label(start_label: str, end_label: str, fraction: float) -> str:
    if fraction <= 0.0 or start_label == end_label:
        return start_label

    try:
        start_value = float(start_label)
        end_value = float(end_label)
    except ValueError:
        return f"{start_label} -> {end_label} ({fraction:.0%})"

    interpolated = interpolate_scalar(start_value, end_value, fraction)
    if abs(interpolated - round(interpolated)) < 1e-9:
        return str(int(round(interpolated)))
    return f"{interpolated:.2f}".rstrip("0").rstrip(".")


class Tumor3DViewerWindow(QMainWindow):
    """Inspect tumor geometry over time using measured ellipsoid axes."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset: Optional[TumorGeometryDataset] = None
        self._updating_day_table = False
        self.slider_substeps = SLIDER_SUBSTEPS_PER_DAY
        self.timer = QTimer(self)
        self.timer.setInterval(PLAYBACK_INTERVAL_MS)
        self.timer.timeout.connect(self.advance_day)
        self.setWindowTitle("Tumor 3D viewer")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)

        root_layout.addWidget(self._build_file_group())
        root_layout.addWidget(self._build_control_group())

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_info_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Open an Excel file with a-b-c tumor measurements.")
        self._set_controls_enabled(False)

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Input", self)
        layout = QHBoxLayout(group)

        self.file_label = QLabel("No file loaded", self)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.file_label, 1)

        open_button = QPushButton("Open Excel file", self)
        open_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(open_button)
        return group

    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("View controls", self)
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Tumor"), 0, 0)
        self.rat_combo = QComboBox(self)
        self.rat_combo.currentIndexChanged.connect(self.on_selection_changed)
        layout.addWidget(self.rat_combo, 0, 1)

        layout.addWidget(QLabel("Render"), 0, 2)
        self.render_combo = QComboBox(self)
        self.render_combo.addItem("Wireframe", "wireframe")
        self.render_combo.addItem("Surface", "surface")
        self.render_combo.currentIndexChanged.connect(self.refresh_view)
        layout.addWidget(self.render_combo, 0, 3)

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button, 0, 4)

        self.day_value_label = QLabel("Day: -", self)
        layout.addWidget(self.day_value_label, 1, 0)

        self.day_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.day_slider.setMinimum(0)
        self.day_slider.setSingleStep(1)
        self.day_slider.setPageStep(self.slider_substeps)
        self.day_slider.valueChanged.connect(self.refresh_view)
        layout.addWidget(self.day_slider, 1, 1, 1, 4)

        return group

    def _build_canvas_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        return panel

    def _build_info_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("Geometry summary"))
        self.info_text = QPlainTextEdit(self)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text, 1)

        layout.addWidget(QLabel("Per-day geometry"))
        self.day_table = QTableWidget(self)
        self.day_table.setColumnCount(6)
        self.day_table.setHorizontalHeaderLabels(["Day", "a", "b", "c", "Volume", "Source"])
        self.day_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.day_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.day_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.day_table.verticalHeader().setVisible(False)
        self.day_table.horizontalHeader().setStretchLastSection(True)
        self.day_table.cellClicked.connect(self.on_day_table_clicked)
        layout.addWidget(self.day_table, 2)
        return panel

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.rat_combo.setEnabled(enabled)
        self.render_combo.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.day_slider.setEnabled(enabled)
        self.day_table.setEnabled(enabled)

    def dragEnterEvent(self, event) -> None:  # pragma: no cover - GUI event
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".xlsx"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # pragma: no cover - GUI event
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".xlsx":
                self.load_dataset(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open tumor geometry Excel file",
            str(Path.cwd()),
            "Excel files (*.xlsx)",
        )
        if file_path:
            self.load_dataset(Path(file_path))

    def load_dataset(self, path: Path) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.dataset = process_tumor_geometry_excel(path)
            self.populate_dataset_controls()
            self.update_day_table()
            self.refresh_view()
            self.file_label.setText(str(path.resolve()))
            self.statusBar().showMessage(
                f"Loaded {path.name}: {self.dataset.rat_count} rats, {self.dataset.time_count} time points."
            )
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.dataset = None
            self._set_controls_enabled(False)
            self.day_table.clearContents()
            self.day_table.setRowCount(0)
            self.file_label.setText("No file loaded")
            self.info_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Failed to open file", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def populate_dataset_controls(self) -> None:
        assert self.dataset is not None
        self.rat_combo.blockSignals(True)
        self.rat_combo.clear()
        self.rat_combo.addItem("Mean across rats", None)
        for rat_index, rat_label in enumerate(self.dataset.rat_labels):
            self.rat_combo.addItem(rat_label, rat_index)
        self.rat_combo.blockSignals(False)

        self.day_slider.blockSignals(True)
        self.day_slider.setMaximum(max((self.dataset.time_count - 1) * self.slider_substeps, 0))
        self.day_slider.setValue(0)
        self.day_slider.blockSignals(False)

        self.play_button.setText("Play")
        self.timer.stop()
        self._set_controls_enabled(True)

    def on_selection_changed(self) -> None:
        self.update_day_table()
        self.refresh_view()

    def toggle_playback(self) -> None:
        if self.dataset is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")

    def advance_day(self) -> None:
        if self.dataset is None:
            self.timer.stop()
            return
        current_day = self.day_slider.value()
        if current_day >= self.day_slider.maximum():
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self.day_slider.setValue(min(current_day + 1, self.day_slider.maximum()))

    def current_position(self) -> float:
        if self.dataset is None:
            return 0.0
        return self.day_slider.value() / float(self.slider_substeps)

    def current_position_components(self) -> tuple[int, int, float]:
        if self.dataset is None or self.dataset.time_count == 0:
            return 0, 0, 0.0
        position = min(max(self.current_position(), 0.0), self.dataset.time_count - 1)
        lower_index = int(math.floor(position))
        upper_index = min(lower_index + 1, self.dataset.time_count - 1)
        fraction = position - lower_index
        return lower_index, upper_index, fraction

    def reference_day_index(self) -> int:
        if self.dataset is None or self.dataset.time_count == 0:
            return 0
        lower_index, upper_index, fraction = self.current_position_components()
        if upper_index == lower_index:
            return lower_index
        return upper_index if fraction >= 0.5 else lower_index

    @staticmethod
    def is_valid_geometry(a: float, b: float, c: float, volume: float) -> bool:
        return bool(all(np.isfinite([a, b, c, volume])) and min(a, b, c) > 0.0)

    def geometry_for_day_index(self, day_index: int) -> tuple[float, float, float, float, str]:
        assert self.dataset is not None
        rat_index = self.rat_combo.currentData()
        if rat_index is None:
            mean_a, mean_b, mean_c, mean_volume = self.dataset.mean_geometry()
            return (
                float(mean_a[day_index]),
                float(mean_b[day_index]),
                float(mean_c[day_index]),
                float(mean_volume[day_index]),
                "Mean ellipsoid across rats",
            )

        return (
            float(self.dataset.axis_a[rat_index, day_index]),
            float(self.dataset.axis_b[rat_index, day_index]),
            float(self.dataset.axis_c[rat_index, day_index]),
            float(self.dataset.volumes[rat_index, day_index]),
            "Explicit a-b-c axes"
            if self.dataset.explicit_axes_mask[rat_index, day_index]
            else "Equivalent sphere from scalar volume",
        )

    def current_geometry(self) -> Optional[dict]:
        if self.dataset is None:
            return None

        lower_index, upper_index, fraction = self.current_position_components()
        start_a, start_b, start_c, start_volume, start_source = self.geometry_for_day_index(lower_index)
        end_a, end_b, end_c, end_volume, end_source = self.geometry_for_day_index(upper_index)

        start_valid = self.is_valid_geometry(start_a, start_b, start_c, start_volume)
        end_valid = self.is_valid_geometry(end_a, end_b, end_c, end_volume)

        if upper_index == lower_index or fraction <= 0.0:
            a, b, c, volume, shape_source = start_a, start_b, start_c, start_volume, start_source
            fraction = 0.0
        elif start_valid and end_valid:
            a = interpolate_scalar(start_a, end_a, fraction)
            b = interpolate_scalar(start_b, end_b, fraction)
            c = interpolate_scalar(start_c, end_c, fraction)
            volume = interpolate_scalar(start_volume, end_volume, fraction)
            shape_source = f"Interpolated geometry: {start_source} -> {end_source}"
        elif start_valid and not end_valid:
            a, b, c, volume, shape_source = start_a, start_b, start_c, start_volume, start_source
        elif end_valid:
            a, b, c, volume, shape_source = end_a, end_b, end_c, end_volume, end_source
        else:
            a, b, c, volume, shape_source = start_a, start_b, start_c, start_volume, start_source

        displayed_volume = (
            ellipsoid_volume_from_diameters(a, b, c)
            if all(np.isfinite([a, b, c])) and min(a, b, c) > 0.0
            else float("nan")
        )
        return {
            "a": a,
            "b": b,
            "c": c,
            "volume": volume,
            "shape_source": shape_source,
            "displayed_volume": displayed_volume,
            "lower_index": lower_index,
            "upper_index": upper_index,
            "fraction": fraction,
            "day_label": format_interpolated_day_label(
                self.dataset.time_data[lower_index],
                self.dataset.time_data[upper_index],
                fraction,
            ),
        }

    def geometry_rows_for_current_selection(self) -> list[tuple[str, float, float, float, float, str]]:
        if self.dataset is None:
            return []

        rat_index = self.rat_combo.currentData()
        if rat_index is None:
            mean_a, mean_b, mean_c, mean_volume = self.dataset.mean_geometry()
            return [
                (
                    self.dataset.time_data[day_index],
                    float(mean_a[day_index]),
                    float(mean_b[day_index]),
                    float(mean_c[day_index]),
                    float(mean_volume[day_index]),
                    f"{int(np.sum(self.dataset.explicit_axes_mask[:, day_index]))}/{self.dataset.rat_count} axes",
                )
                for day_index in range(self.dataset.time_count)
            ]

        return [
            (
                self.dataset.time_data[day_index],
                float(self.dataset.axis_a[rat_index, day_index]),
                float(self.dataset.axis_b[rat_index, day_index]),
                float(self.dataset.axis_c[rat_index, day_index]),
                float(self.dataset.volumes[rat_index, day_index]),
                "axes" if self.dataset.explicit_axes_mask[rat_index, day_index] else "sphere",
            )
            for day_index in range(self.dataset.time_count)
        ]

    @staticmethod
    def format_number(value: float) -> str:
        if not np.isfinite(value):
            return "-"
        return f"{value:.3f}"

    def update_day_table(self) -> None:
        rows = self.geometry_rows_for_current_selection()
        self._updating_day_table = True
        try:
            self.day_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = [
                    row[0],
                    self.format_number(row[1]),
                    self.format_number(row[2]),
                    self.format_number(row[3]),
                    self.format_number(row[4]),
                    row[5],
                ]
                for column_index, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if 0 < column_index < 5:
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                        )
                    self.day_table.setItem(row_index, column_index, item)
            self.day_table.resizeColumnsToContents()
            self.highlight_day_row(self.reference_day_index())
        finally:
            self._updating_day_table = False

    def highlight_day_row(self, day_index: int) -> None:
        if self.day_table.rowCount() == 0 or day_index < 0 or day_index >= self.day_table.rowCount():
            return
        self._updating_day_table = True
        try:
            self.day_table.selectRow(day_index)
        finally:
            self._updating_day_table = False

    def on_day_table_clicked(self, row: int, _column: int) -> None:
        if self._updating_day_table:
            return
        target_value = row * self.slider_substeps
        if target_value != self.day_slider.value():
            self.day_slider.setValue(target_value)

    def current_axis_limit(self) -> float:
        rows = self.geometry_rows_for_current_selection()
        extents = [
            max(row[1], row[2], row[3])
            for row in rows
            if all(np.isfinite([row[1], row[2], row[3]])) and min(row[1], row[2], row[3]) > 0.0
        ]
        if not extents:
            return 1.0
        return max(max(extents) * 0.65, 1.0)

    def refresh_view(self) -> None:
        if self.dataset is None:
            self.figure.clear()
            self.canvas.draw_idle()
            self.info_text.clear()
            self.day_table.clearContents()
            self.day_table.setRowCount(0)
            self.day_value_label.setText("Day: -")
            return

        reference_day_index = self.reference_day_index()
        geometry = self.current_geometry()
        day_label = geometry["day_label"] if geometry is not None else "-"
        self.day_value_label.setText(f"Day: {day_label}")
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.view_init(elev=22, azim=34)

        if geometry is None:
            ax.set_title("No geometry available")
            self.info_text.setPlainText(self.build_info_text(None, reference_day_index))
            self.highlight_day_row(reference_day_index)
            self.canvas.draw_idle()
            return

        a = float(geometry["a"])
        b = float(geometry["b"])
        c = float(geometry["c"])
        volume = float(geometry["volume"])
        if not self.is_valid_geometry(a, b, c, volume):
            ax.set_title(f"No measurement for day {day_label}")
            ax.set_axis_off()
            self.info_text.setPlainText(self.build_info_text(None, reference_day_index, geometry))
            self.highlight_day_row(reference_day_index)
            self.canvas.draw_idle()
            return

        x, y, z = self.build_ellipsoid_mesh(a, b, c)
        if self.render_combo.currentData() == "surface":
            ax.plot_surface(
                x,
                y,
                z,
                color="#8ecae6",
                edgecolor="#1d3557",
                linewidth=0.25,
                alpha=0.8,
                shade=True,
            )
        else:
            ax.plot_wireframe(
                x,
                y,
                z,
                color="#1d3557",
                linewidth=0.8,
                rstride=1,
                cstride=1,
            )

        # Overlay the three measured diameters.
        ax.plot([-a / 2.0, a / 2.0], [0.0, 0.0], [0.0, 0.0], color="#d62828", linewidth=2.0)
        ax.plot([0.0, 0.0], [-b / 2.0, b / 2.0], [0.0, 0.0], color="#2a9d8f", linewidth=2.0)
        ax.plot([0.0, 0.0], [0.0, 0.0], [-c / 2.0, c / 2.0], color="#f4a261", linewidth=2.0)

        limit = self.current_axis_limit()
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_xlabel("a-axis")
        ax.set_ylabel("b-axis")
        ax.set_zlabel("c-axis")
        ax.set_title(f"{self.current_selection_label()} | day {day_label}")

        self.info_text.setPlainText(self.build_info_text(geometry, reference_day_index, geometry))
        self.highlight_day_row(reference_day_index)
        self.canvas.draw_idle()

    def current_selection_label(self) -> str:
        if self.dataset is None:
            return "No selection"
        rat_index = self.rat_combo.currentData()
        if rat_index is None:
            return "Mean across rats"
        return str(self.dataset.rat_labels[rat_index])

    @staticmethod
    def build_ellipsoid_mesh(a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.linspace(0.0, 2.0 * math.pi, 48)
        v = np.linspace(0.0, math.pi, 24)
        uu, vv = np.meshgrid(u, v)
        x = (a / 2.0) * np.cos(uu) * np.sin(vv)
        y = (b / 2.0) * np.sin(uu) * np.sin(vv)
        z = (c / 2.0) * np.cos(vv)
        return x, y, z

    def build_info_text(
        self,
        geometry: Optional[dict],
        day_index: Optional[int],
        position_geometry: Optional[dict] = None,
    ) -> str:
        if self.dataset is None:
            return ""

        lines = [
            f"File: {self.dataset.path.name}",
            f"Tumor: {self.current_selection_label()}",
        ]
        if position_geometry is not None:
            lines.append(f"Displayed day: {position_geometry['day_label']}")
        elif day_index is not None:
            lines.append(f"Day: {self.dataset.time_data[day_index]}")
        if self.dataset.experiment_params:
            lines.append("")
            lines.append("Experiment parameters:")
            for param in self.dataset.experiment_params:
                lines.append(f"- {param}")

        if geometry is None:
            lines.append("")
            lines.append("No geometry is available for this day.")
            return "\n".join(lines)

        if position_geometry is not None:
            lower_index = int(position_geometry["lower_index"])
            upper_index = int(position_geometry["upper_index"])
            fraction = float(position_geometry["fraction"])
            if lower_index != upper_index and fraction > 0.0:
                lines.extend(
                    [
                        "",
                        "Transition:",
                        f"- from day {self.dataset.time_data[lower_index]}",
                        f"- to day {self.dataset.time_data[upper_index]}",
                        f"- interpolation = {fraction:.0%}",
                    ]
                )

        lines.extend(
            [
                "",
                f"a = {geometry['a']:.3f}",
                f"b = {geometry['b']:.3f}",
                f"c = {geometry['c']:.3f}",
                f"Recorded volume = {geometry['volume']:.3f}",
                f"Displayed ellipsoid volume = {geometry['displayed_volume']:.3f}",
                f"Shape source = {geometry['shape_source']}",
            ]
        )

        if self.rat_combo.currentData() is None and self.dataset is not None:
            explicit_count = int(np.sum(self.dataset.explicit_axes_mask[:, day_index]))
            lines.append(
                f"Explicit a-b-c measurements on this day: {explicit_count}/{self.dataset.rat_count}"
            )

        return "\n".join(lines)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Tumor3DViewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
