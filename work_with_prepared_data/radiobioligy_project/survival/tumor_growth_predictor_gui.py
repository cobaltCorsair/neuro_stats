# coding: utf-8
"""Standalone PyQt6 window for tumor-growth prediction and ellipsoid dynamics."""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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
    process_tumor_geometry_excel,
)
from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    parse_fractions,
)
from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
    GeometryReference,
    GeometryScalingModel,
    GrowthModelParameters,
    GrowthSimulationResult,
    TreatmentFraction,
    build_default_schedule,
    default_geometry_scaling,
    fit_geometry_scaling,
    fit_gompertz_to_control,
    simulate_growth,
)

PLAYBACK_INTERVAL_MS = 60


def _parse_numeric_day_labels(labels: Sequence[str]) -> np.ndarray:
    values = []
    for index, label in enumerate(labels):
        try:
            values.append(float(label))
        except ValueError:
            values.append(float(index))
    return np.asarray(values, dtype=float)


class TumorGrowthPredictorWindow(QMainWindow):
    """Interactive tumor-growth predictor with plots and a 3D ellipsoid view."""

    def __init__(self, run_results: Optional[Sequence[AnalysisRunResult]] = None) -> None:
        super().__init__()
        self.run_results = list(run_results or [])
        self.geometry_dataset: Optional[TumorGeometryDataset] = None
        self.control_dataset: Optional[TumorGeometryDataset] = None
        self.simulation_result: Optional[GrowthSimulationResult] = None
        self.observed_days: Optional[np.ndarray] = None
        self.observed_volume: Optional[np.ndarray] = None
        self.observed_axis_a: Optional[np.ndarray] = None
        self.observed_axis_b: Optional[np.ndarray] = None
        self.observed_axis_c: Optional[np.ndarray] = None
        self.reference_geometry: Optional[GeometryReference] = None
        self.geometry_scaling_model: Optional[GeometryScalingModel] = None

        self.timer = QTimer(self)
        self.timer.setInterval(PLAYBACK_INTERVAL_MS)
        self.timer.timeout.connect(self.advance_prediction_frame)

        self.setWindowTitle("Tumor growth predictor")
        self.resize(1600, 980)
        self._build_ui()
        self.populate_fit_results()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Load a treated tumor file and simulate growth.")

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_parameter_group())
        layout.addWidget(self._build_schedule_group(), 1)

        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.simulate_button)
        layout.addStretch(1)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self._build_plot_panel())
        splitter.addWidget(self._build_3d_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return panel

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Files and selection", self)
        layout = QGridLayout(group)

        self.geometry_label = QLabel("No treated tumor file loaded", self)
        self.geometry_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        geometry_button = QPushButton("Open treated tumor file", self)
        geometry_button.clicked.connect(self.open_geometry_file)
        layout.addWidget(QLabel("Treated tumor"), 0, 0)
        layout.addWidget(self.geometry_label, 0, 1)
        layout.addWidget(geometry_button, 0, 2)

        self.control_label = QLabel("No control file loaded", self)
        self.control_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        control_button = QPushButton("Open control file", self)
        control_button.clicked.connect(self.open_control_file)
        layout.addWidget(QLabel("Control"), 1, 0)
        layout.addWidget(self.control_label, 1, 1)
        layout.addWidget(control_button, 1, 2)

        self.selection_combo = QComboBox(self)
        self.selection_combo.currentIndexChanged.connect(self.on_selection_changed)
        layout.addWidget(QLabel("Tumor"), 2, 0)
        layout.addWidget(self.selection_combo, 2, 1, 1, 2)

        self.fit_result_combo = QComboBox(self)
        self.fit_result_combo.currentIndexChanged.connect(self.apply_selected_fit_result)
        layout.addWidget(QLabel("Alpha/Beta source"), 3, 0)
        layout.addWidget(self.fit_result_combo, 3, 1, 1, 2)

        fit_button = QPushButton("Fit Gompertz from control", self)
        fit_button.clicked.connect(self.fit_growth_from_control)
        layout.addWidget(fit_button, 4, 0, 1, 3)
        return group

    def _build_parameter_group(self) -> QGroupBox:
        group = QGroupBox("Model parameters", self)
        layout = QGridLayout(group)

        self.alpha_spin = self._create_float_spinbox(0.0, 10.0, 0.013, decimals=6, step=0.001)
        self.beta_spin = self._create_float_spinbox(0.0, 10.0, 0.0016, decimals=6, step=0.0001)
        self.growth_rate_spin = self._create_float_spinbox(0.0, 5.0, 0.15, decimals=4, step=0.01)
        self.carrying_capacity_spin = self._create_float_spinbox(0.0, 100000.0, 100.0, decimals=3, step=1.0)
        self.clearance_rate_spin = self._create_float_spinbox(0.0, 5.0, 0.10, decimals=4, step=0.01)
        self.horizon_spin = self._create_float_spinbox(0.0, 365.0, 30.0, decimals=2, step=1.0)
        self.step_spin = self._create_float_spinbox(0.01, 10.0, 0.25, decimals=3, step=0.05)

        layout.addWidget(QLabel("alpha"), 0, 0)
        layout.addWidget(self.alpha_spin, 0, 1)
        layout.addWidget(QLabel("beta"), 0, 2)
        layout.addWidget(self.beta_spin, 0, 3)

        layout.addWidget(QLabel("Growth rate r"), 1, 0)
        layout.addWidget(self.growth_rate_spin, 1, 1)
        layout.addWidget(QLabel("Carrying capacity K"), 1, 2)
        layout.addWidget(self.carrying_capacity_spin, 1, 3)

        layout.addWidget(QLabel("Clearance rate"), 2, 0)
        layout.addWidget(self.clearance_rate_spin, 2, 1)
        layout.addWidget(QLabel("Horizon (days)"), 2, 2)
        layout.addWidget(self.horizon_spin, 2, 3)

        layout.addWidget(QLabel("Step (days)"), 3, 0)
        layout.addWidget(self.step_spin, 3, 1)

        layout.addWidget(QLabel("Geometry mode"), 3, 2)
        self.geometry_mode_combo = QComboBox(self)
        self.geometry_mode_combo.addItem("Fixed ratios", "fixed")
        self.geometry_mode_combo.addItem("Fit from observed shape", "fitted")
        self.geometry_mode_combo.currentIndexChanged.connect(self.on_geometry_mode_changed)
        layout.addWidget(self.geometry_mode_combo, 3, 3)
        return group

    def _build_schedule_group(self) -> QGroupBox:
        group = QGroupBox("Dose schedule", self)
        layout = QVBoxLayout(group)

        button_row = QHBoxLayout()
        prefill_button = QPushButton("Prefill from treated file fractions", self)
        prefill_button.clicked.connect(self.prefill_schedule_from_geometry)
        button_row.addWidget(prefill_button)

        add_button = QPushButton("Add event", self)
        add_button.clicked.connect(self.add_schedule_row)
        button_row.addWidget(add_button)

        remove_button = QPushButton("Remove selected", self)
        remove_button.clicked.connect(self.remove_selected_schedule_rows)
        button_row.addWidget(remove_button)

        clear_button = QPushButton("Clear schedule", self)
        clear_button.clicked.connect(self.clear_schedule)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.schedule_table = QTableWidget(self)
        self.schedule_table.setColumnCount(2)
        self.schedule_table.setHorizontalHeaderLabels(["Day", "Dose (Gy)"])
        self.schedule_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.schedule_table.setMinimumHeight(220)
        layout.addWidget(self.schedule_table, 1)
        return group

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        self.curve_figure = Figure(figsize=(9, 6))
        self.curve_canvas = FigureCanvasQTAgg(self.curve_figure)
        self.curve_toolbar = NavigationToolbar2QT(self.curve_canvas, self)
        layout.addWidget(self.curve_toolbar)
        layout.addWidget(self.curve_canvas, 1)
        return panel

    def _build_3d_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)

        self.frame_label = QLabel("Day: -", self)
        controls.addWidget(self.frame_label)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.refresh_3d_view)
        layout.addWidget(self.frame_slider)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        figure_panel = QWidget(self)
        figure_layout = QVBoxLayout(figure_panel)
        self.view_figure = Figure(figsize=(7, 5))
        self.view_canvas = FigureCanvasQTAgg(self.view_figure)
        figure_layout.addWidget(self.view_canvas, 1)
        bottom_splitter.addWidget(figure_panel)

        self.summary_text = QPlainTextEdit(self)
        self.summary_text.setReadOnly(True)
        bottom_splitter.addWidget(self.summary_text)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 2)
        layout.addWidget(bottom_splitter, 1)
        return panel

    @staticmethod
    def _create_float_spinbox(
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int,
        step: float,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def populate_fit_results(self) -> None:
        self.fit_result_combo.blockSignals(True)
        self.fit_result_combo.clear()
        self.fit_result_combo.addItem("Manual", None)

        successful_runs = [run for run in self.run_results if run.fit_result is not None]
        for run in successful_runs:
            fit = run.fit_result
            assert fit is not None
            label = f"{run.label} | alpha={fit.alpha:.5f} | beta={fit.beta:.5f}"
            self.fit_result_combo.addItem(label, fit)

        if successful_runs:
            self.fit_result_combo.setCurrentIndex(len(successful_runs))
        self.fit_result_combo.blockSignals(False)
        if successful_runs:
            self.apply_selected_fit_result()

    def apply_selected_fit_result(self) -> None:
        fit = self.fit_result_combo.currentData()
        if fit is None:
            return
        self.alpha_spin.setValue(fit.alpha)
        self.beta_spin.setValue(fit.beta)
        self.statusBar().showMessage("Loaded alpha/beta from fitter result.")

    def open_geometry_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open treated tumor Excel file",
            str(Path.cwd()),
            "Excel files (*.xlsx)",
        )
        if path_str:
            self.load_geometry_dataset(Path(path_str))

    def open_control_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open control Excel file",
            str(Path.cwd()),
            "Excel files (*.xlsx)",
        )
        if path_str:
            self.load_control_dataset(Path(path_str))

    def load_geometry_dataset(self, path: str | Path) -> None:
        path = Path(path)
        try:
            dataset = process_tumor_geometry_excel(path)
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.summary_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Failed to load treated tumor file", str(exc))
            return

        self.geometry_dataset = dataset
        self.geometry_label.setText(str(path.resolve()))
        self.populate_selection_combo()
        self.prefill_schedule_from_geometry()
        self.suggest_horizon_from_geometry()
        self.statusBar().showMessage(f"Loaded treated tumor file: {path.name}")

    def load_control_dataset(self, path: str | Path) -> None:
        path = Path(path)
        try:
            dataset = process_tumor_geometry_excel(path)
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.summary_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Failed to load control file", str(exc))
            return

        self.control_dataset = dataset
        self.control_label.setText(str(path.resolve()))
        self.statusBar().showMessage(f"Loaded control file: {path.name}")

    def populate_selection_combo(self) -> None:
        self.selection_combo.blockSignals(True)
        self.selection_combo.clear()
        self.selection_combo.addItem("Mean across rats", None)
        if self.geometry_dataset is not None:
            for rat_index, rat_label in enumerate(self.geometry_dataset.rat_labels):
                self.selection_combo.addItem(str(rat_label), rat_index)
        self.selection_combo.blockSignals(False)
        self.on_selection_changed()

    def on_selection_changed(self) -> None:
        self.update_observed_series()
        self.clear_prediction_outputs()

    def on_geometry_mode_changed(self) -> None:
        self.update_geometry_scaling_model()
        self.clear_prediction_outputs()

    def observed_geometry_for_selection(
        self,
        dataset: TumorGeometryDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        day_values = _parse_numeric_day_labels(dataset.time_data)
        selected_rat = self.selection_combo.currentData()
        if selected_rat is None:
            axis_a, axis_b, axis_c, volumes = dataset.mean_geometry()
            return day_values, axis_a, axis_b, axis_c, volumes
        return (
            day_values,
            dataset.axis_a[selected_rat],
            dataset.axis_b[selected_rat],
            dataset.axis_c[selected_rat],
            dataset.volumes[selected_rat],
        )

    def update_observed_series(self) -> None:
        if self.geometry_dataset is None:
            self.observed_days = None
            self.observed_volume = None
            self.observed_axis_a = None
            self.observed_axis_b = None
            self.observed_axis_c = None
            self.reference_geometry = None
            return

        day_values, axis_a, axis_b, axis_c, volumes = self.observed_geometry_for_selection(
            self.geometry_dataset
        )
        self.observed_days = np.asarray(day_values, dtype=float)
        self.observed_axis_a = np.asarray(axis_a, dtype=float)
        self.observed_axis_b = np.asarray(axis_b, dtype=float)
        self.observed_axis_c = np.asarray(axis_c, dtype=float)
        self.observed_volume = np.asarray(volumes, dtype=float)
        self.reference_geometry = self.find_reference_geometry()
        self.update_geometry_scaling_model()

    def update_geometry_scaling_model(self) -> None:
        if self.reference_geometry is None:
            self.geometry_scaling_model = None
            return

        mode = self.geometry_mode_combo.currentData()
        if mode == "fitted":
            self.geometry_scaling_model = fit_geometry_scaling(
                self.observed_volume if self.observed_volume is not None else [],
                self.observed_axis_a if self.observed_axis_a is not None else [],
                self.observed_axis_b if self.observed_axis_b is not None else [],
                self.observed_axis_c if self.observed_axis_c is not None else [],
                self.reference_geometry,
            )
            return

        self.geometry_scaling_model = default_geometry_scaling(self.reference_geometry)

    def find_reference_geometry(self) -> Optional[GeometryReference]:
        if (
            self.observed_axis_a is None
            or self.observed_axis_b is None
            or self.observed_axis_c is None
            or self.observed_volume is None
        ):
            return None

        valid_mask = (
            np.isfinite(self.observed_axis_a)
            & np.isfinite(self.observed_axis_b)
            & np.isfinite(self.observed_axis_c)
            & np.isfinite(self.observed_volume)
            & (self.observed_axis_a > 0.0)
            & (self.observed_axis_b > 0.0)
            & (self.observed_axis_c > 0.0)
            & (self.observed_volume > 0.0)
        )
        if not np.any(valid_mask):
            return None

        first_index = int(np.where(valid_mask)[0][0])
        return GeometryReference(
            axis_a=float(self.observed_axis_a[first_index]),
            axis_b=float(self.observed_axis_b[first_index]),
            axis_c=float(self.observed_axis_c[first_index]),
            volume=float(self.observed_volume[first_index]),
        )

    def suggest_horizon_from_geometry(self) -> None:
        if self.geometry_dataset is None:
            return
        observed_days = _parse_numeric_day_labels(self.geometry_dataset.time_data)
        if len(observed_days) > 0:
            self.horizon_spin.setValue(max(float(np.nanmax(observed_days)), self.horizon_spin.value()))

    def fit_growth_from_control(self) -> None:
        if self.control_dataset is None:
            QMessageBox.warning(self, "No control", "Load a control file first.")
            return

        try:
            control_days = _parse_numeric_day_labels(self.control_dataset.time_data)
            _, _, _, control_volumes = self.control_dataset.mean_geometry()
            result = fit_gompertz_to_control(control_days, control_volumes)
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.summary_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Failed to fit control growth", str(exc))
            return

        self.growth_rate_spin.setValue(result.growth_rate)
        self.carrying_capacity_spin.setValue(result.carrying_capacity)
        self.statusBar().showMessage(
            f"Fitted Gompertz control growth: r={result.growth_rate:.4f}, K={result.carrying_capacity:.3f}"
        )

    def prefill_schedule_from_geometry(self) -> None:
        if self.geometry_dataset is None:
            return
        fractions = parse_fractions(list(self.geometry_dataset.experiment_params))
        schedule = build_default_schedule(fractions)
        self.populate_schedule_table(schedule)
        if schedule:
            self.statusBar().showMessage("Schedule prefilled from experiment fractions.")

    def populate_schedule_table(self, schedule: Sequence[TreatmentFraction]) -> None:
        self.schedule_table.setRowCount(len(schedule))
        for row_index, event in enumerate(schedule):
            self.schedule_table.setItem(row_index, 0, QTableWidgetItem(f"{event.day:g}"))
            self.schedule_table.setItem(row_index, 1, QTableWidgetItem(f"{event.dose:g}"))

    def add_schedule_row(self) -> None:
        row_index = self.schedule_table.rowCount()
        self.schedule_table.insertRow(row_index)
        self.schedule_table.setItem(row_index, 0, QTableWidgetItem("0"))
        self.schedule_table.setItem(row_index, 1, QTableWidgetItem("1"))

    def remove_selected_schedule_rows(self) -> None:
        rows = sorted({item.row() for item in self.schedule_table.selectedItems()}, reverse=True)
        for row in rows:
            self.schedule_table.removeRow(row)

    def clear_schedule(self) -> None:
        self.schedule_table.setRowCount(0)

    def schedule_from_table(self) -> list[TreatmentFraction]:
        schedule: list[TreatmentFraction] = []
        for row_index in range(self.schedule_table.rowCount()):
            day_item = self.schedule_table.item(row_index, 0)
            dose_item = self.schedule_table.item(row_index, 1)
            if day_item is None or dose_item is None:
                continue
            try:
                day = float(day_item.text().replace(",", "."))
                dose = float(dose_item.text().replace(",", "."))
            except ValueError as exc:
                raise ValueError(f"Invalid schedule row {row_index + 1}.") from exc
            schedule.append(TreatmentFraction(day=day, dose=dose))
        return schedule

    def build_sample_times(self, schedule: Sequence[TreatmentFraction]) -> np.ndarray:
        step = self.step_spin.value()
        horizon = self.horizon_spin.value()
        if self.observed_days is not None and len(self.observed_days) > 0:
            horizon = max(horizon, float(np.nanmax(self.observed_days)))
        if schedule:
            horizon = max(horizon, max(event.day for event in schedule) + 1.0)

        base_grid = np.arange(0.0, horizon + step * 0.5, step)
        extra_times = [base_grid]
        if self.observed_days is not None:
            extra_times.append(self.observed_days)
        if schedule:
            extra_times.append(np.asarray([event.day for event in schedule], dtype=float))
        return np.unique(np.round(np.concatenate(extra_times), 6))

    def run_simulation(self) -> None:
        if self.geometry_dataset is None:
            QMessageBox.warning(self, "No treated tumor", "Load a treated tumor file first.")
            return
        if self.reference_geometry is None:
            QMessageBox.warning(
                self,
                "No reference geometry",
                "Could not infer a positive baseline a-b-c measurement from the selected tumor.",
            )
            return

        try:
            schedule = self.schedule_from_table()
            sample_times = self.build_sample_times(schedule)
            parameters = GrowthModelParameters(
                alpha=self.alpha_spin.value(),
                beta=self.beta_spin.value(),
                growth_rate=self.growth_rate_spin.value(),
                carrying_capacity=self.carrying_capacity_spin.value(),
                clearance_rate=self.clearance_rate_spin.value(),
            )
            self.simulation_result = simulate_growth(
                sample_times,
                parameters,
                self.reference_geometry,
                schedule,
                self.geometry_scaling_model,
            )
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.summary_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Simulation failed", str(exc))
            return

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(len(self.simulation_result.times) - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.refresh_curves()
        self.refresh_3d_view()
        self.statusBar().showMessage("Tumor-growth simulation complete.")

    def refresh_curves(self) -> None:
        self.curve_figure.clear()
        volume_ax = self.curve_figure.add_subplot(211)
        axis_ax = self.curve_figure.add_subplot(212)

        if self.simulation_result is None:
            volume_ax.set_title("No prediction available")
            axis_ax.set_title("No prediction available")
            self.curve_canvas.draw_idle()
            return

        result = self.simulation_result
        volume_ax.plot(result.times, result.total_volume, label="Predicted total", color="#264653", linewidth=2.0)
        volume_ax.plot(result.times, result.live_volume, label="Predicted live", color="#2a9d8f", linestyle="--")
        volume_ax.plot(result.times, result.dead_volume, label="Predicted dead", color="#e76f51", linestyle=":")

        schedule = self.schedule_from_table()
        for event in schedule:
            volume_ax.axvline(event.day, color="#bdbdbd", linewidth=0.8, linestyle=":")

        if self.observed_days is not None and self.observed_volume is not None:
            valid = np.isfinite(self.observed_days) & np.isfinite(self.observed_volume) & (self.observed_volume > 0.0)
            if np.any(valid):
                volume_ax.scatter(
                    self.observed_days[valid],
                    self.observed_volume[valid],
                    label="Observed volume",
                    color="#1d3557",
                    zorder=3,
                )

        volume_ax.set_ylabel("Volume")
        volume_ax.set_xlabel("Day")
        volume_ax.legend(loc="best")
        volume_ax.grid(True, alpha=0.25)

        axis_ax.plot(result.times, result.axis_a, label="Predicted a", color="#d62828")
        axis_ax.plot(result.times, result.axis_b, label="Predicted b", color="#2a9d8f")
        axis_ax.plot(result.times, result.axis_c, label="Predicted c", color="#f4a261")

        if self.observed_days is not None and self.observed_axis_a is not None:
            valid_a = np.isfinite(self.observed_days) & np.isfinite(self.observed_axis_a) & (self.observed_axis_a > 0.0)
            valid_b = np.isfinite(self.observed_days) & np.isfinite(self.observed_axis_b) & (self.observed_axis_b > 0.0)
            valid_c = np.isfinite(self.observed_days) & np.isfinite(self.observed_axis_c) & (self.observed_axis_c > 0.0)
            if np.any(valid_a):
                axis_ax.scatter(self.observed_days[valid_a], self.observed_axis_a[valid_a], color="#d62828", marker="o", alpha=0.65)
            if np.any(valid_b):
                axis_ax.scatter(self.observed_days[valid_b], self.observed_axis_b[valid_b], color="#2a9d8f", marker="s", alpha=0.65)
            if np.any(valid_c):
                axis_ax.scatter(self.observed_days[valid_c], self.observed_axis_c[valid_c], color="#f4a261", marker="^", alpha=0.65)

        axis_ax.set_ylabel("Axis length")
        axis_ax.set_xlabel("Day")
        axis_ax.legend(loc="best")
        axis_ax.grid(True, alpha=0.25)
        self.curve_figure.tight_layout()
        self.curve_canvas.draw_idle()

    def toggle_playback(self) -> None:
        if self.simulation_result is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")

    def advance_prediction_frame(self) -> None:
        if self.simulation_result is None:
            self.timer.stop()
            self.play_button.setText("Play")
            return
        current_index = self.frame_slider.value()
        if current_index >= self.frame_slider.maximum():
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self.frame_slider.setValue(current_index + 1)

    def current_axis_limit(self) -> float:
        if self.simulation_result is None:
            return 1.0
        maximum_axis = max(
            float(np.nanmax(self.simulation_result.axis_a)),
            float(np.nanmax(self.simulation_result.axis_b)),
            float(np.nanmax(self.simulation_result.axis_c)),
        )
        return max(maximum_axis * 0.7, 1.0)

    @staticmethod
    def build_ellipsoid_mesh(a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.linspace(0.0, 2.0 * math.pi, 48)
        v = np.linspace(0.0, math.pi, 24)
        uu, vv = np.meshgrid(u, v)
        x = (a / 2.0) * np.cos(uu) * np.sin(vv)
        y = (b / 2.0) * np.sin(uu) * np.sin(vv)
        z = (c / 2.0) * np.cos(vv)
        return x, y, z

    def refresh_3d_view(self) -> None:
        self.view_figure.clear()
        ax = self.view_figure.add_subplot(111, projection="3d")
        ax.view_init(elev=22, azim=36)

        if self.simulation_result is None:
            ax.set_title("No prediction available")
            self.view_canvas.draw_idle()
            self.frame_label.setText("Day: -")
            self.summary_text.setPlainText("")
            return

        index = self.frame_slider.value()
        result = self.simulation_result
        day = float(result.times[index])
        axis_a = float(result.axis_a[index])
        axis_b = float(result.axis_b[index])
        axis_c = float(result.axis_c[index])
        total = float(result.total_volume[index])
        live = float(result.live_volume[index])
        dead = float(result.dead_volume[index])

        self.frame_label.setText(f"Day: {day:.2f}")
        if all(np.isfinite([axis_a, axis_b, axis_c])) and min(axis_a, axis_b, axis_c) > 0.0:
            x, y, z = self.build_ellipsoid_mesh(axis_a, axis_b, axis_c)
            ax.plot_surface(
                x,
                y,
                z,
                color="#8ecae6",
                edgecolor="#1d3557",
                linewidth=0.25,
                alpha=0.85,
                shade=True,
            )
            ax.plot([-axis_a / 2.0, axis_a / 2.0], [0.0, 0.0], [0.0, 0.0], color="#d62828", linewidth=2.0)
            ax.plot([0.0, 0.0], [-axis_b / 2.0, axis_b / 2.0], [0.0, 0.0], color="#2a9d8f", linewidth=2.0)
            ax.plot([0.0, 0.0], [0.0, 0.0], [-axis_c / 2.0, axis_c / 2.0], color="#f4a261", linewidth=2.0)
        else:
            ax.set_title("Invalid predicted geometry")

        limit = self.current_axis_limit()
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_xlabel("a-axis")
        ax.set_ylabel("b-axis")
        ax.set_zlabel("c-axis")
        ax.set_title(f"{self.selection_combo.currentText()} | day {day:.2f}")
        self.view_canvas.draw_idle()

        self.summary_text.setPlainText(
            self.build_summary_text(day, axis_a, axis_b, axis_c, total, live, dead)
        )

    def build_summary_text(
        self,
        day: float,
        axis_a: float,
        axis_b: float,
        axis_c: float,
        total: float,
        live: float,
        dead: float,
    ) -> str:
        lines = []
        if self.geometry_dataset is not None:
            lines.append(f"Treated tumor file: {self.geometry_dataset.path.name}")
        if self.control_dataset is not None:
            lines.append(f"Control file: {self.control_dataset.path.name}")
        lines.extend(
            [
                f"Tumor: {self.selection_combo.currentText()}",
                f"alpha = {self.alpha_spin.value():.6f}",
                f"beta = {self.beta_spin.value():.6f}",
                f"growth_rate = {self.growth_rate_spin.value():.4f}",
                f"carrying_capacity = {self.carrying_capacity_spin.value():.3f}",
                f"clearance_rate = {self.clearance_rate_spin.value():.4f}",
                f"geometry_mode = {self.geometry_mode_combo.currentData()}",
                "",
                f"Displayed day = {day:.2f}",
                f"Predicted total volume = {total:.3f}",
                f"Predicted live volume = {live:.3f}",
                f"Predicted dead volume = {dead:.3f}",
                f"a = {axis_a:.3f}",
                f"b = {axis_b:.3f}",
                f"c = {axis_c:.3f}",
            ]
        )

        if self.reference_geometry is not None:
            lines.extend(
                [
                    "",
                    "Reference geometry:",
                    f"- a0 = {self.reference_geometry.axis_a:.3f}",
                    f"- b0 = {self.reference_geometry.axis_b:.3f}",
                    f"- c0 = {self.reference_geometry.axis_c:.3f}",
                    f"- V0 = {self.reference_geometry.volume:.3f}",
                ]
            )

        if self.geometry_scaling_model is not None:
            lines.extend(
                [
                    "",
                    "Geometry scaling:",
                    (
                        f"- a = {self.geometry_scaling_model.coeff_a:.4f} * V^{self.geometry_scaling_model.power_a:.4f}"
                    ),
                    (
                        f"- b = {self.geometry_scaling_model.coeff_b:.4f} * V^{self.geometry_scaling_model.power_b:.4f}"
                    ),
                    (
                        f"- c = {self.geometry_scaling_model.coeff_c:.4f} * V^{self.geometry_scaling_model.power_c:.4f}"
                    ),
                ]
            )

        schedule = self.schedule_from_table()
        if schedule:
            lines.append("")
            lines.append("Dose schedule:")
            for event in schedule:
                lines.append(f"- day {event.day:g}: {event.dose:g} Gy")
        return "\n".join(lines)

    def clear_prediction_outputs(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self.simulation_result = None
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.refresh_curves()
        self.refresh_3d_view()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TumorGrowthPredictorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
