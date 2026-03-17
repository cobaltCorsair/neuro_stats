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
    QTabWidget,
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
    build_schedule_from_intervals,
    parse_irradiation_intervals_days,
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


def _format_time_days(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value) < 1.0:
        return f"{value:.4f} d ({value * 24.0:.2f} h)"
    return f"{value:.2f} d"


class TumorGrowthPredictorWindow(QMainWindow):
    """Interactive tumor-growth predictor with plots and a 3D ellipsoid view."""

    def __init__(self, run_results: Optional[Sequence[AnalysisRunResult]] = None) -> None:
        super().__init__()
        self.run_results = list(run_results or [])
        self.geometry_dataset: Optional[TumorGeometryDataset] = None
        self.control_dataset: Optional[TumorGeometryDataset] = None
        self.geometry_path_text: Optional[str] = None
        self.control_path_text: Optional[str] = None
        self.simulation_result: Optional[GrowthSimulationResult] = None
        self.observed_days: Optional[np.ndarray] = None
        self.observed_volume: Optional[np.ndarray] = None
        self.observed_axis_a: Optional[np.ndarray] = None
        self.observed_axis_b: Optional[np.ndarray] = None
        self.observed_axis_c: Optional[np.ndarray] = None
        self.reference_geometry: Optional[GeometryReference] = None
        self.geometry_scaling_model: Optional[GeometryScalingModel] = None
        self.display_frame_times: Optional[np.ndarray] = None

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
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1170])
        root_layout.addWidget(splitter)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Load a treated tumor file and simulate growth.")

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI behavior
        super().resizeEvent(event)
        self._refresh_path_labels()

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(520)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.results_tabs = QTabWidget(self)
        self.results_tabs.addTab(self._build_plot_panel(), "Curves")
        self.results_tabs.addTab(self._build_3d_panel(), "3D")
        layout.addWidget(self.results_tabs, 1)
        return panel

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Files and selection", self)
        layout = QGridLayout(group)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.geometry_label = QLabel("No treated tumor file loaded", self)
        self.geometry_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        geometry_button = QPushButton("Open treated file", self)
        geometry_button.setToolTip("Load treated tumor file with observed a-b-c measurements.")
        geometry_button.clicked.connect(self.open_geometry_file)
        layout.addWidget(QLabel("Treated tumor"), 0, 0)
        layout.addWidget(self.geometry_label, 0, 1)
        layout.addWidget(geometry_button, 1, 1)

        self.control_label = QLabel("No control file loaded", self)
        self.control_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        control_button = QPushButton("Open control file", self)
        control_button.setToolTip("Load control tumor file for untreated growth fitting.")
        control_button.clicked.connect(self.open_control_file)
        layout.addWidget(QLabel("Control"), 2, 0)
        layout.addWidget(self.control_label, 2, 1)
        layout.addWidget(control_button, 3, 1)

        self.selection_combo = QComboBox(self)
        self.selection_combo.currentIndexChanged.connect(self.on_selection_changed)
        layout.addWidget(QLabel("Tumor"), 4, 0)
        layout.addWidget(self.selection_combo, 4, 1)

        self.fit_result_combo = QComboBox(self)
        self.fit_result_combo.currentIndexChanged.connect(self.apply_selected_fit_result)
        layout.addWidget(QLabel("Alpha/Beta source"), 5, 0)
        layout.addWidget(self.fit_result_combo, 5, 1)

        fit_button = QPushButton("Fit Gompertz from control", self)
        fit_button.clicked.connect(self.fit_growth_from_control)
        layout.addWidget(fit_button, 6, 0, 1, 2)
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
        self.repair_half_time_spin = self._create_float_spinbox(0.0, 240.0, 0.0, decimals=3, step=0.25)
        self.repair_half_time_spin.setToolTip(
            "0 keeps independent per-fraction kill. Positive values enable "
            "repair-aware interaction between closely spaced fractions."
        )

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
        layout.addWidget(QLabel("Repair T1/2 (h)"), 3, 2)
        layout.addWidget(self.repair_half_time_spin, 3, 3)

        layout.addWidget(QLabel("Geometry mode"), 4, 0)
        self.geometry_mode_combo = QComboBox(self)
        self.geometry_mode_combo.addItem("Fixed ratios", "fixed")
        self.geometry_mode_combo.addItem("Fit from observed shape", "fitted")
        self.geometry_mode_combo.currentIndexChanged.connect(self.on_geometry_mode_changed)
        layout.addWidget(self.geometry_mode_combo, 4, 1, 1, 3)
        return group

    def _build_schedule_group(self) -> QGroupBox:
        group = QGroupBox("Dose schedule", self)
        layout = QVBoxLayout(group)

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(6)
        button_grid.setVerticalSpacing(6)
        button_grid.setColumnStretch(0, 1)
        button_grid.setColumnStretch(1, 1)

        prefill_button = QPushButton("Use treated fractions", self)
        prefill_button.setToolTip("Populate the schedule from the fraction labels in the treated tumor file.")
        prefill_button.clicked.connect(self.prefill_schedule_from_geometry)
        button_grid.addWidget(prefill_button, 0, 0, 1, 2)

        add_button = QPushButton("Add event", self)
        add_button.clicked.connect(self.add_schedule_row)
        button_grid.addWidget(add_button, 1, 0)

        remove_button = QPushButton("Remove rows", self)
        remove_button.setToolTip("Remove the selected schedule rows.")
        remove_button.clicked.connect(self.remove_selected_schedule_rows)
        button_grid.addWidget(remove_button, 1, 1)

        clear_button = QPushButton("Clear schedule", self)
        clear_button.clicked.connect(self.clear_schedule)
        button_grid.addWidget(clear_button, 2, 0, 1, 2)
        layout.addLayout(button_grid)

        self.schedule_table = QTableWidget(self)
        self.schedule_table.setColumnCount(2)
        self.schedule_table.setHorizontalHeaderLabels(["Time (days)", "Dose (Gy)"])
        self.schedule_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.schedule_table.setMinimumHeight(150)
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
        controls.addWidget(QLabel("Frames", self))

        self.frame_mode_combo = QComboBox(self)
        self.frame_mode_combo.addItem("Daily snapshots", "daily")
        self.frame_mode_combo.addItem("Raw timeline", "raw")
        self.frame_mode_combo.currentIndexChanged.connect(self.on_frame_mode_changed)
        controls.addWidget(self.frame_mode_combo)

        controls.addWidget(QLabel("Speed", self))
        self.playback_speed_combo = QComboBox(self)
        self.playback_speed_combo.addItem("Auto", "auto")
        self.playback_speed_combo.addItem("1x", 1)
        self.playback_speed_combo.addItem("2x", 2)
        self.playback_speed_combo.addItem("4x", 4)
        self.playback_speed_combo.addItem("8x", 8)
        self.playback_speed_combo.addItem("16x", 16)
        self.playback_speed_combo.setToolTip(
            "Playback speed for the 3D slider. In Auto mode, raw timeline runs faster than daily snapshots."
        )
        controls.addWidget(self.playback_speed_combo)
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
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setSizes([900, 280])
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

    def _apply_path_label(
        self,
        label: QLabel,
        path_text: Optional[str],
        empty_text: str,
    ) -> None:
        if not path_text:
            label.setText(empty_text)
            label.setToolTip("")
            return

        available_width = max(120, label.width() - 4)
        elided = label.fontMetrics().elidedText(
            path_text,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        label.setText(elided)
        label.setToolTip(path_text)

    def _refresh_path_labels(self) -> None:
        self._apply_path_label(
            self.geometry_label,
            self.geometry_path_text,
            "No treated tumor file loaded",
        )
        self._apply_path_label(
            self.control_label,
            self.control_path_text,
            "No control file loaded",
        )

    def apply_selected_fit_result(self) -> None:
        fit = self.fit_result_combo.currentData()
        if fit is None:
            return
        self.alpha_spin.setValue(fit.alpha)
        self.beta_spin.setValue(fit.beta)
        repair_half_time_hours = getattr(fit, "repair_half_time_hours", None)
        self.repair_half_time_spin.setValue(
            float(repair_half_time_hours)
            if repair_half_time_hours is not None and repair_half_time_hours > 0.0
            else 0.0
        )
        self.statusBar().showMessage("Loaded alpha/beta settings from fitter result.")

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
        self.geometry_path_text = str(path.resolve())
        self._refresh_path_labels()
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
        self.control_path_text = str(path.resolve())
        self._refresh_path_labels()
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

    def on_frame_mode_changed(self) -> None:
        self.refresh_display_frames(reset_slider=False)
        if self.simulation_result is not None:
            self.refresh_3d_view()

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
        interval_days = parse_irradiation_intervals_days(self.geometry_dataset.experiment_params)
        schedule = build_schedule_from_intervals(fractions, interval_days)
        self.populate_schedule_table(schedule)
        if schedule:
            if interval_days:
                interval_text = ", ".join(f"{gap * 24.0:g} h" for gap in interval_days)
                self.statusBar().showMessage(
                    f"Schedule prefilled from experiment fractions and t= intervals ({interval_text})."
                )
            else:
                self.statusBar().showMessage("Schedule prefilled from experiment fractions.")

    def populate_schedule_table(self, schedule: Sequence[TreatmentFraction]) -> None:
        self.schedule_table.setRowCount(len(schedule))
        for row_index, event in enumerate(schedule):
            self.schedule_table.setItem(row_index, 0, QTableWidgetItem(f"{event.day:.6g}"))
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
        if len(schedule) > 1:
            schedule_days = np.asarray([event.day for event in schedule], dtype=float)
            gaps = np.diff(np.unique(np.round(schedule_days, 8)))
            positive_gaps = gaps[gaps > 1.0e-12]
            if len(positive_gaps) > 0:
                step = min(step, float(np.min(positive_gaps)) / 4.0)
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
                repair_half_time_hours=self.repair_half_time_spin.value(),
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

        self.refresh_display_frames(reset_slider=True)
        self.refresh_curves()
        self.refresh_3d_view()
        self.statusBar().showMessage("Tumor-growth simulation complete.")

    def refresh_curves(self) -> None:
        self.curve_figure.clear()
        axes = self.curve_figure.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 2]},
        )
        volume_ax, axis_ax = axes

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
        axis_ax.set_xlabel("Time (days)")
        axis_ax.legend(loc="best")
        axis_ax.grid(True, alpha=0.25)
        self.curve_figure.tight_layout(pad=1.2)
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
        frame_mode = str(self.frame_mode_combo.currentData() or "daily")
        step = self.resolve_playback_step(frame_mode, self.playback_speed_combo.currentData())
        self.frame_slider.setValue(min(current_index + step, self.frame_slider.maximum()))

    @staticmethod
    def build_display_days(times: Sequence[float]) -> np.ndarray:
        """Reduce dense simulation times to day-level frames for the 3D viewer."""
        times_array = np.asarray(times, dtype=float)
        if times_array.ndim != 1 or len(times_array) == 0:
            return np.asarray([0.0], dtype=float)

        max_time = float(np.nanmax(times_array))
        whole_day_count = int(math.floor(max_time + 1.0e-9))
        if whole_day_count <= 0:
            return np.asarray([0.0], dtype=float)
        return np.arange(0.0, float(whole_day_count) + 1.0, 1.0, dtype=float)

    @classmethod
    def build_display_times(cls, times: Sequence[float], mode: str) -> np.ndarray:
        """Build display frames for the 3D tab without changing the dense simulation."""
        times_array = np.asarray(times, dtype=float)
        if times_array.ndim != 1 or len(times_array) == 0:
            return np.asarray([0.0], dtype=float)
        if mode == "raw":
            return np.unique(np.round(times_array, 6))
        return cls.build_display_days(times_array)

    @staticmethod
    def resolve_playback_step(mode: str, speed_value: object) -> int:
        """Convert the selected speed option into a frame step."""
        if speed_value == "auto":
            return 8 if mode == "raw" else 1
        try:
            step = int(speed_value)
        except (TypeError, ValueError):
            return 1
        return max(step, 1)

    @staticmethod
    def interpolate_series(times: np.ndarray, values: np.ndarray, day: float) -> float:
        """Sample a dense simulation series at a daily display point."""
        return float(np.interp(day, times, values))

    def refresh_display_frames(self, *, reset_slider: bool) -> None:
        """Rebuild the 3D display frames for the selected mode."""
        if self.simulation_result is None:
            self.display_frame_times = None
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            return

        previous_time = 0.0
        if (
            not reset_slider
            and self.display_frame_times is not None
            and len(self.display_frame_times) > 0
        ):
            previous_index = min(self.frame_slider.value(), len(self.display_frame_times) - 1)
            previous_time = float(self.display_frame_times[previous_index])

        mode = str(self.frame_mode_combo.currentData() or "daily")
        display_frame_times = self.build_display_times(self.simulation_result.times, mode)
        self.display_frame_times = display_frame_times

        if reset_slider:
            target_index = 0
        else:
            insertion_index = int(np.searchsorted(display_frame_times, previous_time, side="left"))
            if insertion_index >= len(display_frame_times):
                target_index = len(display_frame_times) - 1
            elif insertion_index > 0:
                left_time = float(display_frame_times[insertion_index - 1])
                right_time = float(display_frame_times[insertion_index])
                target_index = insertion_index - 1 if abs(previous_time - left_time) <= abs(previous_time - right_time) else insertion_index
            else:
                target_index = 0

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(len(display_frame_times) - 1, 0))
        self.frame_slider.setValue(target_index)
        self.frame_slider.blockSignals(False)

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

        result = self.simulation_result
        frame_mode = str(self.frame_mode_combo.currentData() or "daily")
        display_frame_times = self.display_frame_times
        if display_frame_times is None or len(display_frame_times) == 0:
            display_frame_times = self.build_display_times(result.times, frame_mode)
            self.display_frame_times = display_frame_times

        index = min(self.frame_slider.value(), len(display_frame_times) - 1)
        day = float(display_frame_times[index])
        axis_a = self.interpolate_series(result.times, result.axis_a, day)
        axis_b = self.interpolate_series(result.times, result.axis_b, day)
        axis_c = self.interpolate_series(result.times, result.axis_c, day)
        total = self.interpolate_series(result.times, result.total_volume, day)
        live = self.interpolate_series(result.times, result.live_volume, day)
        dead = self.interpolate_series(result.times, result.dead_volume, day)

        if frame_mode == "raw":
            self.frame_label.setText(f"Time: {_format_time_days(day)}")
        else:
            self.frame_label.setText(f"Day: {int(round(day))}")
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
        if frame_mode == "raw":
            ax.set_title(f"{self.selection_combo.currentText()} | t = {_format_time_days(day)}")
        else:
            ax.set_title(f"{self.selection_combo.currentText()} | day {int(round(day))}")
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
        frame_mode = str(self.frame_mode_combo.currentData() or "daily")
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
                (
                    f"repair_half_time_hours = {self.repair_half_time_spin.value():.3f}"
                    if self.repair_half_time_spin.value() > 0.0
                    else "repair_half_time_hours = disabled"
                ),
                f"frame_mode = {frame_mode}",
                f"geometry_mode = {self.geometry_mode_combo.currentData()}",
                "",
                (
                    f"Displayed time = {_format_time_days(day)}"
                    if frame_mode == "raw"
                    else f"Displayed day = {int(round(day))}"
                ),
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
                lines.append(f"- t = {_format_time_days(event.day)}: {event.dose:g} Gy")
        return "\n".join(lines)

    def clear_prediction_outputs(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self.simulation_result = None
        self.display_frame_times = None
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
