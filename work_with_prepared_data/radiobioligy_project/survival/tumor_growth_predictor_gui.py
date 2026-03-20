# coding: utf-8
"""Standalone PyQt6 window for tumor-growth prediction and ellipsoid dynamics."""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
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
    format_interval_values_days,
    format_schedule_intervals,
    infer_radiation_family,
    parse_fractions,
)
from work_with_prepared_data.radiobioligy_project.survival.gui_csv_export import (
    related_csv_path,
    write_csv_rows,
)
from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
    IntervalSensitivityReport,
    ParameterSensitivityReport,
    ScenarioComparisonReport,
    analyze_interval_sensitivity,
    analyze_parameter_sensitivity,
    compare_treatment_scenarios,
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
    predict_schedule_surviving_fraction,
    simulate_growth,
)

PLAYBACK_INTERVAL_MS = 60
PARAMETER_SENSITIVITY_HEADERS = [
    "Parameter",
    "Delta %",
    "Baseline",
    "Varied",
    "RMSE",
    "Delta RMSE",
    "RMSE ratio",
    "Status",
    "Reason",
]
INFLUENCE_HEADERS = [
    "Parameter",
    "Max |Delta RMSE|",
    "Mean |Delta RMSE|",
    "Cases",
]
INTERVAL_SENSITIVITY_HEADERS = [
    "Interval (h)",
    "RMSE",
    "Delta RMSE",
    "RMSE ratio",
    "Intervals",
]
COMPARISON_HEADERS = [
    "Scenario",
    "Total dose",
    "Families",
    "Min volume",
    "Nadir day",
    "Final volume",
    "AUC",
]


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


def _format_optional_float(value: float | None, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_optional_ratio(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{float(value):.3f}"


def parse_positive_float_list(text: str, default: Sequence[float]) -> list[float]:
    """Parse a comma-separated list of positive floats for analysis controls."""
    raw = text.strip()
    if not raw:
        return [float(item) for item in default]

    values: list[float] = []
    for chunk in raw.split(","):
        value = float(chunk.strip().replace(",", "."))
        if value <= 0.0:
            raise ValueError("Values must be positive.")
        values.append(value)
    return values or [float(item) for item in default]


def parse_percentage_list(text: str, default: Sequence[float]) -> list[float]:
    """Parse percentages like ``10, 20, 30`` into fractions ``0.1, 0.2, 0.3``."""
    return [value / 100.0 for value in parse_positive_float_list(text, default)]


def parse_positive_scalar(text: str, *, label: str) -> float:
    """Parse one positive scalar value from a GUI text field."""
    normalized = text.strip().replace(",", ".")
    if not normalized:
        raise ValueError(f"{label} is required.")

    value = float(normalized)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive number.")
    return value


def _normalize_family_label(value: str | None) -> str | None:
    if value is None:
        return None
    family = str(value).strip().lower()
    return family or None


def build_family_parameter_overrides(
    run_results: Sequence[AnalysisRunResult],
    base_parameters: GrowthModelParameters,
    schedule: Sequence[TreatmentFraction],
    *,
    default_family: str | None = None,
) -> tuple[dict[str, GrowthModelParameters], tuple[str, ...]]:
    """Build per-family predictor overrides from fitter results for mixed schedules."""
    needed_families = {
        family
        for family in (_normalize_family_label(item.family) for item in schedule)
        if family is not None
    }
    default_family = _normalize_family_label(default_family)
    if not needed_families:
        return {}, ()

    overrides: dict[str, GrowthModelParameters] = {}
    missing: list[str] = []
    for family in sorted(needed_families):
        if family == default_family:
            overrides[family] = base_parameters
            continue

        matched_run = next(
            (
                run
                for run in reversed(tuple(run_results))
                if run.fit_result is not None and _normalize_family_label(run.summary.family) == family
            ),
            None,
        )
        if matched_run is None or matched_run.fit_result is None:
            missing.append(family)
            continue

        fit = matched_run.fit_result
        repair_half_time_hours = getattr(fit, "repair_half_time_hours", None)
        overrides[family] = GrowthModelParameters(
            alpha=fit.alpha,
            beta=fit.beta,
            growth_rate=base_parameters.growth_rate,
            carrying_capacity=base_parameters.carrying_capacity,
            clearance_rate=base_parameters.clearance_rate,
            repair_half_time_hours=(
                float(repair_half_time_hours)
                if repair_half_time_hours is not None and repair_half_time_hours > 0.0
                else base_parameters.repair_half_time_hours
            ),
        )

    return overrides, tuple(missing)


def build_parameter_sensitivity_table_rows(
    report: Optional[ParameterSensitivityReport],
) -> list[list[str]]:
    rows = list(report.rows) if report is not None else []
    return [
        [
            row.parameter,
            f"{row.perturbation_fraction * 100.0:+.1f}",
            _format_optional_float(row.baseline_value, digits=6),
            _format_optional_float(row.varied_value, digits=6),
            _format_optional_float(row.rmse, digits=6),
            _format_optional_float(row.delta_rmse, digits=6),
            _format_optional_ratio(row.rmse_ratio),
            row.status,
            row.reason,
        ]
        for row in rows
    ]


def build_influence_table_rows(report: Optional[ParameterSensitivityReport]) -> list[list[str]]:
    rows = list(report.influence) if report is not None else []
    return [
        [
            row.parameter,
            _format_optional_float(row.max_abs_delta_rmse, digits=6),
            _format_optional_float(row.mean_abs_delta_rmse, digits=6),
            str(row.tested_cases),
        ]
        for row in rows
    ]


def build_interval_sensitivity_table_rows(
    report: Optional[IntervalSensitivityReport],
) -> list[list[str]]:
    rows = list(report.rows) if report is not None else []
    return [
        [
            _format_optional_float(row.interval_hours, digits=3),
            _format_optional_float(row.rmse, digits=6),
            _format_optional_float(row.delta_rmse, digits=6),
            _format_optional_ratio(row.rmse_ratio),
            format_schedule_intervals(row.schedule_days),
        ]
        for row in rows
    ]


def build_comparison_table_rows(report: Optional[ScenarioComparisonReport]) -> list[list[str]]:
    rows = list(report.rows) if report is not None else []
    return [
        [
            row.scenario,
            _format_optional_float(row.total_physical_dose, digits=3),
            TumorGrowthPredictorWindow.format_family_sequence(row.family_sequence),
            _format_optional_float(row.min_total_volume, digits=6),
            _format_optional_float(row.min_total_volume_day, digits=4),
            _format_optional_float(row.final_total_volume, digits=6),
            _format_optional_float(row.auc_total_volume, digits=6),
        ]
        for row in rows
    ]


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
        self.active_family_overrides: dict[str, GrowthModelParameters] = {}
        self.missing_schedule_families: tuple[str, ...] = ()
        self.parameter_sensitivity_report: Optional[ParameterSensitivityReport] = None
        self.interval_sensitivity_report: Optional[IntervalSensitivityReport] = None
        self.scenario_comparison_report: Optional[ScenarioComparisonReport] = None
        self.comparison_curves: dict[str, np.ndarray] = {}

        self.timer = QTimer(self)
        self.timer.setInterval(PLAYBACK_INTERVAL_MS)
        self.timer.timeout.connect(self.advance_prediction_frame)

        self.setWindowTitle("Tumor growth predictor")
        self.resize(1360, 840)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._apply_window_style()
        self.populate_fit_results()
        self.refresh_schedule_preview_labels()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 970])
        root_layout.addWidget(splitter)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Load a treated tumor file and simulate growth.")

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI behavior
        super().resizeEvent(event)
        self._refresh_path_labels()

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(460)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.setObjectName("PrimaryAction")
        self.simulate_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.simulate_button)

        tabs = QTabWidget(self)

        files_page = QWidget(self)
        files_layout = QVBoxLayout(files_page)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.addWidget(self._build_file_group())
        tabs.addTab(files_page, "Files")

        model_container = QWidget(self)
        model_layout = QVBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self._build_parameter_group())
        model_layout.addStretch(1)

        model_scroll = QScrollArea(self)
        model_scroll.setWidgetResizable(True)
        model_scroll.setWidget(model_container)
        tabs.addTab(model_scroll, "Model")

        schedule_container = QWidget(self)
        schedule_layout = QVBoxLayout(schedule_container)
        schedule_layout.setContentsMargins(0, 0, 0, 0)
        schedule_layout.addWidget(self._build_schedule_group())
        schedule_layout.addStretch(1)

        schedule_scroll = QScrollArea(self)
        schedule_scroll.setWidgetResizable(True)
        schedule_scroll.setWidget(schedule_container)
        tabs.addTab(schedule_scroll, "Schedule")

        layout.addWidget(tabs, 1)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.results_tabs = QTabWidget(self)
        self.results_tabs.addTab(self._build_plot_panel(), "Curves")
        self.results_tabs.addTab(self._build_sensitivity_panel(), "Sensitivity")
        self.results_tabs.addTab(self._build_comparison_panel(), "Comparison")
        self.results_tabs.addTab(self._build_3d_panel(), "3D")
        layout.addWidget(self.results_tabs, 1)
        return panel

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Files and series", self)
        layout = QGridLayout(group)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.geometry_label = QLabel("No treated tumor file loaded", self)
        self.geometry_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        geometry_button = QPushButton("Open treated file", self)
        geometry_button.setToolTip("Load treated tumor file with observed a-b-c measurements.")
        geometry_button.clicked.connect(self.open_geometry_file)
        layout.addWidget(QLabel("Treated tumor"), 0, 0)
        layout.addWidget(self.geometry_label, 0, 1)
        layout.addWidget(geometry_button, 0, 2)

        self.control_label = QLabel("No control file loaded", self)
        self.control_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        control_button = QPushButton("Open control file", self)
        control_button.setToolTip("Load control tumor file for untreated growth fitting.")
        control_button.clicked.connect(self.open_control_file)
        layout.addWidget(QLabel("Control"), 1, 0)
        layout.addWidget(self.control_label, 1, 1)
        layout.addWidget(control_button, 1, 2)

        self.selection_combo = QComboBox(self)
        self._configure_combo_box(self.selection_combo)
        self.selection_combo.currentIndexChanged.connect(self.on_selection_changed)
        layout.addWidget(QLabel("Tumor"), 2, 0)
        layout.addWidget(self.selection_combo, 2, 1, 1, 2)

        self.fit_result_combo = QComboBox(self)
        self._configure_combo_box(self.fit_result_combo)
        self.fit_result_combo.currentIndexChanged.connect(self.apply_selected_fit_result)
        layout.addWidget(QLabel("Alpha/Beta source"), 3, 0)
        layout.addWidget(self.fit_result_combo, 3, 1, 1, 2)

        fit_button = QPushButton("Fit growth from control", self)
        fit_button.clicked.connect(self.fit_growth_from_control)
        layout.addWidget(fit_button, 4, 0, 1, 3)
        layout.setRowStretch(5, 1)
        return group

    def _build_parameter_group(self) -> QGroupBox:
        group = QGroupBox("Model parameters", self)
        layout = QGridLayout(group)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

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
        self.tcp_cell_density_edit = QLineEdit("1e7", self)
        self.tcp_cell_density_edit.setPlaceholderText("1e7")
        self.tcp_cell_density_edit.setToolTip(
            "Clonogenic cell density in cells/cm^3 used for the TCP estimate in the summary."
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
        self._configure_combo_box(self.geometry_mode_combo)
        self.geometry_mode_combo.addItem("Fixed ratios", "fixed")
        self.geometry_mode_combo.addItem("Fit from observed shape", "fitted")
        self.geometry_mode_combo.currentIndexChanged.connect(self.on_geometry_mode_changed)
        layout.addWidget(self.geometry_mode_combo, 4, 1)

        layout.addWidget(QLabel("TCP cell density"), 4, 2)
        layout.addWidget(self.tcp_cell_density_edit, 4, 3)
        return group

    def _build_schedule_group(self) -> QGroupBox:
        group = QGroupBox("Dose schedule", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

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
        self.schedule_table.setColumnCount(3)
        self.schedule_table.setHorizontalHeaderLabels(["Time (days)", "Dose (Gy)", "Family"])
        self.schedule_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.schedule_table.verticalHeader().setVisible(False)
        self.schedule_table.setMinimumHeight(120)
        self.schedule_table.itemChanged.connect(self.refresh_schedule_preview_labels)
        layout.addWidget(self.schedule_table, 1)

        self.schedule_hint_label = QLabel("Intervals: -", self)
        self.schedule_hint_label.setWordWrap(True)
        self.schedule_hint_label.setObjectName("ScheduleHint")
        layout.addWidget(self.schedule_hint_label)
        return group

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.curve_figure = Figure(figsize=(9, 6))
        self.curve_canvas = FigureCanvasQTAgg(self.curve_figure)
        self.curve_toolbar = NavigationToolbar2QT(self.curve_canvas, self)
        layout.addWidget(self.curve_toolbar)
        layout.addWidget(self.curve_canvas, 1)
        return panel

    def _build_sensitivity_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_group = QGroupBox("Sensitivity setup", self)
        controls_layout = QGridLayout(controls_group)
        controls_layout.setContentsMargins(10, 14, 10, 10)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        controls_layout.addWidget(QLabel("Perturbations (%)"), 0, 0)
        self.sensitivity_perturb_edit = QLineEdit("10, 20, 30", self)
        self.sensitivity_perturb_edit.setPlaceholderText("10, 20, 30")
        controls_layout.addWidget(self.sensitivity_perturb_edit, 0, 1)

        controls_layout.addWidget(QLabel("Intervals (h)"), 0, 2)
        self.interval_hours_edit = QLineEdit("0.5, 1, 2.5, 24", self)
        self.interval_hours_edit.setPlaceholderText("0.5, 1, 2.5, 24")
        controls_layout.addWidget(self.interval_hours_edit, 0, 3)

        self.refresh_sensitivity_button = QPushButton("Run sensitivity", self)
        self.refresh_sensitivity_button.clicked.connect(self.refresh_sensitivity_views)
        controls_layout.addWidget(self.refresh_sensitivity_button, 0, 4)

        self.export_sensitivity_button = QPushButton("Export CSV", self)
        self.export_sensitivity_button.clicked.connect(self.export_sensitivity_csv)
        controls_layout.addWidget(self.export_sensitivity_button, 0, 5)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(3, 1)
        layout.addWidget(controls_group)

        self.sensitivity_figure = Figure(figsize=(8, 4.5))
        self.sensitivity_canvas = FigureCanvasQTAgg(self.sensitivity_figure)
        self.sensitivity_canvas.setMinimumHeight(220)
        layout.addWidget(self.sensitivity_canvas)

        detail_tabs = QTabWidget(self)
        self.parameter_sensitivity_table = self._create_table(PARAMETER_SENSITIVITY_HEADERS)
        detail_tabs.addTab(self.parameter_sensitivity_table, "Parameters")

        self.influence_table = self._create_table(INFLUENCE_HEADERS)
        detail_tabs.addTab(self.influence_table, "Influence")

        self.interval_sensitivity_table = self._create_table(INTERVAL_SENSITIVITY_HEADERS)
        detail_tabs.addTab(self.interval_sensitivity_table, "Intervals")
        layout.addWidget(detail_tabs, 1)

        self.sensitivity_text = QPlainTextEdit(self)
        self.sensitivity_text.setReadOnly(True)
        self.sensitivity_text.setMaximumHeight(130)
        layout.addWidget(self.sensitivity_text)
        return panel

    def _build_comparison_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_group = QGroupBox("Scenario comparison", self)
        controls_layout = QGridLayout(controls_group)
        controls_layout.setContentsMargins(10, 14, 10, 10)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        controls_layout.addWidget(QLabel("Alternative name"), 0, 0)
        self.comparison_name_edit = QLineEdit("Alternative", self)
        self.comparison_name_edit.setPlaceholderText("Alternative")
        controls_layout.addWidget(self.comparison_name_edit, 0, 1)

        self.copy_schedule_button = QPushButton("Copy current schedule", self)
        self.copy_schedule_button.clicked.connect(self.copy_current_schedule_to_comparison)
        controls_layout.addWidget(self.copy_schedule_button, 0, 2)

        self.clear_comparison_button = QPushButton("Clear alternative", self)
        self.clear_comparison_button.clicked.connect(self.clear_comparison_schedule)
        controls_layout.addWidget(self.clear_comparison_button, 0, 3)

        self.compare_button = QPushButton("Compare scenarios", self)
        self.compare_button.clicked.connect(self.refresh_comparison_view)
        controls_layout.addWidget(self.compare_button, 0, 4)

        self.export_comparison_button = QPushButton("Export CSV", self)
        self.export_comparison_button.clicked.connect(self.export_comparison_csv)
        controls_layout.addWidget(self.export_comparison_button, 0, 5)
        controls_layout.setColumnStretch(1, 1)
        layout.addWidget(controls_group)

        self.comparison_schedule_table = self._create_table(["Time (days)", "Dose (Gy)", "Family"])
        self.comparison_schedule_table.setMinimumHeight(150)
        self.comparison_schedule_table.itemChanged.connect(self.refresh_schedule_preview_labels)
        layout.addWidget(self.comparison_schedule_table)

        self.comparison_schedule_hint_label = QLabel("Alternative intervals: -", self)
        self.comparison_schedule_hint_label.setWordWrap(True)
        self.comparison_schedule_hint_label.setObjectName("ScheduleHint")
        layout.addWidget(self.comparison_schedule_hint_label)

        self.comparison_figure = Figure(figsize=(8, 4.6))
        self.comparison_canvas = FigureCanvasQTAgg(self.comparison_figure)
        self.comparison_canvas.setMinimumHeight(220)
        layout.addWidget(self.comparison_canvas)

        self.comparison_table = self._create_table(COMPARISON_HEADERS)
        self.comparison_table.setMinimumHeight(160)
        layout.addWidget(self.comparison_table)

        self.comparison_text = QPlainTextEdit(self)
        self.comparison_text.setReadOnly(True)
        self.comparison_text.setMaximumHeight(120)
        layout.addWidget(self.comparison_text)
        return panel

    def _build_3d_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)

        self.frame_label = QLabel("Day: -", self)
        controls.addWidget(self.frame_label)
        controls.addWidget(QLabel("Frames", self))

        self.frame_mode_combo = QComboBox(self)
        self._configure_combo_box(self.frame_mode_combo)
        self.frame_mode_combo.addItem("Daily snapshots", "daily")
        self.frame_mode_combo.addItem("Raw timeline", "raw")
        self.frame_mode_combo.currentIndexChanged.connect(self.on_frame_mode_changed)
        controls.addWidget(self.frame_mode_combo)

        controls.addWidget(QLabel("Speed", self))
        self.playback_speed_combo = QComboBox(self)
        self._configure_combo_box(self.playback_speed_combo)
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
        bottom_splitter.setSizes([940, 240])
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

    @staticmethod
    def _create_table(headers: Sequence[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        return table

    @staticmethod
    def _configure_combo_box(combo: QComboBox) -> None:
        combo.setMinimumHeight(30)
        combo.setMaxVisibleItems(14)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)

    def _apply_window_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 12px;
            }
            QMainWindow {
                background: #f3f5f9;
            }
            QGroupBox {
                background: #f8fafc;
                border: 1px solid #d7dde8;
                border-radius: 10px;
                margin-top: 14px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #d7dde8;
                border-radius: 10px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #e9eef6;
                border: 1px solid #d7dde8;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 6px 10px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cfd7e4;
                border-radius: 8px;
                padding: 6px 10px;
                min-height: 28px;
            }
            QPushButton:hover {
                background: #f4f8ff;
                border-color: #b8c7dd;
            }
            QPushButton#PrimaryAction {
                background: #dcecff;
                border-color: #9cbde7;
                font-weight: 600;
            }
            QPushButton#PrimaryAction:hover {
                background: #cfe4ff;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #cfd7e4;
                border-radius: 7px;
                padding: 4px 6px;
                min-height: 24px;
            }
            QComboBox {
                background: #ffffff;
                border: 1px solid #cfd7e4;
                border-radius: 7px;
                padding: 2px 28px 2px 6px;
                min-height: 0px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #cfd7e4;
                selection-background-color: #dcecff;
                selection-color: #1f2937;
                outline: 0;
                padding: 2px;
            }
            QTableWidget, QPlainTextEdit, QScrollArea {
                background: #ffffff;
                border: 1px solid #d7dde8;
                border-radius: 8px;
                alternate-background-color: #f7f9fc;
            }
            QHeaderView::section {
                background: #eef3f8;
                border: none;
                border-right: 1px solid #d7dde8;
                border-bottom: 1px solid #d7dde8;
                padding: 5px 6px;
                font-weight: 600;
            }
            QLabel#ScheduleHint {
                color: #475569;
                padding: 2px 2px 0 2px;
            }
            QSplitter::handle {
                background: #e3e8f0;
            }
            """
        )

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

    @staticmethod
    def _build_interval_preview_from_days(
        schedule_days: Sequence[float],
        *,
        label: str,
    ) -> str:
        if len(schedule_days) < 2:
            return f"{label}: -"
        return f"{label}: {format_schedule_intervals(schedule_days)}"

    def _build_interval_preview_from_table(self, table: QTableWidget, *, label: str) -> str:
        schedule_days: list[float] = []
        for row_index in range(table.rowCount()):
            day_item = table.item(row_index, 0)
            if day_item is None or not day_item.text().strip():
                continue
            try:
                schedule_days.append(float(day_item.text().replace(",", ".")))
            except ValueError:
                return f"{label}: invalid time value"
        return self._build_interval_preview_from_days(schedule_days, label=label)

    def refresh_schedule_preview_labels(self) -> None:
        self.schedule_hint_label.setText(
            self._build_interval_preview_from_table(self.schedule_table, label="Intervals")
        )
        self.comparison_schedule_hint_label.setText(
            self._build_interval_preview_from_table(
                self.comparison_schedule_table,
                label="Alternative intervals",
            )
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
        default_family = infer_radiation_family(self.geometry_dataset.path)
        schedule = [
            TreatmentFraction(day=item.day, dose=item.dose, family=default_family)
            for item in build_schedule_from_intervals(fractions, interval_days)
        ]
        self.populate_schedule_table(schedule)
        if self.comparison_schedule_table.rowCount() == 0:
            self.populate_comparison_schedule_table(schedule)
        if schedule:
            if interval_days:
                self.statusBar().showMessage(
                    "Schedule prefilled from experiment fractions and "
                    f"{format_interval_values_days(interval_days)}."
                )
            else:
                self.statusBar().showMessage("Schedule prefilled from experiment fractions.")

    def populate_schedule_table(self, schedule: Sequence[TreatmentFraction]) -> None:
        self.schedule_table.blockSignals(True)
        try:
            self.schedule_table.setRowCount(len(schedule))
            for row_index, event in enumerate(schedule):
                self.schedule_table.setItem(row_index, 0, QTableWidgetItem(f"{event.day:.6g}"))
                self.schedule_table.setItem(row_index, 1, QTableWidgetItem(f"{event.dose:g}"))
                self.schedule_table.setItem(row_index, 2, QTableWidgetItem(event.family or ""))
        finally:
            self.schedule_table.blockSignals(False)
        self.refresh_schedule_preview_labels()

    def add_schedule_row(self) -> None:
        row_index = self.schedule_table.rowCount()
        self.schedule_table.insertRow(row_index)
        default_family = ""
        if self.geometry_dataset is not None:
            default_family = infer_radiation_family(self.geometry_dataset.path) or ""
        self.schedule_table.setItem(row_index, 0, QTableWidgetItem("0"))
        self.schedule_table.setItem(row_index, 1, QTableWidgetItem("1"))
        self.schedule_table.setItem(row_index, 2, QTableWidgetItem(default_family))
        self.refresh_schedule_preview_labels()

    def remove_selected_schedule_rows(self) -> None:
        rows = sorted({item.row() for item in self.schedule_table.selectedItems()}, reverse=True)
        for row in rows:
            self.schedule_table.removeRow(row)
        self.refresh_schedule_preview_labels()

    def clear_schedule(self) -> None:
        self.schedule_table.setRowCount(0)
        self.refresh_schedule_preview_labels()

    def schedule_from_table(self) -> list[TreatmentFraction]:
        return self._schedule_from_widget(self.schedule_table, "schedule")

    def populate_comparison_schedule_table(self, schedule: Sequence[TreatmentFraction]) -> None:
        self.comparison_schedule_table.blockSignals(True)
        try:
            self.comparison_schedule_table.setRowCount(len(schedule))
            for row_index, event in enumerate(schedule):
                self.comparison_schedule_table.setItem(row_index, 0, QTableWidgetItem(f"{event.day:.6g}"))
                self.comparison_schedule_table.setItem(row_index, 1, QTableWidgetItem(f"{event.dose:g}"))
                self.comparison_schedule_table.setItem(row_index, 2, QTableWidgetItem(event.family or ""))
        finally:
            self.comparison_schedule_table.blockSignals(False)
        self.refresh_schedule_preview_labels()

    def clear_comparison_schedule(self) -> None:
        self.comparison_schedule_table.setRowCount(0)
        self.refresh_schedule_preview_labels()
        self.clear_comparison_outputs()

    def copy_current_schedule_to_comparison(self) -> None:
        self.populate_comparison_schedule_table(self.schedule_from_table())
        self.refresh_comparison_view()

    def comparison_schedule_from_table(self) -> list[TreatmentFraction]:
        return self._schedule_from_widget(self.comparison_schedule_table, "alternative schedule")

    def _schedule_from_widget(
        self,
        table: QTableWidget,
        label: str,
    ) -> list[TreatmentFraction]:
        schedule: list[TreatmentFraction] = []
        for row_index in range(table.rowCount()):
            day_item = table.item(row_index, 0)
            dose_item = table.item(row_index, 1)
            family_item = table.item(row_index, 2)
            if day_item is None or dose_item is None:
                continue
            try:
                day = float(day_item.text().replace(",", "."))
                dose = float(dose_item.text().replace(",", "."))
            except ValueError as exc:
                raise ValueError(f"Invalid {label} row {row_index + 1}.") from exc
            family = None if family_item is None else _normalize_family_label(family_item.text())
            schedule.append(TreatmentFraction(day=day, dose=dose, family=family))
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

    def current_parameters(self) -> GrowthModelParameters:
        return GrowthModelParameters(
            alpha=self.alpha_spin.value(),
            beta=self.beta_spin.value(),
            growth_rate=self.growth_rate_spin.value(),
            carrying_capacity=self.carrying_capacity_spin.value(),
            clearance_rate=self.clearance_rate_spin.value(),
            repair_half_time_hours=self.repair_half_time_spin.value(),
        )

    def current_tcp_cell_density(self) -> float:
        return parse_positive_scalar(self.tcp_cell_density_edit.text(), label="TCP cell density")

    def current_default_family(self) -> str | None:
        if self.geometry_dataset is None:
            return None
        return infer_radiation_family(self.geometry_dataset.path)

    def build_family_overrides_for_schedule(
        self,
        schedule: Sequence[TreatmentFraction],
        parameters: GrowthModelParameters,
    ) -> tuple[dict[str, GrowthModelParameters], tuple[str, ...]]:
        return build_family_parameter_overrides(
            self.run_results,
            parameters,
            schedule,
            default_family=self.current_default_family(),
        )

    @staticmethod
    def _fill_row(table: QTableWidget, row_index: int, values: Sequence[str]) -> None:
        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            table.setItem(row_index, column_index, item)

    def refresh_sensitivity_views(self) -> None:
        self.parameter_sensitivity_report = None
        self.interval_sensitivity_report = None

        if (
            self.reference_geometry is None
            or self.observed_days is None
            or self.observed_volume is None
        ):
            self.clear_sensitivity_outputs("Load a treated tumor file and run a simulation first.")
            return

        try:
            schedule = self.schedule_from_table()
            if not schedule:
                raise ValueError("Add at least one irradiation event to the dose schedule.")

            perturbations = parse_percentage_list(
                self.sensitivity_perturb_edit.text(),
                default=(10.0, 20.0, 30.0),
            )
            interval_hours = parse_positive_float_list(
                self.interval_hours_edit.text(),
                default=(0.5, 1.0, 2.5, 24.0),
            )
            parameters = self.current_parameters()
            family_overrides, missing_families = self.build_family_overrides_for_schedule(
                schedule,
                parameters,
            )

            self.parameter_sensitivity_report = analyze_parameter_sensitivity(
                observed_days=self.observed_days,
                observed_volume=self.observed_volume,
                reference=self.reference_geometry,
                parameters=parameters,
                schedule=schedule,
                scaling_model=self.geometry_scaling_model,
                family_parameters=family_overrides or None,
                perturbation_fractions=perturbations,
                vary=(
                    "alpha",
                    "beta",
                    "growth_rate",
                    "carrying_capacity",
                    "clearance_rate",
                    "dose",
                ),
            )

            if len(schedule) >= 2:
                self.interval_sensitivity_report = analyze_interval_sensitivity(
                    observed_days=self.observed_days,
                    observed_volume=self.observed_volume,
                    reference=self.reference_geometry,
                    parameters=parameters,
                    fractions=[event.dose for event in schedule],
                    interval_hours=interval_hours,
                    scaling_model=self.geometry_scaling_model,
                    baseline_schedule=schedule,
                    schedule_template=schedule,
                    family_parameters=family_overrides or None,
                )
            else:
                self.interval_sensitivity_report = None

            self.populate_parameter_sensitivity_table()
            self.populate_influence_table()
            self.populate_interval_sensitivity_table()
            self.refresh_sensitivity_plot()
            self.sensitivity_text.setPlainText(
                self.build_sensitivity_summary_text(missing_families)
            )
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.clear_sensitivity_outputs(traceback.format_exc())
            QMessageBox.critical(self, "Sensitivity analysis failed", str(exc))

    def clear_sensitivity_outputs(self, message: str = "") -> None:
        self.parameter_sensitivity_table.setRowCount(0)
        self.influence_table.setRowCount(0)
        self.interval_sensitivity_table.setRowCount(0)
        self.sensitivity_text.setPlainText(message)
        self.refresh_sensitivity_plot()

    def populate_parameter_sensitivity_table(self) -> None:
        table_rows = build_parameter_sensitivity_table_rows(self.parameter_sensitivity_report)
        self.parameter_sensitivity_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.parameter_sensitivity_table, row_index, values)

    def populate_influence_table(self) -> None:
        table_rows = build_influence_table_rows(self.parameter_sensitivity_report)
        self.influence_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.influence_table, row_index, values)

    def populate_interval_sensitivity_table(self) -> None:
        table_rows = build_interval_sensitivity_table_rows(self.interval_sensitivity_report)
        self.interval_sensitivity_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.interval_sensitivity_table, row_index, values)

    def export_sensitivity_csv(self) -> None:
        has_parameter_rows = bool(build_parameter_sensitivity_table_rows(self.parameter_sensitivity_report))
        has_interval_rows = bool(build_interval_sensitivity_table_rows(self.interval_sensitivity_report))
        if not has_parameter_rows and not has_interval_rows:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Run sensitivity analysis first.",
            )
            return

        default_stem = self.treated_path.stem if self.treated_path is not None else "tumor_growth"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export sensitivity CSV",
            str(Path.cwd() / f"{default_stem}_sensitivity.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        written_paths: list[Path] = []
        parameter_rows = build_parameter_sensitivity_table_rows(self.parameter_sensitivity_report)
        if parameter_rows:
            parameter_path = related_csv_path(path, "parameter_sensitivity")
            write_csv_rows(parameter_path, PARAMETER_SENSITIVITY_HEADERS, parameter_rows)
            written_paths.append(parameter_path)

        influence_rows = build_influence_table_rows(self.parameter_sensitivity_report)
        if influence_rows:
            influence_path = related_csv_path(path, "parameter_influence")
            write_csv_rows(influence_path, INFLUENCE_HEADERS, influence_rows)
            written_paths.append(influence_path)

        interval_rows = build_interval_sensitivity_table_rows(self.interval_sensitivity_report)
        if interval_rows:
            interval_path = related_csv_path(path, "interval_sensitivity")
            write_csv_rows(interval_path, INTERVAL_SENSITIVITY_HEADERS, interval_rows)
            written_paths.append(interval_path)

        self.statusBar().showMessage(
            "Sensitivity CSV export complete: " + ", ".join(str(item) for item in written_paths)
        )

    def refresh_sensitivity_plot(self) -> None:
        self.sensitivity_figure.clear()
        influence_rows = (
            list(self.parameter_sensitivity_report.influence)
            if self.parameter_sensitivity_report is not None
            else []
        )
        interval_rows = (
            list(self.interval_sensitivity_report.rows)
            if self.interval_sensitivity_report is not None
            else []
        )
        if not influence_rows and not interval_rows:
            axis = self.sensitivity_figure.add_subplot(111)
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No sensitivity results yet",
                ha="center",
                va="center",
                fontsize=11,
                color="#5f6b7a",
            )
            self.sensitivity_canvas.draw_idle()
            return

        left_axis, right_axis = self.sensitivity_figure.subplots(1, 2)

        if influence_rows:
            parameters = [row.parameter for row in influence_rows]
            deltas = [row.max_abs_delta_rmse for row in influence_rows]
            left_axis.barh(parameters, deltas, color="#457b9d")
            left_axis.invert_yaxis()
            left_axis.set_title("Parameter influence")
            left_axis.set_xlabel("Max |Delta RMSE|")
            left_axis.grid(True, axis="x", alpha=0.25)
        else:
            left_axis.axis("off")
            left_axis.text(
                0.5,
                0.5,
                "No parameter influence data",
                ha="center",
                va="center",
                transform=left_axis.transAxes,
                fontsize=10,
                color="#5f6b7a",
            )

        if interval_rows:
            interval_rows = sorted(interval_rows, key=lambda row: row.interval_hours)
            right_axis.plot(
                [row.interval_hours for row in interval_rows],
                [row.rmse for row in interval_rows],
                marker="o",
                color="#e76f51",
                linewidth=1.8,
            )
            right_axis.set_title("Interval sensitivity")
            right_axis.set_xlabel("Interval (h)")
            right_axis.set_ylabel("RMSE")
            right_axis.grid(True, alpha=0.25)
        else:
            right_axis.axis("off")
            right_axis.text(
                0.5,
                0.5,
                "Need at least two fractions\nfor interval sensitivity",
                ha="center",
                va="center",
                transform=right_axis.transAxes,
                fontsize=10,
                color="#5f6b7a",
            )

        self.sensitivity_figure.tight_layout(pad=1.1)
        self.sensitivity_canvas.draw_idle()

    def build_sensitivity_summary_text(
        self,
        missing_families: Sequence[str],
    ) -> str:
        lines: list[str] = []
        parameter_report = self.parameter_sensitivity_report
        interval_report = self.interval_sensitivity_report

        if parameter_report is not None:
            lines.append(f"Baseline RMSE = {parameter_report.baseline_rmse:.6f}")
            if parameter_report.influence:
                top = parameter_report.influence[0]
                lines.append(
                    f"Top parameter influence: {top.parameter} | max |Delta RMSE| = {top.max_abs_delta_rmse:.6f}"
                )

        if interval_report is not None and interval_report.rows:
            best = interval_report.rows[0]
            lines.append(
                f"Best interval candidate: {best.interval_hours:.3f} h | RMSE = {best.rmse:.6f}"
            )
        elif interval_report is None:
            lines.append("Interval sensitivity skipped: less than two fractions in the current schedule.")

        if missing_families:
            lines.append(
                "Missing family-specific fits: "
                + ", ".join(missing_families)
                + " | default alpha/beta used."
            )

        lines.append(
            "Sensitivity uses the currently selected tumor, current schedule, and current predictor parameters."
        )
        return "\n".join(lines)

    @staticmethod
    def format_family_sequence(sequence: Sequence[str]) -> str:
        return " -> ".join(sequence) if sequence else "-"

    def refresh_comparison_view(self) -> None:
        self.scenario_comparison_report = None
        self.comparison_curves = {}

        if self.reference_geometry is None:
            self.clear_comparison_outputs("Load a treated tumor file and run a simulation first.")
            return

        try:
            current_schedule = self.schedule_from_table()
            alternative_schedule = self.comparison_schedule_from_table()
            if not current_schedule:
                raise ValueError("Current schedule is empty.")
            if not alternative_schedule:
                self.clear_comparison_outputs("Copy or enter an alternative schedule first.")
                return

            parameters = self.current_parameters()
            combined_schedule = list(current_schedule) + list(alternative_schedule)
            family_overrides, missing_families = self.build_family_overrides_for_schedule(
                combined_schedule,
                parameters,
            )
            sample_times = self.build_sample_times(combined_schedule)
            alternative_name = self.comparison_name_edit.text().strip() or "Alternative"
            scenarios = {
                "Current": current_schedule,
                alternative_name: alternative_schedule,
            }
            self.scenario_comparison_report = compare_treatment_scenarios(
                sample_times=sample_times,
                reference=self.reference_geometry,
                parameters=parameters,
                scenarios=scenarios,
                scaling_model=self.geometry_scaling_model,
                family_parameters=family_overrides or None,
            )
            self.comparison_curves = {}
            for scenario_name, scenario_schedule in scenarios.items():
                result = simulate_growth(
                    sample_times=sample_times,
                    parameters=parameters,
                    reference=self.reference_geometry,
                    schedule=scenario_schedule,
                    scaling_model=self.geometry_scaling_model,
                    family_parameters=family_overrides or None,
                )
                self.comparison_curves[scenario_name] = np.asarray(result.total_volume, dtype=float)

            self.populate_comparison_table()
            self.refresh_comparison_plot(sample_times)
            self.comparison_text.setPlainText(
                self.build_comparison_summary_text(missing_families, alternative_name)
            )
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.clear_comparison_outputs(traceback.format_exc())
            QMessageBox.critical(self, "Scenario comparison failed", str(exc))

    def clear_comparison_outputs(self, message: str = "") -> None:
        self.scenario_comparison_report = None
        self.comparison_curves = {}
        self.comparison_table.setRowCount(0)
        self.comparison_text.setPlainText(message)
        self.refresh_comparison_plot(None)

    def populate_comparison_table(self) -> None:
        table_rows = build_comparison_table_rows(self.scenario_comparison_report)
        self.comparison_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.comparison_table, row_index, values)

    def export_comparison_csv(self) -> None:
        comparison_rows = build_comparison_table_rows(self.scenario_comparison_report)
        if not comparison_rows:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Run scenario comparison first.",
            )
            return

        default_stem = self.treated_path.stem if self.treated_path is not None else "tumor_growth"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export scenario comparison CSV",
            str(Path.cwd() / f"{default_stem}_comparison.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        output_path = write_csv_rows(path, COMPARISON_HEADERS, comparison_rows)
        self.statusBar().showMessage(f"Comparison CSV export complete: {output_path}")

    def refresh_comparison_plot(self, sample_times: Optional[np.ndarray]) -> None:
        self.comparison_figure.clear()
        axis = self.comparison_figure.add_subplot(111)

        if sample_times is None or not self.comparison_curves:
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No scenario comparison yet",
                ha="center",
                va="center",
                fontsize=11,
                color="#5f6b7a",
            )
            self.comparison_canvas.draw_idle()
            return

        for scenario_name, values in self.comparison_curves.items():
            axis.plot(sample_times, values, linewidth=2.0, label=scenario_name)

        if self.observed_days is not None and self.observed_volume is not None:
            valid = np.isfinite(self.observed_days) & np.isfinite(self.observed_volume) & (self.observed_volume > 0.0)
            if np.any(valid):
                axis.scatter(
                    self.observed_days[valid],
                    self.observed_volume[valid],
                    color="#1d3557",
                    label="Observed",
                    zorder=3,
                )

        axis.set_title("Scenario comparison")
        axis.set_xlabel("Time (days)")
        axis.set_ylabel("Total volume")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        self.comparison_figure.tight_layout(pad=1.1)
        self.comparison_canvas.draw_idle()

    def build_comparison_summary_text(
        self,
        missing_families: Sequence[str],
        alternative_name: str,
    ) -> str:
        if self.scenario_comparison_report is None or not self.scenario_comparison_report.rows:
            return ""

        rows = list(self.scenario_comparison_report.rows)
        current_row = next((row for row in rows if row.scenario == "Current"), None)
        alternative_row = next((row for row in rows if row.scenario == alternative_name), None)
        current_schedule = self.schedule_from_table()
        alternative_schedule = self.comparison_schedule_from_table()

        lines: list[str] = []
        if len(current_schedule) >= 2:
            lines.append(
                "Current intervals: "
                + format_schedule_intervals([event.day for event in current_schedule])
            )
        if len(alternative_schedule) >= 2:
            lines.append(
                f"{alternative_name} intervals: "
                + format_schedule_intervals([event.day for event in alternative_schedule])
            )
        if current_row is not None and alternative_row is not None:
            same_dose = abs(current_row.total_physical_dose - alternative_row.total_physical_dose) <= 1.0e-9
            lines.append(
                "Physical dose match: "
                + ("yes" if same_dose else f"no ({current_row.total_physical_dose:.3f} vs {alternative_row.total_physical_dose:.3f} Gy)")
            )
            lines.append(
                f"Lower final volume: {rows[0].scenario} ({rows[0].final_total_volume:.6f})"
            )
            lines.append(
                f"Lower AUC: {min(rows, key=lambda row: row.auc_total_volume).scenario}"
            )

        if missing_families:
            lines.append(
                "Missing family-specific fits: "
                + ", ".join(missing_families)
                + " | default alpha/beta used."
            )

        lines.append(
            "Comparison uses the current predictor parameters and family-specific overrides where available."
        )
        return "\n".join(lines)

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
            parameters = self.current_parameters()
            default_family = self.current_default_family()
            family_overrides, missing_families = build_family_parameter_overrides(
                self.run_results,
                parameters,
                schedule,
                default_family=default_family,
            )
            self.active_family_overrides = family_overrides
            self.missing_schedule_families = missing_families
            self.simulation_result = simulate_growth(
                sample_times,
                parameters,
                self.reference_geometry,
                schedule,
                self.geometry_scaling_model,
                family_overrides or None,
            )
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.summary_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Simulation failed", str(exc))
            return

        self.refresh_display_frames(reset_slider=True)
        self.refresh_curves()
        self.refresh_sensitivity_views()
        self.refresh_comparison_view()
        self.refresh_3d_view()
        if self.missing_schedule_families:
            self.statusBar().showMessage(
                "Tumor-growth simulation complete. Missing family-specific fits: "
                + ", ".join(self.missing_schedule_families)
            )
        elif self.active_family_overrides:
            self.statusBar().showMessage(
                "Tumor-growth simulation complete with family-specific alpha/beta overrides."
            )
        else:
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
        schedule = self.schedule_from_table()
        interval_schedule_label = "-"
        if len(schedule) >= 2:
            interval_schedule_label = format_schedule_intervals([event.day for event in schedule])
        tcp_summary_line: str | None = None
        schedule_sf_line: str | None = None
        initial_volume_line: str | None = None
        if self.reference_geometry is not None and schedule:
            initial_volume_cm3 = float(self.reference_geometry.volume) / 1000.0
            initial_volume_line = f"Initial volume = {initial_volume_cm3:.4f} cm^3"
            try:
                cell_density = self.current_tcp_cell_density()
                schedule_sf = predict_schedule_surviving_fraction(
                    schedule,
                    self.current_parameters(),
                    family_parameters=self.active_family_overrides or None,
                )
                burden = initial_volume_cm3 * cell_density * schedule_sf
                tcp_value = 0.0 if burden >= 700.0 else float(math.exp(-burden))
                schedule_sf_line = f"Predicted schedule SF = {schedule_sf:.6f}"
                tcp_summary_line = (
                    f"Predicted TCP = {tcp_value:.6f} "
                    f"(rho={cell_density:.4g} cells/cm^3)"
                )
            except ValueError as exc:
                tcp_summary_line = f"Predicted TCP = unavailable ({exc})"

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
        if initial_volume_line is not None:
            lines.append(initial_volume_line)
        if schedule_sf_line is not None:
            lines.append(schedule_sf_line)
        if tcp_summary_line is not None:
            lines.append(tcp_summary_line)

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

        if schedule:
            lines.append("")
            lines.append(f"Interval schedule = {interval_schedule_label}")
            lines.append("Dose schedule:")
            for event in schedule:
                family_suffix = f" | family={event.family}" if event.family else ""
                lines.append(f"- t = {_format_time_days(event.day)}: {event.dose:g} Gy{family_suffix}")
        if self.active_family_overrides:
            lines.append("")
            lines.append("Family overrides:")
            for family, parameters in sorted(self.active_family_overrides.items()):
                lines.append(
                    f"- {family}: alpha={parameters.alpha:.6f}, beta={parameters.beta:.6f}, "
                    + (
                        f"repair_half_time_hours={parameters.repair_half_time_hours:.3f}"
                        if parameters.repair_half_time_hours > 0.0
                        else "repair_half_time_hours=disabled"
                    )
                )
        if self.missing_schedule_families:
            lines.append("")
            lines.append("Missing family-specific fits:")
            for family in self.missing_schedule_families:
                lines.append(f"- {family} -> using default alpha/beta")
        return "\n".join(lines)

    def clear_prediction_outputs(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self.simulation_result = None
        self.display_frame_times = None
        self.active_family_overrides = {}
        self.missing_schedule_families = ()
        self.parameter_sensitivity_report = None
        self.interval_sensitivity_report = None
        self.scenario_comparison_report = None
        self.comparison_curves = {}
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.refresh_curves()
        self.clear_sensitivity_outputs()
        self.clear_comparison_outputs()
        self.refresh_3d_view()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TumorGrowthPredictorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
