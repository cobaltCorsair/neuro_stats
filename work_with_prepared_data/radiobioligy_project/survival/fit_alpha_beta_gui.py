# coding: utf-8
"""Simple PyQt6 GUI for LQ alpha/beta fitting from tumor-volume Excel files."""

from __future__ import annotations

from dataclasses import replace
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    FAMILY_LET_DEFAULTS,
    Fitter,
    InventoryReport,
    LQFitResult,
    PredictionRow,
    TumorExperiment,
    analyze_files,
    format_fractions,
    infer_radiation_family,
    is_control_file,
    parse_sf_modes,
)
from work_with_prepared_data.radiobioligy_project.survival.geant4_pipeline_gui import (
    Geant4PipelineWindow,
)
from work_with_prepared_data.radiobioligy_project.survival.gui_csv_export import (
    related_csv_path,
    write_csv_rows,
)
from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
    NTCPFitGroup,
    NTCPFitResult,
    NTCPPoint,
    RBEPoint,
    SFMetricComparisonRow,
    TCPResult,
    build_ntcp_curve,
    build_rbe_let_series,
    build_rbe_series,
    build_tcp_curve,
    compare_sf_metric_sensitivity,
    compute_tcp,
    export_bed_eqd2_table,
    fit_ntcp_lkb_from_groups,
    summarize_skin_reaction_file,
)
from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor_gui import (
    TumorGrowthPredictorWindow,
)

USE_ALL_CONTROLS = "__all_controls__"
UNASSIGNED_CONTROL = "__unassigned_control__"

COMPACT_PLOT_FIGSIZE = (7.0, 3.35)
COMPACT_PLOT_MIN_HEIGHT = 150
COMPACT_PLOT_LABEL_FONT = 11
COMPACT_PLOT_TICK_FONT = 9
COMPACT_PLOT_LEGEND_FONT = 9
COMPACT_PLOT_LINE_WIDTH = 1.8
COMPACT_PLOT_SECONDARY_LINE_WIDTH = 1.5
COMPACT_PLOT_MARKER_SIZE = 5.5


SUMMARY_HEADERS = [
    "SF mode",
    "Response",
    "Model",
    "Family",
    "Status",
    "Total",
    "Single",
    "Fractionated",
    "Train",
    "Validation",
    "Alpha",
    "Beta",
    "Alpha/Beta",
    "BED",
    "EQD2",
    "G",
    "R²",
    "Adj R²",
    "BIC",
    "CV RMSE",
    "Reason",
]

TRAIN_HEADERS = [
    "File",
    "Family",
    "Control",
    "Kind",
    "Fractions",
    "Schedule",
    "D",
    "D2",
    "SF",
    "BED",
    "EQD2",
    "G",
    "Pred TCP",
    "Repeats",
    "SF std",
]

VALIDATION_HEADERS = [
    "File",
    "Family",
    "Control",
    "Kind",
    "Fractions",
    "Schedule",
    "BED",
    "EQD2",
    "G",
    "Pred TCP",
    "Observed SF",
    "Predicted SF",
    "Abs error",
    "Rel error",
    "Log error",
]

CROSS_VALIDATION_HEADERS = [
    "File",
    "Observed SF",
    "Predicted SF",
    "Residual",
    "Abs error",
    "Rel error",
    "Log error",
]

TCP_HEADERS = [
    "File",
    "Kind",
    "Dose",
    "Fractions",
    "Schedule",
    "BED",
    "EQD2",
    "G",
    "Pred SF",
    "TCP",
]

BOOTSTRAP_HEADERS = [
    "Parameter",
    "Mean",
    "Std",
    "Q2.5%",
    "Median",
    "Q97.5%",
]

ASSIGNMENT_HEADERS = [
    "Experiment",
    "Family",
    "Control",
]

INVENTORY_HEADERS = [
    "File",
    "Role",
    "Family",
    "Kind",
    "Fractions",
    "Schedule",
    "Control",
    "Fit ready",
    "Notes",
]

RBE_HEADERS = [
    "Reference",
    "Test family",
    "Test dose",
    "Reference dose",
    "RBE",
    "Test alpha/beta",
    "Model",
]

SF_METRIC_HEADERS = [
    "Family",
    "Response",
    "Model",
    "SF mode",
    "Alpha",
    "Beta",
    "Alpha/Beta",
    "Delta alpha %",
    "Delta beta %",
    "Delta ratio %",
]

LET_ALPHA_HEADERS = [
    "Family",
    "LET (keV/um)",
    "Alpha",
    "Model",
]

NTCP_HEADERS = [
    "Dose",
    "NTCP",
    "TD50",
    "m",
]

NTCP_SOURCE_HEADERS = [
    "File",
    "Dose (Gy)",
    "Subjects",
    "Complications",
    "Rate",
    "Peak RTOG",
]


def parse_positive_float_csv(text: str, default: Sequence[float] = (2.0, 10.0)) -> List[float]:
    """Parse a comma-separated positive float list for GUI analysis controls."""
    raw = text.strip()
    if not raw:
        return [float(value) for value in default]

    values: List[float] = []
    for chunk in raw.split(","):
        value = float(chunk.strip().replace(",", "."))
        if value <= 0.0:
            raise ValueError("Dose values must be positive.")
        values.append(value)
    return values or [float(value) for value in default]


def _format_export_float(value: Optional[float], digits: int) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def parse_positive_scalar(text: str, *, label: str) -> float:
    """Parse one positive scalar value from a GUI text field."""
    normalized = text.strip().replace(",", ".")
    if not normalized:
        raise ValueError(f"{label} is required.")

    value = float(normalized)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive number.")
    return value


def build_rbe_table_rows(rows: Sequence[RBEPoint]) -> List[List[str]]:
    ordered_rows = sorted(rows, key=lambda row: (row.test_family, row.test_dose))
    return [
        [
            row.reference_family,
            row.test_family,
            f"{row.test_dose:.3f}",
            f"{row.reference_dose:.6f}",
            f"{row.rbe:.6f}",
            _format_export_float(row.test_alpha_beta_ratio, digits=3),
            row.test_model_kind,
        ]
        for row in ordered_rows
    ]


def build_sf_metric_table_rows(rows: Sequence[SFMetricComparisonRow]) -> List[List[str]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (row.family, row.response_mode, row.model_kind, row.sf_mode),
    )
    return [
        [
            row.family,
            row.response_mode,
            row.model_kind,
            row.sf_mode,
            _format_export_float(row.alpha, digits=6),
            _format_export_float(row.beta, digits=6),
            _format_export_float(row.alpha_beta_ratio, digits=3),
            _format_export_float(row.delta_alpha_pct, digits=2),
            _format_export_float(row.delta_beta_pct, digits=2),
            _format_export_float(row.delta_ratio_pct, digits=2),
        ]
        for row in ordered_rows
    ]


def build_let_alpha_table_rows(rows: Sequence[tuple[str, float, float, str]]) -> List[List[str]]:
    ordered_rows = sorted(rows, key=lambda item: item[1])
    return [
        [
            family,
            f"{let_kev_um:.3f}",
            f"{alpha:.6f}",
            model_kind,
        ]
        for family, let_kev_um, alpha, model_kind in ordered_rows
    ]


def build_ntcp_table_rows(rows: Sequence[NTCPPoint]) -> List[List[str]]:
    ordered_rows = sorted(rows, key=lambda row: row.dose_total)
    return [
        [
            f"{row.dose_total:.3f}",
            f"{row.ntcp:.6f}",
            f"{row.td50:.3f}",
            f"{row.m:.4f}",
        ]
        for row in ordered_rows
    ]


def build_ntcp_source_table_rows(rows: Sequence[NTCPFitGroup]) -> List[List[str]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (math.inf if row.dose_total is None else float(row.dose_total), row.label.lower()),
    )
    return [
        [
            row.label,
            ("" if row.dose_total is None else f"{float(row.dose_total):.3f}"),
            str(int(row.n_subjects)),
            str(int(row.n_complications)),
            f"{float(row.complication_rate):.6f}",
            f"{float(row.peak_grade_mean):.3f}",
        ]
        for row in ordered_rows
    ]


def build_cross_validation_table_rows(rows: Sequence[PredictionRow]) -> List[List[str]]:
    """Convert leave-one-out rows into displayable table values."""
    return [
        [
            row.experiment.path.name,
            f"{row.observed_sf:.6f}",
            f"{row.predicted_sf:.6f}",
            f"{row.residual:.6f}",
            f"{row.abs_error:.6f}",
            f"{row.rel_error:.2%}",
            f"{row.log_error:.6f}",
        ]
        for row in rows
    ]


def build_tcp_table_rows(
    fit_result: LQFitResult,
    rows: Sequence[TCPResult],
    *,
    n_fractions: int,
    schedule_interval_days: float,
) -> List[List[str]]:
    """Convert a synthetic TCP dose sweep into table rows with BED/EQD2/G."""
    table_rows: List[List[str]] = []
    for row in sorted(rows, key=lambda item: item.dose_total):
        fraction_dose = float(row.dose_total) / float(n_fractions)
        fractions = tuple(float(fraction_dose) for _ in range(n_fractions))
        schedule_days = tuple(
            float(index) * float(schedule_interval_days)
            for index in range(n_fractions)
        )
        experiment = TumorExperiment(
            path=Path("synthetic_tcp.xlsx"),
            fractions=fractions,
            sf=float(row.sf),
            family=row.family or fit_result.family,
            sf_time_day=(schedule_days[-1] + 1.0) if schedule_days else 1.0,
            schedule_days=schedule_days,
            has_explicit_timing=n_fractions > 1,
            initial_volume_mm3=float(row.initial_volume_cm3) * 1000.0,
        )
        table_rows.append(
            [
                experiment.path.name,
                experiment.regimen_kind,
                f"{row.dose_total:.3f}",
                format_fractions(experiment.fractions),
                experiment.schedule_label,
                _format_export_float(fit_result.compute_bed(experiment), digits=4),
                _format_export_float(fit_result.compute_eqd2(experiment), digits=4),
                _format_export_float(fit_result.compute_g_factor(experiment), digits=4),
                _format_export_float(row.sf, digits=6),
                _format_export_float(row.tcp, digits=6),
            ]
        )
    return table_rows


def select_rbe_run_context(
    run_results: Sequence[AnalysisRunResult],
    selected_index: int,
    reference_family: str,
) -> tuple[LQFitResult, dict[str, LQFitResult], str]:
    """Select a coherent RBE comparison context from fitted GUI runs."""
    if selected_index < 0 or selected_index >= len(run_results):
        raise ValueError("Choose a fitted run first.")

    selected_run = run_results[selected_index]
    if selected_run.fit_result is None:
        raise ValueError("Selected run does not contain fitted alpha/beta values.")

    response_mode = selected_run.summary.response_mode
    model_kind = selected_run.summary.model_kind
    sf_mode = selected_run.summary.sf_mode
    reference_family = reference_family.strip().lower()

    matching_runs = [
        run
        for run in run_results
        if run.fit_result is not None
        and run.summary.status == "ok"
        and run.summary.response_mode == response_mode
        and run.summary.model_kind == model_kind
        and run.summary.sf_mode == sf_mode
    ]
    if not matching_runs:
        raise ValueError("No comparable fitted runs are available for RBE analysis.")

    reference_run = next(
        (run for run in matching_runs if (run.summary.family or "").lower() == reference_family),
        None,
    )
    if reference_run is None or reference_run.fit_result is None:
        raise ValueError(
            "No fitted reference run matches the selected response/model/SF context "
            f"for family '{reference_family}'."
        )

    comparison_results: dict[str, LQFitResult] = {}
    for run in matching_runs:
        family = (run.summary.family or "").lower()
        if not family or family == reference_family or run.fit_result is None:
            continue
        comparison_results[family] = run.fit_result

    if not comparison_results:
        raise ValueError(
            "No other fitted families are available in the same response/model/SF context."
        )

    context_label = (
        f"sf={sf_mode} | response={response_mode} | model={model_kind} | reference={reference_family}"
    )
    return reference_run.fit_result, comparison_results, context_label


def select_let_run_context(
    run_results: Sequence[AnalysisRunResult],
    selected_index: int,
) -> tuple[LQFitResult, list[tuple[str, float, float, str]], str]:
    """Select one LET-dependent fit and per-family alpha points for alpha(LET) plots."""
    if selected_index < 0 or selected_index >= len(run_results):
        raise ValueError("Choose a fitted run first.")

    selected_run = run_results[selected_index]
    response_mode = selected_run.summary.response_mode
    sf_mode = selected_run.summary.sf_mode

    compatible_runs = [
        run
        for run in run_results
        if run.fit_result is not None
        and run.summary.status == "ok"
        and run.summary.response_mode == response_mode
        and run.summary.sf_mode == sf_mode
    ]
    if not compatible_runs:
        raise ValueError("No compatible fitted runs are available.")

    let_run = next(
        (
            run
            for run in compatible_runs
            if run.summary.model_kind == "let_dependent"
        ),
        None,
    )
    if let_run is None or let_run.fit_result is None:
        raise ValueError("No LET-dependent fit is available in the selected response/SF context.")

    model_priority = {
        "classic_lq": 0,
        "repair_lq": 1,
        "repair_biexp": 1,
        "glq": 2,
        "lq_l": 2,
        "linear": 3,
        "lq_repop": 4,
        "repair_repop": 4,
    }
    best_by_family: dict[str, tuple[str, float]] = {}
    for run in compatible_runs:
        if run.fit_result is None or run.summary.model_kind == "let_dependent":
            continue
        family = (run.summary.family or "").strip().lower()
        if family not in FAMILY_LET_DEFAULTS:
            continue
        current_priority = model_priority.get(run.summary.model_kind, 99)
        if family in best_by_family:
            previous_model_kind, previous_alpha = best_by_family[family]
            previous_priority = model_priority.get(previous_model_kind, 99)
            if current_priority >= previous_priority:
                continue
        best_by_family[family] = (run.summary.model_kind, run.fit_result.alpha)

    points = sorted(
        (
            (family, float(FAMILY_LET_DEFAULTS[family]), alpha, model_kind)
            for family, (model_kind, alpha) in best_by_family.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    context_label = (
        f"sf={sf_mode} | response={response_mode} | let-model={let_run.summary.model_kind}"
    )
    return let_run.fit_result, points, context_label


class FileDropListWidget(QListWidget):
    """List widget that accepts dropped local files."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class FitAlphaBetaWindow(QMainWindow):
    """Standalone GUI for the survival fitter."""

    def __init__(self) -> None:
        super().__init__()
        self.run_results: List[AnalysisRunResult] = []
        self.inventory_report: Optional[InventoryReport] = None
        self.inventory_stale = False
        self._suspend_assignment_inventory_refresh = False
        self.growth_predictor_window: Optional[TumorGrowthPredictorWindow] = None
        self.geant4_pipeline_window: Optional[Geant4PipelineWindow] = None
        self.rbe_points: List[RBEPoint] = []
        self.sf_metric_rows: List[SFMetricComparisonRow] = []
        self.let_fit_result: Optional[LQFitResult] = None
        self.let_alpha_points: List[tuple[str, float, float, str]] = []
        self.tcp_curve_rows: List[TCPResult] = []
        self.ntcp_curve_rows: List[NTCPPoint] = []
        self.ntcp_fit_groups: List[NTCPFitGroup] = []
        self.ntcp_fit_result: Optional[NTCPFitResult] = None
        self.setWindowTitle("Survival LQ fitter")
        self.resize(1260, 780)
        self.setMinimumSize(1080, 680)
        self._build_ui()
        self._apply_window_style()
        self.statusBar().showMessage("Drop .xlsx files here or add them with the buttons.")

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(self._build_sidebar_panel())

        self.results_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.setHandleWidth(4)
        self.results_splitter.addWidget(self._build_summary_panel())
        self.results_splitter.addWidget(self._build_detail_panel())
        self.results_splitter.setStretchFactor(0, 0)
        self.results_splitter.setStretchFactor(1, 1)
        self.results_splitter.setSizes([160, 610])

        main_splitter.addWidget(self.results_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([430, 900])
        root_layout.addWidget(main_splitter, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Drop .xlsx files сюда или добавьте их кнопками.")
        self.refresh_control_selector()

    def _build_sidebar_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.run_button = QPushButton("Run fit")
        self.run_button.setObjectName("PrimaryAction")
        self.run_button.clicked.connect(self.run_analysis)
        action_row.addWidget(self.run_button)

        self.inventory_button = QPushButton("Inventory")
        self.inventory_button.clicked.connect(self.scan_inventory)
        action_row.addWidget(self.inventory_button)
        layout.addLayout(action_row)

        self.predictor_button = QPushButton("Growth predictor")
        self.predictor_button.clicked.connect(self.open_growth_predictor)
        layout.addWidget(self.predictor_button)

        self.geant4_pipeline_button = QPushButton("GEANT4 pipeline")
        self.geant4_pipeline_button.clicked.connect(self.open_geant4_pipeline)
        layout.addWidget(self.geant4_pipeline_button)

        tabs = QTabWidget(self)

        files_page = QWidget(self)
        files_layout = QVBoxLayout(files_page)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.addWidget(self._build_file_group())
        tabs.addTab(files_page, "Files")

        options_container = QWidget(self)
        options_layout = QVBoxLayout(options_container)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.addWidget(self._build_options_group())
        options_layout.addStretch(1)

        options_scroll = QScrollArea(self)
        options_scroll.setWidgetResizable(True)
        options_scroll.setWidget(options_container)
        tabs.addTab(options_scroll, "Fit setup")

        layout.addWidget(tabs, 1)
        return panel

    def open_growth_predictor(self) -> None:
        if self.growth_predictor_window is None:
            self.growth_predictor_window = TumorGrowthPredictorWindow(self.run_results)
        else:
            self.growth_predictor_window.run_results = list(self.run_results)
            self.growth_predictor_window.populate_fit_results()
        self.growth_predictor_window.show()
        self.growth_predictor_window.raise_()
        self.growth_predictor_window.activateWindow()

    def open_geant4_pipeline(self) -> None:
        summary_csv_path = self.summary_csv_edit.text().strip() or None
        if self.geant4_pipeline_window is None:
            self.geant4_pipeline_window = Geant4PipelineWindow(
                self.run_results,
                summary_csv_path=summary_csv_path,
            )
        else:
            self.geant4_pipeline_window.set_context(
                self.run_results,
                summary_csv_path=summary_csv_path,
            )
        self.geant4_pipeline_window.show()
        self.geant4_pipeline_window.raise_()
        self.geant4_pipeline_window.activateWindow()

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Files and controls", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        add_files_button = QPushButton("Add files")
        add_files_button.clicked.connect(self.add_files_dialog)
        button_row.addWidget(add_files_button)

        add_folder_button = QPushButton("Folder")
        add_folder_button.clicked.connect(self.add_folder_dialog)
        button_row.addWidget(add_folder_button)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_selected_files)
        button_row.addWidget(remove_button)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_files)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)

        layout.addLayout(button_row)

        self.file_list = FileDropListWidget(self)
        self.file_list.files_dropped.connect(self.add_paths)
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setMinimumHeight(140)
        self.file_list.setToolTip(
            "Можно перетаскивать .xlsx файлы или целые папки. "
            "Файлы с 'control' в имени будут использованы как контроль."
        )
        layout.addWidget(self.file_list)

        layout.addWidget(QLabel("Experiment to control mapping"))
        self.assignment_table = self._create_table(ASSIGNMENT_HEADERS)
        self.assignment_table.setMinimumHeight(150)
        layout.addWidget(self.assignment_table)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Fit options", self)
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("SF modes"), 0, 0)
        self.sf_modes_edit = QLineEdit("absolute", self)
        self.sf_modes_edit.setPlaceholderText("absolute, absindex:1")
        layout.addWidget(self.sf_modes_edit, 0, 1)

        layout.addWidget(QLabel("Family"), 0, 2)
        self.family_combo = QComboBox(self)
        self.family_combo.addItem("all", None)
        for family in ("y", "p", "p_peak", "p_through", "n", "e", "c"):
            self.family_combo.addItem(family, family)
        layout.addWidget(self.family_combo, 0, 3)

        self.by_family_check = QCheckBox("By family", self)
        layout.addWidget(self.by_family_check, 0, 4)

        layout.addWidget(QLabel("Default control"), 1, 0)
        self.control_combo = QComboBox(self)
        self.control_combo.setToolTip(
            "Выберите control по умолчанию для новых строк сопоставления "
            "или примените его ко всем экспериментам."
        )
        layout.addWidget(self.control_combo, 1, 1, 1, 4)

        self.apply_default_control_button = QPushButton("Apply to all")
        self.apply_default_control_button.clicked.connect(self.apply_default_control_to_all)
        layout.addWidget(self.apply_default_control_button, 1, 5)

        layout.addWidget(QLabel("Fit kind"), 2, 0)
        self.fit_kind_combo = QComboBox(self)
        self.fit_kind_combo.addItem("all", "all")
        self.fit_kind_combo.addItem("single", "single")
        self.fit_kind_combo.addItem("fractionated", "fractionated")
        layout.addWidget(self.fit_kind_combo, 2, 1)

        layout.addWidget(QLabel("Validate kind"), 2, 2)
        self.validate_kind_combo = QComboBox(self)
        self.validate_kind_combo.addItem("none", "none")
        self.validate_kind_combo.addItem("all", "all")
        self.validate_kind_combo.addItem("single", "single")
        self.validate_kind_combo.addItem("fractionated", "fractionated")
        layout.addWidget(self.validate_kind_combo, 2, 3)

        layout.addWidget(QLabel("Min SF"), 2, 4)
        self.min_sf_spin = QDoubleSpinBox(self)
        self.min_sf_spin.setRange(0.0, 1.0)
        self.min_sf_spin.setDecimals(3)
        self.min_sf_spin.setSingleStep(0.05)
        self.min_sf_spin.setValue(1.0)
        layout.addWidget(self.min_sf_spin, 2, 5)

        layout.addWidget(QLabel("Response mode"), 3, 0)
        self.response_mode_combo = QComboBox(self)
        self.response_mode_combo.addItem("scalar", "scalar")
        self.response_mode_combo.addItem("curve", "curve")
        self.response_mode_combo.setToolTip(
            "Scalar fits one SF per regimen. Curve uses the full normalized tumor-volume response."
        )
        layout.addWidget(self.response_mode_combo, 3, 1)

        layout.addWidget(QLabel("Model"), 3, 2)
        self.model_kind_combo = QComboBox(self)
        self.model_kind_combo.addItem("auto", "auto")
        self.model_kind_combo.addItem("classic_lq", "classic_lq")
        self.model_kind_combo.addItem("repair_lq", "repair_lq")
        self.model_kind_combo.addItem("repair_biexp", "repair_biexp")
        self.model_kind_combo.addItem("glq", "glq")
        self.model_kind_combo.addItem("let_dependent", "let_dependent")
        self.model_kind_combo.addItem("repair_repop", "repair_repop")
        self.model_kind_combo.addItem("lq_l", "lq_l")
        self.model_kind_combo.addItem("lq_repop", "lq_repop")
        self.model_kind_combo.addItem("linear", "linear")
        layout.addWidget(self.model_kind_combo, 3, 3)

        self.compare_models_check = QCheckBox("Compare models", self)
        self.compare_models_check.setToolTip(
            "Run classic LQ, repair-aware LQ, bi-exponential repair, gLQ, LET-dependent LQ, "
            "repair + repopulation, LQ-L, LQ + repopulation and linear candidates, then rank them by fit error."
        )
        layout.addWidget(self.compare_models_check, 3, 4, 1, 2)

        self.fix_alpha_check = QCheckBox("Fix alpha", self)
        self.fix_alpha_check.toggled.connect(self._update_alpha_enabled)
        layout.addWidget(self.fix_alpha_check, 4, 0)

        self.alpha_spin = QDoubleSpinBox(self)
        self.alpha_spin.setRange(0.0, 10.0)
        self.alpha_spin.setDecimals(6)
        self.alpha_spin.setSingleStep(0.001)
        self.alpha_spin.setEnabled(False)
        layout.addWidget(self.alpha_spin, 4, 1)

        layout.addWidget(QLabel("Bootstrap"), 4, 2)
        self.bootstrap_spin = QSpinBox(self)
        self.bootstrap_spin.setRange(0, 100000)
        self.bootstrap_spin.setValue(0)
        layout.addWidget(self.bootstrap_spin, 4, 3)

        layout.addWidget(QLabel("Bootstrap seed"), 4, 4)
        self.bootstrap_seed_edit = QLineEdit(self)
        self.bootstrap_seed_edit.setPlaceholderText("optional")
        layout.addWidget(self.bootstrap_seed_edit, 4, 5)

        layout.addWidget(QLabel("Repair T1/2 (h)"), 5, 0)
        self.repair_half_time_spin = QDoubleSpinBox(self)
        self.repair_half_time_spin.setRange(0.0, 240.0)
        self.repair_half_time_spin.setDecimals(3)
        self.repair_half_time_spin.setSingleStep(0.25)
        self.repair_half_time_spin.setValue(0.0)
        self.repair_half_time_spin.setToolTip(
            "0 keeps the classic schedule-free LQ model. Positive values enable "
            "time-aware repair for fractionated regimens with t= intervals."
        )
        layout.addWidget(self.repair_half_time_spin, 5, 1)

        layout.addWidget(QLabel("Fast repair T1/2 (h)"), 5, 2)
        self.repair_half_time_fast_spin = QDoubleSpinBox(self)
        self.repair_half_time_fast_spin.setRange(0.0, 240.0)
        self.repair_half_time_fast_spin.setDecimals(3)
        self.repair_half_time_fast_spin.setSingleStep(0.25)
        self.repair_half_time_fast_spin.setValue(0.0)
        self.repair_half_time_fast_spin.setToolTip(
            "Fast repair half-time for the bi-exponential repair model. "
            "Leave at 0 to disable the fast component."
        )
        layout.addWidget(self.repair_half_time_fast_spin, 5, 3)

        layout.addWidget(QLabel("Slow repair T1/2 (h)"), 5, 4)
        self.repair_half_time_slow_spin = QDoubleSpinBox(self)
        self.repair_half_time_slow_spin.setRange(0.0, 240.0)
        self.repair_half_time_slow_spin.setDecimals(3)
        self.repair_half_time_slow_spin.setSingleStep(0.25)
        self.repair_half_time_slow_spin.setValue(0.0)
        self.repair_half_time_slow_spin.setToolTip(
            "Slow repair half-time for the bi-exponential repair model. "
            "Leave at 0 to disable the slow component."
        )
        layout.addWidget(self.repair_half_time_slow_spin, 5, 5)

        self.aggregate_check = QCheckBox("Aggregate repeats", self)
        layout.addWidget(self.aggregate_check, 6, 0, 1, 2)

        self.dedupe_check = QCheckBox("Deduplicate", self)
        layout.addWidget(self.dedupe_check, 6, 2, 1, 2)

        layout.addWidget(QLabel("Fast fraction"), 6, 4)
        self.repair_fast_fraction_spin = QDoubleSpinBox(self)
        self.repair_fast_fraction_spin.setRange(0.0, 1.0)
        self.repair_fast_fraction_spin.setDecimals(3)
        self.repair_fast_fraction_spin.setSingleStep(0.05)
        self.repair_fast_fraction_spin.setValue(0.6)
        self.repair_fast_fraction_spin.setToolTip(
            "Weight of the fast repair component in the bi-exponential repair model."
        )
        layout.addWidget(self.repair_fast_fraction_spin, 6, 5)

        self.verbose_check = QCheckBox("Verbose terminal log", self)
        layout.addWidget(self.verbose_check, 7, 0, 1, 3)

        self.exclude_dead_check = QCheckBox("Exclude animals that died mid-experiment", self)
        self.exclude_dead_check.setToolTip(
            "Detects confirmed deaths before the last observed time point in each file "
            "(same death markers used by the Kaplan-Meier tool) and drops those animals' "
            "rows before fitting, for both control and experiment files."
        )
        layout.addWidget(self.exclude_dead_check, 7, 3, 1, 3)

        layout.addWidget(QLabel("Summary CSV"), 8, 0)
        self.summary_csv_edit = QLineEdit(self)
        self.summary_csv_edit.setPlaceholderText("optional path for summary csv")
        layout.addWidget(self.summary_csv_edit, 8, 1, 1, 4)

        summary_browse_button = QPushButton("Browse")
        summary_browse_button.clicked.connect(self.choose_summary_csv)
        layout.addWidget(summary_browse_button, 8, 5)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)
        return group

    def _build_summary_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Run summary"))
        header_row.addStretch(1)
        header_row.addWidget(QLabel("Selected run"))
        self.run_selector = QComboBox(self)
        self.run_selector.setMinimumWidth(220)
        self.run_selector.setMinimumContentsLength(28)
        self.run_selector.setMaxVisibleItems(14)
        self.run_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.run_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.run_selector.currentIndexChanged.connect(self.display_run)
        header_row.addWidget(self.run_selector)
        layout.addLayout(header_row)

        self.summary_table = self._create_table(SUMMARY_HEADERS)
        self.summary_table.currentCellChanged.connect(self._sync_run_selector_with_table)
        self.summary_table.setMinimumHeight(120)
        self.summary_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.summary_table)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.detail_tabs = QTabWidget(self)

        self.train_table = self._create_table(TRAIN_HEADERS)
        self.detail_tabs.addTab(self.train_table, "Training")

        self.validation_table = self._create_table(VALIDATION_HEADERS)
        self.detail_tabs.addTab(self.validation_table, "Validation")

        self.bootstrap_table = self._create_table(BOOTSTRAP_HEADERS)
        self.detail_tabs.addTab(self.bootstrap_table, "Bootstrap")

        self.cross_validation_table = self._create_table(CROSS_VALIDATION_HEADERS)
        self.detail_tabs.addTab(self.cross_validation_table, "Cross-validation")

        inventory_panel = QWidget(self)
        inventory_layout = QVBoxLayout(inventory_panel)
        self.inventory_table = self._create_table(INVENTORY_HEADERS)
        inventory_layout.addWidget(self.inventory_table, 1)
        self.inventory_text = QPlainTextEdit(self)
        self.inventory_text.setReadOnly(True)
        self.inventory_text.setMaximumHeight(150)
        inventory_layout.addWidget(self.inventory_text)
        self.detail_tabs.addTab(inventory_panel, "Inventory")

        self.details_text = QPlainTextEdit(self)
        self.details_text.setReadOnly(True)
        self.detail_tabs.addTab(self.details_text, "Summary text")

        self.detail_tabs.addTab(self._build_analysis_panel(), "Analysis")
        self.detail_tabs.addTab(self._build_tcp_panel(), "TCP")
        self.detail_tabs.addTab(self._build_ntcp_panel(), "NTCP")
        self.detail_tabs.addTab(self._build_bed_eqd2_panel(), "BED/EQD2")

        layout.addWidget(self.detail_tabs, 1)
        return panel

    def _build_analysis_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_group = QGroupBox("Radiobiology analysis", self)
        controls_layout = QGridLayout(controls_group)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        controls_layout.addWidget(QLabel("Reference family"), 0, 0)
        self.rbe_reference_combo = QComboBox(self)
        for family in ("y", "p", "p_peak", "p_through", "n", "e", "c"):
            self.rbe_reference_combo.addItem(family, family)
        self.rbe_reference_combo.setCurrentText("y")
        self.rbe_reference_combo.currentIndexChanged.connect(self.refresh_analysis_views)
        controls_layout.addWidget(self.rbe_reference_combo, 0, 1)

        controls_layout.addWidget(QLabel("RBE mode"), 0, 2)
        self.rbe_mode_combo = QComboBox(self)
        self.rbe_mode_combo.addItem("Per-family fits", "family")
        self.rbe_mode_combo.addItem("LET model", "let")
        self.rbe_mode_combo.currentIndexChanged.connect(self.refresh_analysis_views)
        controls_layout.addWidget(self.rbe_mode_combo, 0, 3)

        controls_layout.addWidget(QLabel("RBE doses (Gy)"), 0, 4)
        self.rbe_doses_edit = QLineEdit("2, 10", self)
        self.rbe_doses_edit.setPlaceholderText("2, 10")
        controls_layout.addWidget(self.rbe_doses_edit, 0, 5)

        self.rbe_button = QPushButton("Build RBE")
        self.rbe_button.clicked.connect(self.refresh_analysis_views)
        controls_layout.addWidget(self.rbe_button, 1, 0, 1, 2)

        self.sf_metric_button = QPushButton("Compare SF modes")
        self.sf_metric_button.clicked.connect(self.refresh_analysis_views)
        controls_layout.addWidget(self.sf_metric_button, 1, 2, 1, 2)

        self.export_analysis_button = QPushButton("Export CSV")
        self.export_analysis_button.clicked.connect(self.export_analysis_csv)
        controls_layout.addWidget(self.export_analysis_button, 1, 4, 1, 2)

        controls_layout.setColumnStretch(5, 1)
        layout.addWidget(controls_group)

        self.analysis_figure = Figure(figsize=(8, 4.8))
        self.analysis_canvas = FigureCanvasQTAgg(self.analysis_figure)
        self.analysis_canvas.setMinimumHeight(220)
        layout.addWidget(self.analysis_canvas)

        analysis_tables = QSplitter(Qt.Orientation.Vertical, self)
        analysis_tables.setChildrenCollapsible(False)

        self.rbe_table = self._create_table(RBE_HEADERS)
        self.rbe_table.setMinimumHeight(140)
        analysis_tables.addWidget(self.rbe_table)

        self.sf_metric_table = self._create_table(SF_METRIC_HEADERS)
        self.sf_metric_table.setMinimumHeight(140)
        analysis_tables.addWidget(self.sf_metric_table)
        analysis_tables.setStretchFactor(0, 1)
        analysis_tables.setStretchFactor(1, 1)
        analysis_tables.setSizes([180, 180])
        layout.addWidget(analysis_tables, 1)

        self.analysis_text = QPlainTextEdit(self)
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(140)
        layout.addWidget(self.analysis_text)
        return panel

    def _build_ntcp_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.ntcp_controls_group = QGroupBox("NTCP analysis", self)
        controls_layout = QGridLayout(self.ntcp_controls_group)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        controls_layout.addWidget(QLabel("TD50"), 0, 0)
        self.ntcp_td50_spin = QDoubleSpinBox(self)
        self.ntcp_td50_spin.setRange(0.001, 1000.0)
        self.ntcp_td50_spin.setDecimals(3)
        self.ntcp_td50_spin.setSingleStep(1.0)
        self.ntcp_td50_spin.setValue(50.0)
        controls_layout.addWidget(self.ntcp_td50_spin, 0, 1)

        controls_layout.addWidget(QLabel("m"), 0, 2)
        self.ntcp_m_spin = QDoubleSpinBox(self)
        self.ntcp_m_spin.setRange(0.001, 10.0)
        self.ntcp_m_spin.setDecimals(4)
        self.ntcp_m_spin.setSingleStep(0.01)
        self.ntcp_m_spin.setValue(0.2)
        controls_layout.addWidget(self.ntcp_m_spin, 0, 3)

        controls_layout.addWidget(QLabel("Threshold grade"), 0, 4)
        self.ntcp_threshold_spin = QSpinBox(self)
        self.ntcp_threshold_spin.setRange(1, 4)
        self.ntcp_threshold_spin.setValue(3)
        controls_layout.addWidget(self.ntcp_threshold_spin, 0, 5)

        controls_layout.addWidget(QLabel("Input scale"), 0, 6)
        self.ntcp_scale_combo = QComboBox(self)
        self.ntcp_scale_combo.addItem("Legacy skin scale", "our")
        self.ntcp_scale_combo.addItem("RTOG grades", "rtog")
        controls_layout.addWidget(self.ntcp_scale_combo, 0, 7)

        controls_layout.addWidget(QLabel("Dose grid (Gy)"), 1, 0)
        self.ntcp_dose_grid_edit = QLineEdit("2, 10, 20, 30, 40, 50, 60", self)
        self.ntcp_dose_grid_edit.setPlaceholderText("2, 10, 20, 30, 40, 50, 60")
        controls_layout.addWidget(self.ntcp_dose_grid_edit, 1, 1, 1, 5)

        self.ntcp_build_button = QPushButton("Build NTCP")
        self.ntcp_build_button.clicked.connect(self.refresh_ntcp_view)
        controls_layout.addWidget(self.ntcp_build_button, 1, 6)

        self.ntcp_export_button = QPushButton("Export CSV")
        self.ntcp_export_button.clicked.connect(self.export_ntcp_csv)
        controls_layout.addWidget(self.ntcp_export_button, 1, 7)

        self.ntcp_load_button = QPushButton("Load skin files")
        self.ntcp_load_button.clicked.connect(self.load_ntcp_skin_files)
        controls_layout.addWidget(self.ntcp_load_button, 2, 0, 1, 2)

        self.ntcp_fit_button = QPushButton("Fit TD50/m")
        self.ntcp_fit_button.clicked.connect(self.fit_ntcp_from_skin_files)
        controls_layout.addWidget(self.ntcp_fit_button, 2, 2, 1, 2)

        self.ntcp_clear_groups_button = QPushButton("Clear skin files")
        self.ntcp_clear_groups_button.clicked.connect(self.clear_ntcp_skin_files)
        controls_layout.addWidget(self.ntcp_clear_groups_button, 2, 4, 1, 2)

        controls_layout.setColumnStretch(5, 1)
        layout.addWidget(self.ntcp_controls_group)

        self.ntcp_content_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.ntcp_content_splitter.setChildrenCollapsible(False)
        self.ntcp_content_splitter.setHandleWidth(4)

        self.ntcp_source_group = QGroupBox("Skin/RTOG groups", self)
        source_layout = QVBoxLayout(self.ntcp_source_group)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_layout.setSpacing(6)

        self.ntcp_source_table = self._create_table(NTCP_SOURCE_HEADERS)
        self.ntcp_source_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.ntcp_source_group.setMinimumHeight(0)
        self.ntcp_source_table.setMinimumHeight(0)
        source_layout.addWidget(self.ntcp_source_table)
        self.ntcp_content_splitter.addWidget(self.ntcp_source_group)
        self.ntcp_content_splitter.setCollapsible(0, True)

        self.ntcp_figure = Figure(figsize=COMPACT_PLOT_FIGSIZE)
        self.ntcp_canvas = FigureCanvasQTAgg(self.ntcp_figure)
        self.ntcp_canvas.setMinimumHeight(COMPACT_PLOT_MIN_HEIGHT)
        self.ntcp_content_splitter.addWidget(self.ntcp_canvas)

        self.ntcp_table = self._create_table(NTCP_HEADERS)
        self.ntcp_table.setMinimumHeight(100)
        self.ntcp_content_splitter.addWidget(self.ntcp_table)

        self.ntcp_text = QPlainTextEdit(self)
        self.ntcp_text.setReadOnly(True)
        self.ntcp_text.setMinimumHeight(80)
        self.ntcp_content_splitter.addWidget(self.ntcp_text)

        self.ntcp_content_splitter.setStretchFactor(0, 2)
        self.ntcp_content_splitter.setStretchFactor(1, 3)
        self.ntcp_content_splitter.setStretchFactor(2, 2)
        self.ntcp_content_splitter.setStretchFactor(3, 1)
        self.ntcp_content_splitter.setSizes([180, 230, 180, 120])
        layout.addWidget(self.ntcp_content_splitter, 1)
        return panel

    def _build_bed_eqd2_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_group = QGroupBox("BED/EQD2 reference table", self)
        controls_layout = QGridLayout(controls_group)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        controls_layout.addWidget(QLabel("Fitted α/β (Gy)"), 0, 0)
        self.bed_alpha_beta_label = QLabel("—", self)
        controls_layout.addWidget(self.bed_alpha_beta_label, 0, 1)

        controls_layout.addWidget(QLabel("Reference α/β for EQD2 (Gy)"), 0, 2)
        self.bed_reference_ab_spin = QDoubleSpinBox(self)
        self.bed_reference_ab_spin.setRange(0.1, 100.0)
        self.bed_reference_ab_spin.setDecimals(1)
        self.bed_reference_ab_spin.setSingleStep(0.5)
        self.bed_reference_ab_spin.setValue(2.0)
        controls_layout.addWidget(self.bed_reference_ab_spin, 0, 3)

        controls_layout.addWidget(QLabel("Dose grid (Gy)"), 1, 0)
        self.bed_dose_grid_edit = QLineEdit(
            "2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60", self
        )
        controls_layout.addWidget(self.bed_dose_grid_edit, 1, 1, 1, 3)

        controls_layout.addWidget(QLabel("Fractions (n)"), 2, 0)
        self.bed_fractions_edit = QLineEdit("1, 3, 5, 10, 15, 20, 30", self)
        controls_layout.addWidget(self.bed_fractions_edit, 2, 1, 1, 2)

        self.bed_build_button = QPushButton("Build")
        self.bed_build_button.clicked.connect(self.refresh_bed_eqd2_view)
        controls_layout.addWidget(self.bed_build_button, 2, 3)

        self.bed_export_button = QPushButton("Export CSV")
        self.bed_export_button.clicked.connect(self.export_bed_eqd2_csv)
        controls_layout.addWidget(self.bed_export_button, 2, 4)

        layout.addWidget(controls_group)

        self.bed_table = self._create_table(["D (Gy)", "n", "d (Gy)", "BED (Gy)", "EQD2 (Gy)"])
        layout.addWidget(self.bed_table, 1)

        return panel

    def _build_tcp_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.tcp_controls_group = QGroupBox("TCP analysis", self)
        controls_layout = QGridLayout(self.tcp_controls_group)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        controls_layout.addWidget(QLabel("Initial volume (cm³)"), 0, 0)
        self.tcp_initial_volume_spin = QDoubleSpinBox(self)
        self.tcp_initial_volume_spin.setRange(0.0001, 1000.0)
        self.tcp_initial_volume_spin.setDecimals(4)
        self.tcp_initial_volume_spin.setSingleStep(0.1)
        self.tcp_initial_volume_spin.setValue(1.0)
        controls_layout.addWidget(self.tcp_initial_volume_spin, 0, 1)

        controls_layout.addWidget(QLabel("Cell density"), 0, 2)
        self.tcp_cell_density_edit = QLineEdit("1e7", self)
        self.tcp_cell_density_edit.setPlaceholderText("1e7")
        controls_layout.addWidget(self.tcp_cell_density_edit, 0, 3)

        controls_layout.addWidget(QLabel("Dose grid (Gy)"), 1, 0)
        self.tcp_dose_grid_edit = QLineEdit("2, 10, 20, 30, 40", self)
        self.tcp_dose_grid_edit.setPlaceholderText("2, 10, 20, 30, 40")
        controls_layout.addWidget(self.tcp_dose_grid_edit, 1, 1, 1, 3)

        controls_layout.addWidget(QLabel("Fractions"), 0, 4)
        self.tcp_fraction_count_spin = QSpinBox(self)
        self.tcp_fraction_count_spin.setRange(1, 100)
        self.tcp_fraction_count_spin.setValue(1)
        controls_layout.addWidget(self.tcp_fraction_count_spin, 0, 5)

        controls_layout.addWidget(QLabel("Interval (days)"), 1, 4)
        self.tcp_interval_spin = QDoubleSpinBox(self)
        self.tcp_interval_spin.setRange(0.0, 365.0)
        self.tcp_interval_spin.setDecimals(4)
        self.tcp_interval_spin.setSingleStep(0.25)
        self.tcp_interval_spin.setValue(1.0)
        controls_layout.addWidget(self.tcp_interval_spin, 1, 5)

        self.tcp_build_button = QPushButton("Build TCP")
        self.tcp_build_button.clicked.connect(self.refresh_tcp_view)
        controls_layout.addWidget(self.tcp_build_button, 0, 6)

        self.tcp_export_button = QPushButton("Export CSV")
        self.tcp_export_button.clicked.connect(self.export_tcp_csv)
        controls_layout.addWidget(self.tcp_export_button, 1, 6)

        controls_layout.setColumnStretch(3, 1)
        layout.addWidget(self.tcp_controls_group)

        self.tcp_content_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.tcp_content_splitter.setChildrenCollapsible(False)
        self.tcp_content_splitter.setHandleWidth(4)

        self.tcp_figure = Figure(figsize=COMPACT_PLOT_FIGSIZE)
        self.tcp_canvas = FigureCanvasQTAgg(self.tcp_figure)
        self.tcp_canvas.setMinimumHeight(COMPACT_PLOT_MIN_HEIGHT)
        self.tcp_content_splitter.addWidget(self.tcp_canvas)

        self.tcp_table = self._create_table(TCP_HEADERS)
        self.tcp_table.setMinimumHeight(100)
        self.tcp_content_splitter.addWidget(self.tcp_table)

        self.tcp_text = QPlainTextEdit(self)
        self.tcp_text.setReadOnly(True)
        self.tcp_text.setMinimumHeight(80)
        self.tcp_content_splitter.addWidget(self.tcp_text)

        self.tcp_content_splitter.setStretchFactor(0, 3)
        self.tcp_content_splitter.setStretchFactor(1, 2)
        self.tcp_content_splitter.setStretchFactor(2, 1)
        self.tcp_content_splitter.setSizes([260, 200, 120])
        layout.addWidget(self.tcp_content_splitter, 1)
        return panel

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
    def _style_compact_plot_axis(axis: object) -> None:
        axis.xaxis.label.set_size(COMPACT_PLOT_LABEL_FONT)
        axis.yaxis.label.set_size(COMPACT_PLOT_LABEL_FONT)
        axis.tick_params(axis="both", labelsize=COMPACT_PLOT_TICK_FONT)

    @staticmethod
    def _style_compact_legend(legend: object) -> None:
        if legend is None:
            return
        for text in legend.get_texts():
            text.set_fontsize(COMPACT_PLOT_LEGEND_FONT)

    def _apply_window_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 12px;
                color: #1f2937;
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
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #cfd7e4;
                border-radius: 7px;
                padding: 4px 6px;
                min-height: 24px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #cfd7e4;
                selection-background-color: #dcecff;
                selection-color: #1f2937;
                outline: 0;
            }
            QTableWidget, QListWidget, QPlainTextEdit, QScrollArea {
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
            QSplitter::handle {
                background: #e3e8f0;
            }
            """
        )

    def _update_alpha_enabled(self, enabled: bool) -> None:
        self.alpha_spin.setEnabled(enabled)

    def add_files_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Excel files",
            str(Path.cwd()),
            "Excel files (*.xlsx)",
        )
        self.add_paths(files)

    def add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder", str(Path.cwd()))
        if not folder:
            return
        paths = sorted(Path(folder).glob("*.xlsx"))
        self.add_paths([str(path) for path in paths])

    def choose_summary_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save summary CSV",
            str(Path.cwd() / "family_summary.csv"),
            "CSV files (*.csv)",
        )
        if path:
            self.summary_csv_edit.setText(path)

    def export_analysis_csv(self) -> None:
        if not self.rbe_points and not self.sf_metric_rows and not self.let_alpha_points:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Build RBE, SF metric, or LET analysis first.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export radiobiology analysis",
            str(Path.cwd() / "radiobiology_analysis.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        written_paths: List[Path] = []
        base_path = Path(path)
        if self.rbe_points:
            rbe_path = related_csv_path(base_path, "rbe")
            write_csv_rows(rbe_path, RBE_HEADERS, build_rbe_table_rows(self.rbe_points))
            written_paths.append(rbe_path)
        if self.sf_metric_rows:
            sf_path = related_csv_path(base_path, "sf_metrics")
            write_csv_rows(sf_path, SF_METRIC_HEADERS, build_sf_metric_table_rows(self.sf_metric_rows))
            written_paths.append(sf_path)
        if self.let_alpha_points:
            let_path = related_csv_path(base_path, "alpha_let")
            write_csv_rows(let_path, LET_ALPHA_HEADERS, build_let_alpha_table_rows(self.let_alpha_points))
            written_paths.append(let_path)

        self.statusBar().showMessage(
            "Analysis CSV export complete: " + ", ".join(str(path) for path in written_paths)
        )

    def add_paths(self, paths: Sequence[str]) -> None:
        known_paths = {
            Path(self.file_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.file_list.count())
        }
        added = 0

        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                for child in sorted(path.glob("*.xlsx")):
                    added += self._add_single_path(child, known_paths)
                continue
            added += self._add_single_path(path, known_paths)

        if added:
            self.statusBar().showMessage(f"Added {added} file(s).")
        else:
            self.statusBar().showMessage("No new .xlsx files were added.")
        self.refresh_control_selector()
        self.scan_inventory()

    def _add_single_path(self, path: Path, known_paths: set[Path]) -> int:
        if path.suffix.lower() != ".xlsx":
            return 0
        resolved = path.expanduser().resolve()
        if resolved in known_paths:
            return 0

        item = QListWidgetItem(str(resolved))
        item.setData(Qt.ItemDataRole.UserRole, str(resolved))
        item.setToolTip(str(resolved))
        self.file_list.addItem(item)
        known_paths.add(resolved)
        return 1

    def remove_selected_files(self) -> None:
        selected = self.file_list.selectedItems()
        for item in selected:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        self.refresh_control_selector()
        self.scan_inventory()
        self.statusBar().showMessage(f"Removed {len(selected)} file(s).")

    def clear_files(self) -> None:
        self.file_list.clear()
        self.refresh_control_selector()
        self.clear_inventory()
        self.clear_results()
        self.statusBar().showMessage("File list cleared.")

    def loaded_paths(self) -> List[Path]:
        return [
            Path(self.file_list.item(index).data(Qt.ItemDataRole.UserRole)).resolve()
            for index in range(self.file_list.count())
        ]

    def control_paths(self) -> List[Path]:
        return [path for path in self.loaded_paths() if is_control_file(path)]

    def experiment_paths(self) -> List[Path]:
        return [path for path in self.loaded_paths() if not is_control_file(path)]

    def read_assignment_controls(self) -> Dict[Path, Optional[Path]]:
        assignments: Dict[Path, Optional[Path]] = {}
        if not hasattr(self, "assignment_table"):
            return assignments

        for row_index in range(self.assignment_table.rowCount()):
            experiment_item = self.assignment_table.item(row_index, 0)
            combo = self.assignment_table.cellWidget(row_index, 2)
            if experiment_item is None or combo is None:
                continue
            experiment_path = Path(str(experiment_item.data(Qt.ItemDataRole.UserRole))).resolve()
            control_value = combo.currentData()
            if control_value == UNASSIGNED_CONTROL:
                continue
            if control_value == USE_ALL_CONTROLS:
                assignments[experiment_path] = None
                continue
            if control_value is not None:
                assignments[experiment_path] = Path(str(control_value)).resolve()
        return assignments

    def _create_assignment_combo(
        self,
        controls: Sequence[Path],
        selected_value: object,
        allow_average: bool,
    ) -> QComboBox:
        combo = QComboBox(self)
        if not controls:
            combo.addItem("No control files loaded", UNASSIGNED_CONTROL)
            combo.setEnabled(False)
            return combo

        if len(controls) > 1:
            combo.addItem("Select control file...", UNASSIGNED_CONTROL)

        for control in controls:
            combo.addItem(control.name, str(control))

        if allow_average:
            combo.addItem("Use all controls (average)", USE_ALL_CONTROLS)

        for index in range(combo.count()):
            if combo.itemData(index) == selected_value:
                combo.setCurrentIndex(index)
                return combo

        if len(controls) == 1:
            combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(0)
        return combo

    def refresh_control_selector(self) -> None:
        previous_default = self.control_combo.currentData() if hasattr(self, "control_combo") else None
        previous_assignments = self.read_assignment_controls()
        controls = self.control_paths()
        experiments = self.experiment_paths()

        self.control_combo.blockSignals(True)
        self.control_combo.clear()

        if not controls:
            self.control_combo.addItem("No control files loaded", None)
            self.control_combo.setEnabled(False)
        else:
            if len(controls) > 1:
                self.control_combo.addItem("Choose default control...", UNASSIGNED_CONTROL)
            for control in controls:
                self.control_combo.addItem(control.name, str(control))
            if len(controls) > 1:
                self.control_combo.addItem("Use all controls (average)", USE_ALL_CONTROLS)
            self.control_combo.setEnabled(True)

            if len(controls) == 1:
                self.control_combo.setCurrentIndex(0)
            elif previous_default == USE_ALL_CONTROLS:
                self.control_combo.setCurrentIndex(self.control_combo.count() - 1)
            elif isinstance(previous_default, str):
                for index in range(self.control_combo.count()):
                    if self.control_combo.itemData(index) == previous_default:
                        self.control_combo.setCurrentIndex(index)
                        break

        self.control_combo.blockSignals(False)
        self.apply_default_control_button.setEnabled(bool(controls and experiments))

        self.assignment_table.setUpdatesEnabled(False)
        try:
            self.assignment_table.setRowCount(len(experiments))
            for row_index, experiment_path in enumerate(experiments):
                experiment_item = QTableWidgetItem(experiment_path.name)
                experiment_item.setData(Qt.ItemDataRole.UserRole, str(experiment_path))
                experiment_item.setToolTip(str(experiment_path))

                family_item = QTableWidgetItem(infer_radiation_family(experiment_path) or "-")

                selected_value: object = UNASSIGNED_CONTROL
                if experiment_path in previous_assignments:
                    previous_control = previous_assignments[experiment_path]
                    if previous_control is None:
                        selected_value = USE_ALL_CONTROLS
                    else:
                        selected_value = str(previous_control)
                else:
                    default_value = self.control_combo.currentData()
                    if default_value not in (None, UNASSIGNED_CONTROL):
                        selected_value = default_value

                combo = self._create_assignment_combo(
                    controls=controls,
                    selected_value=selected_value,
                    allow_average=len(controls) > 1,
                )
                combo.currentIndexChanged.connect(self.on_assignment_control_changed)

                self.assignment_table.setItem(row_index, 0, experiment_item)
                self.assignment_table.setItem(row_index, 1, family_item)
                self.assignment_table.setCellWidget(row_index, 2, combo)
        finally:
            self.assignment_table.setUpdatesEnabled(True)

    def on_assignment_control_changed(self) -> None:
        if self._suspend_assignment_inventory_refresh:
            return
        self.mark_inventory_stale("Control mapping updated. Click Inventory to refresh readiness summary.")

    def mark_inventory_stale(self, message: Optional[str] = None) -> None:
        self.inventory_stale = True
        if message:
            self.statusBar().showMessage(message)

    def apply_default_control_to_all(self) -> None:
        selected_value = self.control_combo.currentData()
        if selected_value in (None, UNASSIGNED_CONTROL):
            QMessageBox.warning(
                self,
                "Default control",
                "Choose a default control before applying it to all experiments.",
            )
            return

        self._suspend_assignment_inventory_refresh = True
        self.assignment_table.setUpdatesEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for row_index in range(self.assignment_table.rowCount()):
                combo = self.assignment_table.cellWidget(row_index, 2)
                if combo is None:
                    continue
                for combo_index in range(combo.count()):
                    if combo.itemData(combo_index) == selected_value:
                        combo.blockSignals(True)
                        try:
                            combo.setCurrentIndex(combo_index)
                        finally:
                            combo.blockSignals(False)
                        break
        finally:
            QApplication.restoreOverrideCursor()
            self.assignment_table.setUpdatesEnabled(True)
            self._suspend_assignment_inventory_refresh = False
        self.mark_inventory_stale(
            "Default control applied to all experiments. Click Inventory to refresh readiness summary."
        )

    def run_analysis(self) -> None:
        loaded_paths = self.loaded_paths()
        if not loaded_paths:
            QMessageBox.warning(self, "No files", "Add at least one .xlsx file.")
            return
        control_paths = self.control_paths()
        if not control_paths:
            QMessageBox.warning(
                self,
                "No control",
                "Add at least one control file. A file is treated as control if its name contains 'control'.",
            )
            return
        experiment_paths = self.experiment_paths()
        if not experiment_paths:
            QMessageBox.warning(
                self,
                "No experiments",
                "Add at least one experimental file in addition to the control files.",
            )
            return

        control_map = self.read_assignment_controls()
        unresolved = [
            experiment_path.name
            for experiment_path in experiment_paths
            if experiment_path not in control_map
        ]
        if unresolved:
            QMessageBox.warning(
                self,
                "Incomplete control mapping",
                "Assign a control to every experimental file before running the fit.\n\n"
                + "\n".join(unresolved),
            )
            return

        files = [str(path) for path in loaded_paths]
        api_control_map = {
            str(experiment_path): (
                None if control_path is None else str(control_path)
            )
            for experiment_path, control_path in control_map.items()
        }

        sf_text = self.sf_modes_edit.text().strip()
        sf_modes = parse_sf_modes([sf_text] if sf_text else None)
        summary_csv = self.summary_csv_edit.text().strip() or None
        alpha = self.alpha_spin.value() if self.fix_alpha_check.isChecked() else None
        repair_half_time_hours = self.repair_half_time_spin.value()
        if repair_half_time_hours <= 0.0:
            repair_half_time_hours = None
        repair_half_time_fast_hours = self.repair_half_time_fast_spin.value()
        if repair_half_time_fast_hours <= 0.0:
            repair_half_time_fast_hours = None
        repair_half_time_slow_hours = self.repair_half_time_slow_spin.value()
        if repair_half_time_slow_hours <= 0.0:
            repair_half_time_slow_hours = None

        seed_text = self.bootstrap_seed_edit.text().strip()
        bootstrap_seed = None
        if seed_text:
            try:
                bootstrap_seed = int(seed_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid seed", "Bootstrap seed must be an integer.")
                return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            _, results = analyze_files(
                files=files,
                sf_modes=sf_modes,
                alpha=alpha,
                repair_half_time_hours=repair_half_time_hours,
                repair_half_time_fast_hours=repair_half_time_fast_hours,
                repair_half_time_slow_hours=repair_half_time_slow_hours,
                repair_fast_fraction=float(self.repair_fast_fraction_spin.value()),
                min_sf=self.min_sf_spin.value(),
                fit_kind=self.fit_kind_combo.currentData(),
                validate_kind=self.validate_kind_combo.currentData(),
                family=self.family_combo.currentData(),
                by_family=self.by_family_check.isChecked(),
                aggregate_regimens=self.aggregate_check.isChecked(),
                dedupe_regimens=self.dedupe_check.isChecked(),
                response_mode=self.response_mode_combo.currentData(),
                requested_model_kind=self.model_kind_combo.currentData(),
                compare_models=self.compare_models_check.isChecked(),
                bootstrap=self.bootstrap_spin.value(),
                bootstrap_seed=bootstrap_seed,
                verbose=self.verbose_check.isChecked(),
                control_map=api_control_map,
                exclude_dead_animals=self.exclude_dead_check.isChecked(),
            )
            self.run_results = results
            self.populate_results()

            if summary_csv and results:
                output_path = Path(summary_csv).expanduser().resolve()
                Fitter.write_analysis_summaries_csv(
                    output_path,
                    [run.summary for run in results],
                    run_results=results,
                )
                self.statusBar().showMessage(
                    f"Analysis complete. Summary CSV written to {output_path}"
                )
            elif results:
                self.statusBar().showMessage(f"Analysis complete: {len(results)} run(s).")
            else:
                self.statusBar().showMessage("Analysis returned no runs.")
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.run_results = []
            self.clear_results()
            self.details_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Analysis failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def populate_results(self) -> None:
        self.summary_table.setRowCount(len(self.run_results))
        self.run_selector.blockSignals(True)
        self.run_selector.clear()

        for row_index, run in enumerate(self.run_results):
            summary = run.summary
            training_metrics = run.training_metrics
            cross_validation = run.cross_validation
            values = [
                summary.sf_mode,
                summary.response_mode,
                summary.model_kind,
                summary.family_label,
                summary.status,
                str(summary.total_count),
                str(summary.single_count),
                str(summary.fractionated_count),
                str(summary.train_count),
                str(summary.validation_count),
                self._format_optional_float(summary.alpha, digits=5),
                self._format_optional_float(summary.beta, digits=6),
                self._format_optional_float(summary.alpha_beta_ratio, digits=2),
                self._format_optional_float(run.mean_train_bed, digits=4),
                self._format_optional_float(run.mean_train_eqd2, digits=4),
                self._format_optional_float(run.mean_train_g_factor, digits=4),
                self._format_optional_float(
                    training_metrics.r_squared if training_metrics is not None else None,
                    digits=4,
                ),
                self._format_optional_float(
                    training_metrics.adjusted_r_squared if training_metrics is not None else None,
                    digits=4,
                ),
                self._format_optional_float(
                    training_metrics.bic if training_metrics is not None else None,
                    digits=4,
                ),
                self._format_optional_float(
                    cross_validation.cv_rmse if cross_validation is not None else None,
                    digits=6,
                ),
                summary.reason or "",
            ]
            self._fill_row(self.summary_table, row_index, values)
            self.run_selector.addItem(run.label)
            self.run_selector.setItemData(
                row_index,
                run.label,
                Qt.ItemDataRole.ToolTipRole,
            )

        self.run_selector.blockSignals(False)

        if self.run_results:
            self.run_selector.setCurrentIndex(0)
            self.summary_table.selectRow(0)
            self.display_run(0)
        else:
            self.clear_detail_tables()

    def clear_results(self) -> None:
        self.run_results = []
        self.summary_table.setRowCount(0)
        self.run_selector.blockSignals(True)
        self.run_selector.clear()
        self.run_selector.blockSignals(False)
        self.clear_detail_tables()

    def clear_detail_tables(self) -> None:
        self.train_table.setRowCount(0)
        self.validation_table.setRowCount(0)
        self.bootstrap_table.setRowCount(0)
        self.cross_validation_table.setRowCount(0)
        self.details_text.clear()
        self.rbe_points = []
        self.sf_metric_rows = []
        self.let_fit_result = None
        self.let_alpha_points = []
        self.tcp_curve_rows = []
        self.ntcp_curve_rows = []
        self.rbe_table.setRowCount(0)
        self.sf_metric_table.setRowCount(0)
        self.tcp_table.setRowCount(0)
        self.ntcp_table.setRowCount(0)
        self.bed_table.setRowCount(0)
        self.bed_alpha_beta_label.setText("—")
        self.analysis_text.clear()
        self.refresh_analysis_plot()
        self.tcp_text.clear()
        self.refresh_tcp_plot()
        self.refresh_ntcp_view()

    def refresh_analysis_views(self, *_args: object) -> None:
        self.rbe_points = []
        self.sf_metric_rows = []
        self.let_fit_result = None
        self.let_alpha_points = []
        rbe_mode = str(self.rbe_mode_combo.currentData() or "family")
        rbe_error: Optional[str] = None
        sf_error: Optional[str] = None
        context_label: Optional[str] = None
        let_error: Optional[str] = None
        let_context_label: Optional[str] = None

        if self.run_results:
            selected_index = self.run_selector.currentIndex()
            try:
                doses = parse_positive_float_csv(self.rbe_doses_edit.text())
                reference_family = str(self.rbe_reference_combo.currentData() or "y")
                if rbe_mode == "let":
                    self.let_fit_result, self.let_alpha_points, let_context_label = select_let_run_context(
                        self.run_results,
                        selected_index,
                    )
                    context_label = (
                        f"{let_context_label} | reference={reference_family}"
                        if let_context_label is not None
                        else f"reference={reference_family}"
                    )
                    if self.let_fit_result is None:
                        raise ValueError("No LET-dependent fit is available.")
                    self.rbe_points = list(
                        build_rbe_let_series(
                            self.let_fit_result,
                            FAMILY_LET_DEFAULTS,
                            reference_family=reference_family,
                            doses=doses,
                        )
                    )
                else:
                    reference_fit, comparison_results, context_label = select_rbe_run_context(
                        self.run_results,
                        selected_index,
                        reference_family,
                    )
                    self.rbe_points = list(
                        build_rbe_series(reference_fit, comparison_results, doses=doses)
                    )
            except Exception as exc:
                rbe_error = str(exc)

            try:
                selected_run = self.run_results[selected_index]
                sf_rows = compare_sf_metric_sensitivity(self.run_results)
                self.sf_metric_rows = [
                    row
                    for row in sf_rows
                    if row.response_mode == selected_run.summary.response_mode
                    and row.model_kind == selected_run.summary.model_kind
                ]
            except Exception as exc:
                sf_error = str(exc)

            if self.let_fit_result is None or not self.let_alpha_points:
                try:
                    self.let_fit_result, self.let_alpha_points, let_context_label = select_let_run_context(
                        self.run_results,
                        selected_index,
                    )
                except Exception as exc:
                    let_error = str(exc)

        self.populate_rbe_table(self.rbe_points)
        self.populate_sf_metric_table(self.sf_metric_rows)
        self.refresh_analysis_plot()
        self.analysis_text.setPlainText(
            self.build_analysis_summary_text(
                rbe_mode,
                context_label,
                rbe_error,
                sf_error,
                let_context_label,
                let_error,
            )
        )

    def populate_rbe_table(self, rows: Sequence[RBEPoint]) -> None:
        table_rows = build_rbe_table_rows(rows)
        self.rbe_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.rbe_table, row_index, values)

    def populate_sf_metric_table(self, rows: Sequence[SFMetricComparisonRow]) -> None:
        table_rows = build_sf_metric_table_rows(rows)
        self.sf_metric_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.sf_metric_table, row_index, values)

    def refresh_analysis_plot(self) -> None:
        self.analysis_figure.clear()
        if not self.rbe_points and self.let_fit_result is None:
            axis = self.analysis_figure.add_subplot(111)
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No RBE or LET points yet",
                ha="center",
                va="center",
                fontsize=11,
                color="#5f6b7a",
            )
            self.analysis_canvas.draw_idle()
            return

        if self.rbe_points and self.let_fit_result is not None:
            dose_axis, ratio_axis, let_axis = self.analysis_figure.subplots(1, 3)
        elif self.rbe_points:
            dose_axis, ratio_axis = self.analysis_figure.subplots(1, 2)
            let_axis = None
        else:
            let_axis = self.analysis_figure.add_subplot(111)
            dose_axis = None
            ratio_axis = None

        if dose_axis is not None and ratio_axis is not None:
            families = sorted({row.test_family for row in self.rbe_points})
            for family in families:
                family_rows = sorted(
                    (row for row in self.rbe_points if row.test_family == family),
                    key=lambda row: row.test_dose,
                )
                dose_axis.plot(
                    [row.test_dose for row in family_rows],
                    [row.rbe for row in family_rows],
                    marker="o",
                    linewidth=1.8,
                    label=family,
                )

                ratio_rows = [
                    row for row in family_rows if row.test_alpha_beta_ratio is not None
                ]
                if ratio_rows:
                    ratio_axis.scatter(
                        [float(row.test_alpha_beta_ratio) for row in ratio_rows],
                        [row.rbe for row in ratio_rows],
                        label=family,
                        s=34,
                    )
                    for row in ratio_rows:
                        ratio_axis.annotate(
                            f"{family}@{row.test_dose:g}",
                            (float(row.test_alpha_beta_ratio), row.rbe),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=8,
                            alpha=0.8,
                        )

            dose_axis.set_title("RBE vs dose")
            dose_axis.set_xlabel("Test dose (Gy)")
            dose_axis.set_ylabel("RBE")
            dose_axis.grid(True, alpha=0.25)
            dose_axis.legend(loc="best")

            ratio_axis.set_title("RBE vs alpha/beta")
            ratio_axis.set_xlabel("Test alpha/beta (Gy)")
            ratio_axis.set_ylabel("RBE")
            ratio_axis.grid(True, alpha=0.25)
            if any(row.test_alpha_beta_ratio is not None for row in self.rbe_points):
                ratio_axis.legend(loc="best")
            else:
                ratio_axis.text(
                    0.5,
                    0.5,
                    "No alpha/beta ratios available",
                    ha="center",
                    va="center",
                    transform=ratio_axis.transAxes,
                    fontsize=10,
                    color="#5f6b7a",
                )

        if let_axis is not None and self.let_fit_result is not None:
            let_values = sorted(FAMILY_LET_DEFAULTS.values())
            let_min = float(min(let_values)) if let_values else 0.0
            let_max = float(max(let_values)) if let_values else 1.0
            padding = max((let_max - let_min) * 0.1, 1.0)
            let_grid = np.linspace(max(let_min - padding, 0.0), let_max + padding, 200)
            alpha_grid = [
                self.let_fit_result.effective_alpha(let_kev_um)
                for let_kev_um in let_grid
            ]
            let_axis.plot(
                let_grid,
                alpha_grid,
                color="#0f766e",
                linewidth=2.0,
                label="alpha(LET) fit",
            )

            if self.let_alpha_points:
                let_axis.scatter(
                    [point[1] for point in self.let_alpha_points],
                    [point[2] for point in self.let_alpha_points],
                    color="#dc2626",
                    s=42,
                    label="Per-family fits",
                    zorder=3,
                )
                for family, let_kev_um, alpha, _model_kind in self.let_alpha_points:
                    let_axis.annotate(
                        family,
                        (let_kev_um, alpha),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                        alpha=0.85,
                    )
            else:
                let_axis.text(
                    0.5,
                    0.12,
                    "No independent per-family alpha fits in this SF/response context",
                    ha="center",
                    va="center",
                    transform=let_axis.transAxes,
                    fontsize=9,
                    color="#5f6b7a",
                )

            let_axis.set_title("alpha(LET)")
            let_axis.set_xlabel("LET (keV/um)")
            let_axis.set_ylabel("Alpha (Gy^-1)")
            let_axis.grid(True, alpha=0.25)
            let_axis.legend(loc="best")

        self.analysis_figure.tight_layout(pad=1.1)
        self.analysis_canvas.draw_idle()

    def build_analysis_summary_text(
        self,
        rbe_mode: str,
        context_label: Optional[str],
        rbe_error: Optional[str],
        sf_error: Optional[str],
        let_context_label: Optional[str],
        let_error: Optional[str],
    ) -> str:
        lines: List[str] = []
        lines.append(f"RBE mode: {'LET model' if rbe_mode == 'let' else 'Per-family fits'}")
        lines.append("")
        if context_label is not None:
            lines.append(f"RBE context: {context_label}")
        if self.rbe_points:
            lines.append(f"RBE points: {len(self.rbe_points)}")
            family_names = sorted({row.test_family for row in self.rbe_points})
            lines.append(f"Compared families: {', '.join(family_names)}")
        elif rbe_error:
            lines.append(f"RBE: {rbe_error}")
        else:
            lines.append("RBE: no comparable fitted families yet.")

        lines.append("")
        if self.sf_metric_rows:
            lines.append(f"SF metric rows: {len(self.sf_metric_rows)}")
            lines.append(
                "Rows are filtered to the selected response/model context."
            )
        elif sf_error:
            lines.append(f"SF metric comparison: {sf_error}")
        else:
            lines.append("SF metric comparison: not enough fitted runs yet.")

        lines.append("")
        if let_context_label is not None:
            lines.append(f"LET context: {let_context_label}")
        if self.let_fit_result is not None:
            lines.append(
                "alpha(LET): "
                f"alpha_0={self._format_optional_float(self.let_fit_result.alpha_0, digits=6) or '-'}, "
                f"lambda_alpha={self._format_optional_float(self.let_fit_result.lambda_alpha, digits=6) or '-'}, "
                f"family points={len(self.let_alpha_points)}"
            )
        elif let_error:
            lines.append(f"alpha(LET): {let_error}")
        else:
            lines.append("alpha(LET): no LET-dependent fit is available in this context.")

        lines.append("")
        lines.append(
            "RBE, SF-metric drift, and alpha(LET) are available on this tab. "
            "Predictor sensitivity and scenario comparison live in the growth predictor window."
        )
        return "\n".join(lines)

    def current_run(self) -> Optional[AnalysisRunResult]:
        index = self.run_selector.currentIndex()
        if 0 <= index < len(self.run_results):
            return self.run_results[index]
        return None

    def current_tcp_cell_density(self) -> float:
        return parse_positive_scalar(self.tcp_cell_density_edit.text(), label="Cell density")

    def autofill_tcp_initial_volume(self, run: AnalysisRunResult) -> None:
        for experiment in run.train:
            if experiment.initial_volume_cm3 is not None and experiment.initial_volume_cm3 > 0.0:
                self.tcp_initial_volume_spin.setValue(float(experiment.initial_volume_cm3))
                return

    def refresh_tcp_view(self) -> None:
        self.tcp_curve_rows = []
        self.tcp_table.setRowCount(0)
        self.tcp_text.clear()

        run = self.current_run()
        if run is None:
            self.refresh_tcp_plot()
            return
        if run.fit_result is None or run.summary.status != "ok":
            self.tcp_text.setPlainText("Select a successful fitted run to build TCP curves.")
            self.refresh_tcp_plot()
            return

        try:
            cell_density = self.current_tcp_cell_density()
            dose_grid = parse_positive_float_csv(
                self.tcp_dose_grid_edit.text(),
                default=(2.0, 10.0, 20.0, 30.0, 40.0),
            )
            self.tcp_curve_rows = list(
                build_tcp_curve(
                    run.fit_result,
                    dose_grid,
                    n_fractions=int(self.tcp_fraction_count_spin.value()),
                    initial_volume_cm3=float(self.tcp_initial_volume_spin.value()),
                    cell_density=cell_density,
                    schedule_interval_days=float(self.tcp_interval_spin.value()),
                    family=run.summary.family,
                )
            )
            self.populate_tcp_table(run)
            self.tcp_text.setPlainText(
                self.build_tcp_summary_text(
                    run,
                    dose_grid=dose_grid,
                    cell_density=cell_density,
                )
            )
        except Exception as exc:
            self.tcp_curve_rows = []
            self.tcp_table.setRowCount(0)
            self.tcp_text.setPlainText(f"TCP analysis: {exc}")

        self.refresh_tcp_plot()

    def populate_tcp_table(self, run: AnalysisRunResult) -> None:
        if run.fit_result is None or not self.tcp_curve_rows:
            self.tcp_table.setRowCount(0)
            return

        table_rows = build_tcp_table_rows(
            run.fit_result,
            self.tcp_curve_rows,
            n_fractions=int(self.tcp_fraction_count_spin.value()),
            schedule_interval_days=float(self.tcp_interval_spin.value()),
        )
        self.tcp_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.tcp_table, row_index, values)

    def refresh_tcp_plot(self) -> None:
        self.tcp_figure.clear()
        if not self.tcp_curve_rows:
            axis = self.tcp_figure.add_subplot(111)
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No TCP curve yet",
                ha="center",
                va="center",
                fontsize=11,
                color="#5f6b7a",
            )
            self.tcp_canvas.draw_idle()
            return

        axis = self.tcp_figure.add_subplot(111)
        ordered_rows = sorted(self.tcp_curve_rows, key=lambda row: row.dose_total)
        doses = [row.dose_total for row in ordered_rows]
        tcps = [row.tcp for row in ordered_rows]
        sfs = [row.sf for row in ordered_rows]

        axis.plot(
            doses,
            tcps,
            marker="o",
            markersize=COMPACT_PLOT_MARKER_SIZE,
            linewidth=COMPACT_PLOT_LINE_WIDTH,
            color="#2563eb",
            label="TCP",
        )
        axis.set_xlabel("Total dose (Gy)", fontsize=COMPACT_PLOT_LABEL_FONT)
        axis.set_ylabel("TCP", fontsize=COMPACT_PLOT_LABEL_FONT)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        self._style_compact_plot_axis(axis)

        sf_axis = axis.twinx()
        sf_axis.plot(
            doses,
            sfs,
            marker="s",
            markersize=COMPACT_PLOT_MARKER_SIZE,
            linewidth=COMPACT_PLOT_SECONDARY_LINE_WIDTH,
            linestyle="--",
            color="#f97316",
            label="Predicted SF",
        )
        sf_axis.set_ylabel("Predicted SF", fontsize=COMPACT_PLOT_LABEL_FONT)
        sf_axis.set_ylim(bottom=0.0)
        self._style_compact_plot_axis(sf_axis)

        lines = axis.get_lines() + sf_axis.get_lines()
        legend = axis.legend(
            lines,
            [line.get_label() for line in lines],
            loc="best",
            fontsize=COMPACT_PLOT_LEGEND_FONT,
        )
        self._style_compact_legend(legend)
        self.tcp_figure.tight_layout(pad=0.8)
        self.tcp_canvas.draw_idle()

    def build_tcp_summary_text(
        self,
        run: AnalysisRunResult,
        *,
        dose_grid: Sequence[float],
        cell_density: float,
    ) -> str:
        if not self.tcp_curve_rows:
            return "No TCP curve computed yet."

        ordered_rows = sorted(self.tcp_curve_rows, key=lambda row: row.dose_total)
        best_row = max(ordered_rows, key=lambda row: row.tcp)
        lines = [
            f"Run: {run.label}",
            f"Family: {run.summary.family_label}",
            f"Model: {run.summary.model_kind}",
            f"Initial volume = {self.tcp_initial_volume_spin.value():.4f} cm^3",
            f"Cell density = {cell_density:.4g} cells/cm^3",
            (
                f"Synthetic schedule: n={self.tcp_fraction_count_spin.value()}, "
                f"interval={self.tcp_interval_spin.value():.4f} d"
            ),
            "Dose grid: " + ", ".join(f"{dose:g}" for dose in dose_grid),
            "",
            (
                f"Max TCP = {best_row.tcp:.6f} "
                f"at total dose {best_row.dose_total:.3f} Gy"
            ),
            (
                f"Predicted SF at max TCP = {best_row.sf:.6f} "
                f"(N0={best_row.n_cells:.4g})"
            ),
        ]
        return "\n".join(lines)

    def export_tcp_csv(self) -> None:
        run = self.current_run()
        if run is None or run.fit_result is None or not self.tcp_curve_rows:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Build a TCP curve first.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export TCP curve",
            str(Path.cwd() / "tcp_curve.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        table_rows = build_tcp_table_rows(
            run.fit_result,
            self.tcp_curve_rows,
            n_fractions=int(self.tcp_fraction_count_spin.value()),
            schedule_interval_days=float(self.tcp_interval_spin.value()),
        )
        output_path = write_csv_rows(path, TCP_HEADERS, table_rows)
        self.statusBar().showMessage(f"TCP CSV export complete: {output_path}")

    def load_ntcp_skin_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load skin-reaction files",
            str(Path.cwd()),
            "Excel files (*.xlsx *.xls)",
        )
        if not paths:
            return

        threshold_grade = int(self.ntcp_threshold_spin.value())
        input_scale = str(self.ntcp_scale_combo.currentData() or "our")
        existing_by_path: Dict[str, NTCPFitGroup] = {
            str(group.path): group
            for group in self.ntcp_fit_groups
            if group.path is not None
        }
        loaded_without_path = [
            group
            for group in self.ntcp_fit_groups
            if group.path is None
        ]
        errors: List[str] = []

        for raw_path in paths:
            path = Path(raw_path)
            try:
                existing_by_path[str(path)] = summarize_skin_reaction_file(
                    path,
                    threshold_grade=threshold_grade,
                    input_scale=input_scale,
                )
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        self.ntcp_fit_groups = list(existing_by_path.values()) + loaded_without_path
        self.ntcp_fit_groups.sort(
            key=lambda group: (math.inf if group.dose_total is None else float(group.dose_total), group.label.lower())
        )
        self.ntcp_fit_result = None
        self.populate_ntcp_source_table()
        self.refresh_ntcp_view()

        loaded_count = len(paths) - len(errors)
        if loaded_count > 0:
            self.statusBar().showMessage(f"Loaded {loaded_count} skin-reaction files for NTCP fitting.")
        if errors:
            QMessageBox.warning(
                self,
                "Some files were skipped",
                "\n".join(errors),
            )

    def clear_ntcp_skin_files(self) -> None:
        self.ntcp_fit_groups = []
        self.ntcp_fit_result = None
        self.ntcp_source_table.setRowCount(0)
        self.refresh_ntcp_view()

    def populate_ntcp_source_table(self) -> None:
        self.ntcp_source_table.setRowCount(len(self.ntcp_fit_groups))
        for row_index, group in enumerate(self.ntcp_fit_groups):
            values = build_ntcp_source_table_rows([group])[0]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column_index == 0:
                    if group.path is not None:
                        item.setData(Qt.ItemDataRole.UserRole, str(group.path))
                        item.setToolTip(str(group.path))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                elif column_index == 1:
                    item.setToolTip("Editable total dose for NTCP fitting.")
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.ntcp_source_table.setItem(row_index, column_index, item)

    def read_ntcp_groups_from_table(self) -> List[NTCPFitGroup]:
        resolved_groups: List[NTCPFitGroup] = []
        for row_index, group in enumerate(self.ntcp_fit_groups):
            dose_item = self.ntcp_source_table.item(row_index, 1)
            dose_text = "" if dose_item is None else dose_item.text().strip()
            dose_total: Optional[float] = None
            if dose_text:
                dose_total = parse_positive_scalar(dose_text, label=f"Dose for {group.label}")
            resolved_groups.append(replace(group, dose_total=dose_total))
        return resolved_groups

    def fit_ntcp_from_skin_files(self) -> None:
        if not self.ntcp_fit_groups:
            QMessageBox.information(
                self,
                "No skin files",
                "Load at least two skin-reaction files before fitting TD50/m.",
            )
            return

        try:
            resolved_groups = self.read_ntcp_groups_from_table()
            fit_result = fit_ntcp_lkb_from_groups(
                resolved_groups,
                initial_td50=float(self.ntcp_td50_spin.value()),
                initial_m=float(self.ntcp_m_spin.value()),
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "NTCP fit failed",
                str(exc),
            )
            return

        self.ntcp_fit_groups = list(resolved_groups)
        self.ntcp_fit_result = fit_result
        self.ntcp_td50_spin.setValue(fit_result.td50)
        self.ntcp_m_spin.setValue(fit_result.m)
        self.populate_ntcp_source_table()
        self.refresh_ntcp_view()
        self.statusBar().showMessage(
            f"Fitted NTCP from {len(fit_result.groups)} groups: TD50={fit_result.td50:.3f} Gy, m={fit_result.m:.4f}"
        )

    def refresh_ntcp_view(self) -> None:
        self.ntcp_curve_rows = []
        self.ntcp_table.setRowCount(0)
        self.ntcp_text.clear()

        try:
            dose_grid = parse_positive_float_csv(
                self.ntcp_dose_grid_edit.text(),
                default=(2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
            )
            self.ntcp_curve_rows = list(
                build_ntcp_curve(
                    dose_grid,
                    td50=float(self.ntcp_td50_spin.value()),
                    m=float(self.ntcp_m_spin.value()),
                )
            )
            self.populate_ntcp_table()
            self.ntcp_text.setPlainText(self.build_ntcp_summary_text(dose_grid))
        except Exception as exc:
            self.ntcp_curve_rows = []
            self.ntcp_table.setRowCount(0)
            self.ntcp_text.setPlainText(f"NTCP analysis: {exc}")

        self.refresh_ntcp_plot()

    def populate_ntcp_table(self) -> None:
        table_rows = build_ntcp_table_rows(self.ntcp_curve_rows)
        self.ntcp_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.ntcp_table, row_index, values)

    def refresh_ntcp_plot(self) -> None:
        self.ntcp_figure.clear()
        if not self.ntcp_curve_rows:
            axis = self.ntcp_figure.add_subplot(111)
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No NTCP curve yet",
                ha="center",
                va="center",
                fontsize=11,
                color="#5f6b7a",
            )
            self.ntcp_canvas.draw_idle()
            return

        axis = self.ntcp_figure.add_subplot(111)
        ordered_rows = sorted(self.ntcp_curve_rows, key=lambda row: row.dose_total)
        doses = [row.dose_total for row in ordered_rows]
        ntcps = [row.ntcp for row in ordered_rows]
        axis.plot(
            doses,
            ntcps,
            marker="o",
            markersize=COMPACT_PLOT_MARKER_SIZE,
            linewidth=COMPACT_PLOT_LINE_WIDTH,
            color="#2563eb",
            label="NTCP",
        )

        observed_groups = [
            group
            for group in self.ntcp_fit_groups
            if group.dose_total is not None and group.n_subjects > 0
        ]
        if observed_groups:
            observed_groups = sorted(observed_groups, key=lambda group: float(group.dose_total))
            axis.plot(
                [float(group.dose_total) for group in observed_groups],
                [group.complication_rate for group in observed_groups],
                marker="s",
                markersize=COMPACT_PLOT_MARKER_SIZE,
                linewidth=COMPACT_PLOT_SECONDARY_LINE_WIDTH,
                linestyle="--",
                color="#f97316",
                label="Observed complication rate",
            )

        axis.set_xlabel("Total dose (Gy)", fontsize=COMPACT_PLOT_LABEL_FONT)
        axis.set_ylabel("NTCP", fontsize=COMPACT_PLOT_LABEL_FONT)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        self._style_compact_plot_axis(axis)
        legend = axis.legend(loc="upper right", fontsize=COMPACT_PLOT_LEGEND_FONT)
        self._style_compact_legend(legend)
        self.ntcp_figure.tight_layout(pad=0.8)
        self.ntcp_canvas.draw_idle()

    def build_ntcp_summary_text(self, dose_grid: Sequence[float]) -> str:
        if not self.ntcp_curve_rows:
            return "No NTCP curve computed yet."

        ordered_rows = sorted(self.ntcp_curve_rows, key=lambda row: row.dose_total)
        nearest_half = min(ordered_rows, key=lambda row: abs(row.ntcp - 0.5))
        lines = [
            f"Current TD50 = {self.ntcp_td50_spin.value():.3f} Gy",
            f"Current m = {self.ntcp_m_spin.value():.4f}",
            "Dose grid: " + ", ".join(f"{dose:g}" for dose in dose_grid),
            "",
            f"Closest point to NTCP=0.5: dose={nearest_half.dose_total:.3f} Gy, NTCP={nearest_half.ntcp:.6f}",
        ]
        if self.ntcp_fit_groups:
            usable_groups = [
                group
                for group in self.ntcp_fit_groups
                if group.dose_total is not None and group.n_subjects > 0
            ]
            threshold_grade = self.ntcp_fit_groups[0].threshold_grade
            lines.extend(
                [
                    "",
                    f"Loaded skin/RTOG groups: {len(self.ntcp_fit_groups)}",
                    f"Usable dose groups: {len(usable_groups)}",
                    f"Binary endpoint: peak RTOG >= {threshold_grade}",
                ]
            )
            missing_dose_count = len(self.ntcp_fit_groups) - len(usable_groups)
            if missing_dose_count > 0:
                lines.append(f"Excluded from fit due to missing dose: {missing_dose_count}")
        else:
            lines.extend(
                [
                    "",
                    "No skin/RTOG files loaded.",
                    "You can still use this tab as a manual LKB curve builder.",
                ]
            )

        if self.ntcp_fit_result is not None:
            lines.extend(
                [
                    "",
                    "Last automatic fit:",
                    f"TD50 = {self.ntcp_fit_result.td50:.3f} Gy",
                    f"m = {self.ntcp_fit_result.m:.4f}",
                    f"Groups = {len(self.ntcp_fit_result.groups)}",
                    f"Subjects = {self.ntcp_fit_result.subject_count}",
                    f"Negative log-likelihood = {self.ntcp_fit_result.negative_log_likelihood:.6f}",
                ]
            )
        elif self.ntcp_fit_groups:
            lines.extend(
                [
                    "",
                    "Skin/RTOG groups are loaded.",
                    "Click 'Fit TD50/m' to estimate LKB parameters from grouped complications.",
                ]
            )
        return "\n".join(lines)

    def export_ntcp_csv(self) -> None:
        if not self.ntcp_curve_rows:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Build an NTCP curve first.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export NTCP curve",
            str(Path.cwd() / "ntcp_curve.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return

        output_path = write_csv_rows(path, NTCP_HEADERS, build_ntcp_table_rows(self.ntcp_curve_rows))
        extra_paths: List[Path] = []
        if self.ntcp_fit_groups:
            try:
                resolved_groups = self.read_ntcp_groups_from_table()
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Invalid NTCP source table",
                    str(exc),
                )
                return
            group_rows = build_ntcp_source_table_rows(resolved_groups)
            groups_path = write_csv_rows(
                related_csv_path(path, "skin_groups"),
                NTCP_SOURCE_HEADERS,
                group_rows,
            )
            extra_paths.append(groups_path)
        if extra_paths:
            exported = ", ".join(str(item) for item in [output_path, *extra_paths])
            self.statusBar().showMessage(f"NTCP CSV export complete: {exported}")
        else:
            self.statusBar().showMessage(f"NTCP CSV export complete: {output_path}")

    def refresh_bed_eqd2_view(self) -> None:
        self.bed_table.setRowCount(0)

        current_index = self.run_selector.currentIndex()
        if current_index < 0 or current_index >= len(self.run_results):
            self.bed_alpha_beta_label.setText("—")
            return

        run = self.run_results[current_index]
        if run.fit_result is None:
            self.bed_alpha_beta_label.setText("—")
            return

        ab_ratio = run.summary.alpha_beta_ratio
        if ab_ratio is None or not math.isfinite(float(ab_ratio)) or float(ab_ratio) <= 0.0:
            self.bed_alpha_beta_label.setText("—")
            return

        self.bed_alpha_beta_label.setText(f"{float(ab_ratio):.2f}")

        try:
            _default_doses = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
            _default_fracs = (1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
            dose_grid = parse_positive_float_csv(self.bed_dose_grid_edit.text(), default=_default_doses)
            fractions = [int(f) for f in parse_positive_float_csv(
                self.bed_fractions_edit.text(), default=_default_fracs
            )]
            reference_ab = self.bed_reference_ab_spin.value()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

        df = export_bed_eqd2_table(
            {run.label: run.fit_result},
            dose_grid=np.array(dose_grid),
            fractions=fractions,
            reference_ab=reference_ab,
        )
        if df.empty:
            return

        self.bed_table.setRowCount(len(df))
        for row_index, row_data in df.iterrows():
            self._fill_row(self.bed_table, row_index, [
                f"{row_data['total_dose_gy']:.1f}",
                str(int(row_data['n_fractions'])),
                f"{row_data['dose_per_fraction_gy']:.2f}",
                f"{row_data['bed']:.2f}",
                f"{row_data['eqd2']:.2f}",
            ])

    def export_bed_eqd2_csv(self) -> None:
        if self.bed_table.rowCount() == 0:
            QMessageBox.information(self, "Nothing to export", "Build the BED/EQD2 table first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export BED/EQD2 table", str(Path.cwd() / "bed_eqd2.csv"), "CSV files (*.csv)"
        )
        if not path:
            return

        current_index = self.run_selector.currentIndex()
        if current_index < 0 or current_index >= len(self.run_results):
            return
        run = self.run_results[current_index]
        if run.fit_result is None:
            return

        try:
            _default_doses = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
            _default_fracs = (1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
            dose_grid = parse_positive_float_csv(self.bed_dose_grid_edit.text(), default=_default_doses)
            fractions = [int(f) for f in parse_positive_float_csv(
                self.bed_fractions_edit.text(), default=_default_fracs
            )]
            reference_ab = self.bed_reference_ab_spin.value()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

        export_bed_eqd2_table(
            {run.label: run.fit_result},
            dose_grid=np.array(dose_grid),
            fractions=fractions,
            reference_ab=reference_ab,
            output_csv=Path(path),
        )
        self.statusBar().showMessage(f"BED/EQD2 table exported to {path}")

    def clear_inventory(self) -> None:
        self.inventory_report = None
        self.inventory_stale = False
        self.inventory_table.setRowCount(0)
        self.inventory_text.clear()

    def scan_inventory(self) -> None:
        loaded_paths = self.loaded_paths()
        if not loaded_paths:
            self.clear_inventory()
            return

        control_map = self.read_assignment_controls()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.inventory_report = Fitter.inspect_files(loaded_paths, control_map=control_map)
        except Exception as exc:  # pragma: no cover - GUI exception path
            self.clear_inventory()
            self.inventory_text.setPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Inventory scan failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.populate_inventory_table(self.inventory_report)
        self.inventory_text.setPlainText(self.build_inventory_summary_text(self.inventory_report))
        self.inventory_stale = False
        self.statusBar().showMessage(
            f"Inventory scanned: {self.inventory_report.experiment_count} experiment file(s), "
            f"{self.inventory_report.control_count} control file(s)."
        )

    def populate_inventory_table(self, report: InventoryReport) -> None:
        self.inventory_table.setRowCount(len(report.rows))
        for row_index, row in enumerate(report.rows):
            values = [
                row.path.name,
                row.role,
                row.family or "-",
                row.kind,
                row.fractions_label,
                row.schedule_label,
                row.control_label,
                "yes" if row.fit_ready else "no",
                row.notes_label,
            ]
            self._fill_row(self.inventory_table, row_index, values)

    @staticmethod
    def build_inventory_summary_text(report: InventoryReport) -> str:
        lines = [
            f"Controls loaded: {report.control_count}",
            f"Experimental files: {report.experiment_count}",
        ]
        if not report.family_summaries:
            lines.append("No analyzable families detected yet.")
            return "\n".join(lines)

        lines.append("")
        lines.append("Family summary:")
        for summary in report.family_summaries:
            recommended_text = (
                f" | recommended: {', '.join(summary.recommended_models)}"
                if summary.recommended_models
                else ""
            )
            possible_text = (
                f" | possible: {', '.join(summary.possible_models)}"
                if summary.possible_models
                else ""
            )
            note_text = f" | notes: {'; '.join(summary.notes)}" if summary.notes else ""
            lines.append(
                f"- {summary.family}: parsed={summary.parsed_count}, "
                f"analyzable={summary.analyzable_count}, "
                f"distinct_regimens={summary.distinct_regimen_count}, "
                f"single={summary.single_count}, "
                f"fractionated={summary.fractionated_count}, "
                f"fit_ready={'yes' if summary.fit_ready else 'no'}"
                f"{recommended_text}{possible_text}{note_text}"
            )
            lines.append("  model guidance:")
            for item in summary.model_suitability:
                lines.append(f"  - {item.model_kind}: {item.status} | {item.reason}")
        return "\n".join(lines)

    def _sync_run_selector_with_table(self, current_row: int, *_args: int) -> None:
        if 0 <= current_row < len(self.run_results):
            self.run_selector.blockSignals(True)
            self.run_selector.setCurrentIndex(current_row)
            self.run_selector.blockSignals(False)
            self.display_run(current_row)

    def display_run(self, index: int) -> None:
        if index < 0 or index >= len(self.run_results):
            self.clear_detail_tables()
            return

        run = self.run_results[index]
        self.autofill_tcp_initial_volume(run)
        self.populate_train_table(run)
        self.populate_validation_table(run)
        self.populate_bootstrap_table(run)
        self.populate_cross_validation_table(run)
        self.details_text.setPlainText(self.build_run_summary_text(run))
        self.refresh_analysis_views()
        self.refresh_tcp_view()
        self.refresh_ntcp_view()
        self.refresh_bed_eqd2_view()

    def populate_train_table(self, run: AnalysisRunResult) -> None:
        experiments = run.train
        self.train_table.clearSpans()
        self.train_table.setRowCount(len(experiments))
        for row_index, experiment in enumerate(experiments):
            bed = eqd2 = g_factor = predicted_tcp = None
            if run.fit_result is not None:
                bed = run.fit_result.compute_bed(experiment)
                eqd2 = run.fit_result.compute_eqd2(experiment)
                g_factor = run.fit_result.compute_g_factor(experiment)
                try:
                    predicted_tcp = compute_tcp(run.fit_result, experiment).tcp
                except ValueError:
                    predicted_tcp = None
            values = [
                experiment.path.name,
                experiment.family or "-",
                experiment.control_label,
                experiment.regimen_kind,
                format_fractions(experiment.fractions),
                experiment.schedule_label,
                f"{experiment.dose_sum:.3f}",
                f"{experiment.dose2_sum:.3f}",
                f"{experiment.sf:.6f}",
                self._format_optional_float(bed, digits=4),
                self._format_optional_float(eqd2, digits=4),
                self._format_optional_float(g_factor, digits=4),
                self._format_optional_float(predicted_tcp, digits=6),
                str(experiment.repeat_count),
                f"{experiment.sf_std:.6f}",
            ]
            self._fill_row(self.train_table, row_index, values)

    def _validation_placeholder_message(self, run: AnalysisRunResult) -> str:
        if run.summary.status != "ok":
            return f"Validation not available: {run.summary.reason or 'run did not complete.'}"
        if run.validation_kind == "none":
            return "Validation not computed: Validate kind = none."
        if run.summary.response_mode == "curve":
            return (
                "Validation table is available only for scalar response; "
                "see Summary text for curve metrics."
            )
        if not run.validation:
            return "Validation not computed: no holdout experiments matched Validate kind."
        return "Validation results are not available for this run."

    def _cross_validation_placeholder_message(self, run: AnalysisRunResult) -> str:
        if run.summary.status != "ok":
            return (
                f"Cross-validation not available: "
                f"{run.summary.reason or 'run did not complete.'}"
            )
        if run.summary.response_mode != "scalar":
            return "Cross-validation is available only for scalar response."
        if len(run.train) < 3:
            return (
                "Cross-validation requires at least 3 training experiments "
                f"(found {len(run.train)})."
            )
        return "Cross-validation results are not available for this run."

    def _bootstrap_placeholder_message(self, run: AnalysisRunResult) -> str:
        if run.summary.status != "ok":
            return f"Bootstrap not available: {run.summary.reason or 'run did not complete.'}"
        return "Bootstrap not computed: Bootstrap = 0."

    @staticmethod
    def _show_table_placeholder(table: QTableWidget, message: str) -> None:
        table.clearSpans()
        column_count = table.columnCount()
        table.setRowCount(1)

        item = QTableWidgetItem(message)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsSelectable)
        item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        table.setItem(0, 0, item)

        for column_index in range(1, column_count):
            filler = QTableWidgetItem("")
            filler.setFlags(
                filler.flags() & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsSelectable
            )
            table.setItem(0, column_index, filler)

        if column_count > 1:
            table.setSpan(0, 0, 1, column_count)

    def populate_validation_table(self, run: AnalysisRunResult) -> None:
        self.validation_table.clearSpans()
        if run.summary.status != "ok" or run.validation_kind == "none":
            self._show_table_placeholder(
                self.validation_table,
                self._validation_placeholder_message(run),
            )
            return

        rows = []
        if run.validation_summary is not None:
            rows = list(run.validation_summary.rows)

        if run.validation_summary is not None and run.validation_summary.response_mode == "curve":
            self._show_table_placeholder(
                self.validation_table,
                self._validation_placeholder_message(run),
            )
            return

        if not run.validation:
            self._show_table_placeholder(
                self.validation_table,
                self._validation_placeholder_message(run),
            )
            return

        self.validation_table.setRowCount(len(run.validation))
        for row_index, experiment in enumerate(run.validation):
            matched_row = None
            for prediction in rows:
                if prediction.experiment.path == experiment.path:
                    matched_row = prediction
                    break

            values = [
                experiment.path.name,
                experiment.family or "-",
                experiment.control_label,
                experiment.regimen_kind,
                format_fractions(experiment.fractions),
                experiment.schedule_label,
                self._format_optional_float(
                    run.fit_result.compute_bed(experiment) if run.fit_result is not None else None,
                    digits=4,
                ),
                self._format_optional_float(
                    run.fit_result.compute_eqd2(experiment) if run.fit_result is not None else None,
                    digits=4,
                ),
                self._format_optional_float(
                    run.fit_result.compute_g_factor(experiment) if run.fit_result is not None else None,
                    digits=4,
                ),
                self._format_optional_float(
                    compute_tcp(run.fit_result, experiment).tcp if run.fit_result is not None and experiment.initial_volume_cm3 is not None else None,
                    digits=6,
                ),
                f"{experiment.sf:.6f}",
                self._format_optional_float(
                    matched_row.predicted_sf if matched_row is not None else None,
                    digits=6,
                ),
                self._format_optional_float(
                    matched_row.abs_error if matched_row is not None else None,
                    digits=6,
                ),
                self._format_optional_percent(
                    matched_row.rel_error if matched_row is not None else None,
                ),
                self._format_optional_float(
                    matched_row.log_error if matched_row is not None else None,
                    digits=6,
                ),
            ]
            self._fill_row(self.validation_table, row_index, values)

    def populate_cross_validation_table(self, run: AnalysisRunResult) -> None:
        self.cross_validation_table.clearSpans()
        if run.cross_validation is None:
            self._show_table_placeholder(
                self.cross_validation_table,
                self._cross_validation_placeholder_message(run),
            )
            return

        table_rows = build_cross_validation_table_rows(
            list(run.cross_validation.rows) if run.cross_validation is not None else []
        )
        self.cross_validation_table.setRowCount(len(table_rows))
        for row_index, values in enumerate(table_rows):
            self._fill_row(self.cross_validation_table, row_index, values)

    def populate_bootstrap_table(self, run: AnalysisRunResult) -> None:
        self.bootstrap_table.clearSpans()
        summary = run.bootstrap_summary
        if summary is None:
            self._show_table_placeholder(
                self.bootstrap_table,
                self._bootstrap_placeholder_message(run),
            )
            return

        rows = [
            ("alpha", summary.alpha),
            ("beta", summary.beta),
        ]
        if summary.alpha_beta_ratio is not None:
            rows.append(("alpha/beta", summary.alpha_beta_ratio))

        self.bootstrap_table.setRowCount(len(rows))
        for row_index, (label, stat) in enumerate(rows):
            values = [
                label,
                f"{stat.mean:.6f}",
                f"{stat.std:.6f}",
                f"{stat.q025:.6f}",
                f"{stat.median:.6f}",
                f"{stat.q975:.6f}",
            ]
            self._fill_row(self.bootstrap_table, row_index, values)

    def build_run_summary_text(self, run: AnalysisRunResult) -> str:
        summary = run.summary
        control_to_experiments: Dict[str, List[str]] = {}
        for experiment in list(run.train) + list(run.validation):
            control_to_experiments.setdefault(experiment.control_label, []).append(
                experiment.path.name
            )

        lines = [
            f"Run: {run.label}",
            f"Train kind: {run.train_kind}",
            f"Validation kind: {run.validation_kind}",
            (
                "Matched regimens: "
                f"total={summary.total_count}, "
                f"single={summary.single_count}, "
                f"fractionated={summary.fractionated_count}, "
                f"train={summary.train_count}, "
                f"validation={summary.validation_count}"
            ),
            f"Response mode: {summary.response_mode}",
            f"Model: {summary.model_kind}",
        ]

        if control_to_experiments:
            lines.append("Control mapping:")
            for control_label, experiment_names in sorted(control_to_experiments.items()):
                lines.append(f"- {control_label}: {', '.join(experiment_names)}")

        if summary.reason:
            lines.append(f"Reason: {summary.reason}")

        if run.fit_result is not None:
            ratio = self._format_optional_float(run.fit_result.alpha_beta_ratio, digits=2) or "-"
            lines.extend(
                [
                    f"alpha = {run.fit_result.alpha:.6f} Gy^-1",
                    f"beta = {run.fit_result.beta:.6f} Gy^-2",
                    f"alpha/beta = {ratio} Gy",
                ]
            )
            if run.fit_result.alpha_0 is not None:
                lines.append(f"alpha_0 = {run.fit_result.alpha_0:.6f} Gy^-1")
            if run.fit_result.lambda_alpha is not None:
                lines.append(
                    f"lambda_alpha = {run.fit_result.lambda_alpha:.6f} Gy^-1 per keV/um"
                )
            if run.fit_result.curve_clearance_rate is not None:
                lines.append(
                    f"curve_clearance_rate = {run.fit_result.curve_clearance_rate:.6f} per day"
                )
            if run.fit_result.transition_dose is not None:
                lines.append(f"transition_dose = {run.fit_result.transition_dose:.6f} Gy")
            if run.fit_result.saturation_dose is not None:
                lines.append(f"saturation_dose = {run.fit_result.saturation_dose:.6f} Gy")
            if run.fit_result.lag_days is not None:
                lines.append(f"lag_days = {run.fit_result.lag_days:.6f}")
            if run.fit_result.repopulation_rate is not None:
                lines.append(
                    f"repopulation_rate = {run.fit_result.repopulation_rate:.6f} per day"
                )
            if run.fit_result.model_kind == "linear":
                lines.append("repair_half_time = n/a for linear model")
            elif run.fit_result.model_kind == "glq":
                lines.append("repair_half_time = not used by gLQ")
            elif run.fit_result.model_kind == "let_dependent":
                lines.append("repair_half_time = not used by LET-dependent LQ")
            elif run.fit_result.model_kind == "lq_l":
                lines.append("repair_half_time = not used by LQ-L")
            elif run.fit_result.model_kind == "lq_repop":
                lines.append("repair_half_time = not used by LQ + repopulation")
            elif run.fit_result.model_kind == "repair_biexp":
                lines.append(
                    "repair_biexp = "
                    f"fast={self._format_optional_float(run.fit_result.repair_half_time_fast_hours, digits=3) or '-'} h, "
                    f"slow={self._format_optional_float(run.fit_result.repair_half_time_slow_hours, digits=3) or '-'} h, "
                    f"fast_fraction={self._format_optional_float(run.fit_result.repair_fast_fraction, digits=3) or '-'}"
                )
            elif run.fit_result.model_kind == "repair_repop":
                lines.append(
                    f"repair_half_time = {run.fit_result.repair_half_time_hours:.3f} h"
                )
            elif (
                run.fit_result.repair_half_time_hours is not None
                and run.fit_result.repair_half_time_hours > 0.0
            ):
                lines.append(
                    f"repair_half_time = {run.fit_result.repair_half_time_hours:.3f} h"
                )
            else:
                lines.append("repair_half_time = disabled (classic LQ)")

        if run.training_metrics is not None:
            lines.append(
                "Training metrics: "
                f"points={run.training_metrics.point_count}, "
                f"MAE={run.training_metrics.mae:.6f}, "
                f"RMSE={run.training_metrics.rmse:.6f}, "
                f"mean_abs_log_error={run.training_metrics.mean_abs_log_error:.6f}"
            )
            if run.training_metrics.aic is not None:
                lines.append(f"Training AIC = {run.training_metrics.aic:.4f}")
            if run.training_metrics.r_squared is not None:
                lines.append(f"Training R^2 = {run.training_metrics.r_squared:.4f}")
            if run.training_metrics.adjusted_r_squared is not None:
                lines.append(
                    f"Training adjusted R^2 = {run.training_metrics.adjusted_r_squared:.4f}"
                )
            if run.training_metrics.bic is not None:
                lines.append(f"Training BIC = {run.training_metrics.bic:.4f}")

        if run.cross_validation is not None:
            lines.append(
                "LOO cross-validation: "
                f"folds={run.cross_validation.n_successful}/{run.cross_validation.n_experiments}, "
                f"MAE={run.cross_validation.cv_mae:.6f}, "
                f"RMSE={run.cross_validation.cv_rmse:.6f}"
            )
            if run.cross_validation.cv_r_squared is not None:
                lines.append(f"LOO CV R^2 = {run.cross_validation.cv_r_squared:.4f}")
        elif summary.response_mode != "scalar":
            lines.append("LOO cross-validation is available only for scalar response.")
        elif len(run.train) < 3:
            lines.append(
                f"LOO cross-validation skipped: need at least 3 training experiments (found {len(run.train)})."
            )
        else:
            lines.append("LOO cross-validation did not produce reportable folds.")

        mean_bed = run.mean_train_bed
        mean_eqd2 = run.mean_train_eqd2
        mean_g = run.mean_train_g_factor
        mean_tcp = run.estimate_mean_train_tcp()
        if any(value is not None for value in (mean_bed, mean_eqd2, mean_g, mean_tcp)):
            lines.append(
                "Mean training radiobiology: "
                f"BED={self._format_optional_float(mean_bed, digits=4) or '-'}, "
                f"EQD2={self._format_optional_float(mean_eqd2, digits=4) or '-'}, "
                f"G={self._format_optional_float(mean_g, digits=4) or '-'}, "
                f"TCP={self._format_optional_float(mean_tcp, digits=6) or '-'}"
            )

        diagnostics = run.timing_diagnostics
        if diagnostics is not None and diagnostics.repair_model_enabled:
            lines.extend(
                [
                    "Timing diagnostics:",
                    (
                        f"- train={diagnostics.train_count}, "
                        f"fractionated={diagnostics.fractionated_count}, "
                        f"explicit_timing={diagnostics.explicit_timing_count}, "
                        f"explicit_fractionated={diagnostics.explicit_fractionated_count}"
                    ),
                    (
                        f"- unique_fractionated_schedules={diagnostics.unique_schedule_count}, "
                        f"matched_patterns_with_multi_timing={diagnostics.same_fractions_multi_timing_count}, "
                        f"unique_quadratic={diagnostics.unique_quadratic_count}, "
                        f"design_rank={diagnostics.design_rank}"
                    ),
                ]
            )
            if diagnostics.condition_number is not None:
                lines.append(f"- condition_number={diagnostics.condition_number:.2g}")
            if diagnostics.repair_half_time_fast_hours is not None:
                lines.append(
                    f"- repair_fast_half_time={diagnostics.repair_half_time_fast_hours:.3f} h"
                )
            if diagnostics.repair_half_time_slow_hours is not None:
                lines.append(
                    f"- repair_slow_half_time={diagnostics.repair_half_time_slow_hours:.3f} h"
                )
            if diagnostics.repair_fast_fraction is not None:
                lines.append(f"- repair_fast_fraction={diagnostics.repair_fast_fraction:.3f}")
            if diagnostics.warnings:
                lines.append("Timing warnings:")
                for warning in diagnostics.warnings:
                    lines.append(f"- {warning}")

        if run.validation_summary is not None:
            lines.extend(
                [
                    (
                        f"Validation metrics ({run.validation_summary.response_mode}): "
                        f"points={run.validation_summary.point_count}, "
                        f"MAE={run.validation_summary.mae:.6f}, "
                        f"RMSE={run.validation_summary.rmse:.6f}, "
                        f"mean_abs_log_error={run.validation_summary.mean_abs_log_error:.6f}"
                    )
                ]
            )
        elif run.validation_kind == "none":
            lines.append("Validation disabled: Validate kind = none.")
        elif run.validation_kind != "none":
            lines.append("Validation subset requested, but no holdout experiments matched it.")

        if run.bootstrap_summary is not None:
            lines.append(
                (
                    "Bootstrap: "
                    f"requested={run.bootstrap_summary.requested_repeats}, "
                    f"successful={run.bootstrap_summary.successful_repeats}, "
                    f"failed={run.bootstrap_summary.failed_repeats}"
                )
            )
            if (
                run.bootstrap_summary.alpha_beta_ratio is not None
                and run.bootstrap_summary.beta.q025 <= 1.0e-8
            ):
                lines.append(
                    "alpha/beta is unstable in bootstrap because beta approaches zero."
                )
        else:
            lines.append("Bootstrap skipped: Bootstrap = 0.")

        if run.model_comparison:
            lines.append("")
            lines.append("Model comparison:")
            for row in run.model_comparison:
                metrics_text = ""
                if row.metrics is not None:
                    metrics_text = (
                        f" | points={row.metrics.point_count}"
                        f" | RMSE={row.metrics.rmse:.6f}"
                        f" | MAE={row.metrics.mae:.6f}"
                    )
                    if row.metrics.aic is not None:
                        metrics_text += f" | AIC={row.metrics.aic:.4f}"
                    if row.metrics.r_squared is not None:
                        metrics_text += f" | R^2={row.metrics.r_squared:.4f}"
                    if row.metrics.adjusted_r_squared is not None:
                        metrics_text += f" | Adj R^2={row.metrics.adjusted_r_squared:.4f}"
                    if row.metrics.bic is not None:
                        metrics_text += f" | BIC={row.metrics.bic:.4f}"
                if row.transition_dose is not None:
                    metrics_text += f" | transition_dose={row.transition_dose:.6f}"
                if row.saturation_dose is not None:
                    metrics_text += f" | saturation_dose={row.saturation_dose:.6f}"
                if row.lambda_alpha is not None:
                    metrics_text += f" | lambda_alpha={row.lambda_alpha:.6f}"
                if row.lag_days is not None:
                    metrics_text += f" | lag_days={row.lag_days:.6f}"
                if row.repopulation_rate is not None:
                    metrics_text += f" | repop={row.repopulation_rate:.6f}"
                note_text = f" | {row.reason}" if row.reason else ""
                lines.append(
                    f"- {row.model_kind}: status={row.status}{metrics_text}{note_text}"
                )

        if run.train:
            lines.append("")
            lines.append("Training files:")
            for experiment in run.train:
                lines.append(
                    f"- {experiment.path.name}: "
                    f"family={experiment.family or '-'}, "
                    f"kind={experiment.regimen_kind}, "
                    f"fractions={format_fractions(experiment.fractions)}, "
                    f"schedule={experiment.schedule_label}, "
                    f"SF={experiment.sf:.6f}"
                )

        if run.validation:
            lines.append("")
            lines.append("Validation files:")
            for experiment in run.validation:
                lines.append(
                    f"- {experiment.path.name}: "
                    f"family={experiment.family or '-'}, "
                    f"kind={experiment.regimen_kind}, "
                    f"fractions={format_fractions(experiment.fractions)}, "
                    f"schedule={experiment.schedule_label}, "
                    f"SF={experiment.sf:.6f}"
                )

        return "\n".join(lines)

    @staticmethod
    def _fill_row(table: QTableWidget, row_index: int, values: Sequence[str]) -> None:
        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row_index, column_index, item)

    @staticmethod
    def _format_optional_float(value: Optional[float], digits: int) -> str:
        if value is None or not math.isfinite(float(value)):
            return ""
        return f"{value:.{digits}f}"

    @staticmethod
    def _format_optional_percent(value: Optional[float]) -> str:
        if value is None or not math.isfinite(float(value)):
            return ""
        return f"{value:.2%}"


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FitAlphaBetaWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
