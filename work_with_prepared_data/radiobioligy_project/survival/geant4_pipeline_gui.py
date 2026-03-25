# coding: utf-8
"""PyQt6 window for running the GEANT4 / RT Dose -> prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from work_with_prepared_data.radiobioligy_project.survival.dose_reader import is_rt_dose_path
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
        AnalysisRunResult,
        Fitter,
    )
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
    )
    from work_with_prepared_data.radiobioligy_project.survival.pipeline_geant4_to_prediction import (
        DEFAULT_COMPONENT_FAMILY_MAP,
        resolve_pipeline_input_paths,
        run_prediction_pipeline,
    )
except ModuleNotFoundError:
    from survival.dose_reader import is_rt_dose_path
    from survival.fit_alpha_beta_using_processor import AnalysisRunResult, Fitter
    from survival.let_parametrization import LETDependentParams
    from survival.pipeline_geant4_to_prediction import (
        DEFAULT_COMPONENT_FAMILY_MAP,
        resolve_pipeline_input_paths,
        run_prediction_pipeline,
    )


SUCCESS_STATUSES = {"", "ok", "success"}


@dataclass(frozen=True)
class PipelineRadiobiologySource:
    """Resolved radiobiology input for the GEANT4 prediction pipeline."""

    fit_results_csv: Optional[Path] = None
    let_params: Optional[LETDependentParams] = None
    manual_alpha_beta: Optional[tuple[float, float]] = None
    description: str = ""


def resolve_radiobiology_source_from_run_results(
    run_results: Sequence[AnalysisRunResult],
    export_path: Path,
) -> PipelineRadiobiologySource:
    """Resolve pipeline inputs from in-memory fitter results."""
    successful_runs = [
        run
        for run in run_results
        if run.fit_result is not None and str(run.summary.status).strip().lower() in SUCCESS_STATUSES
    ]
    if not successful_runs:
        raise ValueError(
            "No successful fit results are available in the fitter window. "
            "Run the fitter first or switch to Summary CSV / manual alpha-beta."
        )

    family_runs = [
        run
        for run in successful_runs
        if _normalize_family_label(run.summary.family) is not None
    ]
    if family_runs:
        export_path = Path(export_path)
        Fitter.write_analysis_summaries_csv(
            export_path,
            [run.summary for run in successful_runs],
            successful_runs,
        )
        return PipelineRadiobiologySource(
            fit_results_csv=export_path,
            description=(
                f"Using {len(family_runs)} family-specific fitter result(s) exported to {export_path}."
            ),
        )

    selected_run = successful_runs[-1]
    fit_result = selected_run.fit_result
    if fit_result is None:
        raise ValueError("Internal error: fitter context does not contain a fit result.")

    if fit_result.model_kind == "let_dependent" or fit_result.alpha_0 is not None or fit_result.lambda_alpha is not None:
        alpha_0 = float(fit_result.alpha_0) if fit_result.alpha_0 is not None else float(fit_result.alpha)
        let_params = LETDependentParams(
            alpha_0=alpha_0,
            lambda_alpha=float(fit_result.lambda_alpha or 0.0),
            beta_0=float(fit_result.beta),
            lambda_beta=0.0,
        )
        return PipelineRadiobiologySource(
            let_params=let_params,
            description=f"Using the latest LET-dependent fitter result ({selected_run.label}).",
        )

    return PipelineRadiobiologySource(
        manual_alpha_beta=(float(fit_result.alpha), float(fit_result.beta)),
        description=f"Using the latest fitter result as constant alpha/beta ({selected_run.label}).",
    )


def _normalize_family_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == "all":
        return None
    return normalized


def _parse_optional_float_csv(text: str) -> Optional[list[float]]:
    raw = text.strip()
    if not raw:
        return None
    values: list[float] = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        values.append(float(piece.replace(",", ".")))
    return values or None


def _parse_optional_int_csv(text: str) -> Optional[list[int]]:
    values = _parse_optional_float_csv(text)
    if values is None:
        return None
    return [int(value) for value in values]


def _parse_component_family_map(text: str) -> Optional[dict[str, str]]:
    raw = text.strip()
    if not raw:
        return None
    mapping: dict[str, str] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid component-family mapping '{item}'. Expected component=family pairs."
            )
        component, family = item.split("=", 1)
        component_key = component.strip()
        family_value = family.strip().lower()
        if not component_key or not family_value:
            raise ValueError(f"Invalid component-family mapping '{item}'.")
        mapping[component_key] = family_value
    return mapping or None


def _parse_required_existing_path(text: str, *, label: str) -> Path:
    raw = text.strip()
    if not raw:
        raise ValueError(f"{label} is required.")
    path = Path(raw).expanduser()
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    return path.resolve()


def _parse_optional_existing_path(text: str) -> Optional[Path]:
    raw = text.strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        raise ValueError(f"Optional path does not exist: {path}")
    return path.resolve()


def _parse_optional_float(text: str, *, label: str, positive: bool = False) -> Optional[float]:
    raw = text.strip()
    if not raw:
        return None
    value = float(raw.replace(",", "."))
    if positive and value <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return float(value)


def _parse_required_float(text: str, *, label: str, positive: bool = False) -> float:
    value = _parse_optional_float(text, label=label, positive=positive)
    if value is None:
        raise ValueError(f"{label} is required.")
    return value


def _parse_optional_int(text: str, *, label: str, positive: bool = False) -> Optional[int]:
    raw = text.strip()
    if not raw:
        return None
    value = int(float(raw.replace(",", ".")))
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


class Geant4PipelineWindow(QMainWindow):
    """GUI wrapper around the GEANT4 prediction pipeline."""

    def __init__(
        self,
        run_results: Optional[Sequence[AnalysisRunResult]] = None,
        *,
        summary_csv_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.run_results = list(run_results or ())
        self.last_result: Optional[dict] = None
        self.setWindowTitle("GEANT4 prediction pipeline")
        self.resize(980, 860)
        self.setMinimumSize(860, 720)
        self._build_ui()
        self.set_context(run_results=run_results, summary_csv_path=summary_csv_path)

    def set_context(
        self,
        run_results: Optional[Sequence[AnalysisRunResult]] = None,
        *,
        summary_csv_path: Optional[str] = None,
    ) -> None:
        self.run_results = list(run_results or ())
        if summary_csv_path and not self.fit_results_csv_edit.text().strip():
            self.fit_results_csv_edit.setText(summary_csv_path)
        self.current_fit_label.setText(self._build_current_fit_context_text())
        self._update_source_controls()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        scroll_content = QWidget(self)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)
        scroll_layout.addWidget(self._build_input_group())
        scroll_layout.addWidget(self._build_radiobiology_group())
        scroll_layout.addWidget(self._build_pipeline_group())
        scroll_layout.addWidget(self._build_growth_group())
        scroll_layout.addWidget(self._build_output_group())
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        root_layout.addWidget(scroll, 1)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.run_button = QPushButton("Run GEANT4 pipeline")
        self.run_button.clicked.connect(self.run_pipeline)
        action_row.addWidget(self.run_button)
        action_row.addStretch(1)
        root_layout.addLayout(action_row)

        root_layout.addWidget(QLabel("Run summary"))
        self.summary_text = QPlainTextEdit(self)
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(220)
        root_layout.addWidget(self.summary_text, 0)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Select GEANT4 files or an input folder, then run the pipeline.")

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("GEANT4 inputs", self)
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("Dose protobuf / RT Dose / Folder"), 0, 0)
        self.dose_pb_edit = QLineEdit(self)
        self.dose_pb_edit.setPlaceholderText("totDoseVoxelMap.pb, fullVoxelMap.pb, RT Dose .dcm, or an input folder")
        layout.addWidget(self.dose_pb_edit, 0, 1)
        dose_button = QPushButton("Browse")
        dose_button.clicked.connect(self.choose_dose_pb)
        layout.addWidget(dose_button, 0, 2)
        dose_folder_button = QPushButton("Folder")
        dose_folder_button.clicked.connect(self.choose_dose_input_dir)
        layout.addWidget(dose_folder_button, 0, 3)

        layout.addWidget(QLabel("Geometry ivz (protobuf only)"), 1, 0)
        self.geometry_ivz_edit = QLineEdit(self)
        self.geometry_ivz_edit.setPlaceholderText("InputVoxelMap .ivz/.pb; leave empty for RT Dose")
        layout.addWidget(self.geometry_ivz_edit, 1, 1)
        geometry_button = QPushButton("Browse")
        geometry_button.clicked.connect(self.choose_geometry_ivz)
        layout.addWidget(geometry_button, 1, 2)

        layout.addWidget(QLabel("Contour protobuf / NIfTI / RTSTRUCT"), 2, 0)
        self.contour_pb_edit = QLineEdit(self)
        self.contour_pb_edit.setPlaceholderText("Optional ContourMeta .pb, Slicer mask .nii/.nii.gz, or RTSTRUCT .dcm")
        layout.addWidget(self.contour_pb_edit, 2, 1)
        contour_button = QPushButton("Browse")
        contour_button.clicked.connect(self.choose_contour_pb)
        layout.addWidget(contour_button, 2, 2)
        return group

    def _build_radiobiology_group(self) -> QGroupBox:
        group = QGroupBox("Radiobiology source", self)
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("Source mode"), 0, 0)
        self.source_mode_combo = QComboBox(self)
        self.source_mode_combo.addItem("Current fitter results", "current_fitter")
        self.source_mode_combo.addItem("Summary CSV", "summary_csv")
        self.source_mode_combo.addItem("Manual alpha/beta", "manual_alpha_beta")
        self.source_mode_combo.addItem("LET profile", "let_profile")
        self.source_mode_combo.currentIndexChanged.connect(self._update_source_controls)
        layout.addWidget(self.source_mode_combo, 0, 1, 1, 2)

        self.current_fit_label = QLabel(self)
        self.current_fit_label.setWordWrap(True)
        layout.addWidget(self.current_fit_label, 1, 0, 1, 4)

        layout.addWidget(QLabel("Summary CSV"), 2, 0)
        self.fit_results_csv_edit = QLineEdit(self)
        self.fit_results_csv_edit.setPlaceholderText("Optional fitter summary CSV")
        layout.addWidget(self.fit_results_csv_edit, 2, 1, 1, 2)
        self.fit_results_csv_button = QPushButton("Browse")
        self.fit_results_csv_button.clicked.connect(self.choose_fit_results_csv)
        layout.addWidget(self.fit_results_csv_button, 2, 3)

        layout.addWidget(QLabel("Alpha"), 3, 0)
        self.alpha_edit = QLineEdit("0.10", self)
        layout.addWidget(self.alpha_edit, 3, 1)

        layout.addWidget(QLabel("Beta"), 3, 2)
        self.beta_edit = QLineEdit("0.02", self)
        layout.addWidget(self.beta_edit, 3, 3)

        layout.addWidget(QLabel("LET alpha_0"), 4, 0)
        self.let_alpha0_edit = QLineEdit("0.08", self)
        layout.addWidget(self.let_alpha0_edit, 4, 1)

        layout.addWidget(QLabel("LET lambda_alpha"), 4, 2)
        self.let_lambda_alpha_edit = QLineEdit("0.003", self)
        layout.addWidget(self.let_lambda_alpha_edit, 4, 3)

        layout.addWidget(QLabel("LET beta_0"), 5, 0)
        self.let_beta0_edit = QLineEdit("0.02", self)
        layout.addWidget(self.let_beta0_edit, 5, 1)

        layout.addWidget(QLabel("LET lambda_beta"), 5, 2)
        self.let_lambda_beta_edit = QLineEdit("0.0", self)
        layout.addWidget(self.let_lambda_beta_edit, 5, 3)
        return group

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Pipeline options", self)
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("Structure"), 0, 0)
        self.structure_name_edit = QLineEdit("tumor", self)
        layout.addWidget(self.structure_name_edit, 0, 1)

        self.mixed_field_check = QCheckBox("Mixed field (fullVoxelMap)", self)
        self.mixed_field_check.toggled.connect(self._update_source_controls)
        layout.addWidget(self.mixed_field_check, 0, 2, 1, 2)

        layout.addWidget(QLabel("Component map"), 1, 0)
        self.component_family_map_edit = QLineEdit(
            ",".join(f"{name}={family}" for name, family in DEFAULT_COMPONENT_FAMILY_MAP.items()),
            self,
        )
        self.component_family_map_edit.setPlaceholderText("protonDose=p,mainDose=y,midDose=n,stuffDose=e")
        layout.addWidget(self.component_family_map_edit, 1, 1, 1, 3)

        layout.addWidget(QLabel("Schedule days"), 2, 0)
        self.schedule_days_edit = QLineEdit("0", self)
        self.schedule_days_edit.setPlaceholderText("0,1,2,3,4")
        layout.addWidget(self.schedule_days_edit, 2, 1)

        layout.addWidget(QLabel("N fractions"), 2, 2)
        self.n_fractions_edit = QLineEdit("", self)
        self.n_fractions_edit.setPlaceholderText("Optional if schedule is set")
        layout.addWidget(self.n_fractions_edit, 2, 3)

        layout.addWidget(QLabel("Model kind"), 3, 0)
        self.model_kind_combo = QComboBox(self)
        for model_kind in ("classic_lq", "repair_lq", "linear", "let_dependent"):
            self.model_kind_combo.addItem(model_kind, model_kind)
        layout.addWidget(self.model_kind_combo, 3, 1)

        layout.addWidget(QLabel("Repair T1/2 (h)"), 3, 2)
        self.repair_half_time_edit = QLineEdit("", self)
        self.repair_half_time_edit.setPlaceholderText("Optional")
        layout.addWidget(self.repair_half_time_edit, 3, 3)
        return group

    def _build_growth_group(self) -> QGroupBox:
        group = QGroupBox("Growth model", self)
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("Initial volume mm^3"), 0, 0)
        self.initial_volume_edit = QLineEdit("", self)
        self.initial_volume_edit.setPlaceholderText("Optional, inferred from voxel count if empty")
        layout.addWidget(self.initial_volume_edit, 0, 1)

        layout.addWidget(QLabel("Growth rate"), 0, 2)
        self.growth_rate_edit = QLineEdit("0.05", self)
        layout.addWidget(self.growth_rate_edit, 0, 3)

        layout.addWidget(QLabel("Carrying capacity"), 1, 0)
        self.carrying_capacity_edit = QLineEdit("5000", self)
        layout.addWidget(self.carrying_capacity_edit, 1, 1)

        layout.addWidget(QLabel("Clearance rate"), 1, 2)
        self.clearance_rate_edit = QLineEdit("0.1", self)
        layout.addWidget(self.clearance_rate_edit, 1, 3)

        layout.addWidget(QLabel("Growth duration (days)"), 2, 0)
        self.growth_duration_edit = QLineEdit("30", self)
        layout.addWidget(self.growth_duration_edit, 2, 1)

        layout.addWidget(QLabel("Time step (days)"), 2, 2)
        self.growth_time_step_edit = QLineEdit("0.5", self)
        layout.addWidget(self.growth_time_step_edit, 2, 3)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Outputs", self)
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("Output directory"), 0, 0)
        self.output_dir_edit = QLineEdit(str(Path.cwd() / "prediction_output"), self)
        layout.addWidget(self.output_dir_edit, 0, 1, 1, 2)
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.choose_output_dir)
        layout.addWidget(output_button, 0, 3)

        layout.addWidget(QLabel("BED/EQD2 fractions"), 1, 0)
        self.bed_eqd2_fractions_edit = QLineEdit("1,3,5,10,20,30", self)
        layout.addWidget(self.bed_eqd2_fractions_edit, 1, 1, 1, 3)
        return group

    def _build_current_fit_context_text(self) -> str:
        if not self.run_results:
            return "No in-memory fitter results are attached to this window."
        successful_runs = [
            run
            for run in self.run_results
            if run.fit_result is not None and str(run.summary.status).strip().lower() in SUCCESS_STATUSES
        ]
        if not successful_runs:
            return "Fitter context is attached, but there are no successful fit results yet."
        family_labels = sorted(
            {
                _normalize_family_label(run.summary.family) or "all"
                for run in successful_runs
            }
        )
        return (
            f"Attached fitter context: {len(successful_runs)} successful run(s), "
            f"families={', '.join(family_labels)}."
        )

    def _update_source_controls(self) -> None:
        source_mode = str(self.source_mode_combo.currentData() or "")
        self.fit_results_csv_edit.setEnabled(source_mode == "summary_csv")
        self.fit_results_csv_button.setEnabled(source_mode == "summary_csv")
        self.alpha_edit.setEnabled(source_mode == "manual_alpha_beta")
        self.beta_edit.setEnabled(source_mode == "manual_alpha_beta")
        self.let_alpha0_edit.setEnabled(source_mode == "let_profile")
        self.let_lambda_alpha_edit.setEnabled(source_mode == "let_profile")
        self.let_beta0_edit.setEnabled(source_mode == "let_profile")
        self.let_lambda_beta_edit.setEnabled(source_mode == "let_profile")
        self.component_family_map_edit.setEnabled(self.mixed_field_check.isChecked())
        if source_mode == "current_fitter":
            self.current_fit_label.setText(self._build_current_fit_context_text())

    def choose_dose_pb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dose input",
            str(Path.cwd()),
            "Dose files (*.pb *.dcm *.dicom);;All files (*)",
        )
        if path:
            self.dose_pb_edit.setText(path)
            if not self.output_dir_edit.text().strip():
                self.output_dir_edit.setText(str(Path(path).resolve().parent / "prediction_output"))

    def choose_dose_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select input directory",
            self.dose_pb_edit.text().strip() or str(Path.cwd()),
        )
        if path:
            resolved = Path(path).resolve()
            self.dose_pb_edit.setText(str(resolved))
            if not self.output_dir_edit.text().strip():
                self.output_dir_edit.setText(str(resolved / "prediction_output"))

    def choose_geometry_ivz(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select geometry ivz",
            str(Path.cwd()),
            "Voxel geometry (*.ivz *.pb);;All files (*)",
        )
        if path:
            self.geometry_ivz_edit.setText(path)

    def choose_contour_pb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select contour protobuf, NIfTI mask, or RTSTRUCT",
            str(Path.cwd()),
            "Contour files (*.pb *.nii *.nii.gz *.dcm *.dicom);;All files (*)",
        )
        if path:
            self.contour_pb_edit.setText(path)

    def choose_fit_results_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select fitter summary CSV",
            str(Path.cwd()),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self.fit_results_csv_edit.setText(path)

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self.output_dir_edit.text().strip() or str(Path.cwd()),
        )
        if path:
            self.output_dir_edit.setText(path)

    def _resolve_radiobiology_source(self, output_dir: Path) -> PipelineRadiobiologySource:
        source_mode = str(self.source_mode_combo.currentData() or "")
        if source_mode == "current_fitter":
            return resolve_radiobiology_source_from_run_results(
                self.run_results,
                output_dir / "_gui_fit_results.csv",
            )
        if source_mode == "summary_csv":
            fit_results_csv = _parse_required_existing_path(
                self.fit_results_csv_edit.text(),
                label="Summary CSV",
            )
            return PipelineRadiobiologySource(
                fit_results_csv=fit_results_csv,
                description=f"Using fitter summary CSV: {fit_results_csv}",
            )
        if source_mode == "manual_alpha_beta":
            alpha = _parse_required_float(self.alpha_edit.text(), label="Alpha", positive=True)
            beta = _parse_required_float(self.beta_edit.text(), label="Beta")
            return PipelineRadiobiologySource(
                manual_alpha_beta=(alpha, beta),
                description=f"Using manual alpha/beta: alpha={alpha:.6f}, beta={beta:.6f}",
            )
        if source_mode == "let_profile":
            alpha_0 = _parse_required_float(self.let_alpha0_edit.text(), label="LET alpha_0", positive=True)
            lambda_alpha = _parse_required_float(
                self.let_lambda_alpha_edit.text(),
                label="LET lambda_alpha",
            )
            beta_0 = _parse_required_float(self.let_beta0_edit.text(), label="LET beta_0")
            lambda_beta = _parse_required_float(
                self.let_lambda_beta_edit.text(),
                label="LET lambda_beta",
            )
            return PipelineRadiobiologySource(
                let_params=LETDependentParams(
                    alpha_0=alpha_0,
                    lambda_alpha=lambda_alpha,
                    beta_0=beta_0,
                    lambda_beta=lambda_beta,
                ),
                description=(
                    "Using explicit LET profile: "
                    f"alpha_0={alpha_0:.6f}, lambda_alpha={lambda_alpha:.6f}, "
                    f"beta_0={beta_0:.6f}, lambda_beta={lambda_beta:.6f}"
                ),
            )
        raise ValueError("Unknown radiobiology source mode.")

    def run_pipeline(self) -> None:
        cursor_set = False
        try:
            dose_input_path = _parse_required_existing_path(
                self.dose_pb_edit.text(),
                label="Dose input",
            )
            geometry_text = self.geometry_ivz_edit.text().strip()
            geometry_ivz_path = (
                None
                if (dose_input_path.is_dir() or is_rt_dose_path(dose_input_path)) and not geometry_text
                else _parse_required_existing_path(
                    geometry_text,
                    label="Geometry ivz",
                )
            )
            contour_path = _parse_optional_existing_path(self.contour_pb_edit.text())
            dose_pb_path, geometry_ivz_path, contour_path = resolve_pipeline_input_paths(
                dose_input_path,
                geometry_ivz_path,
                contour_path,
                mixed_field=self.mixed_field_check.isChecked(),
            )

            output_dir_raw = self.output_dir_edit.text().strip()
            output_dir = (
                Path(output_dir_raw).expanduser().resolve()
                if output_dir_raw
                else ((dose_input_path if dose_input_path.is_dir() else dose_pb_path.parent) / "prediction_output").resolve()
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            source = self._resolve_radiobiology_source(output_dir)
            schedule_days = _parse_optional_float_csv(self.schedule_days_edit.text())
            n_fractions = _parse_optional_int(
                self.n_fractions_edit.text(),
                label="N fractions",
                positive=True,
            )
            if schedule_days is None and n_fractions is None:
                n_fractions = 1

            component_family_map = (
                _parse_component_family_map(self.component_family_map_edit.text())
                if self.mixed_field_check.isChecked()
                else None
            )
            bed_eqd2_fractions = _parse_optional_int_csv(self.bed_eqd2_fractions_edit.text()) or [
                1,
                3,
                5,
                10,
                20,
                30,
            ]

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            cursor_set = True
            self.statusBar().showMessage("Running GEANT4 prediction pipeline...")
            QApplication.processEvents()

            result = run_prediction_pipeline(
                dose_pb_path=dose_pb_path,
                geometry_ivz_path=geometry_ivz_path,
                contour_path=contour_path,
                fit_results_csv=source.fit_results_csv,
                let_params=source.let_params,
                manual_alpha_beta=source.manual_alpha_beta,
                structure_name=self.structure_name_edit.text().strip() or "tumor",
                initial_volume_mm3=_parse_optional_float(
                    self.initial_volume_edit.text(),
                    label="Initial volume",
                    positive=True,
                ),
                growth_rate=_parse_required_float(
                    self.growth_rate_edit.text(),
                    label="Growth rate",
                    positive=True,
                ),
                carrying_capacity=_parse_required_float(
                    self.carrying_capacity_edit.text(),
                    label="Carrying capacity",
                    positive=True,
                ),
                clearance_rate=_parse_required_float(
                    self.clearance_rate_edit.text(),
                    label="Clearance rate",
                ),
                schedule_days=schedule_days,
                n_fractions=n_fractions,
                mixed_field=self.mixed_field_check.isChecked(),
                component_family_map=component_family_map,
                output_dir=output_dir,
                growth_duration_days=_parse_required_float(
                    self.growth_duration_edit.text(),
                    label="Growth duration",
                    positive=True,
                ),
                growth_time_step_days=_parse_required_float(
                    self.growth_time_step_edit.text(),
                    label="Growth time step",
                    positive=True,
                ),
                model_kind=str(self.model_kind_combo.currentData() or "classic_lq"),
                repair_half_time_hours=_parse_optional_float(
                    self.repair_half_time_edit.text(),
                    label="Repair half-time",
                    positive=True,
                ),
                bed_eqd2_fractions=bed_eqd2_fractions,
            )
            self.last_result = result
            self.summary_text.setPlainText(self._build_summary_text(result, source))
            self.statusBar().showMessage(f"GEANT4 prediction pipeline complete. Output: {output_dir}")
        except Exception as exc:
            self.summary_text.setPlainText(traceback.format_exc())
            self.statusBar().showMessage(f"GEANT4 prediction pipeline failed: {exc}")
            QMessageBox.critical(self, "GEANT4 pipeline failed", str(exc))
        finally:
            if cursor_set:
                QApplication.restoreOverrideCursor()

    def _build_summary_text(
        self,
        result: dict,
        source: PipelineRadiobiologySource,
    ) -> str:
        volumetric_sf = result["volumetric_sf"]
        growth_prediction = result["growth_prediction"]
        output_files = [Path(path) for path in result["output_files"]]
        summary_lines = [
            source.description,
            "",
            f"mean_dose_gy = {float(volumetric_sf.mean_dose_gy):.6f}",
            f"mean_sf = {float(volumetric_sf.mean_sf):.6f}",
            f"effective_alpha = {float(volumetric_sf.effective_alpha):.6f}",
            f"effective_beta = {float(volumetric_sf.effective_beta):.6f}",
            f"equivalent_uniform_dose = {float(volumetric_sf.equivalent_uniform_dose):.6f}",
            f"n_voxels = {len(volumetric_sf.voxel_results)}",
            f"growth_samples = {len(growth_prediction.times)}",
            "",
            "Output files:",
        ]
        summary_lines.extend(f"- {path}" for path in output_files)
        return "\n".join(summary_lines)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = Geant4PipelineWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
