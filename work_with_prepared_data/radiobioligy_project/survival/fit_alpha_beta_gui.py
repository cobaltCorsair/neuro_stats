# coding: utf-8
"""Simple PyQt6 GUI for LQ alpha/beta fitting from tumor-volume Excel files."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    Fitter,
    InventoryReport,
    TumorExperiment,
    analyze_files,
    format_fractions,
    infer_radiation_family,
    is_control_file,
    parse_sf_modes,
)
from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor_gui import (
    TumorGrowthPredictorWindow,
)

USE_ALL_CONTROLS = "__all_controls__"
UNASSIGNED_CONTROL = "__unassigned_control__"


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
    "Observed SF",
    "Predicted SF",
    "Abs error",
    "Rel error",
    "Log error",
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
        self.growth_predictor_window: Optional[TumorGrowthPredictorWindow] = None
        self.setWindowTitle("Survival LQ fitter")
        self.resize(1400, 900)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)

        root_layout.addWidget(self._build_file_group())
        root_layout.addWidget(self._build_options_group())

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run fit")
        self.run_button.clicked.connect(self.run_analysis)
        action_row.addWidget(self.run_button)

        self.inventory_button = QPushButton("Scan inventory")
        self.inventory_button.clicked.connect(self.scan_inventory)
        action_row.addWidget(self.inventory_button)

        self.predictor_button = QPushButton("Open growth predictor")
        self.predictor_button.clicked.connect(self.open_growth_predictor)
        action_row.addWidget(self.predictor_button)
        action_row.addStretch(1)
        root_layout.addLayout(action_row)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self._build_summary_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Drop .xlsx files сюда или добавьте их кнопками.")
        self.refresh_control_selector()

    def open_growth_predictor(self) -> None:
        if self.growth_predictor_window is None:
            self.growth_predictor_window = TumorGrowthPredictorWindow(self.run_results)
        else:
            self.growth_predictor_window.run_results = list(self.run_results)
            self.growth_predictor_window.populate_fit_results()
        self.growth_predictor_window.show()
        self.growth_predictor_window.raise_()
        self.growth_predictor_window.activateWindow()

    def _build_file_group(self) -> QGroupBox:
        group = QGroupBox("Input files", self)
        layout = QVBoxLayout(group)

        button_row = QHBoxLayout()
        add_files_button = QPushButton("Add files")
        add_files_button.clicked.connect(self.add_files_dialog)
        button_row.addWidget(add_files_button)

        add_folder_button = QPushButton("Add folder")
        add_folder_button.clicked.connect(self.add_folder_dialog)
        button_row.addWidget(add_folder_button)

        remove_button = QPushButton("Remove selected")
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
        self.file_list.setMinimumHeight(180)
        self.file_list.setToolTip(
            "Можно перетаскивать .xlsx файлы или целые папки. "
            "Файлы с 'control' в имени будут использованы как контроль."
        )
        layout.addWidget(self.file_list)

        layout.addWidget(QLabel("Experiment to control mapping"))
        self.assignment_table = self._create_table(ASSIGNMENT_HEADERS)
        self.assignment_table.setMinimumHeight(180)
        layout.addWidget(self.assignment_table)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Options", self)
        layout = QGridLayout(group)

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

        self.by_family_check = QCheckBox("Analyze by family", self)
        layout.addWidget(self.by_family_check, 0, 4)

        layout.addWidget(QLabel("Default control"), 1, 0)
        self.control_combo = QComboBox(self)
        self.control_combo.setToolTip(
            "Выберите control по умолчанию для новых строк сопоставления "
            "или примените его ко всем экспериментам."
        )
        layout.addWidget(self.control_combo, 1, 1, 1, 4)

        self.apply_default_control_button = QPushButton("Apply to all experiments")
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
        self.model_kind_combo.addItem("lq_l", "lq_l")
        self.model_kind_combo.addItem("linear", "linear")
        layout.addWidget(self.model_kind_combo, 3, 3)

        self.compare_models_check = QCheckBox("Compare models", self)
        self.compare_models_check.setToolTip(
            "Run classic LQ, repair-aware LQ, LQ-L and linear candidates, then rank them by fit error."
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

        self.aggregate_check = QCheckBox("Aggregate repeated regimens", self)
        layout.addWidget(self.aggregate_check, 5, 2, 1, 2)

        self.dedupe_check = QCheckBox("Deduplicate regimens", self)
        layout.addWidget(self.dedupe_check, 5, 4, 1, 2)

        self.verbose_check = QCheckBox("Verbose CLI logging in terminal", self)
        layout.addWidget(self.verbose_check, 6, 0, 1, 3)

        layout.addWidget(QLabel("Summary CSV"), 7, 0)
        self.summary_csv_edit = QLineEdit(self)
        self.summary_csv_edit.setPlaceholderText("optional path for summary csv")
        layout.addWidget(self.summary_csv_edit, 7, 1, 1, 4)

        summary_browse_button = QPushButton("Browse")
        summary_browse_button.clicked.connect(self.choose_summary_csv)
        layout.addWidget(summary_browse_button, 7, 5)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)
        return group

    def _build_summary_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("Run summary"))
        self.summary_table = self._create_table(SUMMARY_HEADERS)
        self.summary_table.currentCellChanged.connect(self._sync_run_selector_with_table)
        layout.addWidget(self.summary_table)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Selected run"))
        self.run_selector = QComboBox(self)
        self.run_selector.currentIndexChanged.connect(self.display_run)
        header_row.addWidget(self.run_selector, 1)
        layout.addLayout(header_row)

        self.detail_tabs = QTabWidget(self)

        self.train_table = self._create_table(TRAIN_HEADERS)
        self.detail_tabs.addTab(self.train_table, "Training")

        self.validation_table = self._create_table(VALIDATION_HEADERS)
        self.detail_tabs.addTab(self.validation_table, "Validation")

        self.bootstrap_table = self._create_table(BOOTSTRAP_HEADERS)
        self.detail_tabs.addTab(self.bootstrap_table, "Bootstrap")

        inventory_panel = QWidget(self)
        inventory_layout = QVBoxLayout(inventory_panel)
        self.inventory_table = self._create_table(INVENTORY_HEADERS)
        inventory_layout.addWidget(self.inventory_table, 1)
        self.inventory_text = QPlainTextEdit(self)
        self.inventory_text.setReadOnly(True)
        self.inventory_text.setMaximumHeight(180)
        inventory_layout.addWidget(self.inventory_text)
        self.detail_tabs.addTab(inventory_panel, "Inventory")

        self.details_text = QPlainTextEdit(self)
        self.details_text.setReadOnly(True)
        self.detail_tabs.addTab(self.details_text, "Summary text")

        layout.addWidget(self.detail_tabs, 1)
        return panel

    @staticmethod
    def _create_table(headers: Sequence[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        return table

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
            combo.currentIndexChanged.connect(lambda *_args: self.scan_inventory())

            self.assignment_table.setItem(row_index, 0, experiment_item)
            self.assignment_table.setItem(row_index, 1, family_item)
            self.assignment_table.setCellWidget(row_index, 2, combo)

    def apply_default_control_to_all(self) -> None:
        selected_value = self.control_combo.currentData()
        if selected_value in (None, UNASSIGNED_CONTROL):
            QMessageBox.warning(
                self,
                "Default control",
                "Choose a default control before applying it to all experiments.",
            )
            return

        for row_index in range(self.assignment_table.rowCount()):
            combo = self.assignment_table.cellWidget(row_index, 2)
            if combo is None:
                continue
            for combo_index in range(combo.count()):
                if combo.itemData(combo_index) == selected_value:
                    combo.setCurrentIndex(combo_index)
                    break
        self.scan_inventory()

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
            )
            self.run_results = results
            self.populate_results()

            if summary_csv and results:
                output_path = Path(summary_csv).expanduser().resolve()
                Fitter.write_analysis_summaries_csv(
                    output_path,
                    [run.summary for run in results],
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
                summary.reason or "",
            ]
            self._fill_row(self.summary_table, row_index, values)
            self.run_selector.addItem(run.label)

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
        self.details_text.clear()

    def clear_inventory(self) -> None:
        self.inventory_report = None
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
            note_text = f" | notes: {'; '.join(summary.notes)}" if summary.notes else ""
            lines.append(
                f"- {summary.family}: analyzable={summary.analyzable_count}, "
                f"single={summary.single_count}, "
                f"fractionated={summary.fractionated_count}, "
                f"fit_ready={'yes' if summary.fit_ready else 'no'}{note_text}"
            )
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
        self.populate_train_table(run.train)
        self.populate_validation_table(run)
        self.populate_bootstrap_table(run)
        self.details_text.setPlainText(self.build_run_summary_text(run))

    def populate_train_table(self, experiments: Sequence[TumorExperiment]) -> None:
        self.train_table.setRowCount(len(experiments))
        for row_index, experiment in enumerate(experiments):
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
                str(experiment.repeat_count),
                f"{experiment.sf_std:.6f}",
            ]
            self._fill_row(self.train_table, row_index, values)

    def populate_validation_table(self, run: AnalysisRunResult) -> None:
        rows = []
        if run.validation_summary is not None:
            rows = list(run.validation_summary.rows)

        if run.validation_summary is not None and run.validation_summary.response_mode == "curve":
            self.validation_table.setRowCount(0)
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

    def populate_bootstrap_table(self, run: AnalysisRunResult) -> None:
        summary = run.bootstrap_summary
        if summary is None:
            self.bootstrap_table.setRowCount(0)
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
            if run.fit_result.curve_clearance_rate is not None:
                lines.append(
                    f"curve_clearance_rate = {run.fit_result.curve_clearance_rate:.6f} per day"
                )
            if run.fit_result.transition_dose is not None:
                lines.append(f"transition_dose = {run.fit_result.transition_dose:.6f} Gy")
            if run.fit_result.model_kind == "linear":
                lines.append("repair_half_time = n/a for linear model")
            elif run.fit_result.model_kind == "lq_l":
                lines.append("repair_half_time = not used by LQ-L")
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
                if row.transition_dose is not None:
                    metrics_text += f" | transition_dose={row.transition_dose:.6f}"
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
        if value is None:
            return ""
        return f"{value:.{digits}f}"

    @staticmethod
    def _format_optional_percent(value: Optional[float]) -> str:
        if value is None:
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
