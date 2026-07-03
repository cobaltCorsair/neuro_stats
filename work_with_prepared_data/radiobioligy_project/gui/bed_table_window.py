import os

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QVBoxLayout,
)

try:
    from gui.dataframe_table_widget import DataFrameTableMixin
    from data_processing.excel_data_processor import extract_dose_fractions_from_params
    from survival.radiobiology_analysis import compute_bed
except ImportError:
    from work_with_prepared_data.radiobioligy_project.gui.dataframe_table_widget import DataFrameTableMixin
    from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (
        extract_dose_fractions_from_params,
    )
    from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import compute_bed


def _format_dose_scheme(fraction_doses: list[float], schedule_hours: list[float]) -> str:
    """Возвращает короткую строку схемы облучения: '32 Гр', '3×16 Гр', '2×23 Гр, 6ч'."""
    if not fraction_doses:
        return "—"
    n = len(fraction_doses)
    total = sum(fraction_doses)
    if n == 1:
        d = fraction_doses[0]
        return f"{d:g} Гр"
    # Если все фракции одинаковые — компактная запись N×d
    if len(set(round(d, 4) for d in fraction_doses)) == 1:
        d = fraction_doses[0]
        scheme = f"{n}×{d:g} Гр"
    else:
        # Разные дозы — перечисляем
        scheme = "+".join(f"{d:g}" for d in fraction_doses) + " Гр"

    # Добавляем интервал, если есть уникальный (один или все одинаковые)
    if schedule_hours:
        unique_gaps = set(round(h, 2) for h in schedule_hours)
        if len(unique_gaps) == 1:
            h = next(iter(unique_gaps))
            scheme += f", {h:g}ч"
        else:
            scheme += ", " + "/".join(f"{h:g}ч" for h in schedule_hours)
    return scheme


class BedTableWindow(QDialog, DataFrameTableMixin):
    """Окно с таблицей BED и EQD2 по выбранным группам.

    Для каждой загруженной группы (файла эксперимента) читает схему облучения
    из experiment_params и вычисляет BED при двух значениях α/β (по умолчанию 10 и 3),
    соответствующих острому и позднему эффекту.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("BED по группам")
        self.setMinimumSize(750, 400)

        # (file_path, experiment_params) для каждой загруженной группы
        self._groups: list[tuple[str, list[str]]] = []

        self.table = QTableWidget()
        self._ab1_spin: QDoubleSpinBox
        self._ab2_spin: QDoubleSpinBox
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Биологически эффективная доза (BED) по группам")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(title)

        # Панель α/β
        ab_group = QGroupBox("Параметры расчёта")
        ab_layout = QFormLayout(ab_group)
        ab_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        self._ab1_spin = QDoubleSpinBox()
        self._ab1_spin.setRange(0.1, 100.0)
        self._ab1_spin.setDecimals(1)
        self._ab1_spin.setSingleStep(0.5)
        self._ab1_spin.setValue(10.0)
        self._ab1_spin.setSuffix(" Гр")
        ab_layout.addRow("α/β (острый эффект):", self._ab1_spin)

        self._ab2_spin = QDoubleSpinBox()
        self._ab2_spin.setRange(0.1, 100.0)
        self._ab2_spin.setDecimals(1)
        self._ab2_spin.setSingleStep(0.5)
        self._ab2_spin.setValue(3.0)
        self._ab2_spin.setSuffix(" Гр")
        ab_layout.addRow("α/β (поздний эффект):", self._ab2_spin)

        self._rbe_spin = QDoubleSpinBox()
        self._rbe_spin.setRange(0.1, 10.0)
        self._rbe_spin.setDecimals(2)
        self._rbe_spin.setSingleStep(0.05)
        self._rbe_spin.setValue(1.0)
        ab_layout.addRow("ОБЭ:", self._rbe_spin)

        recalc_btn = QPushButton("Пересчитать")
        recalc_btn.clicked.connect(self._refresh)
        ab_layout.addRow("", recalc_btn)

        layout.addWidget(ab_group)

        self._hint = QLabel()
        self._hint.setStyleSheet("color: #556; font-size: 11px;")
        layout.addWidget(self._hint)
        self._update_hint()

        self._configure_data_table(self.table, stretch_last_section=False)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ── public API ─────────────────────────────────────────────────────────

    def set_groups(self, groups: list[tuple[str, list[str]]]) -> None:
        """Принимает список (file_path, experiment_params) и перестраивает таблицу."""
        self._groups = groups
        self._refresh()

    # ── internal ───────────────────────────────────────────────────────────

    def _update_hint(self):
        rbe = self._rbe_spin.value()
        if abs(rbe - 1.0) < 1e-9:
            self._hint.setText(
                "ОБЭ = 1.0 (физическая доза). "
                "Для справочной таблицы BED/EQD2 по найденному α/β — используйте вкладку BED/EQD2 в LQ-фиттере."
            )
        else:
            self._hint.setText(
                f"ОБЭ = {rbe:.2f}: предписанная доза делится на ОБЭ перед расчётом BED "
                f"(dфиз = dпредп / {rbe:.2f})."
            )

    def _refresh(self):
        self._update_hint()
        ab1 = self._ab1_spin.value()
        ab2 = self._ab2_spin.value()
        rbe = self._rbe_spin.value()

        col_ab1 = f"BED₁₀ (α/β={ab1:g} Гр)"   # BED₁₀ (α/β=10 Гр)
        col_ab2 = f"BED₃ (α/β={ab2:g} Гр)"         # BED₃  (α/β=3 Гр)

        rows = []
        for file_path, experiment_params in self._groups:
            fraction_doses, schedule_hours = extract_dose_fractions_from_params(experiment_params)
            if not fraction_doses:
                continue

            n = len(fraction_doses)
            dose_total = sum(fraction_doses)
            d_per_fraction = dose_total / n
            scheme = _format_dose_scheme(fraction_doses, schedule_hours)
            group_name = os.path.splitext(os.path.basename(file_path))[0]

            try:
                bed1 = compute_bed(dose_total, n, ab1, rbe_factor=rbe)
                bed2 = compute_bed(dose_total, n, ab2, rbe_factor=rbe)
            except ValueError:
                bed1 = float("nan")
                bed2 = float("nan")

            rows.append({
                "Группа": group_name,
                "Схема": scheme,
                "D, Гр": dose_total,
                "n": n,
                "d, Гр": d_per_fraction,
                col_ab1: bed1,
                col_ab2: bed2,
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["Группа", "Схема", "D, Гр", "n", "d, Гр", col_ab1, col_ab2]
        )
        self._populate_table(self.table, df, {}, show_missing_as_dash=True)
        # Числовые колонки — точнее 1 знак для BED
        self._format_bed_columns(df, col_ab1, col_ab2)

    def _format_bed_columns(self, df: pd.DataFrame, *bed_cols: str):
        """Переформатирует BED-ячейки в таблице: ровно 1 знак после запятой."""
        if df.empty:
            return
        for col_name in bed_cols:
            if col_name not in df.columns:
                continue
            col_idx = list(df.columns).index(col_name)
            for row_idx, val in enumerate(df[col_name]):
                item = self.table.item(row_idx, col_idx)
                if item is None:
                    continue
                try:
                    item.setText(f"{float(val):.1f}")
                except (ValueError, TypeError):
                    pass
