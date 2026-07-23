# coding: utf-8
"""Recalculate effective in-vivo LQ parameters from tumor-volume experiments.

This analysis is intentionally separate from ``fit_alpha_beta_using_processor.py``.
It treats the scalar response as an effective tumor-volume endpoint, not as a
clonogenic surviving fraction:

    SF_eff = min_{t > 0} [(V_exp(t) / V_exp(0)) / (V_ctrl(t) / V_ctrl(0))].

Main safeguards:

* exact/explicit control matches are preferred and nearest-date proxy controls
  can be used under an explicit, fully audited policy;
* exact duplicate files are not counted twice, but genuine repeats are retained;
* animal-level uncertainty is propagated to SF_eff;
* non-negative linear and LQ models are compared using AICc;
* validation leaves out complete experimental series and years;
* the parameter bootstrap resamples series, then animals within each series;
* alpha/beta is suppressed when beta is compatible with zero or prediction is poor.

The script scans every Excel file under ``exps`` and writes a complete audit trail
to ``alpha_beta_eff_results``. Proxy control assignments are never silent: the
selected file and calendar gap are saved for every experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import lsq_linear


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (  # noqa: E402
    process_tumor_data_excel,
)
from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (  # noqa: E402
    infer_radiation_family,
    parse_time_days,
)


DEFAULT_EXPS_DIR = Path(__file__).resolve().with_name("exps")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().with_name("alpha_beta_eff_results")
DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[.\-_](\d{1,2})[.\-_](\d{4})(?!\d)")
NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
DOSE_UNIT_RE = re.compile(r"(?:гр|gy)\.?", re.IGNORECASE)

# These are the two non-coincident date matches already used in the preceding
# analysis. Same-date controls are discovered automatically. Any additional
# biological-series matches must be supplied explicitly through --control-map.
DEFAULT_CONTROL_OVERRIDES = {
    "2016-01-27": "control_13.01.2016.xlsx",
    "2025-10-22": "control_17.11.2025.xlsx",
}

# User-confirmed resolution of an exact-data duplicate carrying two different
# dates. The 16 May copy is retained because a same-date control exists; the
# 3 May copy is kept in the inventory as an excluded duplicate.
PREFERRED_DATE_CONFLICT_FILES = {"16.05.2018_y32.xlsx"}

FAMILY_LABELS_RU = {
    "e": "электроны",
    "y": "гамма-излучение",
    "p_peak": "протоны в пике",
    "p_through": "прострельные протоны",
    "n": "нейтроны",
    "c": "углерод",
}


@dataclass
class ControlSeries:
    path: Path
    experiment_date: Optional[date]
    times: np.ndarray
    volumes: np.ndarray


@dataclass
class ExperimentRecord:
    path: Path
    relative_path: str
    experiment_date: Optional[date] = None
    family: Optional[str] = None
    fractions: tuple[float, ...] = ()
    beam_labels: tuple[str, ...] = ()
    times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    volumes: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    n_animals: int = 0
    control: Optional[ControlSeries] = None
    control_match: str = ""
    nearest_control: str = ""
    nearest_control_gap_days: Optional[int] = None
    duplicate_of: str = ""
    data_quality_note: str = ""
    exclusion_reason: str = ""
    sf_eff: Optional[float] = None
    sf_time_day: Optional[float] = None
    sf_se: Optional[float] = None
    sf_ci_low: Optional[float] = None
    sf_ci_high: Optional[float] = None
    sf_bootstrap_success: int = 0
    aligned_time_points: int = 0

    @property
    def series_id(self) -> str:
        return self.experiment_date.isoformat() if self.experiment_date else "unknown"

    @property
    def year(self) -> str:
        return str(self.experiment_date.year) if self.experiment_date else "unknown"

    @property
    def dose_sum(self) -> float:
        return float(sum(self.fractions))

    @property
    def dose2_sum(self) -> float:
        return float(sum(value * value for value in self.fractions))

    @property
    def regimen(self) -> str:
        return "+".join(f"{value:g}" for value in self.fractions)

    @property
    def usable(self) -> bool:
        return not self.exclusion_reason and self.sf_eff is not None and self.sf_se is not None


@dataclass
class FitResult:
    model: str
    alpha: float
    beta: float
    n: int
    k: int
    log_likelihood: float
    aic: float
    aicc: float
    rmse_log: float
    rmse_sf: float
    r2_sf: float
    condition_number: float


@dataclass
class CVSummary:
    group_type: str
    model: str
    group_count: int
    valid_folds: int
    prediction_count: int
    coverage: float
    rmse_log: float
    rmse_sf: float
    r2_sf: float


def parse_date_from_name(path: Path) -> Optional[date]:
    match = DATE_RE.search(path.stem)
    if not match:
        return None
    day, month, year = (int(value) for value in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date_token(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%d_%m_%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date {value!r}; use YYYY-MM-DD or DD.MM.YYYY.")


def parse_dose_entries(params: Sequence[str]) -> tuple[tuple[str, float], ...]:
    """Parse only the actual dose, avoiding isotope mass numbers such as C12."""
    entries: list[tuple[str, float]] = []
    for raw_token in params:
        token = str(raw_token).strip()
        unit_match = DOSE_UNIT_RE.search(token)
        if not unit_match:
            continue
        before_unit = token[: unit_match.start()]
        if "=" in before_unit:
            beam_text, dose_text = before_unit.rsplit("=", 1)
        else:
            beam_text, dose_text = "", before_unit
        numbers = NUMBER_RE.findall(dose_text)
        if not numbers:
            continue
        beam_label = beam_text.strip().lower()
        for number in numbers:
            dose_value = float(number.replace(",", "."))
            if np.isfinite(dose_value) and dose_value > 0.0:
                entries.append((beam_label, dose_value))
    return tuple(entries)


def normalized_beam_kind(label: str) -> str:
    label = label.lower().replace("γ", "y").replace("у", "y")
    if re.search(r"(?:^|\W)n(?:\d|\W|$)", label):
        return "n"
    if re.search(r"(?:^|\W)p(?:\d|\W|$)", label):
        return "p"
    if "c12" in label or "с12" in label:
        return "c"
    if re.search(r"(?:^|\W)e(?:\W|$)", label):
        return "e"
    if re.search(r"(?:^|\W)y(?:\W|$)", label):
        return "y"
    return label.strip() or "unknown"


def analysis_radiation_family(path: Path) -> Optional[str]:
    """Apply the project rule: a proton file is through-beam unless peak is explicit."""
    family = infer_radiation_family(path)
    return "p_through" if family == "p" else family


def finite_column_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.empty(0, dtype=float)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=float)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def normalized_curve_and_sf(
    exp_times: np.ndarray,
    exp_mean: np.ndarray,
    control_times: np.ndarray,
    control_mean: np.ndarray,
) -> tuple[float, float, int]:
    """Return SF_eff, the day of its minimum, and aligned point count."""
    exp_times = np.asarray(exp_times, dtype=float)
    exp_mean = np.asarray(exp_mean, dtype=float)
    control_times = np.asarray(control_times, dtype=float)
    control_mean = np.asarray(control_mean, dtype=float)

    exp_len = min(len(exp_times), len(exp_mean))
    ctrl_len = min(len(control_times), len(control_mean))
    exp_times, exp_mean = exp_times[:exp_len], exp_mean[:exp_len]
    control_times, control_mean = control_times[:ctrl_len], control_mean[:ctrl_len]

    ctrl_mask = np.isfinite(control_times) & np.isfinite(control_mean) & (control_mean > 0.0)
    if ctrl_mask.sum() < 2:
        raise ValueError("Control curve has fewer than two finite positive points.")
    ctrl_x, unique_indices = np.unique(control_times[ctrl_mask], return_index=True)
    ctrl_y = control_mean[ctrl_mask][unique_indices]
    if len(ctrl_x) < 2:
        raise ValueError("Control curve has fewer than two distinct times.")

    mask = (
        np.isfinite(exp_times)
        & np.isfinite(exp_mean)
        & (exp_mean > 0.0)
        & (exp_times >= ctrl_x[0])
        & (exp_times <= ctrl_x[-1])
    )
    if mask.sum() < 2:
        raise ValueError("Experiment and control have fewer than two aligned time points.")
    times = exp_times[mask]
    treated = exp_mean[mask]
    control = np.interp(times, ctrl_x, ctrl_y)
    ratio = treated / control
    if not np.isfinite(ratio[0]) or ratio[0] <= 0.0:
        raise ValueError("The baseline treated/control ratio is invalid.")
    response = ratio / ratio[0]
    post_mask = np.isfinite(response) & (response > 0.0) & (times > times[0])
    if not post_mask.any():
        raise ValueError("No finite positive post-baseline response value.")
    post_indices = np.flatnonzero(post_mask)
    selected = int(post_indices[np.argmin(response[post_indices])])
    return float(response[selected]), float(times[selected] - times[0]), int(mask.sum())


def resample_rows(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("No animals available for resampling.")
    indices = rng.integers(0, values.shape[0], size=values.shape[0])
    return values[indices]


def materialize_sf(
    record: ExperimentRecord,
    rng: Optional[np.random.Generator] = None,
    control_mean: Optional[np.ndarray] = None,
) -> tuple[float, float, int]:
    if record.control is None:
        raise ValueError("No matched control.")
    exp_values = record.volumes if rng is None else resample_rows(record.volumes, rng)
    if control_mean is None:
        ctrl_values = record.control.volumes if rng is None else resample_rows(record.control.volumes, rng)
        control_mean = finite_column_mean(ctrl_values)
    return normalized_curve_and_sf(
        record.times,
        finite_column_mean(exp_values),
        record.control.times,
        control_mean,
    )


def data_fingerprint(record: ExperimentRecord) -> str:
    digest = hashlib.sha256()
    digest.update((record.family or "").encode("utf-8"))
    digest.update(np.asarray(record.fractions, dtype=np.float64).tobytes())
    digest.update(np.asarray(record.times, dtype=np.float64).tobytes())
    values = np.nan_to_num(record.volumes, nan=-9.87654321e307)
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    digest.update(str(record.volumes.shape).encode("ascii"))
    return digest.hexdigest()


def load_control_map(path: Optional[Path]) -> dict[str, str]:
    mapping = dict(DEFAULT_CONTROL_OVERRIDES)
    if path is None:
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        required = {"experiment_date", "control_file"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Control map needs columns experiment_date and control_file.")
        for row in reader:
            if not row.get("experiment_date") or not row.get("control_file"):
                continue
            mapping[parse_date_token(row["experiment_date"]).isoformat()] = row["control_file"].strip()
    return mapping


def load_inputs(
    exps_dir: Path,
    control_overrides: Mapping[str, str],
    control_policy: str,
) -> tuple[list[ExperimentRecord], dict[Path, ControlSeries]]:
    all_files = sorted(
        path for path in exps_dir.rglob("*.xlsx") if not path.name.startswith("~$")
    )
    control_files = [path for path in all_files if "controls" in {part.lower() for part in path.parts}]
    controls: dict[Path, ControlSeries] = {}
    controls_by_date: dict[date, ControlSeries] = {}
    controls_by_name: dict[str, ControlSeries] = {}
    for path in control_files:
        _, time_labels, _, volumes = process_tumor_data_excel(str(path))
        control = ControlSeries(
            path=path.resolve(),
            experiment_date=parse_date_from_name(path),
            times=np.asarray(parse_time_days(time_labels), dtype=float),
            volumes=np.asarray(volumes, dtype=float),
        )
        controls[control.path] = control
        controls_by_name[path.name.lower()] = control
        if control.experiment_date is not None:
            controls_by_date[control.experiment_date] = control

    records: list[ExperimentRecord] = []
    for path in all_files:
        if path in control_files:
            continue
        relative = str(path.relative_to(exps_dir))
        record = ExperimentRecord(path=path.resolve(), relative_path=relative)
        if "skin" in {part.lower() for part in path.parts} or "skin" in path.stem.lower():
            record.exclusion_reason = "skin_endpoint_not_tumor_volume"
            records.append(record)
            continue
        try:
            params, time_labels, _, volumes = process_tumor_data_excel(str(path))
            entries = parse_dose_entries(params)
            record.experiment_date = parse_date_from_name(path)
            record.family = analysis_radiation_family(path)
            record.fractions = tuple(dose for _, dose in entries)
            record.beam_labels = tuple(normalized_beam_kind(label) for label, _ in entries)
            record.times = np.asarray(parse_time_days(time_labels), dtype=float)
            record.volumes = np.asarray(volumes, dtype=float)
            record.n_animals = int(record.volumes.shape[0]) if record.volumes.ndim == 2 else 0
        except Exception as exc:
            record.exclusion_reason = f"parse_error:{type(exc).__name__}:{exc}"
            records.append(record)
            continue

        if record.experiment_date is not None:
            override_name = control_overrides.get(record.experiment_date.isoformat())
            if override_name:
                record.control = controls_by_name.get(override_name.lower())
                record.control_match = "explicit_date_map"
                if record.control is None:
                    record.exclusion_reason = f"mapped_control_not_found:{override_name}"
            elif record.experiment_date in controls_by_date:
                record.control = controls_by_date[record.experiment_date]
                record.control_match = "same_date"

            dated_controls = [control for control in controls.values() if control.experiment_date]
            if dated_controls:
                nearest = min(
                    dated_controls,
                    key=lambda control: abs((control.experiment_date - record.experiment_date).days),  # type: ignore[operator]
                )
                record.nearest_control = nearest.path.name
                record.nearest_control_gap_days = abs(
                    (nearest.experiment_date - record.experiment_date).days  # type: ignore[operator]
                )
                if record.control is None and control_policy == "nearest":
                    record.control = nearest
                    record.control_match = "nearest_available_proxy"
                    record.data_quality_note = "nearest_date_proxy_control"

        if not record.exclusion_reason:
            if record.experiment_date is None:
                record.exclusion_reason = "experiment_date_not_parsed"
            elif record.control is None:
                record.exclusion_reason = "no_date_or_explicit_control"
            elif not record.fractions:
                record.exclusion_reason = "dose_not_parsed"
            elif record.family is None:
                record.exclusion_reason = "radiation_family_not_inferred"
            elif len(set(record.beam_labels) - {"unknown"}) > 1:
                record.exclusion_reason = "mixed_radiation_field"
            elif record.volumes.ndim != 2 or record.volumes.shape[0] < 2:
                record.exclusion_reason = "fewer_than_two_animals"
            elif record.volumes.shape[1] < 2:
                record.exclusion_reason = "fewer_than_two_time_points"
        records.append(record)

    fingerprint_groups: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        if record.volumes.size == 0 or not record.fractions:
            continue
        fingerprint = data_fingerprint(record)
        fingerprint_groups.setdefault(fingerprint, []).append(record)

    for group in fingerprint_groups.values():
        if len(group) < 2:
            continue
        dates = {record.experiment_date for record in group}
        if len(dates) > 1:
            preferred = [
                record for record in group
                if record.path.name in PREFERRED_DATE_CONFLICT_FILES
            ]
            if len(preferred) == 1:
                canonical = preferred[0]
                canonical.data_quality_note = "user_selected_from_duplicate_date_conflict"
                for record in group:
                    if record is canonical:
                        continue
                    record.duplicate_of = canonical.relative_path
                    record.exclusion_reason = "exact_duplicate_file_user_resolved_date_conflict"
            else:
                canonical = min(group, key=lambda record: record.relative_path)
                for record in group:
                    record.exclusion_reason = "exact_duplicate_conflicting_dates"
                    if record is not canonical:
                        record.duplicate_of = canonical.relative_path
        else:
            canonical = min(
                group,
                key=lambda record: (
                    record.control is None,
                    bool(record.exclusion_reason),
                    record.relative_path,
                ),
            )
            for record in group:
                if record is canonical:
                    continue
                record.duplicate_of = canonical.relative_path
                record.exclusion_reason = "exact_duplicate_file"
    return records, controls


def estimate_sf_uncertainty(
    records: Sequence[ExperimentRecord],
    iterations: int,
    rng: np.random.Generator,
) -> None:
    for record in records:
        if record.exclusion_reason:
            continue
        try:
            record.sf_eff, record.sf_time_day, record.aligned_time_points = materialize_sf(record)
        except Exception as exc:
            record.exclusion_reason = f"sf_calculation_failed:{type(exc).__name__}:{exc}"
            continue
        bootstrap_values: list[float] = []
        for _ in range(iterations):
            try:
                value, _, _ = materialize_sf(record, rng)
            except (ValueError, FloatingPointError):
                continue
            if np.isfinite(value) and value > 0.0:
                bootstrap_values.append(value)
        record.sf_bootstrap_success = len(bootstrap_values)
        if len(bootstrap_values) < max(50, math.ceil(iterations * 0.8)):
            record.exclusion_reason = "insufficient_animal_bootstrap_success"
            continue
        values = np.asarray(bootstrap_values, dtype=float)
        record.sf_se = float(np.std(values, ddof=1))
        record.sf_ci_low, record.sf_ci_high = (
            float(value) for value in np.quantile(values, [0.025, 0.975])
        )


def fit_model(
    records: Sequence[ExperimentRecord],
    model: str,
    sigma_log_floor: float,
) -> FitResult:
    if not records:
        raise ValueError("No observations for fitting.")
    sf = np.asarray([record.sf_eff for record in records], dtype=float)
    dose = np.asarray([record.dose_sum for record in records], dtype=float)
    dose2 = np.asarray([record.dose2_sum for record in records], dtype=float)
    sigma_sf = np.asarray([record.sf_se for record in records], dtype=float)
    if np.any(~np.isfinite(sf)) or np.any(sf <= 0.0):
        raise ValueError("SF_eff must be finite and positive.")
    sigma_log = np.maximum(sigma_sf / sf, sigma_log_floor)
    y = -np.log(sf)
    design = dose[:, None] if model == "linear" else np.column_stack((dose, dose2))
    k = design.shape[1]
    if len(records) < k + 1 or np.linalg.matrix_rank(design) < k:
        raise ValueError(f"The {model} design is not identifiable.")
    weighted_design = design / sigma_log[:, None]
    weighted_y = y / sigma_log
    solution = lsq_linear(weighted_design, weighted_y, bounds=(0.0, np.inf), method="trf")
    if not solution.success:
        raise RuntimeError(solution.message)
    params = solution.x
    alpha = float(params[0])
    beta = float(params[1]) if model == "lq" else 0.0
    predicted_y = design @ params
    predicted_sf = np.exp(-predicted_y)
    residual_y = y - predicted_y
    standardized = residual_y / sigma_log
    log_likelihood = float(
        -0.5 * np.sum(standardized * standardized + np.log(2.0 * np.pi * sigma_log * sigma_log))
    )
    aic = float(2 * k - 2.0 * log_likelihood)
    denominator = len(records) - k - 1
    aicc = float(aic + (2.0 * k * (k + 1)) / denominator) if denominator > 0 else math.inf
    sf_mean = float(np.mean(sf))
    sf_sst = float(np.sum((sf - sf_mean) ** 2))
    sf_sse = float(np.sum((sf - predicted_sf) ** 2))
    r2_sf = 1.0 - sf_sse / sf_sst if sf_sst > 0.0 else math.nan
    condition_number = float(np.linalg.cond(weighted_design))
    return FitResult(
        model=model,
        alpha=alpha,
        beta=beta,
        n=len(records),
        k=k,
        log_likelihood=log_likelihood,
        aic=aic,
        aicc=aicc,
        rmse_log=float(np.sqrt(np.mean(residual_y * residual_y))),
        rmse_sf=float(np.sqrt(np.mean((sf - predicted_sf) ** 2))),
        r2_sf=float(r2_sf),
        condition_number=condition_number,
    )


def predict_sf(records: Sequence[ExperimentRecord], fit: FitResult) -> np.ndarray:
    dose = np.asarray([record.dose_sum for record in records], dtype=float)
    dose2 = np.asarray([record.dose2_sum for record in records], dtype=float)
    return np.exp(-(fit.alpha * dose + fit.beta * dose2))


def grouped_cross_validation(
    records: Sequence[ExperimentRecord],
    family: str,
    model: str,
    group_type: str,
    sigma_log_floor: float,
) -> tuple[CVSummary, list[dict[str, object]]]:
    group_value = (lambda record: record.series_id) if group_type == "series" else (lambda record: record.year)
    groups = sorted({group_value(record) for record in records})
    actual_all: list[float] = []
    predicted_all: list[float] = []
    fold_rows: list[dict[str, object]] = []
    valid_folds = 0
    for group in groups:
        train = [record for record in records if group_value(record) != group]
        test = [record for record in records if group_value(record) == group]
        status = "ok"
        try:
            fit = fit_model(train, model, sigma_log_floor)
            predicted = predict_sf(test, fit)
        except (ValueError, RuntimeError) as exc:
            status = f"failed:{type(exc).__name__}"
            predicted = np.empty(0, dtype=float)
        if len(predicted) == len(test):
            valid_folds += 1
            actual = np.asarray([record.sf_eff for record in test], dtype=float)
            actual_all.extend(actual.tolist())
            predicted_all.extend(predicted.tolist())
            rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        else:
            rmse = math.nan
        fold_rows.append(
            {
                "row_type": "fold",
                "family": family,
                "model": model,
                "group_type": group_type,
                "held_out_group": group,
                "n_train": len(train),
                "n_test": len(test),
                "status": status,
                "rmse_sf": rmse,
            }
        )
    actual_array = np.asarray(actual_all, dtype=float)
    predicted_array = np.asarray(predicted_all, dtype=float)
    if len(actual_array):
        residual = actual_array - predicted_array
        rmse_sf = float(np.sqrt(np.mean(residual * residual)))
        rmse_log = float(np.sqrt(np.mean((np.log(actual_array) - np.log(predicted_array)) ** 2)))
        sst = float(np.sum((actual_array - np.mean(actual_array)) ** 2))
        r2_sf = 1.0 - float(np.sum(residual * residual)) / sst if sst > 0.0 else math.nan
    else:
        rmse_sf = rmse_log = r2_sf = math.nan
    summary = CVSummary(
        group_type=group_type,
        model=model,
        group_count=len(groups),
        valid_folds=valid_folds,
        prediction_count=len(actual_array),
        coverage=len(actual_array) / len(records) if records else 0.0,
        rmse_log=rmse_log,
        rmse_sf=rmse_sf,
        r2_sf=r2_sf,
    )
    fold_rows.insert(
        0,
        {
            "row_type": "aggregate",
            "family": family,
            "model": model,
            "group_type": group_type,
            "held_out_group": "ALL",
            "n_train": "",
            "n_test": len(records),
            "status": "ok" if valid_folds == len(groups) and groups else "incomplete",
            "rmse_sf": rmse_sf,
            "rmse_log": rmse_log,
            "r2_sf": r2_sf,
            "group_count": len(groups),
            "valid_folds": valid_folds,
            "prediction_count": len(actual_array),
            "coverage": summary.coverage,
        },
    )
    return summary, fold_rows


def hierarchical_bootstrap(
    records: Sequence[ExperimentRecord],
    family: str,
    iterations: int,
    sigma_log_floor: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    by_series: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        by_series.setdefault(record.series_id, []).append(record)
    series_ids = sorted(by_series)
    output: list[dict[str, object]] = []
    for iteration in range(iterations):
        selected = rng.choice(series_ids, size=len(series_ids), replace=True).tolist()
        sampled_records: list[ExperimentRecord] = []
        for series_id in selected:
            control_means: dict[Path, np.ndarray] = {}
            for original in by_series[series_id]:
                if original.control is None:
                    continue
                if original.control.path not in control_means:
                    control_means[original.control.path] = finite_column_mean(
                        resample_rows(original.control.volumes, rng)
                    )
                try:
                    sf_value, _, _ = materialize_sf(
                        original,
                        rng,
                        control_mean=control_means[original.control.path],
                    )
                except (ValueError, FloatingPointError):
                    continue
                if not np.isfinite(sf_value) or sf_value <= 0.0:
                    continue
                sampled = ExperimentRecord(
                    path=original.path,
                    relative_path=original.relative_path,
                    experiment_date=original.experiment_date,
                    family=original.family,
                    fractions=original.fractions,
                    sf_eff=sf_value,
                    sf_se=original.sf_se,
                )
                sampled_records.append(sampled)
        for model in ("linear", "lq"):
            try:
                fit = fit_model(sampled_records, model, sigma_log_floor)
            except (ValueError, RuntimeError):
                continue
            output.append(
                {
                    "iteration": iteration,
                    "family": family,
                    "model": model,
                    "alpha_eff": fit.alpha,
                    "beta_eff": fit.beta,
                    "alpha_beta_ratio_gy": fit.alpha / fit.beta if fit.beta > 1.0e-12 else math.nan,
                    "n_materialized": len(sampled_records),
                    "series_draw": ",".join(selected),
                }
            )
    return output


def quantiles(
    rows: Sequence[dict[str, object]],
    model: str,
    column: str,
) -> tuple[float, float, float]:
    values = np.asarray(
        [float(row[column]) for row in rows if row["model"] == model and np.isfinite(float(row[column]))],
        dtype=float,
    )
    if not len(values):
        return math.nan, math.nan, math.nan
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.5, 0.975]))  # type: ignore[return-value]


def finite_or_blank(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_blank(row.get(key, "")) for key in fieldnames})


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}g}"


def build_report(
    records: Sequence[ExperimentRecord],
    controls: Mapping[Path, ControlSeries],
    summaries: Sequence[dict[str, object]],
    animal_bootstrap_iterations: int,
    hierarchical_bootstrap_iterations: int,
    seed: int,
    control_policy: str,
) -> str:
    tumor_records = [record for record in records if "skin_endpoint" not in record.exclusion_reason]
    usable = [record for record in records if record.usable]
    unmatched = [record for record in tumor_records if record.exclusion_reason == "no_date_or_explicit_control"]
    redundant_duplicates = [record for record in records if record.duplicate_of]
    conflicting_duplicate_files = [
        record for record in records
        if record.exclusion_reason == "exact_duplicate_conflicting_dates"
    ]
    resolved_duplicate_conflicts = [
        record for record in records
        if record.data_quality_note == "user_selected_from_duplicate_date_conflict"
    ]
    proxy_control_records = [
        record for record in usable
        if record.control_match == "nearest_available_proxy"
    ]
    proxy_gaps = [
        record.nearest_control_gap_days for record in proxy_control_records
        if record.nearest_control_gap_days is not None
    ]
    if control_policy == "nearest":
        control_policy_text = (
            "По решению автора при отсутствии согласованного контроля каждому такому эксперименту назначен "
            "доступный control-файл с минимальной абсолютной календарной разницей. Это суррогатное историческое "
            "сопоставление; имя контроля и разрыв в днях сохранены для каждой строки."
        )
    else:
        control_policy_text = (
            "В строгом режиме использованы только контроли с той же датой серии и два явно заданных "
            "в предыдущей карте соответствия."
        )
    lines = [
        "# Пересчёт эффективных параметров α и β",
        "",
        f"Дата расчёта: {date.today().isoformat()}.",
        "",
        "## Что рассчитано",
        "",
        "Отклик обозначен как эффективный объёмный показатель in vivo, а не как клоногенная выживаемость:",
        "",
        r"\[SF_\mathrm{eff}=\min_{t>0}\frac{V_\mathrm{exp}(t)/V_\mathrm{exp}(0)}{V_\mathrm{ctrl}(t)/V_\mathrm{ctrl}(0)}.\]",
        "",
        "Соответственно, параметры ниже — **α_eff** и **β_eff**. Их нельзя напрямую переносить в расчёт клинического TCP.",
        "",
        f"Просканировано опухолевых файлов: {len(tumor_records)}; контролей: {len(controls)}; "
        f"в расчёт вошло: {len(usable)}; без контроля, сопоставленного по дате или явной карте: {len(unmatched)}; "
        f"с ближайшим суррогатным контролем: {len(proxy_control_records)}; "
        f"лишних точных файлов-дубликатов: {len(redundant_duplicates)}; "
        f"неразрешённых файлов в конфликте одинаковых данных и разных дат: {len(conflicting_duplicate_files)}; "
        f"разрешённых пользователем конфликтов дат: {len(resolved_duplicate_conflicts)}.",
        "",
        control_policy_text,
        "" if not proxy_gaps else f"Диапазон календарных разрывов для суррогатных контролей: {min(proxy_gaps)}–{max(proxy_gaps)} дней.",
        "",
        "## Результаты моделей",
        "",
        "| Семейство | n / серий (всего файлов) | Выбрана модель | α_eff, Гр⁻¹ (95% ДИ) | β_eff, Гр⁻² (95% ДИ) | α/β, Гр | AICc | CV по сериям R² | Вывод |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        selected_model = row.get("selected_model")
        if selected_model == "not_fitted":
            alpha_text = "—"
            beta_text = "—"
            model_text = "не оценивалась"
        else:
            alpha_text = f"{fmt(row.get('alpha_eff'))} ({fmt(row.get('alpha_ci_low'))}; {fmt(row.get('alpha_ci_high'))})"
            model_text = "LQ" if selected_model == "lq" else "линейная"
        if selected_model == "lq":
            beta_text = f"{fmt(row.get('beta_eff'))} ({fmt(row.get('beta_ci_low'))}; {fmt(row.get('beta_ci_high'))})"
        elif selected_model != "not_fitted":
            beta_text = "0 (линейная модель)"
        conclusion = "можно обсуждать как эффективную LQ-оценку" if row.get("publish_alpha_beta") else str(row.get("recommendation", "не публиковать α/β"))
        lines.append(
            "| {family} | {n} / {series} | {model} | {alpha} | {beta} | {ratio} | {aicc} | {cv} | {conclusion} |".format(
                family=FAMILY_LABELS_RU.get(str(row.get("family")), str(row.get("family"))),
                n=row.get("n_experiments"),
                series=f"{row.get('n_series')} ({row.get('n_files_total')})",
                model=model_text,
                alpha=alpha_text,
                beta=beta_text,
                ratio=fmt(row.get("alpha_beta_ratio_gy")),
                aicc=fmt(row.get("selected_aicc")),
                cv=fmt(row.get("series_cv_r2_sf")),
                conclusion=conclusion,
            )
        )
    if not summaries:
        lines.append("| — | — | — | — | — | — | — | — | Недостаточно сопоставленных данных |")
    lines.extend(
        [
            "",
            "## Правила выбора и ограничения",
            "",
            "- Линейная и LQ-модели подогнаны во взвешенной шкале `-ln(SF_eff)` с ограничениями α_eff ≥ 0 и β_eff ≥ 0.",
            "- Вес каждого файла получен из bootstrap животных экспериментальной и контрольной групп; минимальная стандартная ошибка в лог-шкале равна 0,05.",
            "- Для сравнения моделей использован AICc. LQ выбирается только при преимуществе AICc более 2 и положительной нижней границе bootstrap-интервала β_eff.",
            "- Кросс-валидация исключает целиком дату-серию, а отдельно — год. Повторные режимы остаются отдельными наблюдениями.",
            "- Иерархический bootstrap сначала повторно выбирает серии, затем животных внутри экспериментальной и контрольной групп. Контрольная выборка общая для файлов одной серии.",
            "- Углерод, семейства с недостаточным числом точек, отрицательный прогнозный R² и интервалы β_eff, включающие ноль, автоматически получают запрет на публикацию α/β.",
            "- Назначение ближайшего доступного контроля позволяет получить расчётные оценки для всех семейств, но само по себе не превращает исторический контроль в биологически согласованный.",
            "- Межгодовые двухточечные «изоэффектные» пары в расчёт не входят.",
            "- Для одинаковых данных под датами 03.05.2018 и 16.05.2018 по решению пользователя оставлен файл `16.05.2018_y32.xlsx`, поскольку ему соответствует контроль той же даты; копия от 03.05 исключена.",
            "",
            "## Важное исправление исходных доз",
            "",
            "Доза берётся только справа от знака `=` до единицы `Гр`. Поэтому обозначение изотопа `C12` больше не создаёт ложную дополнительную фракцию 12 Гр. Например, `C12 = 34 Гр` теперь корректно интерпретируется как одна фракция 34 Гр.",
            "",
            "## Что нужно уточнить для расширения анализа",
            "",
            "Других контрольных групп в наборе нет, поэтому для закрытия расчётной задачи использован ближайший доступный контроль. В тексте диссертации эти оценки следует называть исследовательскими α_eff и β_eff, полученными с суррогатными историческими контролями.",
            "",
            "## Воспроизводимость",
            "",
            f"Bootstrap SF_eff: {animal_bootstrap_iterations} итераций на файл; иерархический bootstrap: {hierarchical_bootstrap_iterations} итераций на семейство; seed: {seed}.",
            "",
            "Полная трассировка решений находится в `experiment_sf_eff.csv`, кандидаты моделей — в `model_fits.csv`, результаты групповой проверки — в `grouped_cv.csv`, bootstrap-реализации — в `bootstrap_parameters.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    exps_dir = args.exps_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    control_map = load_control_map(args.control_map)
    rng = np.random.default_rng(args.seed)
    records, controls = load_inputs(exps_dir, control_map, args.control_policy)
    estimate_sf_uncertainty(records, args.animal_bootstrap, rng)

    experiment_rows: list[dict[str, object]] = []
    for record in records:
        experiment_rows.append(
            {
                "relative_path": record.relative_path,
                "date": record.experiment_date.isoformat() if record.experiment_date else "",
                "year": record.year,
                "series_id": record.series_id,
                "family": record.family or "",
                "regimen_gy": record.regimen,
                "dose_sum_gy": record.dose_sum if record.fractions else "",
                "dose2_sum_gy2": record.dose2_sum if record.fractions else "",
                "n_animals": record.n_animals,
                "control_file": record.control.path.name if record.control else "",
                "control_match": record.control_match,
                "nearest_control": record.nearest_control,
                "nearest_control_gap_days": record.nearest_control_gap_days,
                "duplicate_of": record.duplicate_of,
                "data_quality_note": record.data_quality_note,
                "included_analysis": int(record.usable),
                "included_strict": int(
                    record.usable and record.control_match in {"same_date", "explicit_date_map"}
                ),
                "exclusion_reason": record.exclusion_reason,
                "sf_eff": record.sf_eff,
                "sf_time_day": record.sf_time_day,
                "sf_se": record.sf_se,
                "sf_ci_low": record.sf_ci_low,
                "sf_ci_high": record.sf_ci_high,
                "sf_bootstrap_success": record.sf_bootstrap_success,
                "aligned_time_points": record.aligned_time_points,
            }
        )
    experiment_fields = list(experiment_rows[0]) if experiment_rows else []
    write_csv(output_dir / "experiment_sf_eff.csv", experiment_rows, experiment_fields)

    usable_by_family: dict[str, list[ExperimentRecord]] = {}
    observed_by_family: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        if record.family:
            observed_by_family.setdefault(record.family, []).append(record)
        if record.usable and record.family:
            usable_by_family.setdefault(record.family, []).append(record)

    model_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    all_bootstrap_rows: list[dict[str, object]] = []
    for family in sorted(observed_by_family):
        family_records = usable_by_family.get(family, [])
        family_all_records = observed_by_family[family]
        n_unmatched = sum(
            record.exclusion_reason == "no_date_or_explicit_control"
            for record in family_all_records
        )
        n_proxy_controls = sum(
            record.control_match == "nearest_available_proxy"
            for record in family_records
        )
        proxy_gaps = [
            record.nearest_control_gap_days for record in family_records
            if record.control_match == "nearest_available_proxy"
            and record.nearest_control_gap_days is not None
        ]
        if len(family_records) < 2:
            blockers = ["менее двух экспериментов с формально сопоставленным контролем"]
            if n_unmatched:
                blockers.append(f"без контроля по дате или явной карте: {n_unmatched}")
            if family == "p_through" and len(family_all_records) < 3:
                blockers.append("для прострельных протонов всего две исходные точки")
            if family == "c":
                blockers.append("для углерода недостаточна вариативность доз и схем")
            summary_rows.append(
                {
                    "family": family,
                    "n_files_total": len(family_all_records),
                    "n_unmatched_control": n_unmatched,
                    "n_proxy_controls": n_proxy_controls,
                    "max_proxy_control_gap_days": max(proxy_gaps) if proxy_gaps else math.nan,
                    "n_experiments": len(family_records),
                    "n_series": len({record.series_id for record in family_records}),
                    "n_years": len({record.year for record in family_records}),
                    "n_distinct_regimens": len({record.fractions for record in family_records}),
                    "selected_model": "not_fitted",
                    "selection_reason": "insufficient_matched_data",
                    "alpha_eff": math.nan,
                    "alpha_ci_low": math.nan,
                    "alpha_ci_high": math.nan,
                    "beta_eff": math.nan,
                    "beta_ci_low": math.nan,
                    "beta_ci_high": math.nan,
                    "alpha_beta_ratio_gy": math.nan,
                    "alpha_beta_ratio_ci_low": math.nan,
                    "alpha_beta_ratio_ci_high": math.nan,
                    "selected_aicc": math.nan,
                    "series_cv_r2_sf": math.nan,
                    "series_cv_rmse_sf": math.nan,
                    "series_cv_coverage": 0.0,
                    "year_cv_r2_sf": math.nan,
                    "year_cv_rmse_sf": math.nan,
                    "publish_alpha_beta": 0,
                    "recommendation": "; ".join(blockers),
                }
            )
            continue
        fits: dict[str, FitResult] = {}
        cv_summaries: dict[tuple[str, str], CVSummary] = {}
        for model in ("linear", "lq"):
            try:
                fits[model] = fit_model(family_records, model, args.sigma_log_floor)
            except (ValueError, RuntimeError):
                continue
            for group_type in ("series", "year"):
                cv_summary, rows = grouped_cross_validation(
                    family_records,
                    family,
                    model,
                    group_type,
                    args.sigma_log_floor,
                )
                cv_summaries[(model, group_type)] = cv_summary
                cv_rows.extend(rows)

        bootstrap_rows = hierarchical_bootstrap(
            family_records,
            family,
            args.hierarchical_bootstrap,
            args.sigma_log_floor,
            rng,
        )
        all_bootstrap_rows.extend(bootstrap_rows)
        linear_alpha_ci = quantiles(bootstrap_rows, "linear", "alpha_eff")
        lq_alpha_ci = quantiles(bootstrap_rows, "lq", "alpha_eff")
        lq_beta_ci = quantiles(bootstrap_rows, "lq", "beta_eff")
        lq_ratio_ci = quantiles(bootstrap_rows, "lq", "alpha_beta_ratio_gy")
        linear_success = sum(row["model"] == "linear" for row in bootstrap_rows)
        lq_success = sum(row["model"] == "lq" for row in bootstrap_rows)

        selected_model = "linear"
        selection_reason = "linear_fallback"
        if "linear" in fits and "lq" in fits:
            lq_supported = (
                np.isfinite(fits["lq"].aicc)
                and fits["lq"].aicc + 2.0 < fits["linear"].aicc
                and fits["lq"].beta > 1.0e-12
                and np.isfinite(lq_beta_ci[0])
                and lq_beta_ci[0] > 1.0e-12
                and lq_success >= math.ceil(args.hierarchical_bootstrap * 0.8)
            )
            if lq_supported:
                selected_model = "lq"
                selection_reason = "lq_aicc_and_bootstrap_supported"
            elif lq_beta_ci[0] <= 1.0e-12 or fits["lq"].beta <= 1.0e-12:
                selection_reason = "beta_interval_includes_zero_or_boundary"
            elif not np.isfinite(fits["lq"].aicc) or fits["lq"].aicc + 2.0 >= fits["linear"].aicc:
                selection_reason = "aicc_prefers_linear"
            else:
                selection_reason = "lq_bootstrap_unstable"
        elif "linear" not in fits:
            continue

        selected_fit = fits[selected_model]
        selected_alpha_ci = lq_alpha_ci if selected_model == "lq" else linear_alpha_ci
        series_cv = cv_summaries.get((selected_model, "series"))
        year_cv = cv_summaries.get((selected_model, "year"))
        n_series = len({record.series_id for record in family_records})
        n_years = len({record.year for record in family_records})
        n_regimens = len({tuple(round(value, 8) for value in record.fractions) for record in family_records})

        blockers: list[str] = []
        if selected_model != "lq":
            blockers.append("выбрана линейная модель; α/β не определяется")
        if family == "p_through" and len(family_records) < 3:
            blockers.append("для прострельных протонов слишком мало точек")
        if family == "c":
            blockers.append("для углерода недостаточна вариативность доз и схем")
        if n_proxy_controls:
            blockers.append(
                f"суррогатный ближайший контроль использован для {n_proxy_controls}/{len(family_records)} точек"
            )
        if n_series < 3:
            blockers.append("менее трёх независимых серий")
        if n_regimens < 3:
            blockers.append("менее трёх различных режимов")
        if series_cv is None or not np.isfinite(series_cv.r2_sf) or series_cv.r2_sf <= 0.0:
            blockers.append("прогнозный R² по сериям не положителен")
        elif series_cv.coverage < 0.8:
            blockers.append("неполное покрытие групповой кросс-валидации")
        if selected_model == "lq" and (not np.isfinite(lq_beta_ci[0]) or lq_beta_ci[0] <= 1.0e-12):
            blockers.append("95% bootstrap-интервал β_eff включает ноль")
        if selected_fit.condition_number > 1.0e4:
            blockers.append("плохо обусловленный дизайн")
        if selected_model == "lq" and selected_fit.alpha <= 1.0e-8:
            blockers.append("α_eff находится на границе неотрицательного ограничения")
        publish_alpha_beta = selected_model == "lq" and not blockers
        recommendation = "; ".join(dict.fromkeys(blockers)) or "допустима осторожная эффективная LQ-интерпретация"

        for model, fit in fits.items():
            alpha_ci = lq_alpha_ci if model == "lq" else linear_alpha_ci
            beta_ci = lq_beta_ci if model == "lq" else (0.0, 0.0, 0.0)
            model_cv = cv_summaries.get((model, "series"))
            model_rows.append(
                {
                    "family": family,
                    "model": model,
                    "selected": int(model == selected_model),
                    "selection_reason": selection_reason if model == selected_model else "",
                    "n_experiments": len(family_records),
                    "n_series": n_series,
                    "n_years": n_years,
                    "n_distinct_regimens": n_regimens,
                    "alpha_eff": fit.alpha,
                    "alpha_ci_low": alpha_ci[0],
                    "alpha_ci_high": alpha_ci[2],
                    "beta_eff": fit.beta,
                    "beta_ci_low": beta_ci[0],
                    "beta_ci_high": beta_ci[2],
                    "alpha_beta_ratio_gy": fit.alpha / fit.beta if model == "lq" and fit.beta > 1.0e-12 else math.nan,
                    "aic": fit.aic,
                    "aicc": fit.aicc,
                    "rmse_log": fit.rmse_log,
                    "rmse_sf": fit.rmse_sf,
                    "r2_sf": fit.r2_sf,
                    "condition_number": fit.condition_number,
                    "series_cv_r2_sf": model_cv.r2_sf if model_cv else math.nan,
                    "series_cv_rmse_sf": model_cv.rmse_sf if model_cv else math.nan,
                    "series_cv_coverage": model_cv.coverage if model_cv else 0.0,
                    "bootstrap_success": lq_success if model == "lq" else linear_success,
                }
            )

        summary_rows.append(
                {
                    "family": family,
                    "n_files_total": len(family_all_records),
                    "n_unmatched_control": n_unmatched,
                    "n_proxy_controls": n_proxy_controls,
                    "max_proxy_control_gap_days": max(proxy_gaps) if proxy_gaps else math.nan,
                    "n_experiments": len(family_records),
                "n_series": n_series,
                "n_years": n_years,
                "n_distinct_regimens": n_regimens,
                "selected_model": selected_model,
                "selection_reason": selection_reason,
                "alpha_eff": selected_fit.alpha,
                "alpha_ci_low": selected_alpha_ci[0],
                "alpha_ci_high": selected_alpha_ci[2],
                "beta_eff": selected_fit.beta,
                "beta_ci_low": lq_beta_ci[0] if selected_model == "lq" else 0.0,
                "beta_ci_high": lq_beta_ci[2] if selected_model == "lq" else 0.0,
                "alpha_beta_ratio_gy": selected_fit.alpha / selected_fit.beta if selected_model == "lq" and selected_fit.beta > 1.0e-12 else math.nan,
                "alpha_beta_ratio_ci_low": lq_ratio_ci[0] if selected_model == "lq" else math.nan,
                "alpha_beta_ratio_ci_high": lq_ratio_ci[2] if selected_model == "lq" else math.nan,
                "selected_aicc": selected_fit.aicc,
                "series_cv_r2_sf": series_cv.r2_sf if series_cv else math.nan,
                "series_cv_rmse_sf": series_cv.rmse_sf if series_cv else math.nan,
                "series_cv_coverage": series_cv.coverage if series_cv else 0.0,
                "year_cv_r2_sf": year_cv.r2_sf if year_cv else math.nan,
                "year_cv_rmse_sf": year_cv.rmse_sf if year_cv else math.nan,
                "publish_alpha_beta": int(publish_alpha_beta),
                "recommendation": recommendation,
            }
        )

    model_fields = list(model_rows[0]) if model_rows else ["family", "model"]
    summary_fields = list(summary_rows[0]) if summary_rows else ["family", "selected_model"]
    cv_fields = [
        "row_type", "family", "model", "group_type", "held_out_group", "n_train", "n_test",
        "status", "rmse_sf", "rmse_log", "r2_sf", "group_count", "valid_folds",
        "prediction_count", "coverage",
    ]
    bootstrap_fields = [
        "iteration", "family", "model", "alpha_eff", "beta_eff", "alpha_beta_ratio_gy",
        "n_materialized", "series_draw",
    ]
    write_csv(output_dir / "model_fits.csv", model_rows, model_fields)
    write_csv(output_dir / "model_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "grouped_cv.csv", cv_rows, cv_fields)
    write_csv(output_dir / "bootstrap_parameters.csv", all_bootstrap_rows, bootstrap_fields)

    config = {
        "exps_dir": str(exps_dir),
        "output_dir": str(output_dir),
        "animal_bootstrap_iterations": args.animal_bootstrap,
        "hierarchical_bootstrap_iterations": args.hierarchical_bootstrap,
        "sigma_log_floor": args.sigma_log_floor,
        "seed": args.seed,
        "control_map": control_map,
        "control_policy": args.control_policy,
        "preferred_date_conflict_files": sorted(PREFERRED_DATE_CONFLICT_FILES),
        "proton_family_rule": "proton files without explicit peak marker are p_through",
        "sf_eff_definition": "min_t>0 [(Vexp(t)/Vexp(0))/(Vctrl(t)/Vctrl(0))]",
        "primary_control_policy": (
            "same date or explicit map preferred; otherwise nearest available calendar-date proxy"
            if args.control_policy == "nearest"
            else "same date or explicit map only"
        ),
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(
        records,
        controls,
        summary_rows,
        args.animal_bootstrap,
        args.hierarchical_bootstrap,
        args.seed,
        args.control_policy,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Scanned experiment files: {len(records)}")
    print(f"Usable experiments: {sum(record.usable for record in records)}")
    print(f"Families fitted: {sum(row['selected_model'] != 'not_fitted' for row in summary_rows)}")
    print(f"Results: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exps-dir", type=Path, default=DEFAULT_EXPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--control-map",
        type=Path,
        default=None,
        help="Optional CSV with experiment_date and control_file columns.",
    )
    parser.add_argument("--animal-bootstrap", type=int, default=500)
    parser.add_argument("--hierarchical-bootstrap", type=int, default=1000)
    parser.add_argument("--sigma-log-floor", type=float, default=0.05)
    parser.add_argument(
        "--control-policy",
        choices=("strict", "nearest"),
        default="nearest",
        help="Use only date/explicit matches or assign the nearest available dated control.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
