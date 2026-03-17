# coding: utf-8
"""
Fit alpha and beta for the LQ model using tumor-volume Excel files.

Workflow:
1. Read all `.xlsx` files from the current working directory or only files
   passed via `--files`.
2. Build an average control curve from files whose name contains `control`.
3. Normalize each experimental mean volume curve by the control curve.
4. Compute surviving fraction (SF) from the normalized curve:
   - `absolute`: `min(mean_norm[1:]) / mean_norm[0]`
   - `absindex:N`: `mean_norm[N] / mean_norm[0]`
5. Fit the LQ model `SF = exp(-alpha * D - beta * sum(d_i^2))`.

The script can be started from CLI or by setting inline parameters below.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from work_with_prepared_data.radiobioligy_project.data_processing.data_processing import (
    TumorDataProcessor,
)
from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (
    process_tumor_data_excel,
)

RegimenKind = Literal["all", "single", "fractionated"]
ValidationKind = Literal["none", "all", "single", "fractionated"]

# ---------------------- INLINE CONFIG -----------------
# Set USE_INLINE_PARAMS = True to run the script without CLI arguments.
USE_INLINE_PARAMS = False
INLINE_FILES: Optional[List[str]] = None
INLINE_SF_MODES: List[str] = ["absolute"]
INLINE_ALPHA: Optional[float] = None
INLINE_REPAIR_HALF_TIME_HOURS: Optional[float] = None
INLINE_MIN_SF = 1.0
INLINE_FIT_KIND: RegimenKind = "all"
INLINE_VALIDATE_KIND: ValidationKind = "none"
INLINE_FAMILY: Optional[str] = None
INLINE_BY_FAMILY = False
INLINE_SUMMARY_CSV: Optional[str] = None
INLINE_AGGREGATE_REGIMENS = False
INLINE_DEDUPE_REGIMENS = False
INLINE_BOOTSTRAP = 0
INLINE_BOOTSTRAP_SEED: Optional[int] = None
INLINE_VERBOSE = True
# -----------------------------------------------------

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
GR_SUFFIX = re.compile(r"гр|gy", re.IGNORECASE)
FAMILY_TOKEN = re.compile(r"[A-Za-zА-Яа-я]+\d*|\d+")
KNOWN_FAMILIES = ("y", "p", "n", "e")
TIME_TOKEN = re.compile(r"\bt\s*=")


def parse_fractions(experiment_params: List[str]) -> List[float]:
    """Extract dose fractions from experiment parameters."""
    fractions: List[float] = []
    for token in experiment_params:
        token_lower = token.lower()
        if GR_SUFFIX.search(token_lower):
            for num in NUMBER.findall(token_lower):
                fractions.append(float(num.replace(",", ".")))
    return fractions


def _extract_interval_values_days(token: str) -> List[float]:
    """Extract one or many time gaps from tokens like ``t = 1 ч`` or ``t=2.5 hr``."""
    token_lower = token.strip().lower().replace(",", ".")
    if token_lower.startswith("irradiation time="):
        token_lower = token_lower.split("=", 1)[1].strip()
    if not TIME_TOKEN.search(token_lower):
        return []

    values = [float(num.replace(",", ".")) for num in NUMBER.findall(token_lower)]
    if not values:
        return []

    factor = 1.0
    if any(unit in token_lower for unit in ("ч", "час", "hour", "hours", "hr", "hrs")):
        factor = 1.0 / 24.0
    elif any(unit in token_lower for unit in ("мин", "minute", "minutes", "min", "mins")):
        factor = 1.0 / (24.0 * 60.0)
    elif any(unit in token_lower for unit in ("сут", "дн", "день", "дня", "дней", "day", "days")):
        factor = 1.0
    return [value * factor for value in values if np.isfinite(value) and value >= 0.0]


def _build_schedule_from_intervals(
    fractions: Sequence[float],
    intervals_days: Sequence[float],
    *,
    start_day: float = 0.0,
    default_spacing_days: float = 1.0,
) -> Tuple[float, ...]:
    """Build cumulative fraction times from one or many inter-fraction gaps."""
    if not fractions:
        return ()

    times = [float(start_day)]
    current_time = float(start_day)
    clean_intervals = [
        float(value)
        for value in intervals_days
        if np.isfinite(value) and float(value) >= 0.0
    ]
    for index in range(1, len(fractions)):
        if clean_intervals:
            gap = clean_intervals[index - 1] if index - 1 < len(clean_intervals) else clean_intervals[-1]
        else:
            gap = float(default_spacing_days)
        current_time += float(gap)
        times.append(current_time)
    return tuple(times)


def parse_schedule_days(
    experiment_params: Sequence[str],
    fractions: Optional[Sequence[float]] = None,
    *,
    start_day: float = 0.0,
    default_spacing_days: float = 1.0,
) -> Tuple[Tuple[float, ...], bool]:
    """Infer fraction times in days from ordered experiment metadata."""
    doses = [float(dose) for dose in (fractions or parse_fractions(list(experiment_params)))]
    if not doses:
        return (), False
    if len(doses) == 1:
        has_explicit = any(_extract_interval_values_days(str(token)) for token in experiment_params)
        return (float(start_day),), has_explicit

    token_entries: List[Tuple[int, List[float], List[float]]] = []
    for token_index, raw_token in enumerate(experiment_params):
        token = str(raw_token).strip()
        dose_values: List[float] = []
        token_lower = token.lower()
        if GR_SUFFIX.search(token_lower):
            dose_values = [float(num.replace(",", ".")) for num in NUMBER.findall(token_lower)]
        interval_values = _extract_interval_values_days(token)
        token_entries.append((token_index, dose_values, interval_values))

    interval_positions = [index for index, _, interval_values in token_entries if interval_values]
    if not interval_positions:
        return _build_schedule_from_intervals(
            doses,
            (),
            start_day=float(start_day),
            default_spacing_days=float(default_spacing_days),
        ), False

    dose_positions = [index for index, dose_values, _ in token_entries if dose_values]
    if not dose_positions:
        return _build_schedule_from_intervals(
            doses,
            (),
            start_day=float(start_day),
            default_spacing_days=float(default_spacing_days),
        ), False

    last_dose_position = max(dose_positions)
    if all(index > last_dose_position for index in interval_positions):
        trailing_intervals = [
            interval
            for _, _, interval_values in token_entries
            for interval in interval_values
        ]
        return _build_schedule_from_intervals(
            doses,
            trailing_intervals,
            start_day=float(start_day),
            default_spacing_days=float(default_spacing_days),
        ), True

    times: List[float] = []
    current_time = float(start_day)
    pending_intervals: List[float] = []
    emitted_doses = 0

    for _, dose_values, interval_values in token_entries:
        if dose_values:
            for _ in dose_values:
                if emitted_doses == 0:
                    times.append(current_time)
                else:
                    gap = pending_intervals.pop(0) if pending_intervals else float(default_spacing_days)
                    current_time += gap
                    times.append(current_time)
                emitted_doses += 1
                if emitted_doses >= len(doses):
                    break
        if emitted_doses >= len(doses):
            break
        if interval_values:
            pending_intervals.extend(interval_values)

    while emitted_doses < len(doses):
        gap = pending_intervals.pop(0) if pending_intervals else float(default_spacing_days)
        current_time += gap
        times.append(current_time)
        emitted_doses += 1

    return tuple(times), True


def parse_sf_modes(sf_args: Optional[Sequence[str]]) -> List[str]:
    """Parse one or many SF modes from CLI or inline config."""
    if not sf_args:
        return ["absolute"]

    modes: List[str] = []
    for arg in sf_args:
        for chunk in arg.split(","):
            mode = chunk.strip()
            if mode:
                modes.append(mode)

    return modes or ["absolute"]


def format_fractions(fractions: Sequence[float]) -> str:
    """Format fraction sizes for reports and tables."""
    return "+".join(f"{dose:g}" for dose in fractions)


def is_control_file(path: Path) -> bool:
    """Check whether a file should be treated as a control group."""
    return "control" in path.stem.lower()


def compute_sf(mean_norm: np.ndarray, mode: str) -> float:
    """Compute surviving fraction from a normalized mean volume curve."""
    if len(mean_norm) < 2:
        raise ValueError("Need at least two time points to compute SF.")
    if mode.startswith("absindex:"):
        idx = int(mode.split(":", 1)[1])
        if idx >= len(mean_norm):
            raise IndexError(f"absindex {idx} out of range 0..{len(mean_norm) - 1}")
        return float(mean_norm[idx] / mean_norm[0])
    return float(np.nanmin(mean_norm[1:]) / mean_norm[0])


def is_valid_sf(sf: float) -> bool:
    """Check that SF is valid for the LQ model."""
    return bool(np.isfinite(sf) and 0.0 < sf <= 1.0)


def normalize_family(family: Optional[str]) -> Optional[str]:
    """Normalize a radiation-family label for comparisons."""
    if family is None:
        return None
    family = family.strip().lower()
    return family or None


def resolve_requested_families(
    available_families: Sequence[str],
    family: Optional[str],
    by_family: bool,
) -> List[Optional[str]]:
    """Resolve which families should be analyzed in this run."""
    family = normalize_family(family)
    if by_family:
        if family is not None:
            return [family] if family in available_families else []
        return list(available_families)
    return [family]


def infer_radiation_family(path: Path) -> Optional[str]:
    """Infer a coarse radiation-family label from the file name."""
    tokens = FAMILY_TOKEN.findall(path.stem.lower())
    for token in tokens:
        if token in KNOWN_FAMILIES:
            return token
    for token in tokens:
        for family in KNOWN_FAMILIES:
            suffix = token[len(family) :] if token.startswith(family) else ""
            if token.startswith(family) and suffix.isdigit():
                return family
    return None


@dataclass(frozen=True)
class RawTumorSeries:
    """Raw tumor-volume time series for one file."""

    path: Path
    fractions: Tuple[float, ...]
    family: Optional[str]
    volumes: np.ndarray
    schedule_days: Tuple[float, ...] = ()
    has_explicit_timing: bool = False
    control_path: Optional[Path] = None

    @property
    def regimen_kind(self) -> RegimenKind:
        return "single" if len(self.fractions) == 1 else "fractionated"

    @property
    def dose_sum(self) -> float:
        return float(sum(self.fractions))

    @property
    def dose2_sum(self) -> float:
        return float(sum(dose * dose for dose in self.fractions))


@dataclass(frozen=True)
class TumorExperiment:
    """Derived experiment with computed SF."""

    path: Path
    fractions: Tuple[float, ...]
    sf: float
    family: Optional[str] = None
    schedule_days: Tuple[float, ...] = ()
    repeat_count: int = 1
    sf_std: float = 0.0
    source_paths: Tuple[Path, ...] = ()
    has_explicit_timing: bool = False
    control_path: Optional[Path] = None

    @property
    def dose_sum(self) -> float:
        return float(sum(self.fractions))

    @property
    def dose2_sum(self) -> float:
        return float(sum(dose * dose for dose in self.fractions))

    @property
    def fraction_count(self) -> int:
        return len(self.fractions)

    @property
    def regimen_kind(self) -> RegimenKind:
        return "single" if self.fraction_count == 1 else "fractionated"

    @property
    def resolved_schedule_days(self) -> Tuple[float, ...]:
        if len(self.schedule_days) == self.fraction_count:
            return self.schedule_days
        return tuple(float(index) for index in range(self.fraction_count))

    @property
    def interval_days(self) -> Tuple[float, ...]:
        schedule = self.resolved_schedule_days
        if len(schedule) < 2:
            return ()
        return tuple(
            float(schedule[index] - schedule[index - 1])
            for index in range(1, len(schedule))
        )

    def quadratic_term(self, repair_rate_per_day: Optional[float] = None) -> float:
        if repair_rate_per_day is None or repair_rate_per_day <= 0.0 or self.fraction_count <= 1:
            return self.dose2_sum
        schedule = self.resolved_schedule_days
        term = 0.0
        for i, dose_i in enumerate(self.fractions):
            for j, dose_j in enumerate(self.fractions):
                delta_days = abs(schedule[i] - schedule[j])
                term += dose_i * dose_j * math.exp(-repair_rate_per_day * delta_days)
        return float(term)

    @property
    def path_label(self) -> str:
        if self.repeat_count <= 1:
            return self.path.name
        return f"{self.path.name} [+{self.repeat_count - 1}]"

    @property
    def control_label(self) -> str:
        if self.control_path is None:
            return "all-controls"
        return self.control_path.name

    @property
    def regimen_key(self) -> Tuple[Optional[str], Optional[Path], Tuple[float, ...], Tuple[float, ...]]:
        return (
            self.family,
            self.control_path,
            tuple(round(dose, 8) for dose in self.fractions),
            tuple(round(day, 8) for day in self.resolved_schedule_days),
        )

    @property
    def schedule_label(self) -> str:
        parts = []
        for day in self.resolved_schedule_days:
            hours = day * 24.0
            if abs(hours) < 24.0:
                parts.append(f"{hours:g}h")
            else:
                parts.append(f"{day:g}d")
        return "[" + ", ".join(parts) + "]"

    def report(self) -> str:
        family = self.family or "-"
        line = (
            f"{self.path_label:30s} family={family:>2s} kind={self.regimen_kind:12s} "
            f"({format_fractions(self.fractions)})  D={self.dose_sum:5.1f}  D2={self.dose2_sum:6.1f}  "
            f"SF={self.sf:.4f}  control={self.control_label}"
        )
        if self.repeat_count > 1:
            line += f"  repeats={self.repeat_count}  sf_std={self.sf_std:.4f}"
        if self.fraction_count > 1:
            timing = "explicit" if self.has_explicit_timing else "default-daily"
            line += f"  schedule={self.schedule_label} ({timing})"
        return line


@dataclass(frozen=True)
class LQFitResult:
    """Fitted LQ parameters and metadata."""

    alpha: float
    beta: float
    train_count: int
    train_kind: RegimenKind
    family: Optional[str]
    sf_mode: str
    repair_half_time_hours: Optional[float] = None

    @property
    def alpha_beta_ratio(self) -> Optional[float]:
        if self.beta <= 0.0:
            return None
        return self.alpha / self.beta

    @property
    def repair_rate_per_day(self) -> Optional[float]:
        if self.repair_half_time_hours is None or self.repair_half_time_hours <= 0.0:
            return None
        return math.log(2.0) * 24.0 / self.repair_half_time_hours

    def predict_sf(self, experiment: TumorExperiment) -> float:
        quadratic_term = experiment.quadratic_term(self.repair_rate_per_day)
        return float(np.exp(-(self.alpha * experiment.dose_sum + self.beta * quadratic_term)))


@dataclass(frozen=True)
class PredictionRow:
    """Observed vs predicted SF for one holdout experiment."""

    experiment: TumorExperiment
    predicted_sf: float
    abs_error: float
    rel_error: float
    log_error: float

    def report(self) -> str:
        return (
            f"{self.experiment.path.name:30s} "
            f"obs={self.experiment.sf:.4f} pred={self.predicted_sf:.4f} "
            f"abs_err={self.abs_error:.4f} rel_err={self.rel_error:.2%} "
            f"log_err={self.log_error:+.4f}"
        )


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregate quality metrics for holdout experiments."""

    rows: Tuple[PredictionRow, ...]
    mae: float
    rmse: float
    mean_abs_log_error: float


@dataclass(frozen=True)
class BootstrapStat:
    """Summary statistics for one bootstrapped parameter."""

    mean: float
    std: float
    q025: float
    median: float
    q975: float


@dataclass(frozen=True)
class BootstrapSummary:
    """Bootstrap statistics for alpha, beta and alpha/beta."""

    requested_repeats: int
    successful_repeats: int
    failed_repeats: int
    alpha: BootstrapStat
    beta: BootstrapStat
    alpha_beta_ratio: Optional[BootstrapStat]


@dataclass(frozen=True)
class TimingDiagnostics:
    """How informative the training set is for schedule-aware fitting."""

    repair_model_enabled: bool
    repair_half_time_hours: Optional[float]
    train_count: int
    fractionated_count: int
    explicit_timing_count: int
    explicit_fractionated_count: int
    unique_schedule_count: int
    same_fractions_multi_timing_count: int
    unique_quadratic_count: int
    design_rank: int
    condition_number: Optional[float]
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisRunSummary:
    """Outcome of one (sf_mode, family) analysis run."""

    sf_mode: str
    family: Optional[str]
    total_count: int
    single_count: int
    fractionated_count: int
    train_count: int
    validation_count: int
    status: str
    alpha: Optional[float] = None
    beta: Optional[float] = None
    reason: Optional[str] = None

    @property
    def alpha_beta_ratio(self) -> Optional[float]:
        if self.alpha is None or self.beta is None or self.beta <= 0.0:
            return None
        return self.alpha / self.beta

    @property
    def family_label(self) -> str:
        return self.family or "all"


@dataclass(frozen=True)
class AnalysisRunResult:
    """Structured data for one analysis run, suitable for GUI or tests."""

    summary: AnalysisRunSummary
    train: Tuple[TumorExperiment, ...]
    validation: Tuple[TumorExperiment, ...]
    train_kind: RegimenKind
    validation_kind: ValidationKind
    fit_result: Optional[LQFitResult] = None
    validation_summary: Optional[ValidationSummary] = None
    bootstrap_summary: Optional[BootstrapSummary] = None
    timing_diagnostics: Optional[TimingDiagnostics] = None

    @property
    def label(self) -> str:
        return (
            f"sf={self.summary.sf_mode} | "
            f"family={self.summary.family_label} | "
            f"status={self.summary.status}"
        )


class Fitter:
    """Collect experiments and fit alpha/beta."""

    def __init__(
        self,
        sf_mode: str,
        min_sf: float,
        alpha_fixed: Optional[float],
        verbose: bool,
        repair_half_time_hours: Optional[float] = None,
        aggregate_regimens: bool = False,
        dedupe_regimens: bool = False,
    ):
        self.sf_mode = sf_mode
        self.min_sf = min_sf
        self.alpha_fixed = alpha_fixed
        self.repair_half_time_hours = repair_half_time_hours
        self.verbose = verbose
        self.aggregate_regimens = aggregate_regimens
        self.dedupe_regimens = dedupe_regimens
        self.raw_experiments: List[RawTumorSeries] = []
        self.controls: Dict[Path, np.ndarray] = {}
        self.experiments: List[TumorExperiment] = []
        self.control_curve: Optional[np.ndarray] = None

    @property
    def repair_rate_per_day(self) -> Optional[float]:
        if self.repair_half_time_hours is None or self.repair_half_time_hours <= 0.0:
            return None
        return math.log(2.0) * 24.0 / self.repair_half_time_hours

    @staticmethod
    def _mean_tumor_volumes(volumes: np.ndarray) -> np.ndarray:
        return TumorDataProcessor(np.asarray(volumes, dtype=float)).get_mean_tumor_volumes()

    @staticmethod
    def _resample_volumes(volumes: np.ndarray, rng: Optional[np.random.Generator]) -> np.ndarray:
        if rng is None:
            return np.asarray(volumes, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        if volumes.ndim != 2 or volumes.shape[0] == 0:
            raise ValueError("Volumes must be a non-empty 2D array.")
        indices = rng.integers(0, volumes.shape[0], size=volumes.shape[0])
        return volumes[indices]

    def _build_control_curve(
        self,
        control_paths: Optional[Sequence[Path]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if not self.controls:
            raise RuntimeError("No control files were found. Cannot normalize experiments.")

        if control_paths is None:
            control_volumes = list(self.controls.values())
        else:
            missing = [path for path in control_paths if path not in self.controls]
            if missing:
                raise RuntimeError(
                    f"Assigned control files were not loaded: {[path.name for path in missing]}"
                )
            control_volumes = [self.controls[path] for path in control_paths]

        mean_curves: List[np.ndarray] = []
        for volumes in control_volumes:
            sampled = self._resample_volumes(volumes, rng)
            mean_curves.append(self._mean_tumor_volumes(sampled))

        max_len = max(len(curve) for curve in mean_curves)
        padded = [
            np.pad(curve, (0, max_len - len(curve)), constant_values=np.nan)
            for curve in mean_curves
        ]
        return np.nanmean(padded, axis=0)

    def _materialize_experiment(
        self,
        raw_experiment: RawTumorSeries,
        sf_mode: str,
        control_curve: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[TumorExperiment]:
        sampled_volumes = self._resample_volumes(raw_experiment.volumes, rng)
        mean_abs = self._mean_tumor_volumes(sampled_volumes)
        curve_len = min(len(mean_abs), len(control_curve))
        mean_norm = mean_abs[:curve_len] / control_curve[:curve_len]

        sf = compute_sf(mean_norm, sf_mode)
        if not is_valid_sf(sf):
            if self.verbose:
                print(f"WARNING {raw_experiment.path.name}: invalid SF={sf!r} -> skip")
            return None

        if sf >= self.min_sf:
            if self.verbose:
                print(f"INFO {raw_experiment.path.name}: SF={sf:.4f} >= {self.min_sf} -> skip")
            return None

        return TumorExperiment(
            path=raw_experiment.path,
            fractions=raw_experiment.fractions,
            sf=sf,
            family=raw_experiment.family,
            schedule_days=raw_experiment.schedule_days,
            repeat_count=1,
            sf_std=0.0,
            source_paths=(raw_experiment.path,),
            has_explicit_timing=raw_experiment.has_explicit_timing,
            control_path=raw_experiment.control_path,
        )

    def materialize_experiments(
        self,
        sf_mode: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[TumorExperiment]:
        sf_mode = sf_mode or self.sf_mode
        control_paths = {experiment.control_path for experiment in self.raw_experiments}
        control_curve_cache: Dict[Optional[Path], np.ndarray] = {}
        if not control_paths or None in control_paths:
            control_curve_cache[None] = self._build_control_curve(rng=rng)
        for control_path in sorted(
            (path for path in control_paths if path is not None),
            key=lambda path: str(path),
        ):
            control_curve_cache[control_path] = self._build_control_curve(
                control_paths=[control_path],
                rng=rng,
            )
        experiments: List[TumorExperiment] = []

        for raw_experiment in self.raw_experiments:
            control_curve = control_curve_cache[raw_experiment.control_path]
            experiment = self._materialize_experiment(
                raw_experiment=raw_experiment,
                sf_mode=sf_mode,
                control_curve=control_curve,
                rng=rng,
            )
            if experiment is not None:
                experiments.append(experiment)
                if self.verbose and rng is None:
                    print(
                        f"OK {experiment.path.name}: family={experiment.family or '-'} "
                        f"kind={experiment.regimen_kind} fractions={list(experiment.fractions)}, "
                        f"SF={experiment.sf:.4f}"
                    )

        if self.aggregate_regimens:
            experiments = self.aggregate_experiments(experiments)

        if self.dedupe_regimens:
            unique_experiments: Dict[
                Tuple[Optional[str], Optional[Path], Tuple[float, ...], Tuple[float, ...]],
                TumorExperiment,
            ] = {}
            for experiment in experiments:
                unique_experiments.setdefault(experiment.regimen_key, experiment)
            experiments = list(unique_experiments.values())

        if rng is None:
            self.control_curve = control_curve_cache.get(None)
            self.experiments = experiments
            self.sf_mode = sf_mode
        return experiments

    @staticmethod
    def aggregate_experiments(
        experiments: Sequence[TumorExperiment],
    ) -> List[TumorExperiment]:
        grouped: Dict[
            Tuple[Optional[str], Optional[Path], Tuple[float, ...], Tuple[float, ...]],
            List[TumorExperiment],
        ] = {}
        for experiment in experiments:
            grouped.setdefault(experiment.regimen_key, []).append(experiment)

        aggregated: List[TumorExperiment] = []
        for group in grouped.values():
            if len(group) == 1:
                aggregated.append(group[0])
                continue

            sfs = np.array([experiment.sf for experiment in group], dtype=float)
            source_paths = tuple(experiment.path for experiment in group)
            first = group[0]
            aggregated.append(
                TumorExperiment(
                    path=first.path,
                    fractions=first.fractions,
                    sf=float(np.mean(sfs)),
                    family=first.family,
                    schedule_days=first.schedule_days,
                    repeat_count=len(group),
                    sf_std=float(np.std(sfs, ddof=0)),
                    source_paths=source_paths,
                    has_explicit_timing=first.has_explicit_timing,
                    control_path=first.control_path,
                )
            )

        return aggregated

    @staticmethod
    def _normalize_control_map(
        control_map: Optional[Mapping[object, Optional[object]]],
    ) -> Optional[Dict[Path, Optional[Path]]]:
        if control_map is None:
            return None
        normalized: Dict[Path, Optional[Path]] = {}
        for experiment_path, control_path in control_map.items():
            experiment = Path(str(experiment_path)).expanduser().resolve()
            control = None
            if control_path is not None:
                control = Path(str(control_path)).expanduser().resolve()
            normalized[experiment] = control
        return normalized

    def collect(
        self,
        files: List[Path],
        control_map: Optional[Mapping[object, Optional[object]]] = None,
    ) -> None:
        controls: List[Path] = []
        others: List[Path] = []
        normalized_control_map = self._normalize_control_map(control_map)

        for path in files:
            if is_control_file(path):
                controls.append(path)
            else:
                others.append(path)

        self.controls = {}
        for path in controls:
            _, _, _, volumes = process_tumor_data_excel(str(path))
            self.controls[path] = np.asarray(volumes, dtype=float)
            if self.verbose:
                print(f"INFO {path.name}: registered as CONTROL")

        sole_control_path = controls[0] if len(controls) == 1 else None
        self.raw_experiments = []
        for path in others:
            params, _, _, volumes = process_tumor_data_excel(str(path))
            fractions = tuple(parse_fractions(params))
            if not fractions:
                if self.verbose:
                    print(f"WARNING {path.name}: dose fractions were not parsed -> skip")
                continue
            schedule_days, has_explicit_timing = parse_schedule_days(params, fractions)

            assigned_control = None
            if normalized_control_map is not None:
                if path in normalized_control_map:
                    assigned_control = normalized_control_map[path]
                elif sole_control_path is not None:
                    assigned_control = sole_control_path
                else:
                    raise ValueError(
                        f"No control was assigned to experiment {path.name}."
                    )
            elif sole_control_path is not None:
                assigned_control = sole_control_path

            if assigned_control is not None and assigned_control not in self.controls:
                raise ValueError(
                    f"Assigned control {assigned_control.name} for {path.name} was not loaded."
                )

            self.raw_experiments.append(
                RawTumorSeries(
                    path=path,
                    fractions=fractions,
                    family=infer_radiation_family(path),
                    volumes=np.asarray(volumes, dtype=float),
                    schedule_days=schedule_days,
                    has_explicit_timing=has_explicit_timing,
                    control_path=assigned_control,
                )
            )
            if self.verbose and len(fractions) > 1:
                if has_explicit_timing:
                    schedule_hours = ", ".join(f"{day * 24.0:g}h" for day in schedule_days)
                    print(f"INFO {path.name}: parsed schedule {schedule_hours}")
                elif self.repair_half_time_hours is not None and self.repair_half_time_hours > 0.0:
                    print(
                        f"INFO {path.name}: no explicit t= intervals found, "
                        "using default 24h spacing between fractions."
                    )

        if len(self.controls) == 1:
            self.control_curve = self._build_control_curve(control_paths=controls)
            if self.verbose:
                print("INFO control curve prepared:", self.control_curve[:5], "...")
        elif self.verbose:
            if normalized_control_map is None:
                pooled_curve = self._build_control_curve()
                print("INFO pooled control curve prepared:", pooled_curve[:5], "...")
            else:
                print("INFO experiment-specific control mapping enabled.")
        self.experiments = self.materialize_experiments(sf_mode=self.sf_mode)

    @staticmethod
    def _filter_by_regimen(
        experiments: Sequence[TumorExperiment],
        regimen_kind: RegimenKind,
    ) -> List[TumorExperiment]:
        if regimen_kind == "all":
            return list(experiments)
        return [experiment for experiment in experiments if experiment.regimen_kind == regimen_kind]

    @staticmethod
    def _filter_experiments(
        experiments: Sequence[TumorExperiment],
        family: Optional[str],
        regimen_kind: RegimenKind,
    ) -> List[TumorExperiment]:
        family = normalize_family(family)
        filtered = list(experiments)
        if family is not None:
            filtered = [experiment for experiment in filtered if experiment.family == family]
        return Fitter._filter_by_regimen(filtered, regimen_kind)

    def select_experiments(
        self,
        family: Optional[str] = None,
        regimen_kind: RegimenKind = "all",
        experiments: Optional[Sequence[TumorExperiment]] = None,
    ) -> List[TumorExperiment]:
        base = list(experiments) if experiments is not None else list(self.experiments)
        return self._filter_experiments(base, family=family, regimen_kind=regimen_kind)

    def available_families(
        self,
        experiments: Optional[Sequence[TumorExperiment]] = None,
    ) -> List[str]:
        base = list(experiments) if experiments is not None else list(self.experiments)
        families = {experiment.family for experiment in base if experiment.family is not None}
        return sorted(families)

    def regimen_counts(
        self,
        family: Optional[str] = None,
        experiments: Optional[Sequence[TumorExperiment]] = None,
    ) -> Tuple[int, int, int]:
        filtered = self.select_experiments(
            family=family,
            regimen_kind="all",
            experiments=experiments,
        )
        total = len(filtered)
        single = sum(1 for experiment in filtered if experiment.regimen_kind == "single")
        fractionated = sum(
            1 for experiment in filtered if experiment.regimen_kind == "fractionated"
        )
        return total, single, fractionated

    def build_train_validation_sets(
        self,
        fit_kind: RegimenKind,
        validate_kind: ValidationKind,
        family: Optional[str] = None,
        experiments: Optional[Sequence[TumorExperiment]] = None,
    ) -> Tuple[List[TumorExperiment], List[TumorExperiment]]:
        selected = self.select_experiments(family=family, regimen_kind="all", experiments=experiments)
        train = self._filter_by_regimen(selected, fit_kind)
        if validate_kind == "none":
            return train, []

        validation = self._filter_by_regimen(selected, validate_kind)
        train_paths = {experiment.path for experiment in train}
        validation = [
            experiment for experiment in validation if experiment.path not in train_paths
        ]
        return train, validation

    @staticmethod
    def _lq_model(
        xdata: Tuple[np.ndarray, np.ndarray],
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        dose_sum, quadratic_term = xdata
        return np.exp(-(alpha * dose_sum + beta * quadratic_term))

    @staticmethod
    def _initial_guess(
        dose_sum: np.ndarray,
        quadratic_term: np.ndarray,
        sf: np.ndarray,
    ) -> np.ndarray:
        design = np.column_stack((dose_sum, quadratic_term))
        y_log = -np.log(sf)
        guess, *_ = np.linalg.lstsq(design, y_log, rcond=None)
        guess = np.asarray(guess, dtype=float)
        guess = np.clip(guess, 1.0e-8, None)
        if not np.all(np.isfinite(guess)) or np.allclose(guess, 0.0):
            return np.array([0.1, 0.01], dtype=float)
        return guess

    @staticmethod
    def _fit_beta_for_fixed_alpha(
        dose_sum: np.ndarray,
        quadratic_term: np.ndarray,
        sf: np.ndarray,
        sigma_log_sf: np.ndarray,
        alpha: float,
    ) -> float:
        y_log = -np.log(sf)
        y_shift = y_log - alpha * dose_sum
        weights = 1.0 / np.square(np.clip(sigma_log_sf, 1.0e-8, None))
        weighted_quadratic = weights * quadratic_term
        denom = float(np.dot(weighted_quadratic, quadratic_term))
        if denom == 0.0:
            raise RuntimeError("Cannot fit beta: denominator is zero.")
        return max(0.0, float(np.dot(weighted_quadratic, y_shift) / denom))

    def _fit_free_alpha_beta(
        self,
        dose_sum: np.ndarray,
        quadratic_term: np.ndarray,
        sf: np.ndarray,
        sigma_sf: np.ndarray,
    ) -> Tuple[float, float]:
        design = np.column_stack((dose_sum, quadratic_term))
        if np.linalg.matrix_rank(design) < 2:
            raise RuntimeError(
                "Need at least two linearly independent regimens to fit both alpha and beta."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                self._lq_model,
                (dose_sum, quadratic_term),
                sf,
                p0=self._initial_guess(dose_sum, quadratic_term, sf),
                bounds=(0.0, np.inf),
                sigma=np.clip(sigma_sf, 1.0e-8, None),
                absolute_sigma=False,
                maxfev=20000,
            )
        alpha, beta = (float(value) for value in params)
        return alpha, beta

    @staticmethod
    def _to_arrays(
        experiments: Sequence[TumorExperiment],
        repair_rate_per_day: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dose_sum = np.array([experiment.dose_sum for experiment in experiments], dtype=float)
        quadratic_term = np.array(
            [experiment.quadratic_term(repair_rate_per_day) for experiment in experiments],
            dtype=float,
        )
        sf = np.array([experiment.sf for experiment in experiments], dtype=float)
        sigma_sf = np.array(
            [
                max(
                    (
                        experiment.sf_std / np.sqrt(experiment.repeat_count)
                        if experiment.repeat_count > 1 and experiment.sf_std > 0.0
                        else experiment.sf / np.sqrt(experiment.repeat_count)
                    ),
                    1.0e-8,
                )
                for experiment in experiments
            ],
            dtype=float,
        )
        sigma_log_sf = np.clip(sigma_sf / np.clip(sf, 1.0e-8, None), 1.0e-8, None)
        return dose_sum, quadratic_term, sf, sigma_sf, sigma_log_sf

    def fit(
        self,
        experiments: Optional[Sequence[TumorExperiment]] = None,
        train_kind: RegimenKind = "all",
        family: Optional[str] = None,
        sf_mode: Optional[str] = None,
    ) -> LQFitResult:
        experiments = list(experiments) if experiments is not None else list(self.experiments)
        if len(experiments) < 2:
            raise RuntimeError("Need at least two valid experiments for fitting.")

        dose_sum, quadratic_term, sf, sigma_sf, sigma_log_sf = self._to_arrays(
            experiments,
            repair_rate_per_day=self.repair_rate_per_day,
        )
        if np.any(~np.isfinite(sf)) or np.any(sf <= 0.0) or np.any(sf > 1.0):
            raise ValueError("All SF values must be finite and in the interval (0, 1].")

        if self.alpha_fixed is not None:
            beta = self._fit_beta_for_fixed_alpha(
                dose_sum=dose_sum,
                quadratic_term=quadratic_term,
                sf=sf,
                sigma_log_sf=sigma_log_sf,
                alpha=self.alpha_fixed,
            )
            alpha = self.alpha_fixed
        else:
            alpha, beta = self._fit_free_alpha_beta(
                dose_sum=dose_sum,
                quadratic_term=quadratic_term,
                sf=sf,
                sigma_sf=sigma_sf,
            )

        return LQFitResult(
            alpha=alpha,
            beta=beta,
            train_count=len(experiments),
            train_kind=train_kind,
            family=normalize_family(family),
            sf_mode=sf_mode or self.sf_mode,
            repair_half_time_hours=self.repair_half_time_hours,
        )

    def compute_timing_diagnostics(
        self,
        experiments: Sequence[TumorExperiment],
    ) -> TimingDiagnostics:
        """Describe whether the training set can really inform time-aware fitting."""
        experiments = list(experiments)
        fractionated = [experiment for experiment in experiments if experiment.regimen_kind == "fractionated"]
        explicit_timing_count = sum(1 for experiment in experiments if experiment.has_explicit_timing)
        explicit_fractionated_count = sum(
            1 for experiment in fractionated if experiment.has_explicit_timing
        )
        unique_schedule_count = len(
            {
                tuple(round(day, 8) for day in experiment.resolved_schedule_days)
                for experiment in fractionated
            }
        )
        schedule_groups: Dict[Tuple[float, ...], set[Tuple[float, ...]]] = {}
        for experiment in fractionated:
            dose_key = tuple(round(dose, 8) for dose in experiment.fractions)
            schedule_key = tuple(round(day, 8) for day in experiment.resolved_schedule_days)
            schedule_groups.setdefault(dose_key, set()).add(schedule_key)
        same_fractions_multi_timing_count = sum(
            1 for schedules in schedule_groups.values() if len(schedules) > 1
        )

        repair_rate = self.repair_rate_per_day
        quadratic_term = np.array(
            [experiment.quadratic_term(repair_rate) for experiment in experiments],
            dtype=float,
        )
        unique_quadratic_count = len({round(value, 8) for value in quadratic_term})

        if experiments:
            dose_sum = np.array([experiment.dose_sum for experiment in experiments], dtype=float)
            design = np.column_stack((dose_sum, quadratic_term))
            design_rank = int(np.linalg.matrix_rank(design))
            condition_number = None
            if len(experiments) >= 2 and design_rank >= 2:
                condition_number = float(np.linalg.cond(design))
        else:
            design_rank = 0
            condition_number = None

        warnings_list: List[str] = []
        if repair_rate is not None:
            if len(experiments) < 3:
                warnings_list.append(
                    "Only two training regimens are available; repair-aware alpha/beta is likely unstable."
                )
            if not fractionated:
                warnings_list.append("No fractionated regimens are present in the training set.")
            if explicit_fractionated_count == 0 and fractionated:
                warnings_list.append(
                    "Fractionated regimens have no explicit t= timing metadata; default 24h spacing is being used."
                )
            if explicit_fractionated_count > 0 and same_fractions_multi_timing_count == 0:
                warnings_list.append(
                    "No matched dose pattern is represented at multiple interval schedules, so timing contrast is weak."
                )
            if unique_quadratic_count < 2:
                warnings_list.append(
                    "The repair-aware quadratic term does not vary across training regimens."
                )
            if design_rank < 2:
                warnings_list.append(
                    "The repair-aware design matrix rank is below 2, so alpha and beta are not identifiable."
                )
            elif condition_number is not None and condition_number > 1.0e4:
                warnings_list.append(
                    f"The repair-aware design matrix is ill-conditioned (cond={condition_number:.2g})."
                )

        return TimingDiagnostics(
            repair_model_enabled=repair_rate is not None,
            repair_half_time_hours=self.repair_half_time_hours,
            train_count=len(experiments),
            fractionated_count=len(fractionated),
            explicit_timing_count=explicit_timing_count,
            explicit_fractionated_count=explicit_fractionated_count,
            unique_schedule_count=unique_schedule_count,
            same_fractions_multi_timing_count=same_fractions_multi_timing_count,
            unique_quadratic_count=unique_quadratic_count,
            design_rank=design_rank,
            condition_number=condition_number,
            warnings=tuple(warnings_list),
        )

    def evaluate(
        self,
        experiments: Sequence[TumorExperiment],
        result: LQFitResult,
    ) -> ValidationSummary:
        rows: List[PredictionRow] = []
        squared_errors: List[float] = []
        abs_log_errors: List[float] = []

        for experiment in experiments:
            predicted_sf = result.predict_sf(experiment)
            abs_error = abs(predicted_sf - experiment.sf)
            rel_error = abs_error / experiment.sf
            log_error = float(np.log(predicted_sf) - np.log(experiment.sf))
            rows.append(
                PredictionRow(
                    experiment=experiment,
                    predicted_sf=predicted_sf,
                    abs_error=abs_error,
                    rel_error=rel_error,
                    log_error=log_error,
                )
            )
            squared_errors.append(abs_error * abs_error)
            abs_log_errors.append(abs(log_error))

        mae = float(np.mean([row.abs_error for row in rows])) if rows else 0.0
        rmse = float(np.sqrt(np.mean(squared_errors))) if rows else 0.0
        mean_abs_log_error = float(np.mean(abs_log_errors)) if rows else 0.0
        return ValidationSummary(
            rows=tuple(rows),
            mae=mae,
            rmse=rmse,
            mean_abs_log_error=mean_abs_log_error,
        )

    @staticmethod
    def _summarize_bootstrap_samples(samples: np.ndarray) -> BootstrapStat:
        return BootstrapStat(
            mean=float(np.mean(samples)),
            std=float(np.std(samples, ddof=0)),
            q025=float(np.quantile(samples, 0.025)),
            median=float(np.quantile(samples, 0.5)),
            q975=float(np.quantile(samples, 0.975)),
        )

    def bootstrap_fit(
        self,
        sf_mode: str,
        fit_kind: RegimenKind,
        family: Optional[str],
        repeats: int,
        seed: Optional[int] = None,
    ) -> BootstrapSummary:
        if repeats <= 0:
            raise ValueError("Bootstrap repeats must be positive.")

        rng = np.random.default_rng(seed)
        alpha_samples: List[float] = []
        beta_samples: List[float] = []
        ratio_samples: List[float] = []
        failed = 0

        for _ in range(repeats):
            sampled_experiments = self.materialize_experiments(sf_mode=sf_mode, rng=rng)
            train = self.select_experiments(
                family=family,
                regimen_kind=fit_kind,
                experiments=sampled_experiments,
            )
            if len(train) < 2:
                failed += 1
                continue
            try:
                result = self.fit(
                    experiments=train,
                    train_kind=fit_kind,
                    family=family,
                    sf_mode=sf_mode,
                )
            except (RuntimeError, ValueError):
                failed += 1
                continue

            alpha_samples.append(result.alpha)
            beta_samples.append(result.beta)
            if result.alpha_beta_ratio is not None:
                ratio_samples.append(result.alpha_beta_ratio)

        if not alpha_samples or not beta_samples:
            raise RuntimeError("Bootstrap produced no successful fits.")

        alpha_arr = np.asarray(alpha_samples, dtype=float)
        beta_arr = np.asarray(beta_samples, dtype=float)
        ratio_summary = None
        if ratio_samples:
            ratio_summary = self._summarize_bootstrap_samples(np.asarray(ratio_samples, dtype=float))

        return BootstrapSummary(
            requested_repeats=repeats,
            successful_repeats=len(alpha_samples),
            failed_repeats=failed,
            alpha=self._summarize_bootstrap_samples(alpha_arr),
            beta=self._summarize_bootstrap_samples(beta_arr),
            alpha_beta_ratio=ratio_summary,
        )

    @staticmethod
    def report_training(experiments: Sequence[TumorExperiment], result: LQFitResult) -> None:
        family = result.family or "all"
        print("\n# Training experiments:")
        print(
            f"sf_mode={result.sf_mode} family={family} "
            f"train_kind={result.train_kind} count={result.train_count}"
        )
        if result.repair_half_time_hours is not None and result.repair_half_time_hours > 0.0:
            print(
                "model=time-aware-lq "
                f"repair_half_time_hours={result.repair_half_time_hours:.3f}"
            )
        else:
            print("model=classic-lq")
        for experiment in experiments:
            print(experiment.report())
        print("\n===== FIT RESULT =====")
        print(f"alpha (Gy^-1): {result.alpha:.5f}")
        print(f"beta  (Gy^-2): {result.beta:.6f}")
        if result.alpha_beta_ratio is not None:
            print(f"alpha/beta   : {result.alpha_beta_ratio:.2f} Gy")
        print("======================")

    @staticmethod
    def report_validation(
        validation_kind: ValidationKind,
        summary: ValidationSummary,
    ) -> None:
        print("\n# Holdout validation:")
        print(f"validate_kind={validation_kind} count={len(summary.rows)}")
        for row in summary.rows:
            print(row.report())
        print(
            "summary: "
            f"MAE={summary.mae:.4f} "
            f"RMSE={summary.rmse:.4f} "
            f"mean_abs_log_error={summary.mean_abs_log_error:.4f}"
        )

    @staticmethod
    def report_bootstrap(summary: BootstrapSummary) -> None:
        print("\n# Bootstrap summary:")
        print(
            f"requested={summary.requested_repeats} "
            f"successful={summary.successful_repeats} failed={summary.failed_repeats}"
        )
        print(
            "alpha: "
            f"mean={summary.alpha.mean:.5f} std={summary.alpha.std:.5f} "
            f"median={summary.alpha.median:.5f} "
            f"CI95=[{summary.alpha.q025:.5f}, {summary.alpha.q975:.5f}]"
        )
        print(
            "beta:  "
            f"mean={summary.beta.mean:.6f} std={summary.beta.std:.6f} "
            f"median={summary.beta.median:.6f} "
            f"CI95=[{summary.beta.q025:.6f}, {summary.beta.q975:.6f}]"
        )
        if summary.alpha_beta_ratio is not None:
            ratio = summary.alpha_beta_ratio
            if summary.beta.q025 <= 1.0e-8:
                print(
                    "alpha/beta: unstable because bootstrap beta approaches zero; "
                    "interpret the ratio with caution."
                )
            else:
                print(
                    "alpha/beta: "
                    f"mean={ratio.mean:.2f} std={ratio.std:.2f} "
                    f"median={ratio.median:.2f} "
                    f"CI95=[{ratio.q025:.2f}, {ratio.q975:.2f}] Gy"
                )

    @staticmethod
    def report_analysis_summaries(summaries: Sequence[AnalysisRunSummary]) -> None:
        print("\n# Batch summary:")
        for summary in summaries:
            family = summary.family or "all"
            line = (
                f"sf_mode={summary.sf_mode} family={family} status={summary.status} "
                f"total={summary.total_count} single={summary.single_count} "
                f"fractionated={summary.fractionated_count} "
                f"train={summary.train_count} validation={summary.validation_count}"
            )
            if summary.alpha is not None and summary.beta is not None:
                line += f" alpha={summary.alpha:.5f} beta={summary.beta:.6f}"
            if summary.reason:
                line += f" reason={summary.reason}"
            print(line)

    @staticmethod
    def write_analysis_summaries_csv(
        path: Path,
        summaries: Sequence[AnalysisRunSummary],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sf_mode",
                    "family",
                    "status",
                    "total_count",
                    "single_count",
                    "fractionated_count",
                    "train_count",
                    "validation_count",
                    "alpha",
                    "beta",
                    "alpha_beta_ratio",
                    "reason",
                ]
            )
            for summary in summaries:
                ratio = None
                if summary.alpha is not None and summary.beta is not None and summary.beta > 0.0:
                    ratio = summary.alpha / summary.beta
                writer.writerow(
                    [
                        summary.sf_mode,
                        summary.family or "all",
                        summary.status,
                        summary.total_count,
                        summary.single_count,
                        summary.fractionated_count,
                        summary.train_count,
                        summary.validation_count,
                        summary.alpha,
                        summary.beta,
                        ratio,
                        summary.reason or "",
                    ]
                )


def analyze_fitter(
    fitter: Fitter,
    sf_modes: Sequence[str],
    fit_kind: RegimenKind,
    validate_kind: ValidationKind,
    family: Optional[str],
    by_family: bool,
    bootstrap: int,
    bootstrap_seed: Optional[int],
) -> List[AnalysisRunResult]:
    """Run one or many analyses on an already collected fitter."""
    results: List[AnalysisRunResult] = []

    for mode_index, sf_mode in enumerate(sf_modes):
        if fitter.raw_experiments:
            experiments = fitter.materialize_experiments(sf_mode=sf_mode)
        else:
            experiments = list(fitter.experiments)
        available_families = fitter.available_families(experiments)
        requested_families = resolve_requested_families(
            available_families=available_families,
            family=family,
            by_family=by_family,
        )

        if not requested_families:
            requested_family = normalize_family(family)
            detected = ", ".join(available_families) if available_families else "none"
            if requested_family is None:
                reason = "No inferred family labels were detected among valid experiments."
            else:
                reason = (
                    f"No matching families available "
                    f"(requested={requested_family}, detected={detected})."
                )
            results.append(
                AnalysisRunResult(
                    summary=AnalysisRunSummary(
                        sf_mode=sf_mode,
                        family=requested_family,
                        total_count=0,
                        single_count=0,
                        fractionated_count=0,
                        train_count=0,
                        validation_count=0,
                        status="skipped",
                        reason=reason,
                    ),
                    train=(),
                    validation=(),
                    train_kind=fit_kind,
                    validation_kind=validate_kind,
                )
            )
            continue

        for family_index, run_family in enumerate(requested_families):
            total_count, single_count, fractionated_count = fitter.regimen_counts(
                family=run_family,
                experiments=experiments,
            )
            train, validation = fitter.build_train_validation_sets(
                fit_kind=fit_kind,
                validate_kind=validate_kind,
                family=run_family,
                experiments=experiments,
            )
            timing_diagnostics = fitter.compute_timing_diagnostics(train) if train else None

            if len(train) < 2:
                reason = (
                    f"Need at least two valid training experiments "
                    f"(fit_kind={fit_kind}, found={len(train)}, "
                    f"total={total_count}, single={single_count}, "
                    f"fractionated={fractionated_count})"
                )
                results.append(
                    AnalysisRunResult(
                        summary=AnalysisRunSummary(
                            sf_mode=sf_mode,
                            family=run_family,
                            total_count=total_count,
                            single_count=single_count,
                            fractionated_count=fractionated_count,
                            train_count=len(train),
                            validation_count=len(validation),
                            status="skipped",
                            reason=reason,
                        ),
                        train=tuple(train),
                        validation=tuple(validation),
                        train_kind=fit_kind,
                        validation_kind=validate_kind,
                        timing_diagnostics=timing_diagnostics,
                    )
                )
                continue

            fit_result = fitter.fit(
                experiments=train,
                train_kind=fit_kind,
                family=run_family,
                sf_mode=sf_mode,
            )
            validation_summary = None
            if validate_kind != "none" and validation:
                validation_summary = fitter.evaluate(validation, fit_result)

            bootstrap_summary = None
            if bootstrap > 0:
                seed = None
                if bootstrap_seed is not None:
                    seed = bootstrap_seed + mode_index * 1000 + family_index
                bootstrap_summary = fitter.bootstrap_fit(
                    sf_mode=sf_mode,
                    fit_kind=fit_kind,
                    family=run_family,
                    repeats=bootstrap,
                    seed=seed,
                )

            results.append(
                AnalysisRunResult(
                    summary=AnalysisRunSummary(
                        sf_mode=sf_mode,
                        family=run_family,
                        total_count=total_count,
                        single_count=single_count,
                        fractionated_count=fractionated_count,
                        train_count=len(train),
                        validation_count=len(validation),
                        status="ok",
                        alpha=fit_result.alpha,
                        beta=fit_result.beta,
                    ),
                    train=tuple(train),
                    validation=tuple(validation),
                    train_kind=fit_kind,
                    validation_kind=validate_kind,
                    fit_result=fit_result,
                    validation_summary=validation_summary,
                    bootstrap_summary=bootstrap_summary,
                    timing_diagnostics=timing_diagnostics,
                )
            )

    return results


def analyze_files(
    files: Optional[List[str]],
    sf_modes: Sequence[str],
    alpha: Optional[float],
    repair_half_time_hours: Optional[float],
    min_sf: float,
    fit_kind: RegimenKind,
    validate_kind: ValidationKind,
    family: Optional[str],
    by_family: bool,
    aggregate_regimens: bool,
    dedupe_regimens: bool,
    bootstrap: int,
    bootstrap_seed: Optional[int],
    verbose: bool,
    control_map: Optional[Mapping[object, Optional[object]]] = None,
) -> Tuple[Fitter, List[AnalysisRunResult]]:
    """Collect Excel files and run structured analysis results."""
    paths = [Path(file).resolve() for file in files] if files else sorted(Path.cwd().glob("*.xlsx"))
    fitter = Fitter(
        sf_mode=sf_modes[0] if sf_modes else "absolute",
        min_sf=min_sf,
        alpha_fixed=alpha,
        repair_half_time_hours=repair_half_time_hours,
        verbose=verbose,
        aggregate_regimens=aggregate_regimens,
        dedupe_regimens=dedupe_regimens,
    )
    fitter.collect(paths, control_map=control_map)
    results = analyze_fitter(
        fitter=fitter,
        sf_modes=sf_modes,
        fit_kind=fit_kind,
        validate_kind=validate_kind,
        family=family,
        by_family=by_family,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    return fitter, results


def report_analysis_run(fitter: Fitter, run: AnalysisRunResult) -> None:
    """Print one structured analysis result in the legacy CLI format."""
    if run.summary.status != "ok" or run.fit_result is None:
        family_label = run.summary.family_label
        print(
            f"{run.summary.reason} for sf_mode={run.summary.sf_mode}, "
            f"family={family_label}",
            file=sys.stderr,
        )
        return

    fitter.report_training(run.train, run.fit_result)
    diagnostics = run.timing_diagnostics
    if diagnostics is not None and diagnostics.repair_model_enabled:
        print("\n# Timing diagnostics:")
        print(
            "timing: "
            f"train={diagnostics.train_count} "
            f"fractionated={diagnostics.fractionated_count} "
            f"explicit_timing={diagnostics.explicit_timing_count} "
            f"explicit_fractionated={diagnostics.explicit_fractionated_count} "
            f"unique_fractionated_schedules={diagnostics.unique_schedule_count} "
            f"matched_patterns_with_multi_timing={diagnostics.same_fractions_multi_timing_count} "
            f"unique_quadratic={diagnostics.unique_quadratic_count} "
            f"design_rank={diagnostics.design_rank}"
        )
        if diagnostics.condition_number is not None:
            print(f"condition_number={diagnostics.condition_number:.2g}")
        for warning in diagnostics.warnings:
            print(f"WARNING timing: {warning}")

    if run.validation_kind != "none":
        if run.validation_summary is None:
            print("\n# Holdout validation:")
            print("No holdout experiments matched the requested validation subset.")
        else:
            fitter.report_validation(run.validation_kind, run.validation_summary)

    if run.bootstrap_summary is not None:
        fitter.report_bootstrap(run.bootstrap_summary)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit alpha and beta from normalized tumor-volume curves."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="List of .xlsx files. If omitted, all .xlsx files in cwd are used.",
    )
    parser.add_argument(
        "--sf",
        action="append",
        help=(
            "SF mode. Repeat the option or pass a comma-separated list. "
            "Examples: --sf absolute --sf absindex:2"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Fix alpha and fit only beta (example: --alpha 0.3)",
    )
    parser.add_argument(
        "--repair-half-time-hours",
        type=float,
        help=(
            "Repair half-time in hours for time-aware fractionation fitting. "
            "When omitted, the classic schedule-free LQ model is used."
        ),
    )
    parser.add_argument(
        "--min-sf",
        type=float,
        default=1.0,
        help="Drop experiments with SF >= MIN_SF (default: 1.0)",
    )
    parser.add_argument(
        "--fit-kind",
        choices=("all", "single", "fractionated"),
        default="all",
        help="Training subset: all | single | fractionated (default: all)",
    )
    parser.add_argument(
        "--validate-kind",
        choices=("none", "all", "single", "fractionated"),
        default="none",
        help="Holdout subset. Training files are excluded from validation (default: none)",
    )
    parser.add_argument(
        "--family",
        help="Restrict experiments to one inferred family label, e.g. y, p, n or e",
    )
    parser.add_argument(
        "--by-family",
        action="store_true",
        help="Run the analysis separately for each detected family label",
    )
    parser.add_argument(
        "--summary-csv",
        help="Write the batch or single-run summary to a CSV file",
    )
    parser.add_argument(
        "--aggregate-regimens",
        action="store_true",
        help=(
            "Average repeated experiments with the same family, fraction sizes, and timing "
            "into one weighted regimen"
        ),
    )
    parser.add_argument(
        "--dedupe-regimens",
        action="store_true",
        help="Keep only one experiment per identical family + fraction/timing regimen",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap replicates over animals/control groups (default: 0)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        help="Random seed for bootstrap replicates",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def run_fit(
    files: Optional[List[str]],
    sf_modes: Sequence[str],
    alpha: Optional[float],
    repair_half_time_hours: Optional[float],
    min_sf: float,
    fit_kind: RegimenKind,
    validate_kind: ValidationKind,
    family: Optional[str],
    by_family: bool,
    summary_csv: Optional[str],
    aggregate_regimens: bool,
    dedupe_regimens: bool,
    bootstrap: int,
    bootstrap_seed: Optional[int],
    verbose: bool,
) -> None:
    fitter, results = analyze_files(
        files=files,
        sf_modes=sf_modes,
        alpha=alpha,
        repair_half_time_hours=repair_half_time_hours,
        min_sf=min_sf,
        fit_kind=fit_kind,
        validate_kind=validate_kind,
        family=family,
        by_family=by_family,
        aggregate_regimens=aggregate_regimens,
        dedupe_regimens=dedupe_regimens,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        verbose=verbose,
        control_map=None,
    )
    summaries = [run.summary for run in results]

    for run in results:
        report_analysis_run(fitter, run)

    if by_family and summaries:
        fitter.report_analysis_summaries(summaries)
    if summary_csv and summaries:
        output_path = Path(summary_csv).expanduser().resolve()
        fitter.write_analysis_summaries_csv(output_path, summaries)
        print(f"\n# Summary CSV written to {output_path}")


def main() -> None:
    if USE_INLINE_PARAMS:
        run_fit(
            files=INLINE_FILES,
            sf_modes=INLINE_SF_MODES,
            alpha=INLINE_ALPHA,
            repair_half_time_hours=INLINE_REPAIR_HALF_TIME_HOURS,
            min_sf=INLINE_MIN_SF,
            fit_kind=INLINE_FIT_KIND,
            validate_kind=INLINE_VALIDATE_KIND,
            family=INLINE_FAMILY,
            by_family=INLINE_BY_FAMILY,
            summary_csv=INLINE_SUMMARY_CSV,
            aggregate_regimens=INLINE_AGGREGATE_REGIMENS,
            dedupe_regimens=INLINE_DEDUPE_REGIMENS,
            bootstrap=INLINE_BOOTSTRAP,
            bootstrap_seed=INLINE_BOOTSTRAP_SEED,
            verbose=INLINE_VERBOSE,
        )
        return

    args = parse_cli()
    run_fit(
        files=args.files,
        sf_modes=parse_sf_modes(args.sf),
        alpha=args.alpha,
        repair_half_time_hours=args.repair_half_time_hours,
        min_sf=args.min_sf,
        fit_kind=args.fit_kind,
        validate_kind=args.validate_kind,
        family=args.family,
        by_family=args.by_family,
        summary_csv=args.summary_csv,
        aggregate_regimens=args.aggregate_regimens,
        dedupe_regimens=args.dedupe_regimens,
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
