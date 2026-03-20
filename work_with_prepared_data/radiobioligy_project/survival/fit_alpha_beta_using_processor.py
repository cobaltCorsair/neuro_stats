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
ResponseMode = Literal["scalar", "curve"]
RequestedModelKind = Literal[
    "auto",
    "classic_lq",
    "repair_lq",
    "repair_biexp",
    "glq",
    "let_dependent",
    "linear",
    "lq_l",
    "lq_repop",
    "repair_repop",
]
ModelKind = Literal[
    "classic_lq",
    "repair_lq",
    "repair_biexp",
    "glq",
    "let_dependent",
    "linear",
    "lq_l",
    "lq_repop",
    "repair_repop",
]

# ---------------------- INLINE CONFIG -----------------
# Set USE_INLINE_PARAMS = True to run the script without CLI arguments.
USE_INLINE_PARAMS = False
INLINE_FILES: Optional[List[str]] = None
INLINE_SF_MODES: List[str] = ["absolute"]
INLINE_ALPHA: Optional[float] = None
INLINE_REPAIR_HALF_TIME_HOURS: Optional[float] = None
INLINE_REPAIR_HALF_TIME_FAST_HOURS: Optional[float] = None
INLINE_REPAIR_HALF_TIME_SLOW_HOURS: Optional[float] = None
INLINE_REPAIR_FAST_FRACTION: float = 0.6
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
INLINE_RESPONSE_MODE: ResponseMode = "scalar"
INLINE_MODEL_KIND: RequestedModelKind = "auto"
INLINE_COMPARE_MODELS = False
INLINE_CROSS_VALIDATE_LOO = True
INLINE_VERBOSE = True
# -----------------------------------------------------

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
GR_SUFFIX = re.compile(r"гр|gy", re.IGNORECASE)
FAMILY_TOKEN = re.compile(r"[A-Za-zА-Яа-я]+\d*|\d+")
KNOWN_FAMILIES = ("y", "p", "p_peak", "p_through", "n", "e", "c")
TIME_TOKEN = re.compile(r"\bt\s*=")
TIME_VALUE_WITH_UNIT = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>"
    r"(?:ч(?:\.|ас(?:а|ов)?)?)|"
    r"(?:hour|hours|hr|hrs)|"
    r"(?:мин(?:\.|ут(?:а|ы|)?)?)|"
    r"(?:minute|minutes|min|mins)|"
    r"(?:сут(?:\.|ки)?)|"
    r"(?:дн(?:\.|я|ей)?)|"
    r"(?:день|дня|дней)|"
    r"(?:day|days)"
    r")",
    re.IGNORECASE,
)
PROTON_PEAK_TOKENS = ("in_peak", "в_пике")
PROTON_THROUGH_TOKENS = ("прострел",)
FAMILY_LET_DEFAULTS: Dict[str, float] = {
    "y": 0.3,
    "e": 0.3,
    "p": 3.0,
    "p_peak": 12.0,
    "p_through": 2.0,
    "n": 20.0,
    "c": 100.0,
}
DURATION_MARKER = re.compile(r"(t_irr|tirr|tau|duration|dur|irradiation\s*duration)", re.IGNORECASE)
TIME_HOUR_MARKERS = ("ч", "час", "hour", "hours", "hr", "hrs")
TIME_MINUTE_MARKERS = ("мин", "minute", "minutes", "min", "mins")
TIME_DAY_MARKERS = ("сут", "дн", "день", "дня", "дней", "day", "days")


def parse_fractions(experiment_params: List[str]) -> List[float]:
    """Extract dose fractions from experiment parameters."""
    fractions: List[float] = []
    for token in experiment_params:
        token_lower = token.lower()
        if GR_SUFFIX.search(token_lower):
            for num in NUMBER.findall(token_lower):
                fractions.append(float(num.replace(",", ".")))
    return fractions


def _strip_time_prefix(token: str) -> Tuple[str, bool]:
    """Remove known time prefixes and report whether the token is a time token."""
    token_lower = token.strip().lower().replace(",", ".")
    is_time_token = False
    if token_lower.startswith("irradiation time="):
        token_lower = token_lower.split("=", 1)[1].strip()
        is_time_token = True
    if TIME_TOKEN.search(token_lower):
        token_lower = TIME_TOKEN.sub("", token_lower, count=1).strip()
        is_time_token = True
    token_lower = token_lower.lstrip("=:").strip()
    return token_lower, is_time_token


def _detect_time_unit_kind(text: str) -> Optional[Literal["hours", "minutes", "days"]]:
    """Return the unique time-unit category used in text or ``None`` if ambiguous."""
    text_lower = text.strip().lower()
    kinds: set[str] = set()
    if any(marker in text_lower for marker in TIME_HOUR_MARKERS):
        kinds.add("hours")
    if any(marker in text_lower for marker in TIME_MINUTE_MARKERS):
        kinds.add("minutes")
    if any(marker in text_lower for marker in TIME_DAY_MARKERS):
        kinds.add("days")
    if len(kinds) != 1:
        return None
    return next(iter(kinds))  # type: ignore[return-value]


def _time_kind_factor(
    kind: Literal["hours", "minutes", "days"],
    output_unit: Literal["days", "hours"],
) -> float:
    """Convert a parsed time-unit category into the requested output scale."""
    factor_days = {
        "minutes": 1.0 / (24.0 * 60.0),
        "hours": 1.0 / 24.0,
        "days": 1.0,
    }[kind]
    if output_unit == "days":
        return factor_days
    return factor_days * 24.0


def extract_time_values(
    token: str,
    *,
    require_t_token: bool,
    output_unit: Literal["days", "hours"],
) -> List[float]:
    """Extract one or many time values, including slash-separated mixed units."""
    token_body, is_time_token = _strip_time_prefix(token)
    if require_t_token and not is_time_token:
        return []
    if not token_body:
        return []

    total_numbers = NUMBER.findall(token_body)
    pair_matches = list(TIME_VALUE_WITH_UNIT.finditer(token_body))
    if pair_matches and len(pair_matches) == len(total_numbers):
        parsed_values: List[float] = []
        for match in pair_matches:
            kind = _detect_time_unit_kind(match.group("unit"))
            if kind is None:
                continue
            value = float(match.group("value").replace(",", "."))
            if not np.isfinite(value) or value < 0.0:
                continue
            parsed_values.append(value * _time_kind_factor(kind, output_unit))
        if parsed_values:
            return parsed_values

    segments = [segment.strip() for segment in re.split(r"\s*[/;|]+\s*", token_body) if segment.strip()]
    if not segments:
        segments = [token_body]

    global_kind = _detect_time_unit_kind(token_body)
    values: List[float] = []
    for segment in segments:
        segment_values = [float(num.replace(",", ".")) for num in NUMBER.findall(segment)]
        if not segment_values:
            continue
        kind = _detect_time_unit_kind(segment) or global_kind
        if kind is None:
            continue
        factor = _time_kind_factor(kind, output_unit)
        values.extend(value * factor for value in segment_values if np.isfinite(value) and value >= 0.0)
    return values


def _extract_interval_values_days(token: str) -> List[float]:
    """Extract one or many inter-fraction gaps in days from one metadata token."""
    return extract_time_values(token, require_t_token=True, output_unit="days")


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


def parse_irradiation_durations_hours(
    experiment_params: Sequence[str],
    fractions: Optional[Sequence[float]] = None,
) -> Tuple[float, ...]:
    """Extract optional per-fraction irradiation durations in hours."""
    doses = [float(dose) for dose in (fractions or parse_fractions(list(experiment_params)))]
    if not doses:
        return ()

    durations: List[float] = []
    for raw_token in experiment_params:
        token = str(raw_token).strip()
        token_lower = token.lower().replace(",", ".")
        if TIME_TOKEN.search(token_lower):
            continue
        if not DURATION_MARKER.search(token_lower):
            continue

        parsed_durations = extract_time_values(token, require_t_token=False, output_unit="hours")
        if parsed_durations:
            durations.extend(parsed_durations)
            continue

        values = [float(num.replace(",", ".")) for num in NUMBER.findall(token_lower)]
        if not values:
            continue

        factor = 1.0
        if any(unit in token_lower for unit in ("РјРёРЅ", "minute", "minutes", "min", "mins")):
            factor = 1.0 / 60.0
        elif any(unit in token_lower for unit in ("СЃСѓС‚", "РґРЅ", "РґРµРЅСЊ", "РґРЅСЏ", "РґРЅРµР№", "day", "days")):
            factor = 24.0
        durations.extend(
            value * factor for value in values if np.isfinite(value) and value >= 0.0
        )

    if not durations:
        return ()
    if len(durations) == 1:
        return tuple(float(durations[0]) for _ in doses)
    if len(durations) < len(doses):
        durations.extend([durations[-1]] * (len(doses) - len(durations)))
    return tuple(float(value) for value in durations[: len(doses)])


def parse_time_days(time_labels: Sequence[str]) -> Tuple[float, ...]:
    """Convert Excel time labels to numeric day values, falling back to indices."""
    days: List[float] = []
    for index, raw_label in enumerate(time_labels):
        label = str(raw_label).strip().replace(",", ".")
        try:
            days.append(float(label))
        except ValueError:
            days.append(float(index))

    if not days:
        return ()

    if any(not np.isfinite(day) for day in days):
        return tuple(float(index) for index in range(len(time_labels)))
    return tuple(days)


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


def format_schedule_days(schedule_days: Sequence[float]) -> str:
    """Format cumulative fraction times for reports and tables."""
    if not schedule_days:
        return "-"
    parts = []
    for day in schedule_days:
        hours = float(day) * 24.0
        if abs(hours) < 24.0:
            parts.append(f"{hours:g}h")
        else:
            parts.append(f"{float(day):g}d")
    return "[" + ", ".join(parts) + "]"


def interval_days_from_schedule(schedule_days: Sequence[float]) -> Tuple[float, ...]:
    """Convert cumulative fraction times into inter-fraction gaps."""
    if len(schedule_days) < 2:
        return ()
    return tuple(
        float(schedule_days[index] - schedule_days[index - 1])
        for index in range(1, len(schedule_days))
    )


def _format_interval_value_days(interval_days: float) -> str:
    """Render one interval in a compact operator-friendly form."""
    minutes = float(interval_days) * 24.0 * 60.0
    hours = float(interval_days) * 24.0
    rounded_hours = int(round(hours))
    if (
        abs(rounded_hours) >= 1
        and abs(rounded_hours) < 24
        and math.isclose(hours, rounded_hours, rel_tol=0.0, abs_tol=1.0e-4)
    ):
        return f"{rounded_hours} ч."
    if abs(minutes) < 60.0 and math.isclose(minutes, round(minutes), rel_tol=0.0, abs_tol=1.0e-4):
        return f"{int(round(minutes))} мин."
    if abs(hours) < 24.0:
        return f"{hours:g} ч."
    return f"{float(interval_days):g} сут."


def format_interval_values_days(interval_days: Sequence[float]) -> str:
    """Format inter-fraction gaps like ``t=1 ч./1 сут./1 ч.``."""
    if not interval_days:
        return "-"
    return "t=" + "/".join(_format_interval_value_days(value) for value in interval_days)


def format_schedule_intervals(schedule_days: Sequence[float]) -> str:
    """Format a cumulative schedule via intervals when possible."""
    intervals = interval_days_from_schedule(schedule_days)
    if not intervals:
        return "-"
    if any(not np.isfinite(value) for value in intervals) or any(value < -1.0e-9 for value in intervals):
        return format_schedule_days(schedule_days)
    return format_interval_values_days(intervals)


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


def compute_sf_time_day(
    mean_norm: np.ndarray,
    time_days: Sequence[float],
    mode: str,
) -> float:
    """Return the time point in days used by the scalar SF endpoint."""
    if len(mean_norm) < 2:
        raise ValueError("Need at least two time points to compute SF timing.")
    relative_time = np.asarray(time_days[: len(mean_norm)], dtype=float)
    if len(relative_time) != len(mean_norm) or np.any(~np.isfinite(relative_time)):
        relative_time = np.arange(len(mean_norm), dtype=float)
    relative_time = relative_time - float(relative_time[0])

    if mode.startswith("absindex:"):
        idx = int(mode.split(":", 1)[1])
        if idx >= len(mean_norm):
            raise IndexError(f"absindex {idx} out of range 0..{len(mean_norm) - 1}")
        return float(relative_time[idx])

    sf_index = int(np.nanargmin(mean_norm[1:])) + 1
    return float(relative_time[sf_index])


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
    stem = path.stem.lower().replace("в пике", "в_пике")
    tokens = [
        ("c" + token[1:]) if token.startswith("с") else token
        for token in FAMILY_TOKEN.findall(stem)
    ]

    is_proton = any(
        token == "p" or (token.startswith("p") and token[1:].isdigit())
        for token in tokens
    )
    if is_proton:
        if any(marker in stem for marker in PROTON_PEAK_TOKENS):
            return "p_peak"
        if any(marker in stem for marker in PROTON_THROUGH_TOKENS):
            return "p_through"

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
    let_kev_um: Optional[float] = None
    time_days: Tuple[float, ...] = ()
    schedule_days: Tuple[float, ...] = ()
    irradiation_duration_hours: Tuple[float, ...] = ()
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
    let_kev_um: Optional[float] = None
    sf_time_day: float = 0.0
    time_days: Tuple[float, ...] = ()
    curve_response: Tuple[float, ...] = ()
    control_relative_curve: Tuple[float, ...] = ()
    schedule_days: Tuple[float, ...] = ()
    repeat_count: int = 1
    sf_std: float = 0.0
    source_paths: Tuple[Path, ...] = ()
    has_explicit_timing: bool = False
    control_path: Optional[Path] = None
    initial_volume_mm3: Optional[float] = None
    irradiation_duration_hours: Tuple[float, ...] = ()

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
        return interval_days_from_schedule(self.resolved_schedule_days)

    @property
    def resolved_irradiation_duration_hours(self) -> Tuple[float, ...]:
        if len(self.irradiation_duration_hours) == self.fraction_count:
            return tuple(max(float(value), 0.0) for value in self.irradiation_duration_hours)
        return tuple(0.0 for _ in range(self.fraction_count))

    @staticmethod
    def _self_repair_g_factor(
        irradiation_duration_days: float,
        repair_rate_per_day: Optional[float],
    ) -> float:
        if (
            repair_rate_per_day is None
            or repair_rate_per_day <= 0.0
            or irradiation_duration_days <= 0.0
        ):
            return 1.0
        x_value = float(repair_rate_per_day) * float(irradiation_duration_days)
        if x_value <= 1.0e-10:
            return 1.0
        return float(2.0 * (x_value + math.exp(-x_value) - 1.0) / (x_value * x_value))

    def quadratic_term(self, repair_rate_per_day: Optional[float] = None) -> float:
        if repair_rate_per_day is None or repair_rate_per_day <= 0.0:
            return self.dose2_sum
        schedule = self.resolved_schedule_days
        durations_days = tuple(value / 24.0 for value in self.resolved_irradiation_duration_hours)
        term = 0.0
        for i, dose_i in enumerate(self.fractions):
            for j, dose_j in enumerate(self.fractions):
                if i == j:
                    g_value = self._self_repair_g_factor(durations_days[i], repair_rate_per_day)
                else:
                    delta_days = abs(schedule[i] - schedule[j])
                    g_value = math.exp(-repair_rate_per_day * delta_days)
                term += dose_i * dose_j * g_value
        return float(term)

    def quadratic_term_biexp(
        self,
        fast_rate_per_day: Optional[float],
        slow_rate_per_day: Optional[float],
        fast_fraction: float = 0.6,
    ) -> float:
        if (
            fast_rate_per_day is None
            or slow_rate_per_day is None
            or fast_rate_per_day <= 0.0
            or slow_rate_per_day <= 0.0
        ):
            return self.dose2_sum
        fast_fraction = float(np.clip(fast_fraction, 0.0, 1.0))
        schedule = self.resolved_schedule_days
        durations_days = tuple(value / 24.0 for value in self.resolved_irradiation_duration_hours)
        term = 0.0
        for i, dose_i in enumerate(self.fractions):
            for j, dose_j in enumerate(self.fractions):
                if i == j:
                    g = (
                        fast_fraction
                        * self._self_repair_g_factor(durations_days[i], fast_rate_per_day)
                        + (1.0 - fast_fraction)
                        * self._self_repair_g_factor(durations_days[i], slow_rate_per_day)
                    )
                else:
                    delta_days = abs(schedule[i] - schedule[j])
                    g = (
                        fast_fraction * math.exp(-fast_rate_per_day * delta_days)
                        + (1.0 - fast_fraction) * math.exp(-slow_rate_per_day * delta_days)
                    )
                term += dose_i * dose_j * g
        return float(term)

    def lea_catcheside_g_factor(self, repair_rate_per_day: Optional[float] = None) -> float:
        dose_total_sq = self.dose_sum * self.dose_sum
        if dose_total_sq <= 0.0:
            return 1.0
        return float(self.quadratic_term(repair_rate_per_day) / dose_total_sq)

    @property
    def initial_volume_cm3(self) -> Optional[float]:
        if self.initial_volume_mm3 is None or self.initial_volume_mm3 <= 0.0:
            return None
        return float(self.initial_volume_mm3) / 1000.0

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
    def regimen_key(
        self,
    ) -> Tuple[Optional[str], Optional[Path], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        return (
            self.family,
            self.control_path,
            tuple(round(dose, 8) for dose in self.fractions),
            tuple(round(day, 8) for day in self.resolved_schedule_days),
            tuple(round(duration, 8) for duration in self.resolved_irradiation_duration_hours),
        )

    @property
    def schedule_label(self) -> str:
        if self.fraction_count <= 1:
            return "-"
        return format_schedule_intervals(self.resolved_schedule_days)

    @property
    def curve_point_count(self) -> int:
        return max(len(self.curve_response) - 1, 0)

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
    model_kind: ModelKind = "classic_lq"
    response_mode: ResponseMode = "scalar"
    repair_half_time_hours: Optional[float] = None
    repair_half_time_fast_hours: Optional[float] = None
    repair_half_time_slow_hours: Optional[float] = None
    repair_fast_fraction: Optional[float] = None
    alpha_0: Optional[float] = None
    lambda_alpha: Optional[float] = None
    curve_clearance_rate: Optional[float] = None
    transition_dose: Optional[float] = None
    saturation_dose: Optional[float] = None
    lag_days: Optional[float] = None
    repopulation_rate: Optional[float] = None

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

    @property
    def repair_fast_rate_per_day(self) -> Optional[float]:
        if self.repair_half_time_fast_hours is None or self.repair_half_time_fast_hours <= 0.0:
            return None
        return math.log(2.0) * 24.0 / self.repair_half_time_fast_hours

    @property
    def repair_slow_rate_per_day(self) -> Optional[float]:
        if self.repair_half_time_slow_hours is None or self.repair_half_time_slow_hours <= 0.0:
            return None
        return math.log(2.0) * 24.0 / self.repair_half_time_slow_hours

    def effective_alpha(
        self,
        let_kev_um: Optional[float] = None,
        family: Optional[str] = None,
    ) -> float:
        if self.model_kind != "let_dependent":
            return float(self.alpha)
        if self.lambda_alpha is None:
            raise ValueError("LET-dependent prediction requires lambda_alpha.")
        if let_kev_um is not None and np.isfinite(let_kev_um):
            let_value = float(let_kev_um)
        else:
            family_key = normalize_family(family) or normalize_family(self.family)
            let_value = FAMILY_LET_DEFAULTS.get(family_key or "", 0.3)
        alpha_0 = self.alpha_0 if self.alpha_0 is not None else self.alpha
        return float(alpha_0 + self.lambda_alpha * let_value)

    def alpha_beta_ratio_for_experiment(self, experiment: TumorExperiment) -> Optional[float]:
        if self.beta <= 0.0:
            return None
        if self.model_kind == "let_dependent":
            return self.effective_alpha(experiment.let_kev_um, family=experiment.family) / self.beta
        return self.alpha_beta_ratio

    def _quadratic_term_for_experiment(self, experiment: TumorExperiment) -> float:
        if self.model_kind in ("repair_lq", "repair_repop"):
            return experiment.quadratic_term(self.repair_rate_per_day)
        if self.model_kind == "repair_biexp":
            return experiment.quadratic_term_biexp(
                self.repair_fast_rate_per_day,
                self.repair_slow_rate_per_day,
                self.repair_fast_fraction or 0.6,
            )
        return experiment.dose2_sum

    def compute_bed(self, experiment: TumorExperiment) -> float:
        alpha_beta_ratio = self.alpha_beta_ratio_for_experiment(experiment)
        if alpha_beta_ratio is None or alpha_beta_ratio <= 0.0:
            return float("nan")
        quadratic_term = self._quadratic_term_for_experiment(experiment)
        return float(experiment.dose_sum + quadratic_term / alpha_beta_ratio)

    def compute_eqd2(self, experiment: TumorExperiment) -> float:
        alpha_beta_ratio = self.alpha_beta_ratio_for_experiment(experiment)
        if alpha_beta_ratio is None or alpha_beta_ratio <= 0.0:
            return float("nan")
        bed = self.compute_bed(experiment)
        return float(bed / (1.0 + 2.0 / alpha_beta_ratio))

    def compute_g_factor(self, experiment: TumorExperiment) -> float:
        if self.model_kind in ("repair_lq", "repair_repop"):
            return experiment.lea_catcheside_g_factor(self.repair_rate_per_day)
        if self.model_kind == "repair_biexp":
            dose_total_sq = experiment.dose_sum * experiment.dose_sum
            if dose_total_sq <= 0.0:
                return 1.0
            return float(
                experiment.quadratic_term_biexp(
                    self.repair_fast_rate_per_day,
                    self.repair_slow_rate_per_day,
                    self.repair_fast_fraction or 0.6,
                )
                / dose_total_sq
            )
        return experiment.lea_catcheside_g_factor(None)

    def predict_sf(self, experiment: TumorExperiment) -> float:
        if self.model_kind == "linear":
            quadratic_term = 0.0
            exponent = self.alpha * experiment.dose_sum + self.beta * quadratic_term
        elif self.model_kind == "glq":
            if self.saturation_dose is None:
                raise ValueError("gLQ prediction requires saturation_dose.")
            exponent = Fitter.glq_exponent(
                experiment,
                alpha=self.alpha,
                beta=self.beta,
                saturation_dose=self.saturation_dose,
            )
        elif self.model_kind == "lq_l":
            if self.transition_dose is None:
                raise ValueError("LQ-L prediction requires transition_dose.")
            exponent = Fitter.lql_exponent(
                experiment,
                alpha=self.alpha,
                beta=self.beta,
                transition_dose=self.transition_dose,
            )
        elif self.model_kind == "lq_repop":
            if self.lag_days is None or self.repopulation_rate is None:
                raise ValueError("LQ + repopulation prediction requires lag_days and repopulation_rate.")
            base_exponent = self.alpha * experiment.dose_sum + self.beta * experiment.dose2_sum
            exponent = base_exponent - self.repopulation_rate * max(
                experiment.sf_time_day - self.lag_days,
                0.0,
            )
        elif self.model_kind == "let_dependent":
            exponent = (
                self.effective_alpha(experiment.let_kev_um, family=experiment.family) * experiment.dose_sum
                + self.beta * experiment.dose2_sum
            )
        elif self.model_kind == "repair_repop":
            if self.lag_days is None or self.repopulation_rate is None:
                raise ValueError(
                    "Repair-aware LQ + repopulation prediction requires lag_days and repopulation_rate."
                )
            quadratic_term = self._quadratic_term_for_experiment(experiment)
            base_exponent = self.alpha * experiment.dose_sum + self.beta * quadratic_term
            exponent = base_exponent - self.repopulation_rate * max(
                experiment.sf_time_day - self.lag_days,
                0.0,
            )
        elif self.model_kind == "repair_lq":
            quadratic_term = self._quadratic_term_for_experiment(experiment)
            exponent = self.alpha * experiment.dose_sum + self.beta * quadratic_term
        elif self.model_kind == "repair_biexp":
            quadratic_term = self._quadratic_term_for_experiment(experiment)
            exponent = self.alpha * experiment.dose_sum + self.beta * quadratic_term
        else:
            quadratic_term = experiment.dose2_sum
            exponent = self.alpha * experiment.dose_sum + self.beta * quadratic_term
        return float(np.exp(-exponent))

    def predict_curve(self, experiment: TumorExperiment) -> np.ndarray:
        if len(experiment.curve_response) == 0:
            raise ValueError("Experiment does not contain curve observations.")
        if self.curve_clearance_rate is None:
            raise ValueError("Curve prediction requires curve_clearance_rate.")

        if self.model_kind == "lq_repop":
            sf = float(np.exp(-(self.alpha * experiment.dose_sum + self.beta * experiment.dose2_sum)))
        elif self.model_kind == "glq":
            if self.saturation_dose is None:
                raise ValueError("gLQ curve prediction requires saturation_dose.")
            sf = float(
                np.exp(
                    -Fitter.glq_exponent(
                        experiment,
                        alpha=self.alpha,
                        beta=self.beta,
                        saturation_dose=self.saturation_dose,
                    )
                )
            )
        elif self.model_kind == "repair_repop":
            sf = float(
                np.exp(
                    -(
                        self.alpha * experiment.dose_sum
                        + self.beta * self._quadratic_term_for_experiment(experiment)
                    )
                )
            )
        else:
            sf = self.predict_sf(experiment)
        time_days = np.asarray(experiment.time_days[: len(experiment.curve_response)], dtype=float)
        time_days = time_days - float(time_days[0])
        control_relative = np.asarray(
            experiment.control_relative_curve[: len(experiment.curve_response)],
            dtype=float,
        )
        control_relative = np.clip(control_relative, 1.0e-8, None)
        response = sf + (1.0 - sf) * np.exp(-self.curve_clearance_rate * time_days) / control_relative
        if self.model_kind in ("lq_repop", "repair_repop"):
            if self.lag_days is None or self.repopulation_rate is None:
                raise ValueError("LQ + repopulation curve prediction requires lag_days and repopulation_rate.")
            response = response * np.exp(
                self.repopulation_rate * np.clip(time_days - self.lag_days, a_min=0.0, a_max=None)
            )
        response[0] = 1.0
        return np.asarray(response, dtype=float)


@dataclass(frozen=True)
class FitMetrics:
    """Error metrics for the fitted model on a given observation set."""

    point_count: int
    mae: float
    rmse: float
    mean_abs_log_error: float
    rss: float
    aic: Optional[float] = None
    r_squared: Optional[float] = None
    adjusted_r_squared: Optional[float] = None
    bic: Optional[float] = None


@dataclass(frozen=True)
class ModelComparisonRow:
    """One fitted candidate model with comparable error metrics."""

    model_kind: ModelKind
    status: str
    response_mode: ResponseMode
    alpha: Optional[float] = None
    beta: Optional[float] = None
    lambda_alpha: Optional[float] = None
    curve_clearance_rate: Optional[float] = None
    transition_dose: Optional[float] = None
    saturation_dose: Optional[float] = None
    lag_days: Optional[float] = None
    repopulation_rate: Optional[float] = None
    metrics: Optional[FitMetrics] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class PredictionRow:
    """Observed vs predicted SF for one holdout experiment."""

    experiment: TumorExperiment
    predicted_sf: float
    abs_error: float
    rel_error: float
    log_error: float

    @property
    def observed_sf(self) -> float:
        return float(self.experiment.sf)

    @property
    def residual(self) -> float:
        return float(self.predicted_sf - self.experiment.sf)

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
    response_mode: ResponseMode = "scalar"
    point_count: int = 0


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
class CrossValidationResult:
    """Leave-one-out predictive performance on the training experiments."""

    n_experiments: int
    n_successful: int
    cv_rmse: float
    cv_mae: float
    cv_r_squared: Optional[float]
    residuals: Tuple[float, ...]
    rows: Tuple[PredictionRow, ...]
    response_mode: ResponseMode = "scalar"


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
    repair_half_time_fast_hours: Optional[float] = None
    repair_half_time_slow_hours: Optional[float] = None
    repair_fast_fraction: Optional[float] = None
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
    response_mode: ResponseMode = "scalar"
    model_kind: str = "classic_lq"
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
    training_metrics: Optional[FitMetrics] = None
    validation_summary: Optional[ValidationSummary] = None
    bootstrap_summary: Optional[BootstrapSummary] = None
    cross_validation: Optional[CrossValidationResult] = None
    timing_diagnostics: Optional[TimingDiagnostics] = None
    model_comparison: Tuple[ModelComparisonRow, ...] = ()

    @property
    def label(self) -> str:
        return (
            f"sf={self.summary.sf_mode} | "
            f"response={self.summary.response_mode} | "
            f"model={self.summary.model_kind} | "
            f"family={self.summary.family_label} | "
            f"status={self.summary.status}"
        )

    def _mean_train_metric(self, getter) -> Optional[float]:
        if self.fit_result is None or not self.train:
            return None
        values = [float(getter(self.fit_result, experiment)) for experiment in self.train]
        finite_values = [value for value in values if np.isfinite(value)]
        if not finite_values:
            return None
        return float(np.mean(finite_values))

    @property
    def mean_train_bed(self) -> Optional[float]:
        return self._mean_train_metric(lambda fit, experiment: fit.compute_bed(experiment))

    @property
    def mean_train_eqd2(self) -> Optional[float]:
        return self._mean_train_metric(lambda fit, experiment: fit.compute_eqd2(experiment))

    @property
    def mean_train_g_factor(self) -> Optional[float]:
        return self._mean_train_metric(lambda fit, experiment: fit.compute_g_factor(experiment))

    def estimate_mean_train_tcp(self, cell_density: float = 1.0e7) -> Optional[float]:
        if self.fit_result is None or not self.train or cell_density <= 0.0:
            return None
        tcp_values: List[float] = []
        for experiment in self.train:
            initial_volume_cm3 = experiment.initial_volume_cm3
            if initial_volume_cm3 is None or initial_volume_cm3 <= 0.0:
                continue
            sf = self.fit_result.predict_sf(experiment)
            burden = initial_volume_cm3 * cell_density * sf
            tcp_values.append(0.0 if burden >= 700.0 else float(math.exp(-burden)))
        if not tcp_values:
            return None
        return float(np.mean(tcp_values))


@dataclass(frozen=True)
class InventoryRow:
    """One parsed file entry in the dataset inventory."""

    path: Path
    role: str
    family: Optional[str] = None
    fractions: Tuple[float, ...] = ()
    schedule_days: Tuple[float, ...] = ()
    has_explicit_timing: bool = False
    time_point_count: int = 0
    time_span_days: float = 0.0
    control_path: Optional[Path] = None
    fit_ready: bool = False
    notes: Tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        if self.role == "control":
            return "control"
        if self.role == "error":
            return "error"
        if not self.fractions:
            return "unparsed"
        return "single" if len(self.fractions) == 1 else "fractionated"

    @property
    def fractions_label(self) -> str:
        return format_fractions(self.fractions) if self.fractions else "-"

    @property
    def schedule_label(self) -> str:
        if self.kind != "fractionated":
            return "-"
        return format_schedule_intervals(self.schedule_days)

    @property
    def control_label(self) -> str:
        if self.role == "control":
            return "-"
        if self.control_path is None:
            return "unassigned"
        return self.control_path.name

    @property
    def notes_label(self) -> str:
        return "; ".join(self.notes)


@dataclass(frozen=True)
class InventoryModelSuitability:
    """Heuristic recommendation for which fitter model matches a dataset family."""

    model_kind: str
    status: str
    reason: str


@dataclass(frozen=True)
class InventoryFamilySummary:
    """Fit-readiness summary for one inferred family."""

    family: str
    parsed_count: int
    analyzable_count: int
    distinct_regimen_count: int
    single_count: int
    fractionated_count: int
    fit_ready: bool
    notes: Tuple[str, ...] = ()
    model_suitability: Tuple[InventoryModelSuitability, ...] = ()

    @property
    def recommended_models(self) -> Tuple[str, ...]:
        return tuple(
            item.model_kind for item in self.model_suitability if item.status == "recommended"
        )

    @property
    def possible_models(self) -> Tuple[str, ...]:
        return tuple(item.model_kind for item in self.model_suitability if item.status == "possible")


@dataclass(frozen=True)
class InventoryReport:
    """Inventory rows plus family-level fit-readiness summary."""

    rows: Tuple[InventoryRow, ...]
    family_summaries: Tuple[InventoryFamilySummary, ...]
    control_count: int
    experiment_count: int


class Fitter:
    """Collect experiments and fit alpha/beta."""

    def __init__(
        self,
        sf_mode: str,
        min_sf: float,
        alpha_fixed: Optional[float],
        verbose: bool,
        repair_half_time_hours: Optional[float] = None,
        repair_half_time_fast_hours: Optional[float] = None,
        repair_half_time_slow_hours: Optional[float] = None,
        repair_fast_fraction: float = 0.6,
        aggregate_regimens: bool = False,
        dedupe_regimens: bool = False,
    ):
        self.sf_mode = sf_mode
        self.min_sf = min_sf
        self.alpha_fixed = alpha_fixed
        self.repair_half_time_hours = repair_half_time_hours
        self.repair_half_time_fast_hours = repair_half_time_fast_hours
        self.repair_half_time_slow_hours = repair_half_time_slow_hours
        self.repair_fast_fraction = float(np.clip(repair_fast_fraction, 0.0, 1.0))
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

    @property
    def repair_fast_rate_per_day(self) -> Optional[float]:
        if self.repair_half_time_fast_hours is None or self.repair_half_time_fast_hours <= 0.0:
            return None
        return math.log(2.0) * 24.0 / self.repair_half_time_fast_hours

    @property
    def repair_slow_rate_per_day(self) -> Optional[float]:
        if self.repair_half_time_slow_hours is None or self.repair_half_time_slow_hours <= 0.0:
            return None
        return math.log(2.0) * 24.0 / self.repair_half_time_slow_hours

    def _quadratic_term_for_model(
        self,
        experiment: TumorExperiment,
        model_kind: ModelKind,
        repair_rate_per_day: Optional[float] = None,
    ) -> float:
        if model_kind in ("repair_lq", "repair_repop"):
            return experiment.quadratic_term(repair_rate_per_day)
        if model_kind == "repair_biexp":
            return experiment.quadratic_term_biexp(
                self.repair_fast_rate_per_day,
                self.repair_slow_rate_per_day,
                self.repair_fast_fraction,
            )
        return experiment.dose2_sum

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
        if curve_len == 0:
            return None

        curve_response = mean_norm / mean_norm[0]
        time_days = raw_experiment.time_days[:curve_len]
        if len(time_days) != curve_len:
            time_days = tuple(float(index) for index in range(curve_len))
        sf_time_day = compute_sf_time_day(mean_norm, time_days, sf_mode)
        control_relative_curve = tuple(
            float(value) for value in (control_curve[:curve_len] / control_curve[0])
        )

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
            let_kev_um=raw_experiment.let_kev_um,
            sf_time_day=float(sf_time_day),
            time_days=tuple(float(day) for day in time_days),
            curve_response=tuple(float(value) for value in curve_response),
            control_relative_curve=control_relative_curve,
            schedule_days=raw_experiment.schedule_days,
            repeat_count=1,
            sf_std=0.0,
            source_paths=(raw_experiment.path,),
            has_explicit_timing=raw_experiment.has_explicit_timing,
            control_path=raw_experiment.control_path,
            initial_volume_mm3=(
                float(mean_abs[0])
                if len(mean_abs) > 0 and np.isfinite(mean_abs[0]) and mean_abs[0] > 0.0
                else None
            ),
            irradiation_duration_hours=raw_experiment.irradiation_duration_hours,
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
                Tuple[Optional[str], Optional[Path], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
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
            Tuple[Optional[str], Optional[Path], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
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
            longest_curve_source = max(group, key=lambda experiment: len(experiment.curve_response))
            curve_response: Tuple[float, ...] = ()
            if any(experiment.curve_response for experiment in group):
                max_curve_len = max(len(experiment.curve_response) for experiment in group)
                padded_curves = [
                    np.pad(
                        np.asarray(experiment.curve_response, dtype=float),
                        (0, max_curve_len - len(experiment.curve_response)),
                        constant_values=np.nan,
                    )
                    for experiment in group
                ]
                curve_response = tuple(
                    float(value) for value in np.nanmean(np.vstack(padded_curves), axis=0)
                )

            aggregated.append(
                TumorExperiment(
                    path=first.path,
                    fractions=first.fractions,
                    sf=float(np.mean(sfs)),
                    family=first.family,
                    let_kev_um=float(
                        np.mean(
                            [
                                experiment.let_kev_um
                                for experiment in group
                                if experiment.let_kev_um is not None
                            ]
                        )
                    )
                    if any(experiment.let_kev_um is not None for experiment in group)
                    else None,
                    sf_time_day=float(np.mean([experiment.sf_time_day for experiment in group])),
                    time_days=longest_curve_source.time_days[: len(curve_response)],
                    curve_response=curve_response,
                    control_relative_curve=longest_curve_source.control_relative_curve[
                        : len(curve_response)
                    ],
                    schedule_days=first.schedule_days,
                    repeat_count=len(group),
                    sf_std=float(np.std(sfs, ddof=0)),
                    source_paths=source_paths,
                    has_explicit_timing=first.has_explicit_timing,
                    control_path=first.control_path,
                    initial_volume_mm3=float(
                        np.mean(
                            [
                                experiment.initial_volume_mm3
                                for experiment in group
                                if experiment.initial_volume_mm3 is not None
                            ]
                        )
                    )
                    if any(experiment.initial_volume_mm3 is not None for experiment in group)
                    else None,
                    irradiation_duration_hours=first.irradiation_duration_hours,
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

    @staticmethod
    def _inventory_regimen_key(row: InventoryRow) -> Tuple[str, Tuple[float, ...], Tuple[float, ...]]:
        return (
            row.family or "",
            tuple(round(dose, 8) for dose in row.fractions),
            tuple(round(day, 8) for day in row.schedule_days),
        )

    @staticmethod
    def _inventory_design_key(row: InventoryRow) -> Tuple[float, float]:
        return (
            round(sum(row.fractions), 8),
            round(sum(dose * dose for dose in row.fractions), 8),
        )

    @staticmethod
    def _build_inventory_model_suitability(
        family_rows: Sequence[InventoryRow],
    ) -> Tuple[InventoryModelSuitability, ...]:
        if not family_rows:
            return ()

        classic_designs = {Fitter._inventory_design_key(row) for row in family_rows}
        fractionated_rows = [row for row in family_rows if row.kind == "fractionated"]
        explicit_fractionated_rows = [
            row for row in fractionated_rows if row.has_explicit_timing and len(row.schedule_days) > 1
        ]
        schedules_by_fraction: Dict[Tuple[float, ...], set[Tuple[float, ...]]] = {}
        has_subday_interval = False
        for row in explicit_fractionated_rows:
            fraction_key = tuple(round(dose, 8) for dose in row.fractions)
            schedule_key = tuple(round(day, 8) for day in row.schedule_days)
            schedules_by_fraction.setdefault(fraction_key, set()).add(schedule_key)
            if any(
                (row.schedule_days[index] - row.schedule_days[index - 1]) < 1.0
                for index in range(1, len(row.schedule_days))
            ):
                has_subday_interval = True
        matched_timing_patterns = sum(
            1 for schedule_keys in schedules_by_fraction.values() if len(schedule_keys) >= 2
        )

        max_fraction_dose = max(
            (max(row.fractions) for row in family_rows if row.fractions),
            default=0.0,
        )
        high_dose_designs = {
            Fitter._inventory_design_key(row)
            for row in family_rows
            if row.fractions and max(row.fractions) >= 10.0
        }
        ablative_designs = {
            Fitter._inventory_design_key(row)
            for row in family_rows
            if row.fractions and max(row.fractions) >= 15.0
        }

        rich_curve_rows = [
            row for row in family_rows if row.time_point_count >= 5 and row.time_span_days >= 10.0
        ]
        medium_curve_rows = [
            row for row in family_rows if row.time_point_count >= 4 and row.time_span_days >= 7.0
        ]

        assessments: List[InventoryModelSuitability] = []

        if len(classic_designs) >= 3:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="classic_lq",
                    status="recommended",
                    reason=f"{len(classic_designs)} distinct (D, Σd²) regimens",
                )
            )
        elif len(classic_designs) >= 2:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="classic_lq",
                    status="possible",
                    reason=f"{len(classic_designs)} distinct dose-response regimens",
                )
            )
        else:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="classic_lq",
                    status="not_ready",
                    reason="need at least 2 distinct (D, Σd²) regimens",
                )
            )

        repair_reason = "need timed fractionated regimens"
        repair_status = "not_ready"
        if matched_timing_patterns >= 1:
            interval_text = " with sub-day intervals" if has_subday_interval else ""
            repair_status = "recommended"
            repair_reason = (
                f"{matched_timing_patterns} matched fractionation pattern(s) with different timing{interval_text}"
            )
        elif len(explicit_fractionated_rows) >= 3:
            repair_status = "possible"
            repair_reason = (
                f"{len(explicit_fractionated_rows)} timed fractionated regimens, but no matched dose pattern"
            )
        elif fractionated_rows and not explicit_fractionated_rows:
            repair_reason = "fractionated regimens exist, but t= timing metadata is missing"
        elif explicit_fractionated_rows:
            repair_status = "possible" if len(explicit_fractionated_rows) >= 2 else "not_ready"
            repair_reason = (
                f"{len(explicit_fractionated_rows)} timed fractionated regimen(s); interval contrast is limited"
            )
        assessments.append(
            InventoryModelSuitability(
                model_kind="repair_lq",
                status=repair_status,
                reason=repair_reason,
            )
        )

        repair_biexp_status = "not_ready"
        repair_biexp_reason = "need timed fractionated regimens with short-interval contrast"
        if matched_timing_patterns >= 1 and has_subday_interval:
            repair_biexp_status = "recommended"
            repair_biexp_reason = (
                f"{matched_timing_patterns} matched fractionation pattern(s) include sub-day timing contrast"
            )
        elif repair_status != "not_ready":
            repair_biexp_status = "possible"
            repair_biexp_reason = (
                "timed fractionation is available, but short-interval contrast is limited for bi-exponential repair"
            )
        assessments.append(
            InventoryModelSuitability(
                model_kind="repair_biexp",
                status=repair_biexp_status,
                reason=repair_biexp_reason,
            )
        )

        if len(classic_designs) >= 3 and len(high_dose_designs) >= 2:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="lq_l",
                    status="recommended",
                    reason=(
                        f"{len(high_dose_designs)} high-dose regimens; "
                        f"max fraction {max_fraction_dose:g} Gy"
                    ),
                )
            )
        elif len(classic_designs) >= 2 and len(high_dose_designs) >= 1:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="lq_l",
                    status="possible",
                    reason=f"high-dose coverage is present, but limited to {len(high_dose_designs)} regimen(s)",
                )
            )
        else:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="lq_l",
                    status="not_ready",
                    reason="need multiple high-dose regimens (max fraction >= 10 Gy)",
                )
            )

        if len(high_dose_designs) >= 3 and len(ablative_designs) >= 2:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="glq",
                    status="recommended",
                    reason=(
                        f"broad high-dose coverage up to {max_fraction_dose:g} Gy "
                        f"across {len(high_dose_designs)} regimens"
                    ),
                )
            )
        elif len(high_dose_designs) >= 2:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="glq",
                    status="possible",
                    reason=f"{len(high_dose_designs)} high-dose regimens, but ablative range is limited",
                )
            )
        else:
            assessments.append(
                InventoryModelSuitability(
                    model_kind="glq",
                    status="not_ready",
                    reason="need a broader high-dose range to probe quadratic saturation",
                )
            )

        repop_status = "not_ready"
        repop_reason = "need longer multi-point follow-up curves"
        if len(rich_curve_rows) >= 3:
            repop_status = "recommended"
            repop_reason = (
                f"{len(rich_curve_rows)} curves with >=5 time points across >=10 days"
            )
        elif len(medium_curve_rows) >= 2:
            repop_status = "possible"
            repop_reason = (
                f"{len(medium_curve_rows)} multi-day curves; lag/repopulation may be weakly constrained"
            )
        assessments.append(
            InventoryModelSuitability(
                model_kind="lq_repop",
                status=repop_status,
                reason=repop_reason,
            )
        )

        repair_repop_status = "not_ready"
        repair_repop_reason = "need both timed fractionation contrast and long follow-up curves"
        if repair_status == "recommended" and repop_status == "recommended":
            repair_repop_status = "recommended"
            repair_repop_reason = "timed fractionation contrast and long follow-up are both present"
        elif repair_status != "not_ready" and repop_status != "not_ready":
            repair_repop_status = "possible"
            repair_repop_reason = "partial timing contrast and delayed follow-up are both available"
        elif repair_status == "not_ready" and repop_status != "not_ready":
            repair_repop_reason = f"repair signal is weak: {repair_reason}"
        elif repair_status != "not_ready" and repop_status == "not_ready":
            repair_repop_reason = f"repopulation signal is weak: {repop_reason}"
        assessments.append(
            InventoryModelSuitability(
                model_kind="repair_repop",
                status=repair_repop_status,
                reason=repair_repop_reason,
            )
        )

        return tuple(assessments)

    @staticmethod
    def inspect_files(
        files: Sequence[Path],
        control_map: Optional[Mapping[object, Optional[object]]] = None,
    ) -> InventoryReport:
        """Build a file-by-file inventory before running the fit."""
        resolved_files = [Path(path).expanduser().resolve() for path in files]
        normalized_control_map = Fitter._normalize_control_map(control_map)
        control_paths = sorted(path for path in resolved_files if is_control_file(path))
        control_set = set(control_paths)
        sole_control_path = control_paths[0] if len(control_paths) == 1 else None

        rows: List[InventoryRow] = []
        base_ready_by_path: Dict[Path, bool] = {}
        parsed_ready_by_path: Dict[Path, bool] = {}

        for path in resolved_files:
            if path in control_set:
                rows.append(
                    InventoryRow(
                        path=path,
                        role="control",
                        notes=("control file",),
                    )
                )
                continue

            notes: List[str] = []
            family = infer_radiation_family(path)
            fractions: Tuple[float, ...] = ()
            schedule_days: Tuple[float, ...] = ()
            has_explicit_timing = False
            time_point_count = 0
            time_span_days = 0.0
            role = "experiment"

            try:
                params, time_data, _, _ = process_tumor_data_excel(str(path))
                fractions = tuple(parse_fractions(params))
                if fractions:
                    schedule_days, has_explicit_timing = parse_schedule_days(params, fractions)
                time_days = parse_time_days(time_data)
                time_point_count = len(time_days)
                if time_days:
                    time_span_days = float(max(time_days) - min(time_days))
                else:
                    notes.append("dose fractions were not parsed")
            except Exception as exc:
                role = "error"
                notes.append(f"parse failed: {exc}")

            if family is None:
                notes.append("family not inferred from file name")
            elif family == "p":
                notes.append("proton context is not specified as peak or through")

            if fractions and len(fractions) > 1 and not has_explicit_timing:
                notes.append("fractionated regimen has no explicit t= timing metadata")

            control_path: Optional[Path] = None
            if not control_paths:
                notes.append("no control files loaded")
            elif normalized_control_map is not None:
                if path in normalized_control_map:
                    control_path = normalized_control_map[path]
                    if control_path is None:
                        notes.append("control is set to pooled average")
                elif sole_control_path is not None:
                    control_path = sole_control_path
                else:
                    notes.append("control is not assigned")
            elif sole_control_path is not None:
                control_path = sole_control_path
            else:
                notes.append("multiple control files loaded; assign one explicitly")

            if control_path is not None and control_path not in control_set:
                notes.append(f"assigned control {control_path.name} is not loaded")

            base_ready = (
                role == "experiment"
                and family is not None
                and bool(fractions)
                and control_path is not None
                and control_path in control_set
            )
            parsed_ready = role == "experiment" and family is not None and bool(fractions)

            rows.append(
                InventoryRow(
                    path=path,
                    role=role,
                    family=family,
                    fractions=fractions,
                    schedule_days=schedule_days,
                    has_explicit_timing=has_explicit_timing,
                    time_point_count=time_point_count,
                    time_span_days=time_span_days,
                    control_path=control_path if control_path in control_set else None,
                    fit_ready=False,
                    notes=tuple(notes),
                )
            )
            base_ready_by_path[path] = base_ready
            parsed_ready_by_path[path] = parsed_ready

        family_rows: Dict[str, List[InventoryRow]] = {}
        analyzable_counts: Dict[str, int] = {}
        regimen_counts: Dict[Tuple[str, Tuple[float, ...], Tuple[float, ...]], int] = {}
        for row in rows:
            if row.family is None or not parsed_ready_by_path.get(row.path, False):
                continue
            family_rows.setdefault(row.family, []).append(row)
            regimen_key = Fitter._inventory_regimen_key(row)
            regimen_counts[regimen_key] = regimen_counts.get(regimen_key, 0) + 1
            if base_ready_by_path.get(row.path, False):
                analyzable_counts[row.family] = analyzable_counts.get(row.family, 0) + 1

        family_summaries: List[InventoryFamilySummary] = []
        updated_rows: List[InventoryRow] = []
        family_summary_map: Dict[str, InventoryFamilySummary] = {}
        for family, family_items in sorted(family_rows.items()):
            analyzable_count = analyzable_counts.get(family, 0)
            single_count = sum(1 for row in family_items if row.kind == "single")
            fractionated_count = sum(1 for row in family_items if row.kind == "fractionated")
            distinct_regimen_count = len(
                {Fitter._inventory_regimen_key(row) for row in family_items}
            )
            notes: List[str] = []
            if len(family_items) < 2:
                notes.append("fewer than 2 parsed experiments")
            if analyzable_count < 2:
                notes.append("fewer than 2 analyzable experiments")
            if analyzable_count < len(family_items):
                notes.append(
                    f"controls unresolved for {len(family_items) - analyzable_count} parsed experiment(s)"
                )
            if single_count == 0:
                notes.append("no single-dose experiments")
            if fractionated_count == 0:
                notes.append("no fractionated experiments")
            if family == "p":
                notes.append("generic proton family; peak/through is not encoded")
            model_suitability = Fitter._build_inventory_model_suitability(family_items)
            summary = InventoryFamilySummary(
                family=family,
                parsed_count=len(family_items),
                analyzable_count=analyzable_count,
                distinct_regimen_count=distinct_regimen_count,
                single_count=single_count,
                fractionated_count=fractionated_count,
                fit_ready=analyzable_count >= 2,
                notes=tuple(notes),
                model_suitability=model_suitability,
            )
            family_summaries.append(summary)
            family_summary_map[family] = summary

        for row in rows:
            notes = list(row.notes)
            summary = family_summary_map.get(row.family or "")
            fit_ready = False
            if base_ready_by_path.get(row.path, False) and summary is not None:
                fit_ready = summary.fit_ready
                regimen_key = Fitter._inventory_regimen_key(row)
                repeat_count = regimen_counts.get(regimen_key, 1)
                if repeat_count > 1:
                    notes.append(f"repeat regimen detected ({repeat_count} files)")
                for summary_note in summary.notes:
                    if summary_note not in notes:
                        notes.append(summary_note)

            updated_rows.append(
                InventoryRow(
                    path=row.path,
                    role=row.role,
                    family=row.family,
                    fractions=row.fractions,
                    schedule_days=row.schedule_days,
                    has_explicit_timing=row.has_explicit_timing,
                    time_point_count=row.time_point_count,
                    time_span_days=row.time_span_days,
                    control_path=row.control_path,
                    fit_ready=fit_ready,
                    notes=tuple(notes),
                )
            )

        return InventoryReport(
            rows=tuple(updated_rows),
            family_summaries=tuple(family_summaries),
            control_count=len(control_paths),
            experiment_count=sum(1 for row in updated_rows if row.role != "control"),
        )

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
            params, time_data, _, volumes = process_tumor_data_excel(str(path))
            fractions = tuple(parse_fractions(params))
            if not fractions:
                if self.verbose:
                    print(f"WARNING {path.name}: dose fractions were not parsed -> skip")
                continue
            schedule_days, has_explicit_timing = parse_schedule_days(params, fractions)
            irradiation_duration_hours = parse_irradiation_durations_hours(params, fractions)
            time_days = parse_time_days(time_data)
            family = infer_radiation_family(path)

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
                    family=family,
                    volumes=np.asarray(volumes, dtype=float),
                    let_kev_um=FAMILY_LET_DEFAULTS.get(family or ""),
                    time_days=time_days,
                    schedule_days=schedule_days,
                    irradiation_duration_hours=irradiation_duration_hours,
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

    @staticmethod
    def _fit_alpha_for_linear_model(
        dose_sum: np.ndarray,
        sf: np.ndarray,
        sigma_log_sf: np.ndarray,
    ) -> float:
        y_log = -np.log(sf)
        weights = 1.0 / np.square(np.clip(sigma_log_sf, 1.0e-8, None))
        weighted_dose = weights * dose_sum
        denom = float(np.dot(weighted_dose, dose_sum))
        if denom == 0.0:
            raise RuntimeError("Cannot fit alpha for the linear model: denominator is zero.")
        return max(0.0, float(np.dot(weighted_dose, y_log) / denom))

    @staticmethod
    def lql_fraction_kill(
        dose: float,
        alpha: float,
        beta: float,
        transition_dose: float,
    ) -> float:
        transition_dose = max(float(transition_dose), 1.0e-8)
        dose = float(dose)
        if dose <= transition_dose:
            return alpha * dose + beta * dose * dose
        slope = alpha + 2.0 * beta * transition_dose
        return (
            alpha * transition_dose
            + beta * transition_dose * transition_dose
            + slope * (dose - transition_dose)
        )

    @staticmethod
    def lql_exponent(
        experiment: TumorExperiment,
        alpha: float,
        beta: float,
        transition_dose: float,
    ) -> float:
        return float(
            sum(
                Fitter.lql_fraction_kill(
                    dose=dose,
                    alpha=alpha,
                    beta=beta,
                    transition_dose=transition_dose,
                )
                for dose in experiment.fractions
            )
        )

    @staticmethod
    def glq_fraction_kill(
        dose: float,
        alpha: float,
        beta: float,
        saturation_dose: float,
    ) -> float:
        saturation_dose = max(float(saturation_dose), 1.0e-8)
        dose = float(dose)
        return alpha * dose + beta * dose * dose / (1.0 + dose / saturation_dose)

    @staticmethod
    def glq_exponent(
        experiment: TumorExperiment,
        alpha: float,
        beta: float,
        saturation_dose: float,
    ) -> float:
        return float(
            sum(
                Fitter.glq_fraction_kill(
                    dose=dose,
                    alpha=alpha,
                    beta=beta,
                    saturation_dose=saturation_dose,
                )
                for dose in experiment.fractions
            )
        )

    @staticmethod
    def let_dependent_exponent(
        experiment: TumorExperiment,
        alpha_0: float,
        lambda_alpha: float,
        beta_0: float,
    ) -> float:
        let_value = (
            float(experiment.let_kev_um)
            if experiment.let_kev_um is not None and np.isfinite(experiment.let_kev_um)
            else FAMILY_LET_DEFAULTS.get(experiment.family or "", 0.3)
        )
        alpha_eff = alpha_0 + lambda_alpha * let_value
        return float(alpha_eff * experiment.dose_sum + beta_0 * experiment.dose2_sum)

    @staticmethod
    def _initial_transition_dose(experiments: Sequence[TumorExperiment]) -> float:
        max_fractions = [max(experiment.fractions) for experiment in experiments if experiment.fractions]
        if not max_fractions:
            return 6.0
        return max(1.0, float(np.median(np.asarray(max_fractions, dtype=float))))

    @staticmethod
    def _initial_saturation_dose(experiments: Sequence[TumorExperiment]) -> float:
        max_fractions = [max(experiment.fractions) for experiment in experiments if experiment.fractions]
        if not max_fractions:
            return 6.0
        return max(1.0, float(np.median(np.asarray(max_fractions, dtype=float))))

    @staticmethod
    def _initial_lag_days(experiments: Sequence[TumorExperiment]) -> float:
        sf_times = [experiment.sf_time_day for experiment in experiments if np.isfinite(experiment.sf_time_day)]
        if not sf_times:
            return 3.0
        return max(0.0, float(np.median(np.asarray(sf_times, dtype=float)) / 2.0))

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
    def _let_values(experiments: Sequence[TumorExperiment]) -> np.ndarray:
        return np.asarray(
            [
                float(experiment.let_kev_um)
                if experiment.let_kev_um is not None and np.isfinite(experiment.let_kev_um)
                else FAMILY_LET_DEFAULTS.get(experiment.family or "", 0.3)
                for experiment in experiments
            ],
            dtype=float,
        )

    def _fit_scalar_let_parameters(
        self,
        experiments: Sequence[TumorExperiment],
    ) -> Tuple[float, float, float]:
        if len(experiments) < 3:
            raise RuntimeError("LET-dependent fitting needs at least three experiments.")

        let_values = self._let_values(experiments)
        if len({round(value, 8) for value in let_values}) < 2:
            raise RuntimeError("LET-dependent fitting needs at least two distinct LET levels.")

        dose_sum = np.asarray([experiment.dose_sum for experiment in experiments], dtype=float)
        dose2_sum = np.asarray([experiment.dose2_sum for experiment in experiments], dtype=float)
        sf = np.asarray([experiment.sf for experiment in experiments], dtype=float)
        sigma_sf = np.asarray(
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

        design = np.column_stack((dose_sum, dose_sum * let_values, dose2_sum))
        y_log = -np.log(np.clip(sf, 1.0e-8, 1.0))
        guess, *_ = np.linalg.lstsq(design, y_log, rcond=None)
        guess = np.asarray(guess, dtype=float)
        if len(guess) != 3 or np.any(~np.isfinite(guess)):
            guess = np.array([0.05, 0.001, 0.01], dtype=float)
        guess = np.clip(guess, 0.0, None)
        experiment_index = np.arange(len(experiments), dtype=float)

        if self.alpha_fixed is not None:
            params, _ = self._safe_curve_fit(
                lambda index, lambda_alpha, beta_0: self._let_scalar_model(
                    index,
                    self.alpha_fixed,
                    lambda_alpha,
                    beta_0,
                    experiments,
                ),
                experiment_index,
                sf,
                p0=(max(guess[1], 1.0e-8), max(guess[2], 1.0e-8)),
                bounds=((0.0, 0.0), (np.inf, np.inf)),
                sigma=np.clip(sigma_sf, 1.0e-8, None),
                absolute_sigma=False,
                maxfev=30000,
            )
            return float(self.alpha_fixed), float(params[0]), float(params[1])

        params, _ = self._safe_curve_fit(
            lambda index, alpha_0, lambda_alpha, beta_0: self._let_scalar_model(
                index,
                alpha_0,
                lambda_alpha,
                beta_0,
                experiments,
            ),
            experiment_index,
            sf,
            p0=(max(guess[0], 1.0e-8), max(guess[1], 1.0e-8), max(guess[2], 1.0e-8)),
            bounds=((0.0, 0.0, 0.0), (np.inf, np.inf, np.inf)),
            sigma=np.clip(sigma_sf, 1.0e-8, None),
            absolute_sigma=False,
            maxfev=30000,
        )
        return float(params[0]), float(params[1]), float(params[2])

    def _fit_curve_let_parameters(
        self,
        experiments: Sequence[TumorExperiment],
    ) -> Tuple[float, float, float, float]:
        let_values = self._let_values(experiments)
        if len({round(value, 8) for value in let_values}) < 2:
            raise RuntimeError("LET-dependent curve fitting needs at least two distinct LET levels.")

        experiment_index, time_days, control_relative, observed_curve = self._to_lql_curve_arrays(experiments)
        alpha_0, lambda_alpha, beta_0 = self._fit_scalar_let_parameters(experiments)

        if self.alpha_fixed is not None:
            params, _ = self._safe_curve_fit(
                lambda xdata, lambda_alpha_param, beta_0_param, clearance_rate: self._let_curve_response_model(
                    xdata,
                    self.alpha_fixed,
                    lambda_alpha_param,
                    beta_0_param,
                    clearance_rate,
                    experiments,
                ),
                (experiment_index, time_days, control_relative),
                observed_curve,
                p0=(max(lambda_alpha, 1.0e-8), max(beta_0, 1.0e-8), 0.10),
                bounds=((0.0, 0.0, 0.0), (np.inf, np.inf, np.inf)),
                maxfev=30000,
            )
            return float(self.alpha_fixed), float(params[0]), float(params[1]), float(params[2])

        params, _ = self._safe_curve_fit(
            lambda xdata, alpha_0_param, lambda_alpha_param, beta_0_param, clearance_rate: self._let_curve_response_model(
                xdata,
                alpha_0_param,
                lambda_alpha_param,
                beta_0_param,
                clearance_rate,
                experiments,
            ),
            (experiment_index, time_days, control_relative),
            observed_curve,
            p0=(max(alpha_0, 1.0e-8), max(lambda_alpha, 1.0e-8), max(beta_0, 1.0e-8), 0.10),
            bounds=((0.0, 0.0, 0.0, 0.0), (np.inf, np.inf, np.inf, np.inf)),
            maxfev=30000,
        )
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])

    @staticmethod
    def _safe_curve_fit(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            return curve_fit(*args, **kwargs)

    @staticmethod
    def _to_arrays(
        experiments: Sequence[TumorExperiment],
        model_kind: ModelKind,
        repair_rate_per_day: Optional[float] = None,
        repair_fast_rate_per_day: Optional[float] = None,
        repair_slow_rate_per_day: Optional[float] = None,
        repair_fast_fraction: float = 0.6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dose_sum = np.array([experiment.dose_sum for experiment in experiments], dtype=float)
        if model_kind == "linear":
            quadratic_term = np.zeros(len(experiments), dtype=float)
        elif model_kind == "repair_biexp":
            quadratic_term = np.array(
                [
                    experiment.quadratic_term_biexp(
                        repair_fast_rate_per_day,
                        repair_slow_rate_per_day,
                        repair_fast_fraction,
                    )
                    for experiment in experiments
                ],
                dtype=float,
            )
        else:
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

    @staticmethod
    def _curve_response_model(
        xdata: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
        beta: float,
        clearance_rate: float,
    ) -> np.ndarray:
        dose_sum, quadratic_term, time_days, control_relative = xdata
        sf = np.exp(-(alpha * dose_sum + beta * quadratic_term))
        control_relative = np.clip(control_relative, 1.0e-8, None)
        return sf + (1.0 - sf) * np.exp(-clearance_rate * time_days) / control_relative

    @staticmethod
    def _lql_scalar_model(
        experiment_index: np.ndarray,
        alpha: float,
        beta: float,
        transition_dose: float,
        experiments: Sequence[TumorExperiment],
    ) -> np.ndarray:
        predicted = [
            math.exp(
                -Fitter.lql_exponent(
                    experiments[int(index)],
                    alpha=alpha,
                    beta=beta,
                    transition_dose=transition_dose,
                )
            )
            for index in np.asarray(experiment_index, dtype=int)
        ]
        return np.asarray(predicted, dtype=float)

    @staticmethod
    def _glq_scalar_model(
        experiment_index: np.ndarray,
        alpha: float,
        beta: float,
        saturation_dose: float,
        experiments: Sequence[TumorExperiment],
    ) -> np.ndarray:
        predicted = [
            math.exp(
                -Fitter.glq_exponent(
                    experiments[int(index)],
                    alpha=alpha,
                    beta=beta,
                    saturation_dose=saturation_dose,
                )
            )
            for index in np.asarray(experiment_index, dtype=int)
        ]
        return np.asarray(predicted, dtype=float)

    @staticmethod
    def _lql_curve_response_model(
        xdata: Tuple[np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
        beta: float,
        transition_dose: float,
        clearance_rate: float,
        experiments: Sequence[TumorExperiment],
    ) -> np.ndarray:
        experiment_index, time_days, control_relative = xdata
        sf = Fitter._lql_scalar_model(
            experiment_index,
            alpha=alpha,
            beta=beta,
            transition_dose=transition_dose,
            experiments=experiments,
        )
        control_relative = np.clip(control_relative, 1.0e-8, None)
        return sf + (1.0 - sf) * np.exp(-clearance_rate * time_days) / control_relative

    @staticmethod
    def _glq_curve_response_model(
        xdata: Tuple[np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
        beta: float,
        saturation_dose: float,
        clearance_rate: float,
        experiments: Sequence[TumorExperiment],
    ) -> np.ndarray:
        experiment_index, time_days, control_relative = xdata
        sf = Fitter._glq_scalar_model(
            experiment_index,
            alpha=alpha,
            beta=beta,
            saturation_dose=saturation_dose,
            experiments=experiments,
        )
        control_relative = np.clip(control_relative, 1.0e-8, None)
        return sf + (1.0 - sf) * np.exp(-clearance_rate * time_days) / control_relative

    @staticmethod
    def _let_scalar_model(
        experiment_index: np.ndarray,
        alpha_0: float,
        lambda_alpha: float,
        beta_0: float,
        experiments: Sequence[TumorExperiment],
    ) -> np.ndarray:
        predicted = [
            math.exp(
                -Fitter.let_dependent_exponent(
                    experiments[int(index)],
                    alpha_0=alpha_0,
                    lambda_alpha=lambda_alpha,
                    beta_0=beta_0,
                )
            )
            for index in np.asarray(experiment_index, dtype=int)
        ]
        return np.asarray(predicted, dtype=float)

    @staticmethod
    def _let_curve_response_model(
        xdata: Tuple[np.ndarray, np.ndarray, np.ndarray],
        alpha_0: float,
        lambda_alpha: float,
        beta_0: float,
        clearance_rate: float,
        experiments: Sequence[TumorExperiment],
    ) -> np.ndarray:
        experiment_index, time_days, control_relative = xdata
        sf = Fitter._let_scalar_model(
            experiment_index,
            alpha_0=alpha_0,
            lambda_alpha=lambda_alpha,
            beta_0=beta_0,
            experiments=experiments,
        )
        control_relative = np.clip(control_relative, 1.0e-8, None)
        return sf + (1.0 - sf) * np.exp(-clearance_rate * time_days) / control_relative

    @staticmethod
    def _lq_repop_scalar_model(
        experiment_index: np.ndarray,
        alpha: float,
        beta: float,
        lag_days: float,
        repopulation_rate: float,
        experiments: Sequence[TumorExperiment],
    ) -> np.ndarray:
        predicted = []
        for index in np.asarray(experiment_index, dtype=int):
            experiment = experiments[index]
            exponent = (
                alpha * experiment.dose_sum
                + beta * experiment.dose2_sum
                - repopulation_rate * max(experiment.sf_time_day - lag_days, 0.0)
            )
            predicted.append(math.exp(-exponent))
        return np.asarray(predicted, dtype=float)

    @staticmethod
    def _repair_repop_scalar_model(
        experiment_index: np.ndarray,
        alpha: float,
        beta: float,
        lag_days: float,
        repopulation_rate: float,
        experiments: Sequence[TumorExperiment],
        repair_rate_per_day: float,
    ) -> np.ndarray:
        predicted = []
        for index in np.asarray(experiment_index, dtype=int):
            experiment = experiments[index]
            exponent = (
                alpha * experiment.dose_sum
                + beta * experiment.quadratic_term(repair_rate_per_day)
                - repopulation_rate * max(experiment.sf_time_day - lag_days, 0.0)
            )
            predicted.append(math.exp(-exponent))
        return np.asarray(predicted, dtype=float)

    @staticmethod
    def _curve_response_model_with_repop(
        xdata: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
        beta: float,
        clearance_rate: float,
        lag_days: float,
        repopulation_rate: float,
    ) -> np.ndarray:
        dose_sum, quadratic_term, time_days, control_relative = xdata
        sf = np.exp(-(alpha * dose_sum + beta * quadratic_term))
        control_relative = np.clip(control_relative, 1.0e-8, None)
        response = sf + (1.0 - sf) * np.exp(-clearance_rate * time_days) / control_relative
        response = response * np.exp(
            repopulation_rate * np.clip(time_days - lag_days, a_min=0.0, a_max=None)
        )
        return response

    def _to_curve_arrays(
        self,
        experiments: Sequence[TumorExperiment],
        model_kind: ModelKind,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        repair_rate = self.repair_rate_per_day if model_kind in ("repair_lq", "repair_repop") else None
        observed_values: List[float] = []
        dose_sum_values: List[float] = []
        quadratic_values: List[float] = []
        time_values: List[float] = []
        control_relative_values: List[float] = []

        for experiment in experiments:
            curve = np.asarray(experiment.curve_response, dtype=float)
            if len(curve) < 2:
                continue
            time_days = np.asarray(experiment.time_days[: len(curve)], dtype=float)
            if len(time_days) != len(curve):
                time_days = np.arange(len(curve), dtype=float)
            time_days = time_days - float(time_days[0])
            control_relative = np.asarray(
                experiment.control_relative_curve[: len(curve)],
                dtype=float,
            )
            if len(control_relative) != len(curve):
                control_relative = np.ones(len(curve), dtype=float)

            valid_mask = (
                np.isfinite(curve)
                & np.isfinite(time_days)
                & np.isfinite(control_relative)
                & (curve > 0.0)
                & (control_relative > 0.0)
            )
            if len(valid_mask) > 0:
                valid_mask[0] = False
            if not np.any(valid_mask):
                continue

            point_count = int(np.sum(valid_mask))
            observed_values.extend(curve[valid_mask].tolist())
            time_values.extend(time_days[valid_mask].tolist())
            control_relative_values.extend(control_relative[valid_mask].tolist())
            dose_sum_values.extend([experiment.dose_sum] * point_count)
            if model_kind == "linear":
                quadratic_values.extend([0.0] * point_count)
            else:
                quadratic_term = self._quadratic_term_for_model(
                    experiment,
                    model_kind=model_kind,
                    repair_rate_per_day=repair_rate,
                )
                quadratic_values.extend([quadratic_term] * point_count)

        if not observed_values:
            raise RuntimeError("Need at least one post-baseline curve point for curve-mode fitting.")

        return (
            np.asarray(dose_sum_values, dtype=float),
            np.asarray(quadratic_values, dtype=float),
            np.asarray(time_values, dtype=float),
            np.asarray(control_relative_values, dtype=float),
            np.asarray(observed_values, dtype=float),
        )

    @staticmethod
    def _to_lql_curve_arrays(
        experiments: Sequence[TumorExperiment],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observed_values: List[float] = []
        experiment_index_values: List[int] = []
        time_values: List[float] = []
        control_relative_values: List[float] = []

        for experiment_index, experiment in enumerate(experiments):
            curve = np.asarray(experiment.curve_response, dtype=float)
            if len(curve) < 2:
                continue
            time_days = np.asarray(experiment.time_days[: len(curve)], dtype=float)
            if len(time_days) != len(curve):
                time_days = np.arange(len(curve), dtype=float)
            time_days = time_days - float(time_days[0])
            control_relative = np.asarray(
                experiment.control_relative_curve[: len(curve)],
                dtype=float,
            )
            if len(control_relative) != len(curve):
                control_relative = np.ones(len(curve), dtype=float)

            valid_mask = (
                np.isfinite(curve)
                & np.isfinite(time_days)
                & np.isfinite(control_relative)
                & (curve > 0.0)
                & (control_relative > 0.0)
            )
            if len(valid_mask) > 0:
                valid_mask[0] = False
            if not np.any(valid_mask):
                continue

            point_count = int(np.sum(valid_mask))
            observed_values.extend(curve[valid_mask].tolist())
            experiment_index_values.extend([experiment_index] * point_count)
            time_values.extend(time_days[valid_mask].tolist())
            control_relative_values.extend(control_relative[valid_mask].tolist())

        if not observed_values:
            raise RuntimeError("Need at least one post-baseline curve point for curve-mode fitting.")

        return (
            np.asarray(experiment_index_values, dtype=float),
            np.asarray(time_values, dtype=float),
            np.asarray(control_relative_values, dtype=float),
            np.asarray(observed_values, dtype=float),
        )

    def _fit_scalar_parameters(
        self,
        experiments: Sequence[TumorExperiment],
        model_kind: ModelKind,
    ) -> Tuple[
        float,
        float,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        if model_kind == "let_dependent":
            alpha_0, lambda_alpha, beta_0 = self._fit_scalar_let_parameters(experiments)
            return alpha_0, beta_0, None, None, None, None, None, lambda_alpha

        repair_rate = None
        if model_kind in ("repair_lq", "repair_repop"):
            repair_rate = self.repair_rate_per_day
            if repair_rate is None:
                raise RuntimeError("repair_half_time_hours must be set for the repair-aware model.")
        repair_fast_rate = None
        repair_slow_rate = None
        if model_kind == "repair_biexp":
            repair_fast_rate = self.repair_fast_rate_per_day
            repair_slow_rate = self.repair_slow_rate_per_day
            if repair_fast_rate is None or repair_slow_rate is None:
                raise RuntimeError(
                    "repair_half_time_fast_hours and repair_half_time_slow_hours must be set for the bi-exponential repair model."
                )

        dose_sum, quadratic_term, sf, sigma_sf, sigma_log_sf = self._to_arrays(
            experiments,
            model_kind=model_kind,
            repair_rate_per_day=repair_rate,
            repair_fast_rate_per_day=repair_fast_rate,
            repair_slow_rate_per_day=repair_slow_rate,
            repair_fast_fraction=self.repair_fast_fraction,
        )
        if np.any(~np.isfinite(sf)) or np.any(sf <= 0.0) or np.any(sf > 1.0):
            raise ValueError("All SF values must be finite and in the interval (0, 1].")

        if model_kind == "linear":
            if self.alpha_fixed is not None:
                alpha = self.alpha_fixed
            else:
                alpha = self._fit_alpha_for_linear_model(dose_sum, sf, sigma_log_sf)
            beta = 0.0
            return alpha, beta, None, None, None, None, None, None

        if model_kind == "glq":
            experiment_index = np.arange(len(experiments), dtype=float)
            initial_saturation = self._initial_saturation_dose(experiments)
            max_fraction = max(max(experiment.fractions) for experiment in experiments if experiment.fractions)
            saturation_upper = max(max_fraction * 2.0, initial_saturation * 2.0, 1.0)

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda index, beta, saturation_dose: self._glq_scalar_model(
                        index,
                        self.alpha_fixed,
                        beta,
                        saturation_dose,
                        experiments,
                    ),
                    experiment_index,
                    sf,
                    p0=(0.01, initial_saturation),
                    bounds=((0.0, 1.0e-6), (np.inf, saturation_upper)),
                    sigma=np.clip(sigma_sf, 1.0e-8, None),
                    absolute_sigma=False,
                    maxfev=30000,
                )
                return self.alpha_fixed, float(params[0]), None, None, float(params[1]), None, None, None

        if model_kind == "lq_l":
            experiment_index = np.arange(len(experiments), dtype=float)
            initial_transition = self._initial_transition_dose(experiments)
            max_fraction = max(max(experiment.fractions) for experiment in experiments if experiment.fractions)
            transition_upper = max(max_fraction * 2.0, initial_transition * 2.0, 1.0)

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda index, beta, transition_dose: self._lql_scalar_model(
                        index,
                        self.alpha_fixed,
                        beta,
                        transition_dose,
                        experiments,
                    ),
                    experiment_index,
                    sf,
                    p0=(0.01, initial_transition),
                    bounds=((0.0, 1.0e-6), (np.inf, transition_upper)),
                    sigma=np.clip(sigma_sf, 1.0e-8, None),
                    absolute_sigma=False,
                    maxfev=30000,
                )
                return self.alpha_fixed, float(params[0]), None, float(params[1]), None, None, None, None

            initial_alpha, initial_beta = self._fit_free_alpha_beta(
                dose_sum=dose_sum,
                quadratic_term=quadratic_term,
                sf=sf,
                sigma_sf=sigma_sf,
            )
            params, _ = self._safe_curve_fit(
                lambda index, alpha, beta, transition_dose: self._lql_scalar_model(
                    index,
                    alpha,
                    beta,
                    transition_dose,
                    experiments,
                ),
                experiment_index,
                sf,
                p0=(max(initial_alpha, 1.0e-8), max(initial_beta, 1.0e-8), initial_transition),
                bounds=((0.0, 0.0, 1.0e-6), (np.inf, np.inf, transition_upper)),
                sigma=np.clip(sigma_sf, 1.0e-8, None),
                absolute_sigma=False,
                maxfev=30000,
            )
            return float(params[0]), float(params[1]), None, float(params[2]), None, None, None, None

        if model_kind == "glq":
            initial_alpha, initial_beta = self._fit_free_alpha_beta(
                dose_sum=dose_sum,
                quadratic_term=quadratic_term,
                sf=sf,
                sigma_sf=sigma_sf,
            )
            params, _ = self._safe_curve_fit(
                lambda index, alpha, beta, saturation_dose: self._glq_scalar_model(
                    index,
                    alpha,
                    beta,
                    saturation_dose,
                    experiments,
                ),
                experiment_index,
                sf,
                p0=(max(initial_alpha, 1.0e-8), max(initial_beta, 1.0e-8), initial_saturation),
                bounds=((0.0, 0.0, 1.0e-6), (np.inf, np.inf, saturation_upper)),
                sigma=np.clip(sigma_sf, 1.0e-8, None),
                absolute_sigma=False,
                maxfev=30000,
            )
            return float(params[0]), float(params[1]), None, None, float(params[2]), None, None, None

        if model_kind == "lq_repop":
            experiment_index = np.arange(len(experiments), dtype=float)
            initial_lag = self._initial_lag_days(experiments)
            max_time = max(experiment.sf_time_day for experiment in experiments) if experiments else 1.0
            lag_upper = max(max_time * 1.5, initial_lag + 1.0, 1.0)
            repop_upper = 2.0

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda index, beta, lag_days, repopulation_rate: self._lq_repop_scalar_model(
                        index,
                        self.alpha_fixed,
                        beta,
                        lag_days,
                        repopulation_rate,
                        experiments,
                    ),
                    experiment_index,
                    sf,
                    p0=(0.01, initial_lag, 0.05),
                    bounds=((0.0, 0.0, 0.0), (np.inf, lag_upper, repop_upper)),
                    sigma=np.clip(sigma_sf, 1.0e-8, None),
                    absolute_sigma=False,
                    maxfev=30000,
                )
                return self.alpha_fixed, float(params[0]), None, None, None, float(params[1]), float(params[2]), None

            initial_alpha, initial_beta = self._fit_free_alpha_beta(
                dose_sum=dose_sum,
                quadratic_term=quadratic_term,
                sf=sf,
                sigma_sf=sigma_sf,
            )
            params, _ = self._safe_curve_fit(
                lambda index, alpha, beta, lag_days, repopulation_rate: self._lq_repop_scalar_model(
                    index,
                    alpha,
                    beta,
                    lag_days,
                    repopulation_rate,
                    experiments,
                ),
                experiment_index,
                sf,
                p0=(max(initial_alpha, 1.0e-8), max(initial_beta, 1.0e-8), initial_lag, 0.05),
                bounds=((0.0, 0.0, 0.0, 0.0), (np.inf, np.inf, lag_upper, repop_upper)),
                sigma=np.clip(sigma_sf, 1.0e-8, None),
                absolute_sigma=False,
                maxfev=30000,
            )
            return float(params[0]), float(params[1]), None, None, None, float(params[2]), float(params[3]), None

        if model_kind == "repair_repop":
            experiment_index = np.arange(len(experiments), dtype=float)
            initial_lag = self._initial_lag_days(experiments)
            max_time = max(experiment.sf_time_day for experiment in experiments) if experiments else 1.0
            lag_upper = max(max_time * 1.5, initial_lag + 1.0, 1.0)
            repop_upper = 2.0

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda index, beta, lag_days, repopulation_rate: self._repair_repop_scalar_model(
                        index,
                        self.alpha_fixed,
                        beta,
                        lag_days,
                        repopulation_rate,
                        experiments,
                        repair_rate,
                    ),
                    experiment_index,
                    sf,
                    p0=(0.01, initial_lag, 0.05),
                    bounds=((0.0, 0.0, 0.0), (np.inf, lag_upper, repop_upper)),
                    sigma=np.clip(sigma_sf, 1.0e-8, None),
                    absolute_sigma=False,
                    maxfev=30000,
                )
                return self.alpha_fixed, float(params[0]), None, None, None, float(params[1]), float(params[2]), None

            params, _ = self._safe_curve_fit(
                lambda index, alpha, beta, lag_days, repopulation_rate: self._repair_repop_scalar_model(
                    index,
                    alpha,
                    beta,
                    lag_days,
                    repopulation_rate,
                    experiments,
                    repair_rate,
                ),
                experiment_index,
                sf,
                p0=(0.05, 0.01, initial_lag, 0.05),
                bounds=((0.0, 0.0, 0.0, 0.0), (np.inf, np.inf, lag_upper, repop_upper)),
                sigma=np.clip(sigma_sf, 1.0e-8, None),
                absolute_sigma=False,
                maxfev=30000,
            )
            return float(params[0]), float(params[1]), None, None, None, float(params[2]), float(params[3]), None

        if self.alpha_fixed is not None:
            beta = self._fit_beta_for_fixed_alpha(
                dose_sum=dose_sum,
                quadratic_term=quadratic_term,
                sf=sf,
                sigma_log_sf=sigma_log_sf,
                alpha=self.alpha_fixed,
            )
            return self.alpha_fixed, beta, None, None, None, None, None, None

        alpha, beta = self._fit_free_alpha_beta(
            dose_sum=dose_sum,
            quadratic_term=quadratic_term,
            sf=sf,
            sigma_sf=sigma_sf,
        )
        return alpha, beta, None, None, None, None, None, None

    def _fit_curve_parameters(
        self,
        experiments: Sequence[TumorExperiment],
        model_kind: ModelKind,
    ) -> Tuple[
        float,
        float,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        if model_kind == "let_dependent":
            alpha_0, lambda_alpha, beta_0, clearance_rate = self._fit_curve_let_parameters(experiments)
            return alpha_0, beta_0, clearance_rate, None, None, None, None, lambda_alpha

        if model_kind == "glq":
            (
                experiment_index,
                time_days,
                control_relative,
                observed_curve,
            ) = self._to_lql_curve_arrays(experiments)
            (
                initial_alpha,
                initial_beta,
                _,
                _,
                initial_saturation,
                _,
                _,
                _,
            ) = self._fit_scalar_parameters(experiments, model_kind)
            if initial_saturation is None:
                initial_saturation = self._initial_saturation_dose(experiments)
            max_fraction = max(
                max(experiment.fractions) for experiment in experiments if experiment.fractions
            )
            saturation_upper = max(max_fraction * 2.0, initial_saturation * 2.0, 1.0)

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda xdata, beta, saturation_dose, clearance_rate: self._glq_curve_response_model(
                        xdata,
                        self.alpha_fixed,
                        beta,
                        saturation_dose,
                        clearance_rate,
                        experiments,
                    ),
                    (experiment_index, time_days, control_relative),
                    observed_curve,
                    p0=(max(initial_beta, 1.0e-8), initial_saturation, 0.10),
                    bounds=((0.0, 1.0e-6, 0.0), (np.inf, saturation_upper, np.inf)),
                    maxfev=30000,
                )
                return self.alpha_fixed, float(params[0]), float(params[2]), None, float(params[1]), None, None

            params, _ = self._safe_curve_fit(
                lambda xdata, alpha, beta, saturation_dose, clearance_rate: self._glq_curve_response_model(
                    xdata,
                    alpha,
                    beta,
                    saturation_dose,
                    clearance_rate,
                    experiments,
                ),
                (experiment_index, time_days, control_relative),
                observed_curve,
                p0=(
                    max(initial_alpha, 1.0e-8),
                    max(initial_beta, 1.0e-8),
                    initial_saturation,
                    0.10,
                ),
                bounds=((0.0, 0.0, 1.0e-6, 0.0), (np.inf, np.inf, saturation_upper, np.inf)),
                maxfev=30000,
            )
            return float(params[0]), float(params[1]), float(params[3]), None, float(params[2]), None, None

        if model_kind == "lq_l":
            (
                experiment_index,
                time_days,
                control_relative,
                observed_curve,
            ) = self._to_lql_curve_arrays(experiments)
            initial_alpha, initial_beta, _, initial_transition, _, _, _, _ = self._fit_scalar_parameters(
                experiments,
                model_kind,
            )
            if initial_transition is None:
                initial_transition = self._initial_transition_dose(experiments)
            max_fraction = max(
                max(experiment.fractions) for experiment in experiments if experiment.fractions
            )
            transition_upper = max(max_fraction * 2.0, initial_transition * 2.0, 1.0)

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda xdata, beta, transition_dose, clearance_rate: self._lql_curve_response_model(
                        xdata,
                        self.alpha_fixed,
                        beta,
                        transition_dose,
                        clearance_rate,
                        experiments,
                    ),
                    (experiment_index, time_days, control_relative),
                    observed_curve,
                    p0=(max(initial_beta, 1.0e-8), initial_transition, 0.10),
                    bounds=((0.0, 1.0e-6, 0.0), (np.inf, transition_upper, np.inf)),
                    maxfev=30000,
                )
                return self.alpha_fixed, float(params[0]), float(params[2]), float(params[1]), None, None, None

            params, _ = self._safe_curve_fit(
                lambda xdata, alpha, beta, transition_dose, clearance_rate: self._lql_curve_response_model(
                    xdata,
                    alpha,
                    beta,
                    transition_dose,
                    clearance_rate,
                    experiments,
                ),
                (experiment_index, time_days, control_relative),
                observed_curve,
                p0=(
                    max(initial_alpha, 1.0e-8),
                    max(initial_beta, 1.0e-8),
                    initial_transition,
                    0.10,
                ),
                bounds=((0.0, 0.0, 1.0e-6, 0.0), (np.inf, np.inf, transition_upper, np.inf)),
                maxfev=30000,
            )
            return float(params[0]), float(params[1]), float(params[3]), float(params[2]), None, None, None

        if model_kind == "lq_repop":
            (
                dose_sum,
                quadratic_term,
                time_days,
                control_relative,
                observed_curve,
            ) = self._to_curve_arrays(experiments, model_kind="classic_lq")
            initial_alpha, initial_beta, _, _, _, initial_lag, initial_repopulation, _ = self._fit_scalar_parameters(
                experiments,
                model_kind,
            )
            if initial_lag is None:
                initial_lag = self._initial_lag_days(experiments)
            if initial_repopulation is None:
                initial_repopulation = 0.05
            max_time = float(np.max(time_days)) if len(time_days) else 1.0
            lag_upper = max(max_time * 1.25, initial_lag + 1.0, 1.0)
            repop_upper = 2.0

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda xdata, beta, clearance_rate, lag_days, repopulation_rate: self._curve_response_model_with_repop(
                        xdata,
                        self.alpha_fixed,
                        beta,
                        clearance_rate,
                        lag_days,
                        repopulation_rate,
                    ),
                    (dose_sum, quadratic_term, time_days, control_relative),
                    observed_curve,
                    p0=(max(initial_beta, 1.0e-8), 0.10, initial_lag, initial_repopulation),
                    bounds=((0.0, 0.0, 0.0, 0.0), (np.inf, np.inf, lag_upper, repop_upper)),
                    maxfev=30000,
                )
                return (
                    self.alpha_fixed,
                    float(params[0]),
                    float(params[1]),
                    None,
                    None,
                    float(params[2]),
                    float(params[3]),
                )

            params, _ = self._safe_curve_fit(
                self._curve_response_model_with_repop,
                (dose_sum, quadratic_term, time_days, control_relative),
                observed_curve,
                p0=(
                    max(initial_alpha, 1.0e-8),
                    max(initial_beta, 1.0e-8),
                    0.10,
                    initial_lag,
                    initial_repopulation,
                ),
                bounds=((0.0, 0.0, 0.0, 0.0, 0.0), (np.inf, np.inf, np.inf, lag_upper, repop_upper)),
                maxfev=30000,
            )
            return (
                float(params[0]),
                float(params[1]),
                float(params[2]),
                None,
                None,
                float(params[3]),
                float(params[4]),
            )

        if model_kind == "repair_repop":
            (
                dose_sum,
                quadratic_term,
                time_days,
                control_relative,
                observed_curve,
            ) = self._to_curve_arrays(experiments, model_kind="repair_lq")
            initial_alpha, initial_beta, _, _, _, initial_lag, initial_repopulation, _ = self._fit_scalar_parameters(
                experiments,
                model_kind,
            )
            if initial_lag is None:
                initial_lag = self._initial_lag_days(experiments)
            if initial_repopulation is None:
                initial_repopulation = 0.05
            max_time = float(np.max(time_days)) if len(time_days) else 1.0
            lag_upper = max(max_time * 1.25, initial_lag + 1.0, 1.0)
            repop_upper = 2.0

            if self.alpha_fixed is not None:
                params, _ = self._safe_curve_fit(
                    lambda xdata, beta, clearance_rate, lag_days, repopulation_rate: self._curve_response_model_with_repop(
                        xdata,
                        self.alpha_fixed,
                        beta,
                        clearance_rate,
                        lag_days,
                        repopulation_rate,
                    ),
                    (dose_sum, quadratic_term, time_days, control_relative),
                    observed_curve,
                    p0=(max(initial_beta, 1.0e-8), 0.10, initial_lag, initial_repopulation),
                    bounds=((0.0, 0.0, 0.0, 0.0), (np.inf, np.inf, lag_upper, repop_upper)),
                    maxfev=30000,
                )
                return (
                    self.alpha_fixed,
                    float(params[0]),
                    float(params[1]),
                    None,
                    None,
                    float(params[2]),
                    float(params[3]),
                )

            params, _ = self._safe_curve_fit(
                self._curve_response_model_with_repop,
                (dose_sum, quadratic_term, time_days, control_relative),
                observed_curve,
                p0=(
                    max(initial_alpha, 1.0e-8),
                    max(initial_beta, 1.0e-8),
                    0.10,
                    initial_lag,
                    initial_repopulation,
                ),
                bounds=((0.0, 0.0, 0.0, 0.0, 0.0), (np.inf, np.inf, np.inf, lag_upper, repop_upper)),
                maxfev=30000,
            )
            return (
                float(params[0]),
                float(params[1]),
                float(params[2]),
                None,
                None,
                float(params[3]),
                float(params[4]),
            )

        (
            dose_sum,
            quadratic_term,
            time_days,
            control_relative,
            observed_curve,
        ) = self._to_curve_arrays(experiments, model_kind=model_kind)

        if model_kind == "linear":
            initial_alpha = self.alpha_fixed
            if initial_alpha is None:
                initial_alpha, _, _, _, _, _, _, _ = self._fit_scalar_parameters(experiments, "linear")
            initial_guess = [max(initial_alpha, 1.0e-8), 0.10]

            if self.alpha_fixed is not None:
                def model_fixed_alpha(
                    xdata: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    clearance_rate: float,
                ) -> np.ndarray:
                    return self._curve_response_model(xdata, self.alpha_fixed, 0.0, clearance_rate)

                params, _ = self._safe_curve_fit(
                    model_fixed_alpha,
                    (dose_sum, quadratic_term, time_days, control_relative),
                    observed_curve,
                    p0=(0.10,),
                    bounds=((0.0,), (np.inf,)),
                    maxfev=20000,
                )
            return self.alpha_fixed, 0.0, float(params[0]), None, None, None, None

            params, _ = self._safe_curve_fit(
                lambda xdata, alpha, clearance_rate: self._curve_response_model(
                    xdata,
                    alpha,
                    0.0,
                    clearance_rate,
                ),
                (dose_sum, quadratic_term, time_days, control_relative),
                observed_curve,
                p0=tuple(initial_guess),
                bounds=((0.0, 0.0), (np.inf, np.inf)),
                maxfev=20000,
            )
            return float(params[0]), 0.0, float(params[1]), None, None, None, None

        initial_alpha, initial_beta, _, _, _, _, _, _ = self._fit_scalar_parameters(experiments, model_kind)
        if self.alpha_fixed is not None:
            params, _ = self._safe_curve_fit(
                lambda xdata, beta, clearance_rate: self._curve_response_model(
                    xdata,
                    self.alpha_fixed,
                    beta,
                    clearance_rate,
                ),
                (dose_sum, quadratic_term, time_days, control_relative),
                observed_curve,
                p0=(max(initial_beta, 1.0e-8), 0.10),
                bounds=((0.0, 0.0), (np.inf, np.inf)),
                maxfev=20000,
            )
            return self.alpha_fixed, float(params[0]), float(params[1]), None, None, None, None

        params, _ = self._safe_curve_fit(
            self._curve_response_model,
            (dose_sum, quadratic_term, time_days, control_relative),
            observed_curve,
            p0=(max(initial_alpha, 1.0e-8), max(initial_beta, 1.0e-8), 0.10),
            bounds=((0.0, 0.0, 0.0), (np.inf, np.inf, np.inf)),
            maxfev=20000,
        )
        return float(params[0]), float(params[1]), float(params[2]), None, None, None, None

    def fit(
        self,
        experiments: Optional[Sequence[TumorExperiment]] = None,
        train_kind: RegimenKind = "all",
        family: Optional[str] = None,
        sf_mode: Optional[str] = None,
        response_mode: ResponseMode = "scalar",
        model_kind: RequestedModelKind = "auto",
    ) -> LQFitResult:
        experiments = list(experiments) if experiments is not None else list(self.experiments)
        if len(experiments) < 2:
            raise RuntimeError("Need at least two valid experiments for fitting.")

        resolved_model_kind: ModelKind
        if model_kind == "auto":
            if self.repair_fast_rate_per_day is not None and self.repair_slow_rate_per_day is not None:
                resolved_model_kind = "repair_biexp"
            elif self.repair_rate_per_day is not None:
                resolved_model_kind = "repair_lq"
            else:
                resolved_model_kind = "classic_lq"
        else:
            resolved_model_kind = model_kind

        lambda_alpha = None
        if response_mode == "curve":
            fit_params = self._fit_curve_parameters(
                experiments,
                model_kind=resolved_model_kind,
            )
        else:
            fit_params = self._fit_scalar_parameters(
                experiments,
                model_kind=resolved_model_kind,
            )
        (
            alpha,
            beta,
            curve_clearance_rate,
            transition_dose,
            saturation_dose,
            lag_days,
            repopulation_rate,
            *extra_params,
        ) = fit_params
        if extra_params:
            lambda_alpha = extra_params[0]

        return LQFitResult(
            alpha=alpha,
            beta=beta,
            train_count=len(experiments),
            train_kind=train_kind,
            family=normalize_family(family),
            sf_mode=sf_mode or self.sf_mode,
            model_kind=resolved_model_kind,
            response_mode=response_mode,
            repair_half_time_hours=self.repair_half_time_hours,
            repair_half_time_fast_hours=self.repair_half_time_fast_hours,
            repair_half_time_slow_hours=self.repair_half_time_slow_hours,
            repair_fast_fraction=self.repair_fast_fraction,
            alpha_0=alpha if resolved_model_kind == "let_dependent" else None,
            lambda_alpha=lambda_alpha,
            curve_clearance_rate=curve_clearance_rate,
            transition_dose=transition_dose,
            saturation_dose=saturation_dose,
            lag_days=lag_days,
            repopulation_rate=repopulation_rate,
        )

    @staticmethod
    def _aic_from_rss(rss: float, point_count: int, parameter_count: int) -> Optional[float]:
        if point_count <= 0 or parameter_count < 0 or rss <= 0.0:
            return None
        return float(point_count * np.log(rss / point_count) + 2 * parameter_count)

    @staticmethod
    def _bic_from_rss(rss: float, point_count: int, parameter_count: int) -> Optional[float]:
        if point_count <= 0 or parameter_count < 0 or rss <= 0.0:
            return None
        return float(point_count * np.log(rss / point_count) + parameter_count * np.log(point_count))

    def parameter_count(self, result: LQFitResult) -> int:
        parameter_count = 1 if result.model_kind == "linear" else 2
        if self.alpha_fixed is not None:
            parameter_count -= 1
        if result.model_kind == "lq_l":
            parameter_count += 1
        if result.model_kind == "glq":
            parameter_count += 1
        if result.model_kind == "let_dependent":
            parameter_count += 1
        if result.model_kind in ("lq_repop", "repair_repop"):
            parameter_count += 2
        if result.response_mode == "curve":
            parameter_count += 1
        return max(parameter_count, 0)

    def compute_fit_metrics(
        self,
        experiments: Sequence[TumorExperiment],
        result: LQFitResult,
    ) -> FitMetrics:
        observed_values: List[float] = []
        predicted_values: List[float] = []

        if result.response_mode == "curve":
            for experiment in experiments:
                curve = np.asarray(experiment.curve_response, dtype=float)
                if len(curve) < 2:
                    continue
                predicted_curve = result.predict_curve(experiment)
                valid_mask = np.isfinite(curve) & np.isfinite(predicted_curve) & (curve > 0.0) & (predicted_curve > 0.0)
                if len(valid_mask) > 0:
                    valid_mask[0] = False
                observed_values.extend(curve[valid_mask].tolist())
                predicted_values.extend(predicted_curve[valid_mask].tolist())
        else:
            for experiment in experiments:
                observed_values.append(experiment.sf)
                predicted_values.append(result.predict_sf(experiment))

        observed = np.asarray(observed_values, dtype=float)
        predicted = np.asarray(predicted_values, dtype=float)
        if len(observed) == 0:
            raise RuntimeError("No comparable observations are available for fit metrics.")

        residuals = predicted - observed
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mean_abs_log_error = float(
            np.mean(
                np.abs(
                    np.log(np.clip(predicted, 1.0e-8, None)) - np.log(np.clip(observed, 1.0e-8, None))
                )
            )
        )
        rss = float(np.sum(np.square(residuals)))
        parameter_count = self.parameter_count(result)
        aic = self._aic_from_rss(rss, len(observed), parameter_count)
        bic = self._bic_from_rss(rss, len(observed), parameter_count)
        r_squared = None
        adjusted_r_squared = None
        ss_tot = float(np.sum(np.square(observed - np.mean(observed))))
        if ss_tot > 0.0:
            r_squared = float(1.0 - rss / ss_tot)
            if len(observed) > parameter_count + 1:
                adjusted_r_squared = float(
                    1.0
                    - (1.0 - r_squared)
                    * (len(observed) - 1)
                    / (len(observed) - parameter_count - 1)
                )
        return FitMetrics(
            point_count=len(observed),
            mae=mae,
            rmse=rmse,
            mean_abs_log_error=mean_abs_log_error,
            rss=rss,
            aic=aic,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            bic=bic,
        )

    def compare_models(
        self,
        experiments: Sequence[TumorExperiment],
        response_mode: ResponseMode,
        family: Optional[str],
        sf_mode: Optional[str],
    ) -> Tuple[ModelComparisonRow, ...]:
        candidates: List[ModelKind] = ["classic_lq", "glq", "lq_l", "lq_repop", "linear"]
        if self.repair_rate_per_day is not None:
            candidates.insert(1, "repair_lq")
            candidates.insert(2, "repair_repop")
        if self.repair_fast_rate_per_day is not None and self.repair_slow_rate_per_day is not None:
            candidates.insert(2, "repair_biexp")
        let_levels = {
            round(
                float(experiment.let_kev_um)
                if experiment.let_kev_um is not None and np.isfinite(experiment.let_kev_um)
                else FAMILY_LET_DEFAULTS.get(experiment.family or "", 0.3),
                8,
            )
            for experiment in experiments
        }
        if len(let_levels) >= 2:
            candidates.append("let_dependent")

        rows: List[ModelComparisonRow] = []
        for model_kind in candidates:
            try:
                fit_result = self.fit(
                    experiments=experiments,
                    train_kind="all",
                    family=family,
                    sf_mode=sf_mode,
                    response_mode=response_mode,
                    model_kind=model_kind,
                )
                metrics = self.compute_fit_metrics(experiments, fit_result)
                rows.append(
                    ModelComparisonRow(
                        model_kind=model_kind,
                        status="ok",
                        response_mode=response_mode,
                        alpha=fit_result.alpha,
                        beta=fit_result.beta,
                        lambda_alpha=fit_result.lambda_alpha,
                        curve_clearance_rate=fit_result.curve_clearance_rate,
                        transition_dose=fit_result.transition_dose,
                        saturation_dose=fit_result.saturation_dose,
                        lag_days=fit_result.lag_days,
                        repopulation_rate=fit_result.repopulation_rate,
                        metrics=metrics,
                    )
                )
            except Exception as exc:
                rows.append(
                    ModelComparisonRow(
                        model_kind=model_kind,
                        status="skipped",
                        response_mode=response_mode,
                        reason=str(exc),
                    )
                )

        successful_rows = [row for row in rows if row.status == "ok" and row.metrics is not None]
        ranking = {
            id(row): rank
            for rank, row in enumerate(
                sorted(
                    successful_rows,
                    key=lambda row: (
                        float("inf") if row.metrics.aic is None else row.metrics.aic,
                        row.metrics.rmse,
                        row.metrics.mean_abs_log_error,
                    ),
                ),
                start=1,
            )
        }
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                1 if row.status != "ok" or row.metrics is None else 0,
                float("inf") if row.metrics is None or row.metrics.aic is None else row.metrics.aic,
                float("inf") if row.metrics is None else row.metrics.rmse,
                float("inf") if row.metrics is None else row.metrics.mean_abs_log_error,
            ),
        )

        ranked_rows: List[ModelComparisonRow] = []
        for row in ordered_rows:
            if row.status == "ok" and row.metrics is not None:
                reason = f"rank={ranking[id(row)]}"
            else:
                reason = row.reason
            ranked_rows.append(
                ModelComparisonRow(
                    model_kind=row.model_kind,
                    status=row.status,
                    response_mode=row.response_mode,
                    alpha=row.alpha,
                    beta=row.beta,
                    lambda_alpha=row.lambda_alpha,
                    curve_clearance_rate=row.curve_clearance_rate,
                    transition_dose=row.transition_dose,
                    saturation_dose=row.saturation_dose,
                    lag_days=row.lag_days,
                    repopulation_rate=row.repopulation_rate,
                    metrics=row.metrics,
                    reason=reason,
                )
            )
        return tuple(ranked_rows)

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
        repair_model_enabled = repair_rate is not None or (
            self.repair_fast_rate_per_day is not None and self.repair_slow_rate_per_day is not None
        )
        if self.repair_fast_rate_per_day is not None and self.repair_slow_rate_per_day is not None:
            quadratic_term = np.array(
                [
                    experiment.quadratic_term_biexp(
                        self.repair_fast_rate_per_day,
                        self.repair_slow_rate_per_day,
                        self.repair_fast_fraction,
                    )
                    for experiment in experiments
                ],
                dtype=float,
            )
        else:
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
        if repair_model_enabled:
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
            repair_model_enabled=repair_model_enabled,
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
            repair_half_time_fast_hours=self.repair_half_time_fast_hours,
            repair_half_time_slow_hours=self.repair_half_time_slow_hours,
            repair_fast_fraction=(
                self.repair_fast_fraction
                if self.repair_fast_rate_per_day is not None and self.repair_slow_rate_per_day is not None
                else None
            ),
            warnings=tuple(warnings_list),
        )

    def evaluate(
        self,
        experiments: Sequence[TumorExperiment],
        result: LQFitResult,
    ) -> ValidationSummary:
        if result.response_mode == "curve":
            metrics = self.compute_fit_metrics(experiments, result)
            return ValidationSummary(
                rows=(),
                mae=metrics.mae,
                rmse=metrics.rmse,
                mean_abs_log_error=metrics.mean_abs_log_error,
                response_mode="curve",
                point_count=metrics.point_count,
            )

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
            response_mode="scalar",
            point_count=len(rows),
        )

    def cross_validate_loo(
        self,
        experiments: Optional[Sequence[TumorExperiment]] = None,
        family: Optional[str] = None,
        sf_mode: Optional[str] = None,
        response_mode: ResponseMode = "scalar",
        model_kind: RequestedModelKind = "auto",
    ) -> CrossValidationResult:
        experiments = list(experiments) if experiments is not None else list(self.experiments)
        if response_mode != "scalar":
            raise ValueError("Leave-one-out cross-validation is currently implemented for scalar response mode only.")
        if len(experiments) < 3:
            raise RuntimeError("Need at least three experiments for leave-one-out cross-validation.")

        rows: List[PredictionRow] = []
        for holdout_index, holdout in enumerate(experiments):
            train = [experiment for index, experiment in enumerate(experiments) if index != holdout_index]
            if len(train) < 2:
                continue
            fit_result = self.fit(
                experiments=train,
                train_kind="all",
                family=family,
                sf_mode=sf_mode,
                response_mode=response_mode,
                model_kind=model_kind,
            )
            predicted_sf = fit_result.predict_sf(holdout)
            abs_error = abs(predicted_sf - holdout.sf)
            rel_error = abs_error / holdout.sf
            log_error = float(np.log(predicted_sf) - np.log(holdout.sf))
            rows.append(
                PredictionRow(
                    experiment=holdout,
                    predicted_sf=predicted_sf,
                    abs_error=abs_error,
                    rel_error=rel_error,
                    log_error=log_error,
                )
            )

        if not rows:
            raise RuntimeError("Leave-one-out cross-validation did not produce any successful folds.")

        residuals = np.asarray([row.residual for row in rows], dtype=float)
        observed = np.asarray([row.observed_sf for row in rows], dtype=float)
        rss = float(np.sum(np.square(residuals)))
        ss_tot = float(np.sum(np.square(observed - np.mean(observed))))
        cv_r_squared = None if ss_tot <= 0.0 else float(1.0 - rss / ss_tot)
        return CrossValidationResult(
            n_experiments=len(experiments),
            n_successful=len(rows),
            cv_rmse=float(np.sqrt(np.mean(np.square(residuals)))),
            cv_mae=float(np.mean(np.abs(residuals))),
            cv_r_squared=cv_r_squared,
            residuals=tuple(float(value) for value in residuals),
            rows=tuple(rows),
            response_mode="scalar",
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
        response_mode: ResponseMode = "scalar",
        model_kind: RequestedModelKind = "auto",
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
                    response_mode=response_mode,
                    model_kind=model_kind,
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
            f"train_kind={result.train_kind} count={result.train_count} "
            f"response_mode={result.response_mode}"
        )
        if result.model_kind == "linear":
            print("model=linear")
        elif result.model_kind == "let_dependent":
            print("model=let-dependent-lq")
        elif result.model_kind == "glq":
            print("model=glq")
        elif result.model_kind == "lq_l":
            print("model=lq-l")
        elif result.model_kind == "lq_repop":
            print("model=lq-repopulation")
        elif result.model_kind == "repair_biexp":
            print(
                "model=repair-aware-lq-biexponential "
                f"repair_half_time_fast_hours={result.repair_half_time_fast_hours:.3f} "
                f"repair_half_time_slow_hours={result.repair_half_time_slow_hours:.3f} "
                f"repair_fast_fraction={(result.repair_fast_fraction or 0.6):.3f}"
            )
        elif result.model_kind == "repair_repop":
            print(
                "model=repair-aware-lq-repopulation "
                f"repair_half_time_hours={result.repair_half_time_hours:.3f}"
            )
        elif result.repair_half_time_hours is not None and result.repair_half_time_hours > 0.0:
            print(
                "model=time-aware-lq "
                f"repair_half_time_hours={result.repair_half_time_hours:.3f}"
            )
        else:
            print("model=classic-lq")
        if result.transition_dose is not None:
            print(f"transition_dose={result.transition_dose:.6f} Gy")
        if result.saturation_dose is not None:
            print(f"saturation_dose={result.saturation_dose:.6f} Gy")
        if result.lag_days is not None:
            print(f"lag_days={result.lag_days:.6f}")
        if result.repopulation_rate is not None:
            print(f"repopulation_rate={result.repopulation_rate:.6f} per day")
        if result.curve_clearance_rate is not None:
            print(f"curve_clearance_rate={result.curve_clearance_rate:.6f} per day")
        for experiment in experiments:
            print(experiment.report())
        print("\n===== FIT RESULT =====")
        print(f"alpha (Gy^-1): {result.alpha:.5f}")
        print(f"beta  (Gy^-2): {result.beta:.6f}")
        if result.alpha_0 is not None:
            print(f"alpha_0 (Gy^-1): {result.alpha_0:.5f}")
        if result.lambda_alpha is not None:
            print(f"lambda_alpha (Gy^-1 per keV/um): {result.lambda_alpha:.6f}")
        if result.alpha_beta_ratio is not None:
            print(f"alpha/beta   : {result.alpha_beta_ratio:.2f} Gy")
        print("======================")

    @staticmethod
    def report_validation(
        validation_kind: ValidationKind,
        summary: ValidationSummary,
    ) -> None:
        print("\n# Holdout validation:")
        print(
            f"validate_kind={validation_kind} "
            f"response_mode={summary.response_mode} "
            f"points={summary.point_count}"
        )
        if summary.response_mode == "scalar":
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
                f"sf_mode={summary.sf_mode} response={summary.response_mode} "
                f"model={summary.model_kind} family={family} status={summary.status} "
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
        run_results: Optional[Sequence[AnalysisRunResult]] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sf_mode",
                    "response_mode",
                    "model_kind",
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
                    "training_r_squared",
                    "training_adjusted_r_squared",
                    "training_bic",
                    "cv_rmse",
                    "cv_mae",
                    "cv_r_squared",
                    "mean_bed",
                    "mean_eqd2",
                    "mean_g_factor",
                    "mean_tcp",
                    "alpha_0",
                    "lambda_alpha",
                    "repair_half_time_hours",
                    "repair_half_time_fast_hours",
                    "repair_half_time_slow_hours",
                    "repair_fast_fraction",
                    "reason",
                ]
            )
            run_lookup: Dict[Tuple[str, str, str, str], AnalysisRunResult] = {}
            if run_results is not None:
                run_lookup = {
                    (
                        run.summary.sf_mode,
                        run.summary.response_mode,
                        run.summary.model_kind,
                        run.summary.family or "all",
                    ): run
                    for run in run_results
                }
            for summary in summaries:
                ratio = None
                if summary.alpha is not None and summary.beta is not None and summary.beta > 0.0:
                    ratio = summary.alpha / summary.beta
                run = run_lookup.get(
                    (
                        summary.sf_mode,
                        summary.response_mode,
                        summary.model_kind,
                        summary.family or "all",
                    )
                )
                training_metrics = run.training_metrics if run is not None else None
                cross_validation = run.cross_validation if run is not None else None
                mean_bed = run.mean_train_bed if run is not None else None
                mean_eqd2 = run.mean_train_eqd2 if run is not None else None
                mean_g_factor = run.mean_train_g_factor if run is not None else None
                mean_tcp = run.estimate_mean_train_tcp() if run is not None else None
                fit_result = run.fit_result if run is not None else None
                writer.writerow(
                    [
                        summary.sf_mode,
                        summary.response_mode,
                        summary.model_kind,
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
                        training_metrics.r_squared if training_metrics is not None else None,
                        training_metrics.adjusted_r_squared if training_metrics is not None else None,
                        training_metrics.bic if training_metrics is not None else None,
                        cross_validation.cv_rmse if cross_validation is not None else None,
                        cross_validation.cv_mae if cross_validation is not None else None,
                        cross_validation.cv_r_squared if cross_validation is not None else None,
                        mean_bed,
                        mean_eqd2,
                        mean_g_factor,
                        mean_tcp,
                        fit_result.alpha_0 if fit_result is not None else None,
                        fit_result.lambda_alpha if fit_result is not None else None,
                        fit_result.repair_half_time_hours if fit_result is not None else None,
                        fit_result.repair_half_time_fast_hours if fit_result is not None else None,
                        fit_result.repair_half_time_slow_hours if fit_result is not None else None,
                        fit_result.repair_fast_fraction if fit_result is not None else None,
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
    response_mode: ResponseMode,
    requested_model_kind: RequestedModelKind,
    compare_models: bool,
    bootstrap: int,
    bootstrap_seed: Optional[int],
    cross_validate_loo: bool = True,
) -> List[AnalysisRunResult]:
    """Run one or many analyses on an already collected fitter."""
    results: List[AnalysisRunResult] = []
    if response_mode == "curve" and sf_modes:
        sf_modes = [sf_modes[0]]

    default_model_kind = (
        (
            "repair_biexp"
            if requested_model_kind == "auto"
            and fitter.repair_fast_rate_per_day is not None
            and fitter.repair_slow_rate_per_day is not None
            else (
                "repair_lq"
                if requested_model_kind == "auto" and fitter.repair_rate_per_day is not None
                else ("classic_lq" if requested_model_kind == "auto" else requested_model_kind)
            )
        )
    )

    for mode_index, sf_mode in enumerate(sf_modes):
        if fitter.raw_experiments:
            experiments = fitter.materialize_experiments(sf_mode=sf_mode)
        else:
            experiments = list(fitter.experiments)
        available_families = fitter.available_families(experiments)
        if requested_model_kind == "let_dependent":
            requested_families = [None]
        else:
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
                        response_mode=response_mode,
                        model_kind=default_model_kind,
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
            effective_family = None if requested_model_kind == "let_dependent" else run_family
            total_count, single_count, fractionated_count = fitter.regimen_counts(
                family=effective_family,
                experiments=experiments,
            )
            train, validation = fitter.build_train_validation_sets(
                fit_kind=fit_kind,
                validate_kind=validate_kind,
                family=effective_family,
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
                            family=effective_family,
                            total_count=total_count,
                            single_count=single_count,
                            fractionated_count=fractionated_count,
                            train_count=len(train),
                            validation_count=len(validation),
                            status="skipped",
                            response_mode=response_mode,
                            model_kind=default_model_kind,
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

            model_comparison: Tuple[ModelComparisonRow, ...] = ()
            if compare_models:
                model_comparison = fitter.compare_models(
                    experiments=train,
                    response_mode=response_mode,
                    family=effective_family,
                    sf_mode=sf_mode,
                )

            if requested_model_kind == "auto":
                successful_rows = [
                    row for row in model_comparison if row.status == "ok"
                ]
                if successful_rows:
                    model_kind: ModelKind = successful_rows[0].model_kind
                else:
                    model_kind = (
                        "repair_biexp"
                        if fitter.repair_fast_rate_per_day is not None
                        and fitter.repair_slow_rate_per_day is not None
                        else ("repair_lq" if fitter.repair_rate_per_day is not None else "classic_lq")
                    )
            else:
                model_kind = requested_model_kind

            try:
                fit_result = fitter.fit(
                    experiments=train,
                    train_kind=fit_kind,
                    family=effective_family,
                    sf_mode=sf_mode,
                    response_mode=response_mode,
                    model_kind=model_kind,
                )
                fit_metrics = fitter.compute_fit_metrics(train, fit_result)
            except (RuntimeError, ValueError) as exc:
                results.append(
                    AnalysisRunResult(
                        summary=AnalysisRunSummary(
                            sf_mode=sf_mode,
                            family=effective_family,
                            total_count=total_count,
                            single_count=single_count,
                            fractionated_count=fractionated_count,
                            train_count=len(train),
                            validation_count=len(validation),
                            status="skipped",
                            response_mode=response_mode,
                            model_kind=model_kind,
                            reason=str(exc),
                        ),
                        train=tuple(train),
                        validation=tuple(validation),
                        train_kind=fit_kind,
                        validation_kind=validate_kind,
                        timing_diagnostics=timing_diagnostics,
                    )
                )
                continue

            validation_summary = None
            if validate_kind != "none" and validation:
                validation_summary = fitter.evaluate(validation, fit_result)

            cross_validation = None
            if cross_validate_loo and response_mode == "scalar" and len(train) >= 3:
                try:
                    cross_validation = fitter.cross_validate_loo(
                        experiments=train,
                        family=effective_family,
                        sf_mode=sf_mode,
                        response_mode=response_mode,
                        model_kind=model_kind,
                    )
                except (RuntimeError, ValueError):
                    cross_validation = None

            bootstrap_summary = None
            if bootstrap > 0:
                seed = None
                if bootstrap_seed is not None:
                    seed = bootstrap_seed + mode_index * 1000 + family_index
                bootstrap_summary = fitter.bootstrap_fit(
                    sf_mode=sf_mode,
                    fit_kind=fit_kind,
                    family=effective_family,
                    repeats=bootstrap,
                    response_mode=response_mode,
                    model_kind=model_kind,
                    seed=seed,
                )

            results.append(
                AnalysisRunResult(
                    summary=AnalysisRunSummary(
                        sf_mode=sf_mode,
                        family=effective_family,
                        total_count=total_count,
                        single_count=single_count,
                        fractionated_count=fractionated_count,
                        train_count=len(train),
                        validation_count=len(validation),
                        status="ok",
                        response_mode=response_mode,
                        model_kind=fit_result.model_kind,
                        alpha=fit_result.alpha,
                        beta=fit_result.beta,
                    ),
                    train=tuple(train),
                    validation=tuple(validation),
                    train_kind=fit_kind,
                    validation_kind=validate_kind,
                    fit_result=fit_result,
                    training_metrics=fit_metrics,
                    validation_summary=validation_summary,
                    bootstrap_summary=bootstrap_summary,
                    cross_validation=cross_validation,
                    timing_diagnostics=timing_diagnostics,
                    model_comparison=model_comparison,
                )
            )

    return results


def analyze_files(
    files: Optional[List[str]],
    sf_modes: Sequence[str],
    alpha: Optional[float],
    repair_half_time_hours: Optional[float],
    repair_half_time_fast_hours: Optional[float],
    repair_half_time_slow_hours: Optional[float],
    repair_fast_fraction: float,
    min_sf: float,
    fit_kind: RegimenKind,
    validate_kind: ValidationKind,
    family: Optional[str],
    by_family: bool,
    aggregate_regimens: bool,
    dedupe_regimens: bool,
    response_mode: ResponseMode,
    requested_model_kind: RequestedModelKind,
    compare_models: bool,
    bootstrap: int,
    bootstrap_seed: Optional[int],
    verbose: bool,
    cross_validate_loo: bool = True,
    control_map: Optional[Mapping[object, Optional[object]]] = None,
) -> Tuple[Fitter, List[AnalysisRunResult]]:
    """Collect Excel files and run structured analysis results."""
    paths = [Path(file).resolve() for file in files] if files else sorted(Path.cwd().glob("*.xlsx"))
    fitter = Fitter(
        sf_mode=sf_modes[0] if sf_modes else "absolute",
        min_sf=min_sf,
        alpha_fixed=alpha,
        repair_half_time_hours=repair_half_time_hours,
        repair_half_time_fast_hours=repair_half_time_fast_hours,
        repair_half_time_slow_hours=repair_half_time_slow_hours,
        repair_fast_fraction=repair_fast_fraction,
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
        response_mode=response_mode,
        requested_model_kind=requested_model_kind,
        compare_models=compare_models,
        cross_validate_loo=cross_validate_loo,
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
    if run.training_metrics is not None:
        print("\n# Training metrics:")
        print(
            f"points={run.training_metrics.point_count} "
            f"MAE={run.training_metrics.mae:.6f} "
            f"RMSE={run.training_metrics.rmse:.6f} "
            f"mean_abs_log_error={run.training_metrics.mean_abs_log_error:.6f} "
            f"RSS={run.training_metrics.rss:.6f}"
        )
        if run.training_metrics.aic is not None:
            print(f"AIC={run.training_metrics.aic:.4f}")
        if run.training_metrics.r_squared is not None:
            print(f"R^2={run.training_metrics.r_squared:.4f}")
        if run.training_metrics.adjusted_r_squared is not None:
            print(f"Adjusted R^2={run.training_metrics.adjusted_r_squared:.4f}")
        if run.training_metrics.bic is not None:
            print(f"BIC={run.training_metrics.bic:.4f}")

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
        if diagnostics.repair_half_time_fast_hours is not None:
            print(f"repair_half_time_fast_hours={diagnostics.repair_half_time_fast_hours:.3f}")
        if diagnostics.repair_half_time_slow_hours is not None:
            print(f"repair_half_time_slow_hours={diagnostics.repair_half_time_slow_hours:.3f}")
        if diagnostics.repair_fast_fraction is not None:
            print(f"repair_fast_fraction={diagnostics.repair_fast_fraction:.3f}")
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

    if run.model_comparison:
        print("\n# Model comparison:")
        for row in run.model_comparison:
            line = (
                f"model={row.model_kind} response={row.response_mode} status={row.status}"
            )
            if row.metrics is not None:
                line += (
                    f" points={row.metrics.point_count}"
                    f" MAE={row.metrics.mae:.6f}"
                    f" RMSE={row.metrics.rmse:.6f}"
                    f" mean_abs_log_error={row.metrics.mean_abs_log_error:.6f}"
                )
                if row.metrics.aic is not None:
                    line += f" AIC={row.metrics.aic:.4f}"
            if row.alpha is not None:
                line += f" alpha={row.alpha:.6f}"
            if row.beta is not None:
                line += f" beta={row.beta:.6f}"
            if row.lambda_alpha is not None:
                line += f" lambda_alpha={row.lambda_alpha:.6f}"
            if row.curve_clearance_rate is not None:
                line += f" clearance={row.curve_clearance_rate:.6f}"
            if row.transition_dose is not None:
                line += f" transition_dose={row.transition_dose:.6f}"
            if row.saturation_dose is not None:
                line += f" saturation_dose={row.saturation_dose:.6f}"
            if row.lag_days is not None:
                line += f" lag_days={row.lag_days:.6f}"
            if row.repopulation_rate is not None:
                line += f" repopulation_rate={row.repopulation_rate:.6f}"
            if row.reason:
                line += f" note={row.reason}"
            print(line)


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
        "--repair-half-time-fast",
        type=float,
        help="Fast repair half-time in hours for the bi-exponential repair model.",
    )
    parser.add_argument(
        "--repair-half-time-slow",
        type=float,
        help="Slow repair half-time in hours for the bi-exponential repair model.",
    )
    parser.add_argument(
        "--repair-fast-fraction",
        type=float,
        default=0.6,
        help="Weight of the fast repair component for the bi-exponential repair model (default: 0.6).",
    )
    parser.add_argument(
        "--min-sf",
        type=float,
        default=1.0,
        help="Drop experiments with SF >= MIN_SF (default: 1.0)",
    )
    parser.add_argument(
        "--response-mode",
        choices=("scalar", "curve"),
        default="scalar",
        help="Fit one SF per regimen or the full normalized response curve (default: scalar)",
    )
    parser.add_argument(
        "--model-kind",
        choices=(
            "auto",
            "classic_lq",
            "repair_lq",
            "repair_biexp",
            "glq",
            "let_dependent",
            "linear",
            "lq_l",
            "lq_repop",
            "repair_repop",
        ),
        default="auto",
        help="Model family to fit (default: auto)",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Fit several model families and compare them by training error",
    )
    parser.add_argument(
        "--cross-validate-loo",
        dest="cross_validate_loo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable explicit leave-one-out cross-validation reporting (default: enabled).",
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
    repair_half_time_fast_hours: Optional[float],
    repair_half_time_slow_hours: Optional[float],
    repair_fast_fraction: float,
    min_sf: float,
    response_mode: ResponseMode,
    requested_model_kind: RequestedModelKind,
    compare_models: bool,
    cross_validate_loo: bool,
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
        repair_half_time_fast_hours=repair_half_time_fast_hours,
        repair_half_time_slow_hours=repair_half_time_slow_hours,
        repair_fast_fraction=repair_fast_fraction,
        min_sf=min_sf,
        fit_kind=fit_kind,
        validate_kind=validate_kind,
        family=family,
        by_family=by_family,
        aggregate_regimens=aggregate_regimens,
        dedupe_regimens=dedupe_regimens,
        response_mode=response_mode,
        requested_model_kind=requested_model_kind,
        compare_models=compare_models,
        cross_validate_loo=cross_validate_loo,
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
        fitter.write_analysis_summaries_csv(output_path, summaries, run_results=results)
        print(f"\n# Summary CSV written to {output_path}")


def main() -> None:
    if USE_INLINE_PARAMS:
        run_fit(
            files=INLINE_FILES,
            sf_modes=INLINE_SF_MODES,
            alpha=INLINE_ALPHA,
            repair_half_time_hours=INLINE_REPAIR_HALF_TIME_HOURS,
            repair_half_time_fast_hours=INLINE_REPAIR_HALF_TIME_FAST_HOURS,
            repair_half_time_slow_hours=INLINE_REPAIR_HALF_TIME_SLOW_HOURS,
            repair_fast_fraction=INLINE_REPAIR_FAST_FRACTION,
            min_sf=INLINE_MIN_SF,
            response_mode=INLINE_RESPONSE_MODE,
            requested_model_kind=INLINE_MODEL_KIND,
            compare_models=INLINE_COMPARE_MODELS,
            cross_validate_loo=INLINE_CROSS_VALIDATE_LOO,
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
        repair_half_time_fast_hours=args.repair_half_time_fast,
        repair_half_time_slow_hours=args.repair_half_time_slow,
        repair_fast_fraction=args.repair_fast_fraction,
        min_sf=args.min_sf,
        response_mode=args.response_mode,
        requested_model_kind=args.model_kind,
        compare_models=args.compare_models,
        cross_validate_loo=args.cross_validate_loo,
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
