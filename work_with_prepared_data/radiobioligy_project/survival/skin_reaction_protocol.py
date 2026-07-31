"""Reusable helpers for the dissertation skin-reaction protocol.

The module deliberately reuses the validated legacy-score to RTOG converter from
``radiobiology_analysis``.  It adds only experiment metadata parsing, the
documented 2026 peak-proton dose conversion, and small statistical utilities
needed by the reproducible task 3.4 orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # package import
    from .radiobiology_analysis import map_skin_score_to_rtog
except ImportError:  # direct import from the survival directory
    from radiobiology_analysis import map_skin_score_to_rtog


PEAK_PROTON_RBE_2026 = 1.1
PATENT_TIME_INTERCEPT = 100.0
PATENT_TIME_SLOPE = 2.5

# Exact table currently implemented in
# stats_methods/from_our_scale_to_rtog.py.  It is repeated in this pure helper
# so batch analyses can apply the documented correction without importing that
# plotting module (which changes global matplotlib/seaborn styles at import).
# The non-monotone 17->18 day entry is intentionally preserved and audited in
# the dissertation TODO instead of being silently corrected.
DAY_CORRECTIONS = {
    2: 95,
    3: 93,
    4: 90,
    5: 87,
    6: 85,
    7: 82,
    8: 80,
    9: 78,
    10: 75,
    11: 73,
    12: 70,
    13: 68,
    14: 65,
    15: 63,
    16: 60,
    17: 54,
    18: 55,
    19: 53,
    20: 50,
    21: 48,
    22: 45,
    23: 43,
    24: 40,
    25: 37,
    26: 35,
}


@dataclass(frozen=True)
class DoseComponent:
    kind: str
    recorded_dose: float
    physical_dose: float
    dose_basis: str


@dataclass(frozen=True)
class SkinExperimentMetadata:
    irradiation_date: str
    calendar_series: str
    family: str
    regime: str
    geometry: str
    components: tuple[DoseComponent, ...]
    recorded_total_dose: float
    physical_total_dose: float
    dose_basis: str
    fraction_count: int
    order_key: str
    composition_key: str


_COMPONENT_RE = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])"
    r"(?P<kind>c\s*12|с\s*12|γ|y|e|n|p)\s*=\s*(?P<dose>\d+(?:[.,]\d+)?)",
    flags=re.IGNORECASE,
)
_DATE_PARAM_RE = re.compile(
    r"date\s*=\s*(?P<day>\d{1,2})[.]"
    r"(?P<month>\d{1,2})[.](?P<year>\d{4})",
    flags=re.IGNORECASE,
)
_DATE_FILENAME_RE = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})[.]"
    r"(?P<month>\d{1,2})[.](?P<year>\d{4})(?!\d)"
)


# Carbon geometry follows the audited/renamed tumour-response files used by
# task 4.2.  Skin files retain their historical names, so the mapping is kept
# explicit and auditable here.
_C12_GEOMETRY_BY_PREFIX = {
    "05.12.2018_c_12": "modified_peak_filter",
    "1_11.04.2016_c_12": "through",
    "1_12.12.2016_c_12": "modified_peak_filter",
    "1_27.03.2017_c_12": "peak_no_filter",
    "2_11.04.2016_": "through",
    "2_12.12.2016_c_12": "modified_peak_filter",
    "2_27.03.2017_c_12": "through",
}


def _normalise_kind(value: str) -> str:
    compact = re.sub(r"\s+", "", value.lower())
    if compact in {"c12", "с12"}:
        return "c12"
    if compact in {"γ", "y"}:
        return "y"
    if compact in {"e", "n", "p"}:
        return compact
    raise ValueError(f"Unknown radiation component: {value!r}")


def parse_recorded_components(params: Iterable[str]) -> tuple[tuple[str, float], ...]:
    """Return every ordered radiation component found in the Excel header.

    ``finditer`` is intentional: some 2026 files store the complete mixed
    sequence in one header cell separated by slashes.
    """

    components: list[tuple[str, float]] = []
    for raw in params:
        for match in _COMPONENT_RE.finditer(str(raw)):
            kind = _normalise_kind(match.group("kind"))
            dose = float(match.group("dose").replace(",", "."))
            components.append((kind, dose))
    if not components:
        raise ValueError("No radiation dose components found in the Excel header")
    return tuple(components)


def extract_irradiation_date(params: Iterable[str], path: Path) -> str:
    """Extract an ISO date from ``Date=`` or, secondarily, the file name."""

    for raw in params:
        match = _DATE_PARAM_RE.search(str(raw))
        if match:
            parsed = date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
            return parsed.isoformat()
    match = _DATE_FILENAME_RE.search(path.name)
    if match:
        parsed = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        return parsed.isoformat()
    return ""


def carbon_geometry(path: Path) -> str:
    name = path.stem.lower()
    for prefix, geometry in _C12_GEOMETRY_BY_PREFIX.items():
        if name.startswith(prefix.lower()):
            return geometry
    return "unclassified"


def build_experiment_metadata(params: Sequence[str], path: Path) -> SkinExperimentMetadata:
    """Parse radiation family, geometry, fractionation and physical dose."""

    irradiation_date = extract_irradiation_date(params, path)
    year = int(irradiation_date[:4]) if irradiation_date else None
    recorded = parse_recorded_components(params)
    kinds = {kind for kind, _ in recorded}
    peak_proton = "in_peak" in path.stem.lower()

    components: list[DoseComponent] = []
    for kind, dose in recorded:
        if kind == "p" and peak_proton and year == 2026:
            components.append(
                DoseComponent(
                    kind=kind,
                    recorded_dose=dose,
                    physical_dose=dose / PEAK_PROTON_RBE_2026,
                    dose_basis="biologically_weighted_divided_by_1.1",
                )
            )
        else:
            components.append(
                DoseComponent(
                    kind=kind,
                    recorded_dose=dose,
                    physical_dose=dose,
                    dose_basis="physical",
                )
            )

    if len(kinds) > 1:
        regime = "mixed"
    elif len(components) > 1:
        regime = "multifraction"
    else:
        regime = "single"

    if kinds == {"y"}:
        family, geometry = "gamma", "photon"
    elif kinds == {"e"}:
        family, geometry = "electrons", "electron"
    elif kinds == {"n"}:
        family, geometry = "neutrons", "neutron"
    elif kinds == {"p"}:
        geometry = "peak" if peak_proton else "through"
        family = "p_peak" if peak_proton else "p_through"
    elif kinds == {"c12"}:
        geometry = carbon_geometry(path)
        family = f"c12_{geometry}"
    elif kinds == {"n", "p"}:
        family = "mixed_NP"
        geometry = "p_peak" if peak_proton else "p_through"
    elif kinds == {"p", "y"}:
        family = "mixed_gammaP"
        geometry = "p_peak" if peak_proton else "p_through"
    else:
        family = "mixed_other"
        geometry = "mixed"

    kind_labels = {"y": "γ", "e": "e", "n": "n", "p": "p", "c12": "C12"}
    order_key = "→".join(kind_labels[item.kind] for item in components)
    composition_key = "+".join(
        f"{kind_labels[item.kind]}:{item.physical_dose:.6g}"
        for item in sorted(components, key=lambda item: (item.kind, item.physical_dose))
    )
    recorded_total = float(sum(item.recorded_dose for item in components))
    physical_total = float(sum(item.physical_dose for item in components))
    bases = {item.dose_basis for item in components}
    dose_basis = "physical" if bases == {"physical"} else "+".join(sorted(bases))

    calendar_series = irradiation_date or f"consolidated:{path.stem}"
    return SkinExperimentMetadata(
        irradiation_date=irradiation_date,
        calendar_series=calendar_series,
        family=family,
        regime=regime,
        geometry=geometry,
        components=tuple(components),
        recorded_total_dose=recorded_total,
        physical_total_dose=physical_total,
        dose_basis=dose_basis,
        fraction_count=len(components),
        order_key=order_key,
        composition_key=composition_key,
    )


def convert_scores_to_rtog(values: Sequence[float]) -> np.ndarray:
    """Convert finite legacy scores using the program's validated converter."""

    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    if finite.any():
        result[finite] = [map_skin_score_to_rtog(value) for value in array[finite]]
    return result


def apply_day_correction(times: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Subtract the legacy program's tabulated time component by observation day.

    ``values`` may be one animal (1-D) or an animals-by-days matrix (2-D).
    Days absent from the implemented table, including days 0 and 1, are left
    unchanged exactly as in ``CorrectedSkinReactionAnalyzer``.

    This table is retained for a software-compatibility sensitivity analysis.
    The patent-defined calculation should use
    :func:`remove_patent_time_component`, which evaluates ``100 - 2.5 * day``
    directly and therefore does not inherit the anomalous day-17 table entry.
    """

    x = np.asarray(times, dtype=float)
    array = np.asarray(values, dtype=float).copy()
    if array.ndim not in {1, 2}:
        raise ValueError("Skin-reaction values must be one- or two-dimensional")
    if array.shape[-1] != x.size:
        raise ValueError("The last value dimension must match the number of days")
    for position, day in enumerate(x):
        rounded = int(round(float(day)))
        if math.isclose(float(day), rounded, abs_tol=1.0e-9) and rounded in DAY_CORRECTIONS:
            array[..., position] = array[..., position] - DAY_CORRECTIONS[rounded]
    return array


def patent_time_component(times: Sequence[float]) -> np.ndarray:
    """Return the time term ``100 - 2.5 * n`` from patent RU 2855329."""

    return PATENT_TIME_INTERCEPT - PATENT_TIME_SLOPE * np.asarray(times, dtype=float)


def remove_patent_time_component(times: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Remove the exact patent time term before translating severity to RTOG.

    ``values`` may be one animal (1-D) or an animals-by-days matrix (2-D).  The
    raw source files use zero as an explicit no-reaction value; negative results
    produced by subtraction are intentionally retained because the established
    RTOG converter maps every non-positive value to grade zero.
    """

    x = np.asarray(times, dtype=float)
    array = np.asarray(values, dtype=float).copy()
    if array.ndim not in {1, 2}:
        raise ValueError("Skin-reaction values must be one- or two-dimensional")
    if array.shape[-1] != x.size:
        raise ValueError("The last value dimension must match the number of days")
    return array - patent_time_component(x)


def peak_with_day(
    times: Sequence[float],
    values: Sequence[float],
    *,
    lower: float = -math.inf,
    upper: float = math.inf,
) -> tuple[float, float]:
    """Return the maximum observed value and the first day attaining it."""

    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y) & (x >= lower) & (x <= upper)
    if not usable.any():
        return math.nan, math.nan
    x = x[usable]
    y = y[usable]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    peak = float(np.max(y))
    position = int(np.flatnonzero(y == peak)[0])
    return peak, float(x[position])


def duration_above_threshold(
    times: Sequence[float],
    values: Sequence[float],
    threshold: float,
    *,
    lower: float = -math.inf,
    upper: float = math.inf,
) -> tuple[float, bool]:
    """Estimate days at or above a threshold using linear crossing times.

    Every episode inside the requested window contributes.  ``censored`` is
    true when the last usable observation remains at or above the threshold.
    No extrapolation is performed outside observed support.
    """

    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if not usable.any():
        return math.nan, False
    source_x = x[usable]
    source_y = y[usable]
    order = np.argsort(source_x)
    source_x = source_x[order]
    source_y = source_y[order]
    unique_x = np.unique(source_x)
    if unique_x.size != source_x.size:
        source_y = np.asarray(
            [np.mean(source_y[source_x == point]) for point in unique_x], dtype=float
        )
        source_x = unique_x

    first = max(float(lower), float(source_x[0]))
    last = min(float(upper), float(source_x[-1]))
    if last < first:
        return math.nan, False
    inside = (source_x > first) & (source_x < last)
    x_eval = np.concatenate(([first], source_x[inside], [last]))
    y_eval = np.interp(x_eval, source_x, source_y)

    total = 0.0
    for index in range(1, len(x_eval)):
        t0, t1 = float(x_eval[index - 1]), float(x_eval[index])
        v0, v1 = float(y_eval[index - 1]), float(y_eval[index])
        if v0 >= threshold and v1 >= threshold:
            total += t1 - t0
        elif v0 < threshold <= v1 and v1 != v0:
            crossing = t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
            total += t1 - crossing
        elif v0 >= threshold > v1 and v1 != v0:
            crossing = t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
            total += crossing - t0
    return float(total), bool(y_eval[-1] >= threshold)


def time_to_sustained_normalisation(
    times: Sequence[float],
    values: Sequence[float],
    *,
    threshold: float = 100.0,
) -> tuple[float, bool]:
    """Return the final downward crossing to ``threshold`` or below.

    If the last observation is still above the threshold, the returned last day
    is a right-censored lower bound.  If the trajectory never rises above the
    threshold, the first observed day is returned as an uncensored value.
    """

    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if not usable.any():
        return math.nan, False
    x = x[usable]
    y = y[usable]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if y[-1] > threshold:
        return float(x[-1]), True
    for index in range(len(x) - 1, 0, -1):
        if y[index - 1] > threshold >= y[index]:
            crossing = x[index - 1] + (threshold - y[index - 1]) / (y[index] - y[index - 1]) * (
                x[index] - x[index - 1]
            )
            return float(crossing), False
    return float(x[0]), False


def auc_without_extrapolation(
    times: Sequence[float],
    values: Sequence[float],
    *,
    lower: float = 0.0,
    upper: float = 24.0,
) -> tuple[float, float, float, float, float]:
    """Trapezoidal AUC inside a window, interpolating only within support.

    Returns ``(auc, time_normalised_auc, duration, first_day, last_day)``.
    All values are NaN when fewer than two usable points remain.
    """

    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 2:
        return (math.nan,) * 5
    x = x[usable]
    y = y[usable]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Average accidental duplicate time points before integration.
    unique_x = np.unique(x)
    if unique_x.size != x.size:
        y = np.asarray([np.mean(y[x == point]) for point in unique_x], dtype=float)
        x = unique_x

    first = max(float(lower), float(x[0]))
    last = min(float(upper), float(x[-1]))
    if last <= first:
        return (math.nan,) * 5
    inside = (x > first) & (x < last)
    x_eval = np.concatenate(([first], x[inside], [last]))
    y_eval = np.interp(x_eval, x, y)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapezoid(y_eval, x_eval))
    duration = float(last - first)
    return auc, auc / duration, duration, first, last


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = events / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values, preserving NaN positions."""

    raw = np.asarray(p_values, dtype=float)
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(raw))
    if finite_positions.size == 0:
        return adjusted
    order = finite_positions[np.argsort(raw[finite_positions])]
    running = 0.0
    count = len(order)
    for rank, position in enumerate(order):
        candidate = min(1.0, (count - rank) * raw[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted
