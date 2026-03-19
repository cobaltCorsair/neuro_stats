# coding: utf-8
"""Phenomenological tumor-growth predictor driven by alpha/beta and dose schedule."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import curve_fit

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class TreatmentFraction:
    """One irradiation event in days and Gy."""

    day: float
    dose: float
    family: str | None = None


@dataclass(frozen=True)
class GeometryReference:
    """Reference ellipsoid used to reconstruct predicted axes from volume."""

    axis_a: float
    axis_b: float
    axis_c: float
    volume: float


@dataclass(frozen=True)
class GeometryScalingModel:
    """Axis-scaling law used to reconstruct ellipsoid shape from volume."""

    mode: str
    coeff_a: float
    power_a: float
    coeff_b: float
    power_b: float
    coeff_c: float
    power_c: float


@dataclass(frozen=True)
class GrowthModelParameters:
    """Parameters for the live/dead tumor-volume model."""

    alpha: float
    beta: float
    growth_rate: float
    carrying_capacity: float
    clearance_rate: float
    repair_half_time_hours: float = 0.0

    @property
    def repair_rate_per_day(self) -> float | None:
        """Convert repair half-time in hours to an exponential decay rate in days."""
        if self.repair_half_time_hours <= 0.0:
            return None
        return math.log(2.0) * 24.0 / self.repair_half_time_hours


@dataclass(frozen=True)
class ControlFitResult:
    """Control-fit result for a Gompertz growth curve."""

    growth_rate: float
    carrying_capacity: float
    initial_volume: float
    fitted_points: int


@dataclass(frozen=True)
class GrowthSimulationResult:
    """Predicted volume compartments and ellipsoid axes over time."""

    times: np.ndarray
    live_volume: np.ndarray
    dead_volume: np.ndarray
    total_volume: np.ndarray
    axis_a: np.ndarray
    axis_b: np.ndarray
    axis_c: np.ndarray


def surviving_fraction(
    alpha: float,
    beta: float,
    dose: float,
    prior_unrepaired_dose: float = 0.0,
) -> float:
    """LQ surviving fraction for one dose fraction, with optional repair memory."""
    prior_unrepaired_dose = max(float(prior_unrepaired_dose), 0.0)
    quadratic_term = dose * dose + 2.0 * dose * prior_unrepaired_dose
    return float(np.exp(-(alpha * dose + beta * quadratic_term)))


def advance_unrepaired_dose(
    unrepaired_dose: float,
    dt_days: float,
    repair_rate_per_day: float | None,
) -> float:
    """Decay the residual quadratic LQ memory between fractions."""
    if repair_rate_per_day is None or repair_rate_per_day <= 0.0:
        return 0.0
    if dt_days <= 0.0:
        return float(unrepaired_dose)
    return float(unrepaired_dose * math.exp(-repair_rate_per_day * dt_days))


def gompertz_volume(
    time_days: np.ndarray | float,
    initial_volume: float,
    growth_rate: float,
    carrying_capacity: float,
) -> np.ndarray:
    """Closed-form Gompertz growth."""
    time_days = np.asarray(time_days, dtype=float)
    if initial_volume <= 0.0:
        return np.zeros_like(time_days)
    if carrying_capacity <= initial_volume:
        return np.full_like(time_days, initial_volume)
    exponent = np.log(initial_volume / carrying_capacity) * np.exp(-growth_rate * time_days)
    return carrying_capacity * np.exp(exponent)


def gompertz_step(volume: float, dt_days: float, growth_rate: float, carrying_capacity: float) -> float:
    """Advance one live-volume state by dt using the exact Gompertz solution."""
    if dt_days <= 0.0:
        return float(volume)
    return float(gompertz_volume(dt_days, volume, growth_rate, carrying_capacity))


def dead_volume_step(volume: float, dt_days: float, clearance_rate: float) -> float:
    """Exponential clearance for the damaged compartment."""
    if dt_days <= 0.0:
        return float(volume)
    return float(volume * math.exp(-clearance_rate * dt_days))


def build_default_schedule(
    fractions: Sequence[float],
    start_day: float = 0.0,
    spacing_days: float = 1.0,
) -> list[TreatmentFraction]:
    """Create a simple editable schedule from dose fractions."""
    return [
        TreatmentFraction(day=float(start_day + index * spacing_days), dose=float(dose))
        for index, dose in enumerate(fractions)
    ]


def parse_irradiation_intervals_days(experiment_params: Sequence[str]) -> list[float]:
    """Extract irradiation intervals from metadata tokens like ``t = 1 ч``."""
    intervals: list[float] = []
    for raw_token in experiment_params:
        token = str(raw_token).strip().lower().replace(",", ".")
        if token.startswith("irradiation time="):
            token = token.split("=", 1)[1].strip()
        if "t" not in token or "=" not in token:
            continue
        if not re.search(r"\bt\s*=", token):
            continue

        values = [float(value) for value in NUMBER.findall(token)]
        if not values:
            continue

        factor = 1.0
        if any(unit in token for unit in ("ч", "час", "hour", "hours", "hr", "hrs")):
            factor = 1.0 / 24.0
        elif any(unit in token for unit in ("мин", "minute", "minutes", "min", "mins")):
            factor = 1.0 / (24.0 * 60.0)
        elif any(unit in token for unit in ("сут", "дн", "день", "дня", "дней", "day", "days")):
            factor = 1.0

        intervals.extend(value * factor for value in values if np.isfinite(value) and value >= 0.0)

    return intervals


def build_schedule_from_intervals(
    fractions: Sequence[float],
    interval_days: Sequence[float] | None = None,
    *,
    start_day: float = 0.0,
    default_spacing_days: float = 1.0,
) -> list[TreatmentFraction]:
    """Create a schedule using explicit inter-fraction gaps in days when available."""
    fractions = [float(dose) for dose in fractions]
    if not fractions:
        return []
    if len(fractions) == 1:
        return [TreatmentFraction(day=float(start_day), dose=fractions[0])]

    intervals = [
        float(value)
        for value in (interval_days or [])
        if np.isfinite(value) and float(value) >= 0.0
    ]
    if not intervals:
        return build_default_schedule(
            fractions,
            start_day=float(start_day),
            spacing_days=float(default_spacing_days),
        )

    schedule: list[TreatmentFraction] = []
    current_day = float(start_day)
    for index, dose in enumerate(fractions):
        schedule.append(TreatmentFraction(day=current_day, dose=dose))
        if index >= len(fractions) - 1:
            continue
        gap = intervals[index] if index < len(intervals) else intervals[-1]
        current_day += gap
    return schedule


def sanitize_schedule(schedule: Iterable[TreatmentFraction]) -> list[TreatmentFraction]:
    """Sort and filter irradiation events."""
    cleaned = [
        TreatmentFraction(
            day=float(item.day),
            dose=float(item.dose),
            family=(None if item.family is None else str(item.family).strip().lower() or None),
        )
        for item in schedule
        if np.isfinite(item.day) and np.isfinite(item.dose) and item.dose > 0.0
    ]
    return sorted(cleaned, key=lambda item: (item.day, item.dose, item.family or ""))


def _fraction_parameter_key(fraction: TreatmentFraction) -> str:
    return fraction.family or "__default__"


def _resolve_fraction_parameters(
    fraction: TreatmentFraction,
    default_parameters: GrowthModelParameters,
    family_parameters: Mapping[str, GrowthModelParameters] | None,
) -> GrowthModelParameters:
    if fraction.family is None:
        return default_parameters
    if family_parameters is None:
        return default_parameters
    family_key = str(fraction.family).strip().lower()
    if family_key not in family_parameters:
        return default_parameters
    return family_parameters[family_key]


def predict_schedule_surviving_fraction(
    schedule: Sequence[TreatmentFraction],
    parameters: GrowthModelParameters,
    family_parameters: Mapping[str, GrowthModelParameters] | None = None,
) -> float:
    """Predict net surviving fraction from the schedule without growth dynamics."""
    schedule = sanitize_schedule(schedule)
    if not schedule:
        return 1.0

    normalized_family_parameters = None
    if family_parameters is not None:
        normalized_family_parameters = {
            str(key).strip().lower(): value
            for key, value in family_parameters.items()
        }

    surviving = 1.0
    unrepaired_dose_by_family: dict[str, float] = {}
    current_time = float(schedule[0].day)

    for event in schedule:
        event_parameters = _resolve_fraction_parameters(
            event,
            parameters,
            normalized_family_parameters,
        )
        dt_event = max(float(event.day) - current_time, 0.0)
        unrepaired_dose_by_family = _decay_unrepaired_dose_map(
            unrepaired_dose_by_family,
            dt_event,
            parameters,
            normalized_family_parameters,
        )
        current_time = max(current_time, float(event.day))

        event_sf = surviving_fraction(
            event_parameters.alpha,
            event_parameters.beta,
            event.dose,
            prior_unrepaired_dose=sum(unrepaired_dose_by_family.values()),
        )
        surviving *= event_sf

        repair_rate = event_parameters.repair_rate_per_day
        if repair_rate is not None and repair_rate > 0.0:
            family_key = _fraction_parameter_key(event)
            unrepaired_dose_by_family[family_key] = (
                unrepaired_dose_by_family.get(family_key, 0.0) + float(event.dose)
            )

    return float(surviving)


def _decay_unrepaired_dose_map(
    unrepaired_dose_by_family: dict[str, float],
    dt_days: float,
    default_parameters: GrowthModelParameters,
    family_parameters: Mapping[str, GrowthModelParameters] | None,
) -> dict[str, float]:
    if dt_days <= 0.0 or not unrepaired_dose_by_family:
        return dict(unrepaired_dose_by_family)

    decayed: dict[str, float] = {}
    for family_key, unrepaired_dose in unrepaired_dose_by_family.items():
        if family_key == "__default__":
            parameters = default_parameters
        else:
            if family_parameters is None or family_key not in family_parameters:
                parameters = default_parameters
            else:
                parameters = family_parameters[family_key]
        value = advance_unrepaired_dose(
            unrepaired_dose,
            dt_days,
            parameters.repair_rate_per_day,
        )
        if value > 1.0e-12:
            decayed[family_key] = value
    return decayed


def default_geometry_scaling(reference: GeometryReference) -> GeometryScalingModel:
    """Build the baseline fixed-ratio scaling model."""
    if reference.volume <= 0.0:
        raise ValueError("Reference volume must be positive.")
    root_volume = reference.volume ** (1.0 / 3.0)
    return GeometryScalingModel(
        mode="fixed",
        coeff_a=reference.axis_a / root_volume,
        power_a=1.0 / 3.0,
        coeff_b=reference.axis_b / root_volume,
        power_b=1.0 / 3.0,
        coeff_c=reference.axis_c / root_volume,
        power_c=1.0 / 3.0,
    )


def fit_geometry_scaling(
    volumes: Sequence[float],
    axis_a: Sequence[float],
    axis_b: Sequence[float],
    axis_c: Sequence[float],
    reference: GeometryReference,
) -> GeometryScalingModel:
    """Fit axis = coeff * volume^power from observed geometry."""
    volumes = np.asarray(volumes, dtype=float)
    axis_a = np.asarray(axis_a, dtype=float)
    axis_b = np.asarray(axis_b, dtype=float)
    axis_c = np.asarray(axis_c, dtype=float)
    valid_mask = (
        np.isfinite(volumes)
        & np.isfinite(axis_a)
        & np.isfinite(axis_b)
        & np.isfinite(axis_c)
        & (volumes > 0.0)
        & (axis_a > 0.0)
        & (axis_b > 0.0)
        & (axis_c > 0.0)
    )
    if int(np.sum(valid_mask)) < 2:
        return default_geometry_scaling(reference)

    log_volume = np.log(volumes[valid_mask])
    if float(np.nanstd(log_volume)) < 1.0e-8:
        return default_geometry_scaling(reference)

    fallback = default_geometry_scaling(reference)

    def fit_axis(
        axis_values: np.ndarray,
        fallback_coeff: float,
        fallback_power: float,
    ) -> tuple[float, float]:
        slope, intercept = np.polyfit(log_volume, np.log(axis_values[valid_mask]), deg=1)
        coeff = float(np.exp(intercept))
        power = float(slope)
        if not np.isfinite(coeff) or not np.isfinite(power):
            return fallback_coeff, fallback_power
        return coeff, power

    coeff_a, power_a = fit_axis(axis_a, fallback.coeff_a, fallback.power_a)
    coeff_b, power_b = fit_axis(axis_b, fallback.coeff_b, fallback.power_b)
    coeff_c, power_c = fit_axis(axis_c, fallback.coeff_c, fallback.power_c)
    return GeometryScalingModel(
        mode="fitted",
        coeff_a=coeff_a,
        power_a=power_a,
        coeff_b=coeff_b,
        power_b=power_b,
        coeff_c=coeff_c,
        power_c=power_c,
    )


def scale_axes_from_volume(
    total_volume: np.ndarray,
    reference: GeometryReference,
    scaling_model: GeometryScalingModel | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct ellipsoid axes from total volume and a scaling law."""
    scaling_model = scaling_model or default_geometry_scaling(reference)
    safe_volume = np.maximum(np.asarray(total_volume, dtype=float), 0.0)
    return (
        scaling_model.coeff_a * np.power(safe_volume, scaling_model.power_a),
        scaling_model.coeff_b * np.power(safe_volume, scaling_model.power_b),
        scaling_model.coeff_c * np.power(safe_volume, scaling_model.power_c),
    )


def simulate_growth(
    sample_times: Sequence[float],
    parameters: GrowthModelParameters,
    reference: GeometryReference,
    schedule: Sequence[TreatmentFraction],
    scaling_model: GeometryScalingModel | None = None,
    family_parameters: Mapping[str, GrowthModelParameters] | None = None,
) -> GrowthSimulationResult:
    """Simulate tumor dynamics with Gompertz growth, LQ kill, and delayed clearance."""
    times = np.asarray(sample_times, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("sample_times must be a non-empty 1D sequence.")
    if np.any(~np.isfinite(times)):
        raise ValueError("sample_times must be finite.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("sample_times must be sorted in ascending order.")
    if reference.volume <= 0.0:
        raise ValueError("Reference volume must be positive.")
    if parameters.carrying_capacity <= reference.volume:
        raise ValueError("carrying_capacity must exceed the reference volume.")

    schedule = sanitize_schedule(schedule)
    live = float(reference.volume)
    dead = 0.0
    normalized_family_parameters = None
    if family_parameters is not None:
        normalized_family_parameters = {
            str(key).strip().lower(): value
            for key, value in family_parameters.items()
        }

    unrepaired_dose_by_family: dict[str, float] = {}
    current_time = float(times[0])
    if current_time < 0.0:
        raise ValueError("sample_times must start at zero or later.")

    live_values = np.empty_like(times)
    dead_values = np.empty_like(times)
    schedule_index = 0
    for time_index, target_time in enumerate(times):
        while schedule_index < len(schedule) and schedule[schedule_index].day <= target_time + 1.0e-12:
            event = schedule[schedule_index]
            event_parameters = _resolve_fraction_parameters(
                event,
                parameters,
                normalized_family_parameters,
            )
            dt_event = max(event.day - current_time, 0.0)
            live = gompertz_step(live, dt_event, parameters.growth_rate, parameters.carrying_capacity)
            dead = dead_volume_step(dead, dt_event, parameters.clearance_rate)
            unrepaired_dose_by_family = _decay_unrepaired_dose_map(
                unrepaired_dose_by_family,
                dt_event,
                parameters,
                normalized_family_parameters,
            )
            current_time = max(current_time, event.day)

            sf = surviving_fraction(
                event_parameters.alpha,
                event_parameters.beta,
                event.dose,
                prior_unrepaired_dose=sum(unrepaired_dose_by_family.values()),
            )
            killed = live * (1.0 - sf)
            live *= sf
            dead += killed
            repair_rate = event_parameters.repair_rate_per_day
            if repair_rate is not None and repair_rate > 0.0:
                family_key = _fraction_parameter_key(event)
                unrepaired_dose_by_family[family_key] = (
                    unrepaired_dose_by_family.get(family_key, 0.0) + event.dose
                )
            schedule_index += 1

        dt_target = max(float(target_time) - current_time, 0.0)
        live = gompertz_step(live, dt_target, parameters.growth_rate, parameters.carrying_capacity)
        dead = dead_volume_step(dead, dt_target, parameters.clearance_rate)
        unrepaired_dose_by_family = _decay_unrepaired_dose_map(
            unrepaired_dose_by_family,
            dt_target,
            parameters,
            normalized_family_parameters,
        )
        current_time = float(target_time)
        live_values[time_index] = live
        dead_values[time_index] = dead

    total_values = live_values + dead_values
    axis_a, axis_b, axis_c = scale_axes_from_volume(total_values, reference, scaling_model)
    return GrowthSimulationResult(
        times=times,
        live_volume=live_values,
        dead_volume=dead_values,
        total_volume=total_values,
        axis_a=axis_a,
        axis_b=axis_b,
        axis_c=axis_c,
    )


def fit_gompertz_to_control(
    time_days: Sequence[float],
    volumes: Sequence[float],
) -> ControlFitResult:
    """Fit a Gompertz curve to a control trajectory."""
    times = np.asarray(time_days, dtype=float)
    data = np.asarray(volumes, dtype=float)
    valid_mask = np.isfinite(times) & np.isfinite(data) & (data > 0.0)
    times = times[valid_mask]
    data = data[valid_mask]
    if len(times) < 3:
        raise ValueError("Need at least three valid control points to fit Gompertz growth.")

    times = times - times[0]
    initial_volume = float(data[0])
    if initial_volume <= 0.0:
        raise ValueError("Initial control volume must be positive.")

    lower_capacity = max(initial_volume * 1.01, float(np.nanmax(data)) * 1.01)
    upper_capacity = max(float(np.nanmax(data)) * 100.0, lower_capacity * 2.0)
    initial_capacity = max(float(np.nanmax(data)) * 2.0, lower_capacity)

    def model(time_value: np.ndarray, growth_rate: float, carrying_capacity: float) -> np.ndarray:
        return gompertz_volume(time_value, initial_volume, growth_rate, carrying_capacity)

    params, _ = curve_fit(
        model,
        times,
        data,
        p0=(0.15, initial_capacity),
        bounds=((1.0e-6, lower_capacity), (5.0, upper_capacity)),
        maxfev=20000,
    )
    return ControlFitResult(
        growth_rate=float(params[0]),
        carrying_capacity=float(params[1]),
        initial_volume=initial_volume,
        fitted_points=len(times),
    )
