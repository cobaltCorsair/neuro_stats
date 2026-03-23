# coding: utf-8
"""Surviving-fraction models for mixed radiation fields."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
    )
except ModuleNotFoundError:
    from survival.let_parametrization import LETDependentParams


@dataclass(frozen=True)
class FieldComponent:
    """One component of a mixed radiation field."""

    family: str
    dose_fraction_gy: float
    mean_let_kev_um: float
    weight: float = 0.0


@dataclass(frozen=True)
class MixedFieldResult:
    """SF summary for a mixed field under Zaider-Rossi and TDRA approximations."""

    components: tuple[FieldComponent, ...]
    total_dose_gy: float
    sf_zaider_rossi: float
    sf_tdra: float
    effective_alpha: float
    effective_beta: float


def compute_mixed_field_sf(
    components: Sequence[FieldComponent],
    let_params: LETDependentParams,
    method: str = "zaider_rossi",
) -> MixedFieldResult:
    """Compute mixed-field SF using both Zaider-Rossi and TDRA summaries."""
    normalized_components = _normalize_components(components)
    if not normalized_components:
        raise ValueError("At least one field component is required.")

    total_dose = float(sum(component.dose_fraction_gy for component in normalized_components))
    if total_dose <= 0.0:
        raise ValueError("Total mixed-field dose must be positive.")

    alpha_values = np.asarray(
        [let_params.alpha(component.mean_let_kev_um) for component in normalized_components],
        dtype=float,
    )
    beta_values = np.asarray(
        [max(let_params.beta(component.mean_let_kev_um), 0.0) for component in normalized_components],
        dtype=float,
    )
    dose_values = np.asarray(
        [component.dose_fraction_gy for component in normalized_components],
        dtype=float,
    )
    weights = _component_weights(normalized_components, dose_values)

    zaider_rossi_exponent = float(np.sum(alpha_values * dose_values + beta_values * np.square(dose_values)))
    sf_zaider_rossi = float(np.exp(-zaider_rossi_exponent))

    tdra_alpha = float(np.sum(weights * alpha_values))
    tdra_beta = float(np.square(np.sum(weights * np.sqrt(beta_values))))
    sf_tdra = float(np.exp(-(tdra_alpha * total_dose + tdra_beta * total_dose * total_dose)))

    resolved_method = str(method).strip().lower()
    if resolved_method == "zaider_rossi":
        effective_alpha = float(np.sum((dose_values / total_dose) * alpha_values))
        effective_beta = max(
            float((zaider_rossi_exponent - effective_alpha * total_dose) / (total_dose * total_dose)),
            0.0,
        )
    elif resolved_method == "tdra":
        effective_alpha = tdra_alpha
        effective_beta = tdra_beta
    else:
        raise ValueError("method must be 'zaider_rossi' or 'tdra'.")

    return MixedFieldResult(
        components=normalized_components,
        total_dose_gy=total_dose,
        sf_zaider_rossi=sf_zaider_rossi,
        sf_tdra=sf_tdra,
        effective_alpha=effective_alpha,
        effective_beta=effective_beta,
    )


def _normalize_components(components: Sequence[FieldComponent]) -> tuple[FieldComponent, ...]:
    normalized: list[FieldComponent] = []
    for component in components:
        dose_value = float(component.dose_fraction_gy)
        let_value = float(component.mean_let_kev_um)
        weight_value = float(component.weight)
        if not np.isfinite(dose_value) or dose_value < 0.0:
            raise ValueError("Component doses must be finite non-negative numbers.")
        if not np.isfinite(let_value) or let_value < 0.0:
            raise ValueError("Component LET values must be finite non-negative numbers.")
        if not np.isfinite(weight_value):
            raise ValueError("Component weights must be finite.")
        if dose_value <= 0.0:
            continue
        normalized.append(
            FieldComponent(
                family=str(component.family).strip().lower(),
                dose_fraction_gy=dose_value,
                mean_let_kev_um=let_value,
                weight=weight_value,
            )
        )
    return tuple(normalized)


def _component_weights(
    components: Iterable[FieldComponent],
    dose_values: np.ndarray,
) -> np.ndarray:
    explicit_weights = np.asarray([component.weight for component in components], dtype=float)
    if explicit_weights.size == 0:
        return np.asarray([], dtype=float)
    if np.any(explicit_weights > 0.0):
        weight_sum = float(np.sum(np.clip(explicit_weights, 0.0, None)))
        if weight_sum <= 0.0:
            return dose_values / float(np.sum(dose_values))
        return np.clip(explicit_weights, 0.0, None) / weight_sum
    return dose_values / float(np.sum(dose_values))
