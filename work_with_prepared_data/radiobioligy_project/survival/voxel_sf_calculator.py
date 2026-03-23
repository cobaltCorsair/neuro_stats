# coding: utf-8
"""Per-voxel surviving-fraction calculations on top of GEANT4 dose maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from work_with_prepared_data.radiobioligy_project.survival.dose_reader import DoseMap
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
    )
    from work_with_prepared_data.radiobioligy_project.survival.mixed_field_model import (
        FieldComponent,
        compute_mixed_field_sf,
    )
    from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
        advance_unrepaired_dose,
        surviving_fraction,
    )
except ModuleNotFoundError:
    from survival.dose_reader import DoseMap
    from survival.let_parametrization import LETDependentParams
    from survival.mixed_field_model import FieldComponent, compute_mixed_field_sf
    from survival.tumor_growth_predictor import advance_unrepaired_dose, surviving_fraction

SupportedVoxelModelKind = str


@dataclass(frozen=True)
class VoxelSF:
    """Radiobiological summary for one voxel."""

    voxel_id: int
    dose_gy: float
    let_kev_um: float
    alpha: float
    beta: float
    sf: float
    bed: float


@dataclass(frozen=True)
class VolumetricSFResult:
    """Aggregated tumor-volume SF result based on voxel-wise calculations."""

    voxel_results: Tuple[VoxelSF, ...]
    mean_sf: float
    volume_weighted_sf: float
    mean_dose_gy: float
    mean_let_kev_um: float
    d90: float
    d50: float
    v20: float
    effective_alpha: float
    effective_beta: float
    equivalent_uniform_dose: float


def compute_voxel_sf(
    dose_map: DoseMap,
    let_params: LETDependentParams,
    structure_name: str = "tumor",
    model_kind: SupportedVoxelModelKind = "classic_lq",
    repair_half_time_hours: Optional[float] = None,
    n_fractions: int = 1,
    schedule_days: Optional[Tuple[float, ...]] = None,
) -> VolumetricSFResult:
    """
    Compute per-voxel SF and volume-level aggregates for one structure.

    Assumption:
    `dose_map` stores total delivered dose per voxel. For multi-fraction schedules,
    that total dose is split uniformly across `n_fractions`.
    """
    voxel_ids = dose_map.tumor_voxel_ids(structure_name)
    if not voxel_ids:
        raise ValueError(f"No voxels matched structure '{structure_name}'.")
    if n_fractions <= 0:
        raise ValueError("n_fractions must be positive.")

    schedule = _build_schedule_days(n_fractions=n_fractions, schedule_days=schedule_days)
    voxel_results = tuple(
        _compute_single_voxel_sf(
            dose_map=dose_map,
            voxel_id=voxel_id,
            let_params=let_params,
            model_kind=model_kind,
            repair_half_time_hours=repair_half_time_hours,
            schedule_days=schedule,
        )
        for voxel_id in voxel_ids
    )
    return _aggregate_voxel_results(voxel_results)


def compute_mixed_voxel_sf(
    component_dose_maps: dict[str, DoseMap],
    component_family_map: dict[str, str],
    let_params: LETDependentParams,
    *,
    structure_name: str = "tumor",
    method: str = "zaider_rossi",
) -> VolumetricSFResult:
    """Compute voxel-level SF for a mixed field assembled from component dose maps."""
    if not component_dose_maps:
        raise ValueError("component_dose_maps must not be empty.")

    base_map = next(iter(component_dose_maps.values()))
    voxel_ids = base_map.tumor_voxel_ids(structure_name)
    if not voxel_ids:
        raise ValueError(f"No voxels matched structure '{structure_name}'.")

    voxel_results = tuple(
        _compute_single_mixed_voxel_sf(
            voxel_id=voxel_id,
            component_dose_maps=component_dose_maps,
            component_family_map=component_family_map,
            let_params=let_params,
            method=method,
        )
        for voxel_id in voxel_ids
    )
    return _aggregate_voxel_results(voxel_results)


def _aggregate_voxel_results(voxel_results: Sequence[VoxelSF]) -> VolumetricSFResult:
    if not voxel_results:
        raise ValueError("At least one voxel result is required.")
    doses = np.asarray([item.dose_gy for item in voxel_results], dtype=float)
    lets = np.asarray([item.let_kev_um for item in voxel_results], dtype=float)
    alphas = np.asarray([item.alpha for item in voxel_results], dtype=float)
    betas = np.asarray([item.beta for item in voxel_results], dtype=float)
    sfs = np.asarray([item.sf for item in voxel_results], dtype=float)

    mean_sf = float(np.mean(sfs))
    effective_alpha = float(np.mean(alphas))
    effective_beta = float(np.mean(betas))
    return VolumetricSFResult(
        voxel_results=voxel_results,
        mean_sf=mean_sf,
        volume_weighted_sf=mean_sf,
        mean_dose_gy=float(np.mean(doses)),
        mean_let_kev_um=float(np.mean(lets)),
        d90=_dose_covering_fraction(doses, 0.90),
        d50=_dose_covering_fraction(doses, 0.50),
        v20=float(np.mean(doses > 20.0)),
        effective_alpha=effective_alpha,
        effective_beta=effective_beta,
        equivalent_uniform_dose=_equivalent_uniform_dose(
            mean_sf=mean_sf,
            alpha=effective_alpha,
            beta=effective_beta,
        ),
    )


def _compute_single_mixed_voxel_sf(
    *,
    voxel_id: int,
    component_dose_maps: dict[str, DoseMap],
    component_family_map: dict[str, str],
    let_params: LETDependentParams,
    method: str,
) -> VoxelSF:
    components: list[FieldComponent] = []
    for component_name, dose_map in component_dose_maps.items():
        if voxel_id not in dose_map.voxels:
            continue
        voxel = dose_map.voxels[voxel_id]
        if voxel.dose_gy <= 0.0:
            continue
        family = component_family_map.get(component_name, component_name)
        components.append(
            FieldComponent(
                family=family,
                dose_fraction_gy=float(voxel.dose_gy),
                mean_let_kev_um=float(voxel.let_kev_um),
            )
        )
    if not components:
        return VoxelSF(
            voxel_id=voxel_id,
            dose_gy=0.0,
            let_kev_um=0.0,
            alpha=0.0,
            beta=0.0,
            sf=1.0,
            bed=0.0,
        )

    mixed = compute_mixed_field_sf(components, let_params, method=method)
    total_dose = float(sum(component.dose_fraction_gy for component in components))
    weighted_let = float(
        sum(component.dose_fraction_gy * component.mean_let_kev_um for component in components) / total_dose
    )
    selected_sf = mixed.sf_zaider_rossi if str(method).strip().lower() == "zaider_rossi" else mixed.sf_tdra
    bed = _compute_bed(mixed.effective_alpha, mixed.effective_beta, [total_dose])
    return VoxelSF(
        voxel_id=voxel_id,
        dose_gy=total_dose,
        let_kev_um=weighted_let,
        alpha=float(mixed.effective_alpha),
        beta=float(mixed.effective_beta),
        sf=float(selected_sf),
        bed=bed,
    )


def _compute_single_voxel_sf(
    dose_map: DoseMap,
    voxel_id: int,
    let_params: LETDependentParams,
    model_kind: SupportedVoxelModelKind,
    repair_half_time_hours: Optional[float],
    schedule_days: Sequence[float],
) -> VoxelSF:
    voxel = dose_map.voxels[voxel_id]
    alpha = float(let_params.alpha(voxel.let_kev_um))
    beta = max(float(let_params.beta(voxel.let_kev_um)), 0.0)
    fraction_doses = _split_total_dose(total_dose=voxel.dose_gy, n_fractions=len(schedule_days))
    sf = _schedule_surviving_fraction(
        alpha=alpha,
        beta=beta,
        fraction_doses=fraction_doses,
        schedule_days=schedule_days,
        model_kind=model_kind,
        repair_half_time_hours=repair_half_time_hours,
    )
    bed = _compute_bed(alpha=alpha, beta=beta, fraction_doses=fraction_doses)
    return VoxelSF(
        voxel_id=voxel_id,
        dose_gy=float(voxel.dose_gy),
        let_kev_um=float(voxel.let_kev_um),
        alpha=alpha,
        beta=beta,
        sf=sf,
        bed=bed,
    )


def _build_schedule_days(
    *,
    n_fractions: int,
    schedule_days: Optional[Tuple[float, ...]],
) -> Tuple[float, ...]:
    if schedule_days is None:
        return tuple(float(index) for index in range(n_fractions))
    if len(schedule_days) != n_fractions:
        raise ValueError("schedule_days length must match n_fractions.")
    cleaned = tuple(float(day) for day in schedule_days)
    if any(not np.isfinite(day) for day in cleaned):
        raise ValueError("schedule_days must contain only finite values.")
    if any(day < 0.0 for day in cleaned):
        raise ValueError("schedule_days cannot contain negative times.")
    if any(cleaned[index] < cleaned[index - 1] for index in range(1, len(cleaned))):
        raise ValueError("schedule_days must be sorted in ascending order.")
    return cleaned


def _split_total_dose(total_dose: float, n_fractions: int) -> Tuple[float, ...]:
    if n_fractions <= 0:
        raise ValueError("n_fractions must be positive.")
    if total_dose <= 0.0:
        return tuple(0.0 for _ in range(n_fractions))
    return tuple(float(total_dose) / float(n_fractions) for _ in range(n_fractions))


def _schedule_surviving_fraction(
    *,
    alpha: float,
    beta: float,
    fraction_doses: Sequence[float],
    schedule_days: Sequence[float],
    model_kind: SupportedVoxelModelKind,
    repair_half_time_hours: Optional[float],
) -> float:
    resolved_kind = str(model_kind).strip().lower()
    if resolved_kind in ("classic_lq", "let_dependent"):
        return float(
            np.prod([surviving_fraction(alpha, beta, dose) for dose in fraction_doses], dtype=float)
        )
    if resolved_kind == "linear":
        return float(np.exp(-alpha * float(np.sum(np.asarray(fraction_doses, dtype=float)))))
    if resolved_kind != "repair_lq":
        raise ValueError(f"Unsupported voxel SF model_kind '{model_kind}'.")
    if repair_half_time_hours is None or repair_half_time_hours <= 0.0:
        raise ValueError("repair_half_time_hours must be positive for repair_lq.")

    repair_rate_per_day = math.log(2.0) * 24.0 / float(repair_half_time_hours)
    surviving = 1.0
    prior_unrepaired_dose = 0.0
    current_day = float(schedule_days[0])
    for day, dose in zip(schedule_days, fraction_doses):
        dt_days = max(float(day) - current_day, 0.0)
        prior_unrepaired_dose = advance_unrepaired_dose(
            prior_unrepaired_dose,
            dt_days,
            repair_rate_per_day,
        )
        current_day = float(day)
        surviving *= surviving_fraction(
            alpha,
            beta,
            float(dose),
            prior_unrepaired_dose=prior_unrepaired_dose,
        )
        prior_unrepaired_dose += float(dose)
    return float(surviving)


def _compute_bed(alpha: float, beta: float, fraction_doses: Iterable[float]) -> float:
    fraction_array = np.asarray(list(fraction_doses), dtype=float)
    if alpha <= 0.0 or beta < 0.0:
        return float("nan")
    if beta == 0.0:
        return float(np.sum(fraction_array))
    alpha_beta_ratio = alpha / beta
    return float(np.sum(fraction_array + np.square(fraction_array) / alpha_beta_ratio))


def _dose_covering_fraction(doses: np.ndarray, fraction: float) -> float:
    if doses.size == 0:
        return 0.0
    fraction = min(max(float(fraction), 0.0), 1.0)
    sorted_desc = np.sort(np.asarray(doses, dtype=float))[::-1]
    index = max(int(math.ceil(fraction * len(sorted_desc))) - 1, 0)
    index = min(index, len(sorted_desc) - 1)
    return float(sorted_desc[index])


def _equivalent_uniform_dose(mean_sf: float, alpha: float, beta: float) -> float:
    if not np.isfinite(mean_sf) or mean_sf <= 0.0 or mean_sf > 1.0:
        return float("nan")
    kill = -math.log(float(mean_sf))
    if beta > 0.0:
        discriminant = alpha * alpha + 4.0 * beta * kill
        if discriminant < 0.0:
            return float("nan")
        return float((-alpha + math.sqrt(discriminant)) / (2.0 * beta))
    if alpha <= 0.0:
        return float("nan")
    return float(kill / alpha)
