# coding: utf-8
"""End-to-end GEANT4 -> radiobiology -> growth prediction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from work_with_prepared_data.radiobioligy_project.survival.dose_reader import (
        DoseMap,
        read_dose_map,
        read_full_dose_map,
    )
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
        FAMILY_LET_DEFAULTS,
        LQFitResult,
    )
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
        fit_let_dependence,
    )
    from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
        GeometryReference,
        GrowthModelParameters,
        TreatmentFraction,
        simulate_growth,
    )
    from work_with_prepared_data.radiobioligy_project.survival.voxel_sf_calculator import (
        VolumetricSFResult,
        compute_mixed_voxel_sf,
        compute_voxel_sf,
    )
except ModuleNotFoundError:
    from survival.dose_reader import DoseMap, read_dose_map, read_full_dose_map
    from survival.fit_alpha_beta_using_processor import FAMILY_LET_DEFAULTS, LQFitResult
    from survival.let_parametrization import LETDependentParams, fit_let_dependence
    from survival.tumor_growth_predictor import (
        GeometryReference,
        GrowthModelParameters,
        TreatmentFraction,
        simulate_growth,
    )
    from survival.voxel_sf_calculator import (
        VolumetricSFResult,
        compute_mixed_voxel_sf,
        compute_voxel_sf,
    )


DEFAULT_COMPONENT_FAMILY_MAP: Dict[str, str] = {
    "protonDose": "p",
    "midDose": "n",
    "mainDose": "y",
    "stuffDose": "e",
}


def run_prediction_pipeline(
    dose_pb_path: Path,
    geometry_ivz_path: Path,
    *,
    contour_path: Optional[Path] = None,
    fit_results_csv: Optional[Path] = None,
    let_params: Optional[LETDependentParams] = None,
    manual_alpha_beta: Optional[Tuple[float, float]] = None,
    structure_name: str = "tumor",
    initial_volume_mm3: Optional[float] = None,
    growth_rate: float = 0.05,
    carrying_capacity: float = 5000.0,
    clearance_rate: float = 0.1,
    schedule_days: Optional[List[float]] = None,
    n_fractions: Optional[int] = None,
    mixed_field: bool = False,
    component_family_map: Optional[Mapping[str, str]] = None,
    output_dir: Path = Path("./prediction_output"),
    growth_duration_days: float = 30.0,
    growth_time_step_days: float = 0.5,
    model_kind: str = "classic_lq",
    repair_half_time_hours: Optional[float] = None,
    bed_eqd2_fractions: Sequence[int] = (1, 3, 5, 10, 20, 30),
) -> dict:
    """Run the voxel-to-growth prediction pipeline and export CSV/JSON artifacts."""
    dose_pb_path = Path(dose_pb_path)
    geometry_ivz_path = Path(geometry_ivz_path)
    contour_path = None if contour_path is None else Path(contour_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_schedule_days = _resolve_schedule_days(schedule_days, n_fractions)
    resolved_let_params = _resolve_let_params(
        let_params=let_params,
        manual_alpha_beta=manual_alpha_beta,
        fit_results_csv=fit_results_csv,
    )

    if mixed_field:
        component_maps = read_full_dose_map(
            dose_pb_path,
            geometry_ivz_path,
            contour_path=contour_path,
            contour_structure_name=structure_name,
        )
        dose_map = component_maps["totDose"]
        volumetric_sf = compute_mixed_voxel_sf(
            component_dose_maps={
                name: dose_map_component
                for name, dose_map_component in component_maps.items()
                if name != "totDose"
            },
            component_family_map=dict(component_family_map or DEFAULT_COMPONENT_FAMILY_MAP),
            let_params=resolved_let_params,
            structure_name=structure_name,
        )
    else:
        dose_map = read_dose_map(
            dose_pb_path,
            geometry_ivz_path,
            contour_path,
            contour_structure_name=structure_name,
        )
        volumetric_sf = compute_voxel_sf(
            dose_map=dose_map,
            let_params=resolved_let_params,
            structure_name=structure_name,
            model_kind=model_kind,
            repair_half_time_hours=repair_half_time_hours,
            n_fractions=len(resolved_schedule_days),
            schedule_days=tuple(resolved_schedule_days),
        )

    tumor_voxel_ids = dose_map.tumor_voxel_ids(structure_name)
    dvh = dose_map.dose_volume_histogram(tumor_voxel_ids)
    reference_volume_mm3 = (
        float(initial_volume_mm3)
        if initial_volume_mm3 is not None
        else _estimate_structure_volume_mm3(dose_map, tumor_voxel_ids)
    )
    reference = _build_geometry_reference(reference_volume_mm3)
    predictor_parameters = GrowthModelParameters(
        alpha=volumetric_sf.effective_alpha,
        beta=volumetric_sf.effective_beta,
        growth_rate=float(growth_rate),
        carrying_capacity=float(carrying_capacity),
        clearance_rate=float(clearance_rate),
        repair_half_time_hours=float(repair_half_time_hours or 0.0),
    )
    predictor_schedule = _build_predictor_schedule(volumetric_sf.mean_dose_gy, resolved_schedule_days)
    sample_times = _build_sample_times(
        schedule_days=resolved_schedule_days,
        growth_duration_days=growth_duration_days,
        growth_time_step_days=growth_time_step_days,
    )
    growth_prediction = simulate_growth(
        sample_times=sample_times,
        parameters=predictor_parameters,
        reference=reference,
        schedule=predictor_schedule,
        volumetric_sf=volumetric_sf,
    )

    output_files = _write_pipeline_outputs(
        output_dir=output_dir,
        structure_name=structure_name,
        dose_map=dose_map,
        tumor_voxel_ids=tumor_voxel_ids,
        volumetric_sf=volumetric_sf,
        growth_prediction=growth_prediction,
        dvh=dvh,
        resolved_let_params=resolved_let_params,
        predictor_schedule=predictor_schedule,
        reference_volume_mm3=reference_volume_mm3,
        mixed_field=mixed_field,
        component_family_map=component_family_map,
        bed_eqd2_fractions=bed_eqd2_fractions,
    )
    return {
        "volumetric_sf": volumetric_sf,
        "growth_prediction": growth_prediction,
        "dvh": dvh,
        "output_files": output_files,
    }


def _resolve_schedule_days(
    schedule_days: Optional[Sequence[float]],
    n_fractions: Optional[int],
) -> List[float]:
    if schedule_days is not None:
        resolved = [float(value) for value in schedule_days]
        if any(not np.isfinite(value) or value < 0.0 for value in resolved):
            raise ValueError("schedule_days must contain finite non-negative values.")
        if any(resolved[index] < resolved[index - 1] for index in range(1, len(resolved))):
            raise ValueError("schedule_days must be sorted in ascending order.")
        if n_fractions is not None and len(resolved) != int(n_fractions):
            raise ValueError("n_fractions must match the length of schedule_days.")
        if not resolved:
            raise ValueError("schedule_days must not be empty.")
        return resolved
    resolved_fractions = int(n_fractions or 1)
    if resolved_fractions <= 0:
        raise ValueError("n_fractions must be positive.")
    return [float(day) for day in range(resolved_fractions)]


def _resolve_let_params(
    *,
    let_params: Optional[LETDependentParams],
    manual_alpha_beta: Optional[Tuple[float, float]],
    fit_results_csv: Optional[Path],
) -> LETDependentParams:
    if let_params is not None:
        return let_params
    if manual_alpha_beta is not None:
        return LETDependentParams(
            alpha_0=float(manual_alpha_beta[0]),
            lambda_alpha=0.0,
            beta_0=float(manual_alpha_beta[1]),
            lambda_beta=0.0,
        )
    if fit_results_csv is not None:
        family_results = _load_fit_results_csv(Path(fit_results_csv))
        if not family_results:
            raise ValueError(f"No usable fit results were found in {fit_results_csv}.")
        if len(family_results) == 1:
            only_result = next(iter(family_results.values()))
            alpha_0 = (
                float(only_result.alpha_0)
                if only_result.alpha_0 is not None
                else float(only_result.alpha)
            )
            lambda_alpha = float(only_result.lambda_alpha or 0.0)
            return LETDependentParams(
                alpha_0=alpha_0,
                lambda_alpha=lambda_alpha,
                beta_0=float(only_result.beta),
                lambda_beta=0.0,
            )
        family_lets = {
            family: FAMILY_LET_DEFAULTS.get(family, 0.3)
            for family in family_results
        }
        return fit_let_dependence(family_results, family_lets)
    raise ValueError("Provide let_params, manual_alpha_beta, or fit_results_csv.")


def _load_fit_results_csv(path: Path) -> Dict[str, LQFitResult]:
    family_results: Dict[str, LQFitResult] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = (row.get("status") or "ok").strip().lower()
            if status not in ("ok", "success", ""):
                continue
            family = _normalize_family_label(row.get("family"))
            if family is None:
                continue
            alpha = _float_or_none(row.get("alpha"))
            beta = _float_or_none(row.get("beta"))
            if alpha is None or beta is None:
                continue
            train_count = int(_float_or_none(row.get("train_count")) or 0)
            model_kind = str(row.get("model_kind") or "classic_lq").strip() or "classic_lq"
            fit_result = LQFitResult(
                alpha=float(alpha),
                beta=float(beta),
                train_count=max(train_count, 1),
                train_kind="all",
                family=family,
                sf_mode=str(row.get("sf_mode") or "absolute"),
                model_kind=model_kind,  # type: ignore[arg-type]
                alpha_0=_float_or_none(row.get("alpha_0")),
                lambda_alpha=_float_or_none(row.get("lambda_alpha")),
            )
            current = family_results.get(family)
            if current is None or fit_result.train_count >= current.train_count:
                family_results[family] = fit_result
    return family_results


def _float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _normalize_family_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == "all":
        return None
    return normalized


def _estimate_structure_volume_mm3(dose_map: DoseMap, voxel_ids: Sequence[int]) -> float:
    if not voxel_ids:
        raise ValueError("No voxels selected for volume estimation.")
    voxel_volume_mm3 = (
        float(dose_map.voxel_size_mm[0])
        * float(dose_map.voxel_size_mm[1])
        * float(dose_map.voxel_size_mm[2])
    )
    return float(len(voxel_ids) * voxel_volume_mm3)


def _build_geometry_reference(volume_mm3: float) -> GeometryReference:
    if volume_mm3 <= 0.0:
        raise ValueError("Initial tumor volume must be positive.")
    axis = float(volume_mm3 ** (1.0 / 3.0))
    return GeometryReference(axis_a=axis, axis_b=axis, axis_c=axis, volume=float(volume_mm3))


def _build_predictor_schedule(
    mean_total_dose_gy: float,
    schedule_days: Sequence[float],
) -> List[TreatmentFraction]:
    fraction_count = len(schedule_days)
    if fraction_count <= 0:
        raise ValueError("schedule_days must not be empty.")
    doses = [float(mean_total_dose_gy) / float(fraction_count) for _ in schedule_days]
    return [
        TreatmentFraction(day=float(day), dose=float(dose))
        for day, dose in zip(schedule_days, doses)
    ]


def _build_sample_times(
    *,
    schedule_days: Sequence[float],
    growth_duration_days: float,
    growth_time_step_days: float,
) -> np.ndarray:
    if growth_time_step_days <= 0.0:
        raise ValueError("growth_time_step_days must be positive.")
    horizon = max(float(schedule_days[-1]) + float(growth_duration_days), float(schedule_days[-1]))
    steps = max(int(np.ceil(horizon / growth_time_step_days)), 1)
    return np.linspace(0.0, float(steps) * growth_time_step_days, steps + 1)


def _write_pipeline_outputs(
    *,
    output_dir: Path,
    structure_name: str,
    dose_map: DoseMap,
    tumor_voxel_ids: Sequence[int],
    volumetric_sf: VolumetricSFResult,
    growth_prediction,
    dvh: Tuple[np.ndarray, np.ndarray],
    resolved_let_params: LETDependentParams,
    predictor_schedule: Sequence[TreatmentFraction],
    reference_volume_mm3: float,
    mixed_field: bool,
    component_family_map: Optional[Mapping[str, str]],
    bed_eqd2_fractions: Sequence[int],
) -> List[Path]:
    output_files: List[Path] = []

    dvh_path = output_dir / f"dvh_{structure_name}.csv"
    with dvh_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dose_gy", "volume_fraction_ge"])
        for dose_value, volume_fraction in zip(*dvh):
            writer.writerow([float(dose_value), float(volume_fraction)])
    output_files.append(dvh_path)

    sf_path = output_dir / "sf_per_voxel.csv"
    with sf_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["voxel_id", "dose_gy", "let_kev_um", "alpha", "beta", "sf", "bed"])
        for row in volumetric_sf.voxel_results:
            writer.writerow(
                [
                    row.voxel_id,
                    row.dose_gy,
                    row.let_kev_um,
                    row.alpha,
                    row.beta,
                    row.sf,
                    row.bed,
                ]
            )
    output_files.append(sf_path)

    aggregated_path = output_dir / "aggregated_params.json"
    aggregated_payload = {
        "structure_name": structure_name,
        "voxel_count": len(tumor_voxel_ids),
        "grid_shape": list(dose_map.grid_shape),
        "voxel_size_mm": list(dose_map.voxel_size_mm),
        "mean_sf": volumetric_sf.mean_sf,
        "volume_weighted_sf": volumetric_sf.volume_weighted_sf,
        "mean_dose_gy": volumetric_sf.mean_dose_gy,
        "mean_let_kev_um": volumetric_sf.mean_let_kev_um,
        "d90": volumetric_sf.d90,
        "d50": volumetric_sf.d50,
        "v20": volumetric_sf.v20,
        "effective_alpha": volumetric_sf.effective_alpha,
        "effective_beta": volumetric_sf.effective_beta,
        "equivalent_uniform_dose": volumetric_sf.equivalent_uniform_dose,
        "initial_volume_mm3": reference_volume_mm3,
        "mixed_field": mixed_field,
    }
    aggregated_path.write_text(json.dumps(aggregated_payload, indent=2), encoding="utf-8")
    output_files.append(aggregated_path)

    growth_path = output_dir / "growth_curve.csv"
    with growth_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_days", "live_volume", "dead_volume", "total_volume", "axis_a", "axis_b", "axis_c"])
        for values in zip(
            growth_prediction.times,
            growth_prediction.live_volume,
            growth_prediction.dead_volume,
            growth_prediction.total_volume,
            growth_prediction.axis_a,
            growth_prediction.axis_b,
            growth_prediction.axis_c,
        ):
            writer.writerow([float(value) for value in values])
    output_files.append(growth_path)

    bed_eqd2_path = output_dir / "bed_eqd2_table.csv"
    with bed_eqd2_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_fractions", "total_dose_gy", "dose_per_fraction_gy", "bed", "eqd2"])
        for row in _bed_eqd2_rows(
            alpha=volumetric_sf.effective_alpha,
            beta=volumetric_sf.effective_beta,
            fractions=bed_eqd2_fractions,
        ):
            writer.writerow(row)
    output_files.append(bed_eqd2_path)

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "structure_name": structure_name,
        "schedule_days": [float(item.day) for item in predictor_schedule],
        "schedule_dose_gy": [float(item.dose) for item in predictor_schedule],
        "mixed_field": mixed_field,
        "component_family_map": dict(component_family_map or {}),
        "let_params": {
            "alpha_0": resolved_let_params.alpha_0,
            "lambda_alpha": resolved_let_params.lambda_alpha,
            "beta_0": resolved_let_params.beta_0,
            "lambda_beta": resolved_let_params.lambda_beta,
            "alpha_r_squared": resolved_let_params.alpha_r_squared,
            "beta_r_squared": resolved_let_params.beta_r_squared,
        },
        "aggregated": aggregated_payload,
        "output_files": [str(path) for path in output_files],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    output_files.append(summary_path)

    return output_files


def _bed_eqd2_rows(
    *,
    alpha: float,
    beta: float,
    fractions: Sequence[int],
    total_dose_grid: Iterable[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0),
) -> List[List[float]]:
    if alpha <= 0.0 or beta < 0.0:
        return []
    alpha_beta_ratio = float("inf") if beta == 0.0 else float(alpha / beta)
    rows: List[List[float]] = []
    for n_fractions in fractions:
        if n_fractions <= 0:
            continue
        for total_dose in total_dose_grid:
            dose_per_fraction = float(total_dose) / float(n_fractions)
            if np.isinf(alpha_beta_ratio):
                bed = float(total_dose)
                eqd2 = float(total_dose)
            else:
                bed = float(total_dose * (1.0 + dose_per_fraction / alpha_beta_ratio))
                eqd2 = float(bed / (1.0 + 2.0 / alpha_beta_ratio))
            rows.append(
                [
                    int(n_fractions),
                    float(total_dose),
                    dose_per_fraction,
                    bed,
                    eqd2,
                ]
            )
    return rows


def _parse_float_csv(raw_value: Optional[str]) -> Optional[List[float]]:
    if raw_value is None:
        return None
    values = [chunk.strip() for chunk in str(raw_value).split(",")]
    parsed: List[float] = []
    for value in values:
        if not value:
            continue
        parsed.append(float(value))
    return parsed or None


def _parse_int_csv(raw_value: Optional[str]) -> Optional[List[int]]:
    values = _parse_float_csv(raw_value)
    if values is None:
        return None
    return [int(value) for value in values]


def _parse_component_family_map(raw_value: Optional[str]) -> Optional[Dict[str, str]]:
    if raw_value is None or not str(raw_value).strip():
        return None
    parsed: Dict[str, str] = {}
    for chunk in str(raw_value).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid component family mapping '{item}'. Expected component=family pairs."
            )
        component_name, family = item.split("=", 1)
        component_key = str(component_name).strip()
        family_value = str(family).strip().lower()
        if not component_key or not family_value:
            raise ValueError(f"Invalid component family mapping '{item}'.")
        parsed[component_key] = family_value
    return parsed or None


def _resolve_cli_let_params(args: argparse.Namespace) -> tuple[Optional[LETDependentParams], Optional[Tuple[float, float]]]:
    if args.fit_results_csv:
        return None, None

    has_let_profile = args.let_alpha0 is not None or args.let_beta0 is not None
    if has_let_profile:
        if args.let_alpha0 is None or args.let_beta0 is None:
            raise ValueError("Both --let-alpha0 and --let-beta0 are required for explicit LET parameters.")
        return (
            LETDependentParams(
                alpha_0=float(args.let_alpha0),
                lambda_alpha=float(args.let_lambda_alpha or 0.0),
                beta_0=float(args.let_beta0),
                lambda_beta=float(args.let_lambda_beta or 0.0),
            ),
            None,
        )

    if args.alpha is not None or args.beta is not None:
        if args.alpha is None or args.beta is None:
            raise ValueError("Both --alpha and --beta are required for manual alpha/beta input.")
        return None, (float(args.alpha), float(args.beta))

    raise ValueError(
        "Provide one radiobiology source: --fit-results-csv, --alpha/--beta, "
        "or --let-alpha0/--let-beta0."
    )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the GEANT4 -> voxel SF -> tumor growth prediction pipeline."
    )
    parser.add_argument("dose_pb_path", help="Path to GEANT4 dose protobuf (.pb).")
    parser.add_argument("geometry_ivz_path", help="Path to geometry InputVoxelMap protobuf (.ivz/.pb).")
    parser.add_argument(
        "--contour-path",
        help="Optional ContourMeta protobuf or 3D Slicer NIfTI mask path.",
    )
    parser.add_argument("--fit-results-csv", help="Optional fitter summary CSV used to derive alpha/beta or LET params.")
    parser.add_argument("--alpha", type=float, help="Manual alpha value (Gy^-1).")
    parser.add_argument("--beta", type=float, help="Manual beta value (Gy^-2).")
    parser.add_argument("--let-alpha0", type=float, help="Explicit LET model alpha_0 (Gy^-1).")
    parser.add_argument("--let-lambda-alpha", type=float, default=0.0, help="Explicit LET model lambda_alpha.")
    parser.add_argument("--let-beta0", type=float, help="Explicit LET model beta_0 (Gy^-2).")
    parser.add_argument("--let-lambda-beta", type=float, default=0.0, help="Explicit LET model lambda_beta.")
    parser.add_argument("--structure-name", default="tumor", help="Target structure name (default: tumor).")
    parser.add_argument("--initial-volume-mm3", type=float, help="Initial tumor volume in mm^3.")
    parser.add_argument("--growth-rate", type=float, default=0.05, help="Gompertz growth rate per day.")
    parser.add_argument("--carrying-capacity", type=float, default=5000.0, help="Tumor carrying capacity in mm^3.")
    parser.add_argument("--clearance-rate", type=float, default=0.1, help="Dead-cell clearance rate per day.")
    parser.add_argument("--schedule-days", help="Comma-separated fraction times in days, e.g. 0,1,2,3,4.")
    parser.add_argument("--n-fractions", type=int, help="Fraction count when schedule days are not provided.")
    parser.add_argument("--mixed-field", action="store_true", help="Treat dose protobuf as fullVoxelMap mixed-field input.")
    parser.add_argument(
        "--component-family-map",
        help="Comma-separated component=family map, e.g. protonDose=p,mainDose=y,midDose=n.",
    )
    parser.add_argument("--output-dir", default="./prediction_output", help="Output directory path.")
    parser.add_argument("--growth-duration-days", type=float, default=30.0, help="Days to simulate after the last fraction.")
    parser.add_argument("--growth-time-step-days", type=float, default=0.5, help="Growth simulation time step in days.")
    parser.add_argument(
        "--model-kind",
        choices=("classic_lq", "repair_lq", "linear", "let_dependent"),
        default="classic_lq",
        help="Voxel SF model kind for single-field mode.",
    )
    parser.add_argument("--repair-half-time-hours", type=float, help="Repair half-time in hours for repair-aware voxel SF.")
    parser.add_argument(
        "--bed-eqd2-fractions",
        default="1,3,5,10,20,30",
        help="Comma-separated fraction counts for BED/EQD2 export.",
    )
    return parser


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_cli_parser()
    return parser.parse_args(argv)


def cli_main(argv: Optional[Sequence[str]] = None) -> dict:
    args = parse_cli(argv)
    let_params, manual_alpha_beta = _resolve_cli_let_params(args)
    schedule_days = _parse_float_csv(args.schedule_days)
    bed_eqd2_fractions = _parse_int_csv(args.bed_eqd2_fractions) or [1, 3, 5, 10, 20, 30]
    component_family_map = _parse_component_family_map(args.component_family_map)
    result = run_prediction_pipeline(
        dose_pb_path=Path(args.dose_pb_path),
        geometry_ivz_path=Path(args.geometry_ivz_path),
        contour_path=(None if args.contour_path is None else Path(args.contour_path)),
        fit_results_csv=(None if args.fit_results_csv is None else Path(args.fit_results_csv)),
        let_params=let_params,
        manual_alpha_beta=manual_alpha_beta,
        structure_name=args.structure_name,
        initial_volume_mm3=args.initial_volume_mm3,
        growth_rate=args.growth_rate,
        carrying_capacity=args.carrying_capacity,
        clearance_rate=args.clearance_rate,
        schedule_days=schedule_days,
        n_fractions=args.n_fractions,
        mixed_field=bool(args.mixed_field),
        component_family_map=component_family_map,
        output_dir=Path(args.output_dir),
        growth_duration_days=args.growth_duration_days,
        growth_time_step_days=args.growth_time_step_days,
        model_kind=args.model_kind,
        repair_half_time_hours=args.repair_half_time_hours,
        bed_eqd2_fractions=bed_eqd2_fractions,
    )
    volumetric_sf = result["volumetric_sf"]
    summary_path = next(
        (path for path in result["output_files"] if Path(path).name == "summary.json"),
        None,
    )
    print(f"mean_dose_gy={volumetric_sf.mean_dose_gy:.6f}")
    print(f"mean_sf={volumetric_sf.mean_sf:.6f}")
    print(f"effective_alpha={volumetric_sf.effective_alpha:.6f}")
    print(f"effective_beta={volumetric_sf.effective_beta:.6f}")
    if summary_path is not None:
        print(f"summary_json={Path(summary_path)}")
    return result


if __name__ == "__main__":
    cli_main()
