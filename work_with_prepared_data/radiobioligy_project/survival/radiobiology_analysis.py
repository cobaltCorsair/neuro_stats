# coding: utf-8
"""Secondary radiobiological analyses built on top of fitter and predictor backends."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    LQFitResult,
    TumorExperiment,
)
from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
    GeometryReference,
    GeometryScalingModel,
    GrowthModelParameters,
    GrowthSimulationResult,
    TreatmentFraction,
    build_schedule_from_intervals,
    simulate_growth,
)


@dataclass(frozen=True)
class RBEPoint:
    """One iso-effect RBE estimate for a test family at a specific dose."""

    reference_family: str
    test_family: str
    test_dose: float
    reference_dose: float
    rbe: float
    test_alpha_beta_ratio: Optional[float]
    reference_alpha_beta_ratio: Optional[float]
    test_model_kind: str
    reference_model_kind: str


@dataclass(frozen=True)
class ParameterSensitivityRow:
    """One one-at-a-time perturbation around the baseline predictor setup."""

    parameter: str
    perturbation_fraction: float
    baseline_value: float
    varied_value: float
    rmse: Optional[float]
    delta_rmse: Optional[float]
    rmse_ratio: Optional[float]
    status: str = "ok"
    reason: str = ""


@dataclass(frozen=True)
class ParameterInfluence:
    """Aggregate relative influence of one parameter on prediction RMSE."""

    parameter: str
    max_abs_delta_rmse: float
    mean_abs_delta_rmse: float
    tested_cases: int


@dataclass(frozen=True)
class ParameterSensitivityReport:
    """One-at-a-time sensitivity results around a baseline predictor state."""

    baseline_rmse: float
    rows: Tuple[ParameterSensitivityRow, ...]
    influence: Tuple[ParameterInfluence, ...]


@dataclass(frozen=True)
class IntervalSensitivityRow:
    """Prediction error for one candidate inter-fraction spacing."""

    interval_hours: float
    rmse: float
    delta_rmse: float
    rmse_ratio: Optional[float]
    schedule_days: Tuple[float, ...]


@dataclass(frozen=True)
class IntervalSensitivityReport:
    """How prediction quality changes when inter-fraction spacing is varied."""

    baseline_rmse: float
    rows: Tuple[IntervalSensitivityRow, ...]


@dataclass(frozen=True)
class SFMetricComparisonRow:
    """Cross-SF-mode parameter drift for the same family/model/response setup."""

    family: str
    response_mode: str
    model_kind: str
    sf_mode: str
    alpha: Optional[float]
    beta: Optional[float]
    alpha_beta_ratio: Optional[float]
    delta_alpha_pct: Optional[float]
    delta_beta_pct: Optional[float]
    delta_ratio_pct: Optional[float]
    baseline_sf_mode: str


@dataclass(frozen=True)
class ScenarioComparisonRow:
    """One simulated treatment scenario summarized by coarse response metrics."""

    scenario: str
    total_physical_dose: float
    family_sequence: Tuple[str, ...]
    min_total_volume: float
    min_total_volume_day: float
    final_total_volume: float
    auc_total_volume: float


@dataclass(frozen=True)
class ScenarioComparisonReport:
    """Side-by-side comparison of several predictor treatment scenarios."""

    rows: Tuple[ScenarioComparisonRow, ...]


@dataclass(frozen=True)
class TCPResult:
    """Tumor-control estimate derived from predicted surviving fraction."""

    dose_total: float
    sf: float
    n_cells: float
    tcp: float
    cell_density: float
    initial_volume_cm3: float
    family: Optional[str]
    model_kind: str


@dataclass(frozen=True)
class NTCPPoint:
    """One point on an NTCP dose-response curve."""

    dose_total: float
    ntcp: float
    td50: float
    m: float


@dataclass(frozen=True)
class NTCPFitGroup:
    """Aggregated skin-reaction outcome counts for one dose group."""

    label: str
    dose_total: Optional[float]
    n_subjects: int
    n_complications: int
    complication_rate: float
    peak_grade_mean: float
    threshold_grade: int
    path: Optional[Path] = None


@dataclass(frozen=True)
class NTCPFitResult:
    """Fitted LKB parameters derived from grouped skin-reaction outcomes."""

    td50: float
    m: float
    negative_log_likelihood: float
    groups: Tuple[NTCPFitGroup, ...]
    subject_count: int
    threshold_grade: int


def _single_fraction_effect(result: LQFitResult, dose: float) -> float:
    """Convert a single fraction dose into the model's biological effect exponent."""
    dose = float(dose)
    if dose < 0.0:
        raise ValueError("Dose must be non-negative.")
    if result.model_kind == "linear":
        return result.alpha * dose
    if result.model_kind == "glq":
        if result.saturation_dose is None:
            raise ValueError("gLQ RBE analysis requires saturation_dose.")
        denominator = 1.0 + dose / result.saturation_dose
        return result.alpha * dose + result.beta * dose * dose / denominator
    if result.model_kind == "lq_l":
        if result.transition_dose is None:
            raise ValueError("LQ-L RBE analysis requires transition_dose.")
        transition_dose = result.transition_dose
        if dose <= transition_dose:
            return result.alpha * dose + result.beta * dose * dose
        linear_slope = result.alpha + 2.0 * result.beta * transition_dose
        transition_effect = result.alpha * transition_dose + result.beta * transition_dose * transition_dose
        return transition_effect + linear_slope * (dose - transition_dose)
    return result.alpha * dose + result.beta * dose * dose


def solve_isoeffective_reference_dose(
    reference_result: LQFitResult,
    target_effect: float,
    *,
    initial_upper_dose: float = 2.0,
    max_upper_dose: float = 1.0e6,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> float:
    """Solve for the reference dose that produces the same single-fraction effect."""
    target_effect = float(target_effect)
    if target_effect < 0.0:
        raise ValueError("target_effect must be non-negative.")
    if target_effect == 0.0:
        return 0.0

    lower = 0.0
    upper = max(float(initial_upper_dose), tolerance)
    while _single_fraction_effect(reference_result, upper) < target_effect:
        upper *= 2.0
        if upper > max_upper_dose:
            raise ValueError("Could not bracket the iso-effective reference dose.")

    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_effect = _single_fraction_effect(reference_result, midpoint)
        if abs(midpoint_effect - target_effect) <= tolerance:
            return midpoint
        if midpoint_effect < target_effect:
            lower = midpoint
        else:
            upper = midpoint

    return 0.5 * (lower + upper)


def compute_rbe(
    reference_result: LQFitResult,
    test_result: LQFitResult,
    test_dose: float,
    *,
    reference_family_label: Optional[str] = None,
    test_family_label: Optional[str] = None,
) -> RBEPoint:
    """Compute single-fraction RBE as D_ref / D_test at equal biological effect."""
    test_dose = float(test_dose)
    if test_dose <= 0.0:
        raise ValueError("test_dose must be positive.")

    target_effect = _single_fraction_effect(test_result, test_dose)
    reference_dose = solve_isoeffective_reference_dose(reference_result, target_effect)
    return RBEPoint(
        reference_family=reference_family_label or (reference_result.family or "reference"),
        test_family=test_family_label or (test_result.family or "test"),
        test_dose=test_dose,
        reference_dose=reference_dose,
        rbe=reference_dose / test_dose,
        test_alpha_beta_ratio=test_result.alpha_beta_ratio,
        reference_alpha_beta_ratio=reference_result.alpha_beta_ratio,
        test_model_kind=test_result.model_kind,
        reference_model_kind=reference_result.model_kind,
    )


def compute_rbe_let(
    fit_result: LQFitResult,
    test_let: float,
    reference_let: float,
    dose: float,
    *,
    reference_label: Optional[str] = None,
    test_label: Optional[str] = None,
) -> RBEPoint:
    """Compute RBE directly from a LET-dependent fit without per-family refits."""
    if fit_result.model_kind != "let_dependent":
        raise ValueError("compute_rbe_let requires a let_dependent fit result.")

    dose = float(dose)
    if dose <= 0.0:
        raise ValueError("dose must be positive.")

    target_effect = (
        fit_result.effective_alpha(float(test_let)) * dose
        + fit_result.beta * dose * dose
    )

    alpha_ref = fit_result.effective_alpha(float(reference_let))

    def reference_effect(candidate_dose: float) -> float:
        return alpha_ref * candidate_dose + fit_result.beta * candidate_dose * candidate_dose

    lower = 0.0
    upper = max(dose, 1.0)
    while reference_effect(upper) < target_effect:
        upper *= 2.0
        if upper > 1.0e6:
            raise ValueError("Could not bracket the iso-effective reference dose for LET-based RBE.")

    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        midpoint_effect = reference_effect(midpoint)
        if abs(midpoint_effect - target_effect) <= 1.0e-8:
            upper = midpoint
            break
        if midpoint_effect < target_effect:
            lower = midpoint
        else:
            upper = midpoint

    reference_dose = 0.5 * (lower + upper)
    alpha_beta_test = fit_result.effective_alpha(float(test_let)) / fit_result.beta if fit_result.beta > 0.0 else None
    alpha_beta_ref = (
        fit_result.effective_alpha(float(reference_let)) / fit_result.beta
        if fit_result.beta > 0.0
        else None
    )
    return RBEPoint(
        reference_family=reference_label or f"LET {float(reference_let):g}",
        test_family=test_label or f"LET {float(test_let):g}",
        test_dose=dose,
        reference_dose=reference_dose,
        rbe=reference_dose / dose,
        test_alpha_beta_ratio=alpha_beta_test,
        reference_alpha_beta_ratio=alpha_beta_ref,
        test_model_kind=fit_result.model_kind,
        reference_model_kind=fit_result.model_kind,
    )


def build_rbe_series(
    reference_result: LQFitResult,
    comparison_results: Mapping[str, LQFitResult],
    doses: Iterable[float] = (2.0, 10.0),
) -> Tuple[RBEPoint, ...]:
    """Build RBE points for several families and several doses."""
    rows = []
    for family_label, fit_result in comparison_results.items():
        for dose in doses:
            rows.append(
                compute_rbe(
                    reference_result=reference_result,
                    test_result=fit_result,
                    test_dose=float(dose),
                    reference_family_label=reference_result.family or "y",
                    test_family_label=family_label,
                )
            )
    return tuple(rows)


def build_rbe_let_series(
    fit_result: LQFitResult,
    family_lets: Mapping[str, float],
    *,
    reference_family: str = "y",
    doses: Iterable[float] = (2.0, 10.0),
) -> Tuple[RBEPoint, ...]:
    """Build LET-based RBE points directly from one let_dependent fit."""
    reference_family_key = reference_family.strip().lower()
    if reference_family_key not in family_lets:
        raise ValueError(f"Unknown reference family '{reference_family}'.")
    reference_let = float(family_lets[reference_family_key])

    rows = []
    for family, test_let in sorted(family_lets.items()):
        family_key = family.strip().lower()
        if family_key == reference_family_key:
            continue
        for dose in doses:
            rows.append(
                compute_rbe_let(
                    fit_result,
                    test_let=float(test_let),
                    reference_let=reference_let,
                    dose=float(dose),
                    reference_label=reference_family_key,
                    test_label=family_key,
                )
            )
    return tuple(rows)


def compute_ntcp_lkb(
    dose_total: float,
    td50: float,
    m: float,
) -> float:
    """Lyman-Kutcher-Burman NTCP with a standard normal CDF via erf."""
    dose_total = float(dose_total)
    td50 = float(td50)
    m = float(m)
    if td50 <= 0.0:
        raise ValueError("td50 must be positive.")
    if m <= 0.0:
        raise ValueError("m must be positive.")
    t_value = (dose_total - td50) / (m * td50)
    return float(0.5 * (1.0 + math.erf(t_value / math.sqrt(2.0))))


def build_ntcp_curve(
    dose_range: Sequence[float],
    *,
    td50: float,
    m: float,
) -> Tuple[NTCPPoint, ...]:
    """Build a simple NTCP dose-response curve over a total-dose grid."""
    rows = []
    for dose_total in dose_range:
        dose_value = float(dose_total)
        if dose_value < 0.0:
            raise ValueError("dose_range must contain non-negative doses.")
        rows.append(
            NTCPPoint(
                dose_total=dose_value,
                ntcp=compute_ntcp_lkb(dose_value, td50=td50, m=m),
                td50=float(td50),
                m=float(m),
            )
        )
    return tuple(rows)


def map_skin_score_to_rtog(score: float) -> int:
    """Map the legacy skin-reaction scale to the integer RTOG grade."""
    score = float(score)
    if score <= 0.0:
        return 0
    if score <= 200.0:
        return 1
    if score <= 400.0:
        return 2
    if score <= 600.0:
        return 3
    return 4


def infer_total_dose_from_filename(path: Path) -> Optional[float]:
    """Infer total dose from a survival/skin-reaction file name when possible."""
    tokens = [token for token in path.stem.lower().replace(",", ".").split("_") if token]
    doses: list[float] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"y", "e", "n", "c", "p"}:
            next_index = index + 1
            if next_index < len(tokens):
                try:
                    doses.append(float(tokens[next_index]))
                    index = next_index + 1
                    continue
                except ValueError:
                    pass
        for prefix in ("y", "e", "n", "c", "p"):
            if token.startswith(prefix) and len(token) > len(prefix):
                suffix = token[len(prefix) :]
                try:
                    doses.append(float(suffix))
                    break
                except ValueError:
                    continue
        index += 1
    if not doses:
        return None
    return float(sum(doses))


def summarize_skin_reaction_file(
    path: Path,
    *,
    threshold_grade: int = 3,
    input_scale: str = "our",
    dose_total: Optional[float] = None,
) -> NTCPFitGroup:
    """Summarize one skin-reaction Excel file into grouped complication counts."""
    frame = pd.read_excel(path)
    numeric = frame.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError(f"{path.name}: no numeric skin-reaction data found.")

    matrix = numeric.to_numpy(dtype=float)
    if input_scale == "our":
        rtog_matrix = np.vectorize(map_skin_score_to_rtog)(matrix)
    elif input_scale == "rtog":
        rtog_matrix = np.asarray(np.rint(matrix), dtype=int)
    else:
        raise ValueError("input_scale must be 'our' or 'rtog'.")

    peak_grades = np.nanmax(rtog_matrix, axis=1)
    valid_mask = np.isfinite(peak_grades)
    peak_grades = peak_grades[valid_mask]
    if peak_grades.size == 0:
        raise ValueError(f"{path.name}: no valid peak grades found.")

    events = peak_grades >= int(threshold_grade)
    resolved_dose = dose_total if dose_total is not None else infer_total_dose_from_filename(path)
    return NTCPFitGroup(
        label=path.name,
        dose_total=(None if resolved_dose is None else float(resolved_dose)),
        n_subjects=int(peak_grades.size),
        n_complications=int(np.sum(events)),
        complication_rate=float(np.mean(events)),
        peak_grade_mean=float(np.mean(peak_grades)),
        threshold_grade=int(threshold_grade),
        path=path,
    )


def fit_ntcp_lkb_from_groups(
    groups: Sequence[NTCPFitGroup],
    *,
    initial_td50: Optional[float] = None,
    initial_m: float = 0.2,
) -> NTCPFitResult:
    """Fit TD50 and m by binomial MLE from grouped complication counts."""
    usable_groups = tuple(
        group
        for group in groups
        if group.dose_total is not None and group.n_subjects > 0
    )
    if len(usable_groups) < 2:
        raise ValueError("Need at least two dose groups with known dose_total to fit NTCP.")

    doses = np.asarray([float(group.dose_total) for group in usable_groups], dtype=float)
    n_subjects = np.asarray([group.n_subjects for group in usable_groups], dtype=float)
    n_events = np.asarray([group.n_complications for group in usable_groups], dtype=float)
    threshold_grade = usable_groups[0].threshold_grade

    def negative_log_likelihood(params: np.ndarray) -> float:
        td50_value = float(params[0])
        m_value = float(params[1])
        probabilities = np.clip(
            np.asarray([compute_ntcp_lkb(dose, td50=td50_value, m=m_value) for dose in doses]),
            1.0e-9,
            1.0 - 1.0e-9,
        )
        return float(
            -np.sum(n_events * np.log(probabilities) + (n_subjects - n_events) * np.log(1.0 - probabilities))
        )

    initial = np.array(
        [
            float(initial_td50 if initial_td50 is not None else np.median(doses)),
            float(initial_m),
        ],
        dtype=float,
    )
    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=((1.0e-6, None), (1.0e-4, 5.0)),
    )
    if not result.success:
        raise ValueError(f"NTCP fit failed: {result.message}")

    td50_value = float(result.x[0])
    m_value = float(result.x[1])
    return NTCPFitResult(
        td50=td50_value,
        m=m_value,
        negative_log_likelihood=float(result.fun),
        groups=usable_groups,
        subject_count=int(np.sum(n_subjects)),
        threshold_grade=int(threshold_grade),
    )


def compute_tcp(
    fit_result: LQFitResult,
    experiment: TumorExperiment,
    initial_volume_cm3: Optional[float] = None,
    *,
    cell_density: float = 1.0e7,
) -> TCPResult:
    """Compute TCP = exp(-N0 * SF) for one experiment."""
    if cell_density <= 0.0:
        raise ValueError("cell_density must be positive.")

    resolved_volume_cm3 = (
        float(initial_volume_cm3)
        if initial_volume_cm3 is not None
        else experiment.initial_volume_cm3
    )
    if resolved_volume_cm3 is None or resolved_volume_cm3 <= 0.0:
        raise ValueError("initial_volume_cm3 must be provided or derivable from the experiment.")

    sf = float(fit_result.predict_sf(experiment))
    n_cells = float(cell_density * resolved_volume_cm3)
    burden = n_cells * sf
    tcp = 0.0 if burden >= 700.0 else float(math.exp(-burden))
    return TCPResult(
        dose_total=experiment.dose_sum,
        sf=sf,
        n_cells=n_cells,
        tcp=tcp,
        cell_density=float(cell_density),
        initial_volume_cm3=resolved_volume_cm3,
        family=experiment.family,
        model_kind=fit_result.model_kind,
    )


def build_tcp_curve(
    fit_result: LQFitResult,
    dose_range: Sequence[float],
    *,
    n_fractions: int,
    initial_volume_cm3: float,
    cell_density: float = 1.0e7,
    schedule_interval_days: float = 1.0,
    family: Optional[str] = None,
) -> Tuple[TCPResult, ...]:
    """Compute TCP on a synthetic regular-fractionation dose grid."""
    if n_fractions <= 0:
        raise ValueError("n_fractions must be positive.")
    if schedule_interval_days < 0.0:
        raise ValueError("schedule_interval_days must be non-negative.")

    rows = []
    for total_dose in dose_range:
        dose_total = float(total_dose)
        if dose_total <= 0.0:
            raise ValueError("dose_range must contain only positive doses.")
        fraction_dose = dose_total / n_fractions
        fractions = tuple(float(fraction_dose) for _ in range(n_fractions))
        schedule_days = tuple(float(index) * schedule_interval_days for index in range(n_fractions))
        experiment = TumorExperiment(
            path=Path("synthetic_tcp.xlsx"),
            fractions=fractions,
            sf=1.0,
            family=family or fit_result.family,
            sf_time_day=(schedule_days[-1] + 1.0) if schedule_days else 1.0,
            schedule_days=schedule_days,
            has_explicit_timing=n_fractions > 1,
            initial_volume_mm3=float(initial_volume_cm3) * 1000.0,
        )
        rows.append(
            compute_tcp(
                fit_result,
                experiment,
                initial_volume_cm3=initial_volume_cm3,
                cell_density=cell_density,
            )
        )
    return tuple(rows)


def prediction_rmse(
    simulation_result: GrowthSimulationResult,
    observed_days: Sequence[float],
    observed_volume: Sequence[float],
) -> float:
    """Interpolate predicted total volume onto observed days and compute RMSE."""
    days = np.asarray(observed_days, dtype=float)
    volume = np.asarray(observed_volume, dtype=float)
    valid_mask = np.isfinite(days) & np.isfinite(volume) & (volume >= 0.0)
    if int(np.sum(valid_mask)) == 0:
        raise ValueError("Need at least one valid observed point to compute RMSE.")

    days = days[valid_mask]
    volume = volume[valid_mask]
    predicted = np.interp(
        days,
        np.asarray(simulation_result.times, dtype=float),
        np.asarray(simulation_result.total_volume, dtype=float),
    )
    return float(np.sqrt(np.mean(np.square(predicted - volume))))


def _vary_parameters(
    parameters: GrowthModelParameters,
    parameter: str,
    factor: float,
) -> GrowthModelParameters:
    if parameter == "alpha":
        return replace(parameters, alpha=parameters.alpha * factor)
    if parameter == "beta":
        return replace(parameters, beta=parameters.beta * factor)
    if parameter == "growth_rate":
        return replace(parameters, growth_rate=parameters.growth_rate * factor)
    if parameter == "carrying_capacity":
        return replace(parameters, carrying_capacity=parameters.carrying_capacity * factor)
    if parameter == "clearance_rate":
        return replace(parameters, clearance_rate=parameters.clearance_rate * factor)
    raise KeyError(f"Unsupported parameter '{parameter}'.")


def _scaled_schedule(schedule: Sequence[TreatmentFraction], factor: float) -> list[TreatmentFraction]:
    return [
        TreatmentFraction(day=item.day, dose=item.dose * factor, family=item.family)
        for item in schedule
    ]


def analyze_parameter_sensitivity(
    observed_days: Sequence[float],
    observed_volume: Sequence[float],
    reference: GeometryReference,
    parameters: GrowthModelParameters,
    schedule: Sequence[TreatmentFraction],
    *,
    scaling_model: GeometryScalingModel | None = None,
    family_parameters: Mapping[str, GrowthModelParameters] | None = None,
    perturbation_fractions: Sequence[float] = (0.10, 0.20, 0.30),
    vary: Sequence[str] = ("alpha", "beta", "growth_rate", "carrying_capacity", "dose"),
) -> ParameterSensitivityReport:
    """One-at-a-time sensitivity report using total-volume RMSE."""
    baseline = simulate_growth(
        sample_times=observed_days,
        parameters=parameters,
        reference=reference,
        schedule=schedule,
        scaling_model=scaling_model,
        family_parameters=family_parameters,
    )
    baseline_rmse = prediction_rmse(baseline, observed_days, observed_volume)
    rows: list[ParameterSensitivityRow] = []

    for parameter in vary:
        if parameter == "dose":
            baseline_value = 1.0
        else:
            baseline_value = float(getattr(parameters, parameter))

        for perturbation in perturbation_fractions:
            for sign in (-1.0, 1.0):
                factor = 1.0 + sign * float(perturbation)
                if factor <= 0.0:
                    continue
                try:
                    if parameter == "dose":
                        varied_schedule = _scaled_schedule(schedule, factor)
                        varied_parameters = parameters
                        varied_value = baseline_value * factor
                    else:
                        varied_parameters = _vary_parameters(parameters, parameter, factor)
                        if (
                            parameter == "carrying_capacity"
                            and varied_parameters.carrying_capacity <= reference.volume
                        ):
                            raise ValueError("carrying_capacity must exceed reference volume.")
                        varied_schedule = list(schedule)
                        varied_value = baseline_value * factor

                    simulated = simulate_growth(
                        sample_times=observed_days,
                        parameters=varied_parameters,
                        reference=reference,
                        schedule=varied_schedule,
                        scaling_model=scaling_model,
                        family_parameters=family_parameters,
                    )
                    rmse = prediction_rmse(simulated, observed_days, observed_volume)
                    rows.append(
                        ParameterSensitivityRow(
                            parameter=parameter,
                            perturbation_fraction=sign * float(perturbation),
                            baseline_value=baseline_value,
                            varied_value=varied_value,
                            rmse=rmse,
                            delta_rmse=rmse - baseline_rmse,
                            rmse_ratio=(rmse / baseline_rmse) if baseline_rmse > 0.0 else None,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        ParameterSensitivityRow(
                            parameter=parameter,
                            perturbation_fraction=sign * float(perturbation),
                            baseline_value=baseline_value,
                            varied_value=baseline_value * factor,
                            rmse=None,
                            delta_rmse=None,
                            rmse_ratio=None,
                            status="failed",
                            reason=str(exc),
                        )
                    )

    influence_rows: list[ParameterInfluence] = []
    for parameter in vary:
        parameter_rows = [
            row for row in rows if row.parameter == parameter and row.delta_rmse is not None
        ]
        if not parameter_rows:
            continue
        deltas = np.asarray([abs(row.delta_rmse) for row in parameter_rows], dtype=float)
        influence_rows.append(
            ParameterInfluence(
                parameter=parameter,
                max_abs_delta_rmse=float(np.max(deltas)),
                mean_abs_delta_rmse=float(np.mean(deltas)),
                tested_cases=len(parameter_rows),
            )
        )
    influence_rows.sort(key=lambda row: row.max_abs_delta_rmse, reverse=True)

    return ParameterSensitivityReport(
        baseline_rmse=baseline_rmse,
        rows=tuple(rows),
        influence=tuple(influence_rows),
    )


def analyze_interval_sensitivity(
    observed_days: Sequence[float],
    observed_volume: Sequence[float],
    reference: GeometryReference,
    parameters: GrowthModelParameters,
    fractions: Sequence[float],
    interval_hours: Sequence[float],
    *,
    scaling_model: GeometryScalingModel | None = None,
    start_day: float = 0.0,
    baseline_schedule: Optional[Sequence[TreatmentFraction]] = None,
    schedule_template: Optional[Sequence[TreatmentFraction]] = None,
    family: Optional[str] = None,
    family_parameters: Mapping[str, GrowthModelParameters] | None = None,
) -> IntervalSensitivityReport:
    """Compare prediction RMSE for several candidate inter-fraction intervals."""
    if len(fractions) < 2:
        raise ValueError("Interval sensitivity requires at least two fractions.")

    template_families: Optional[Tuple[Optional[str], ...]] = None
    if schedule_template is not None:
        template_schedule = tuple(schedule_template)
        if len(template_schedule) != len(fractions):
            raise ValueError("schedule_template must match the number of fractions.")
        template_families = tuple(item.family for item in template_schedule)
        start_day = float(template_schedule[0].day) if template_schedule else float(start_day)

    if baseline_schedule is None:
        baseline_schedule = build_schedule_from_intervals(fractions, start_day=start_day)
    baseline_simulation = simulate_growth(
        sample_times=observed_days,
        parameters=parameters,
        reference=reference,
        schedule=baseline_schedule,
        scaling_model=scaling_model,
        family_parameters=family_parameters,
    )
    baseline_rmse = prediction_rmse(baseline_simulation, observed_days, observed_volume)

    rows: list[IntervalSensitivityRow] = []
    for interval in interval_hours:
        candidate_schedule = build_schedule_from_intervals(
            fractions,
            [float(interval) / 24.0],
            start_day=start_day,
        )
        if template_families is not None:
            candidate_schedule = [
                TreatmentFraction(
                    day=item.day,
                    dose=item.dose,
                    family=template_families[index],
                )
                for index, item in enumerate(candidate_schedule)
            ]
        elif family is not None:
            candidate_schedule = [
                TreatmentFraction(day=item.day, dose=item.dose, family=family)
                for item in candidate_schedule
            ]
        simulated = simulate_growth(
            sample_times=observed_days,
            parameters=parameters,
            reference=reference,
            schedule=candidate_schedule,
            scaling_model=scaling_model,
            family_parameters=family_parameters,
        )
        rmse = prediction_rmse(simulated, observed_days, observed_volume)
        rows.append(
            IntervalSensitivityRow(
                interval_hours=float(interval),
                rmse=rmse,
                delta_rmse=rmse - baseline_rmse,
                rmse_ratio=(rmse / baseline_rmse) if baseline_rmse > 0.0 else None,
                schedule_days=tuple(item.day for item in candidate_schedule),
            )
        )
    rows.sort(key=lambda row: row.rmse)
    return IntervalSensitivityReport(
        baseline_rmse=baseline_rmse,
        rows=tuple(rows),
    )


def _safe_percent_delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    if abs(baseline) < 1.0e-12:
        return None
    return 100.0 * (value - baseline) / baseline


def compare_sf_metric_sensitivity(
    runs: Sequence[AnalysisRunResult],
    *,
    baseline_sf_mode: str = "absolute",
) -> Tuple[SFMetricComparisonRow, ...]:
    """Compare parameter drift between scalar SF modes for the same family/model setup."""
    grouped: dict[tuple[str, str, str], list[AnalysisRunResult]] = {}
    for run in runs:
        if run.fit_result is None:
            continue
        grouped.setdefault(
            (
                run.summary.family_label,
                run.summary.response_mode,
                run.summary.model_kind,
            ),
            [],
        ).append(run)

    rows: list[SFMetricComparisonRow] = []
    for (family_label, response_mode, model_kind), group_runs in grouped.items():
        baseline_run = next(
            (run for run in group_runs if run.summary.sf_mode == baseline_sf_mode),
            group_runs[0],
        )
        baseline_fit = baseline_run.fit_result
        assert baseline_fit is not None
        for run in sorted(group_runs, key=lambda item: item.summary.sf_mode):
            fit = run.fit_result
            assert fit is not None
            rows.append(
                SFMetricComparisonRow(
                    family=family_label,
                    response_mode=response_mode,
                    model_kind=model_kind,
                    sf_mode=run.summary.sf_mode,
                    alpha=fit.alpha,
                    beta=fit.beta,
                    alpha_beta_ratio=fit.alpha_beta_ratio,
                    delta_alpha_pct=_safe_percent_delta(fit.alpha, baseline_fit.alpha),
                    delta_beta_pct=_safe_percent_delta(fit.beta, baseline_fit.beta),
                    delta_ratio_pct=_safe_percent_delta(
                        fit.alpha_beta_ratio,
                        baseline_fit.alpha_beta_ratio,
                    ),
                    baseline_sf_mode=baseline_run.summary.sf_mode,
                )
            )
    return tuple(rows)


def compare_treatment_scenarios(
    sample_times: Sequence[float],
    reference: GeometryReference,
    parameters: GrowthModelParameters,
    scenarios: Mapping[str, Sequence[TreatmentFraction]],
    *,
    scaling_model: GeometryScalingModel | None = None,
    family_parameters: Mapping[str, GrowthModelParameters] | None = None,
) -> ScenarioComparisonReport:
    """Simulate several schedules and summarize their predicted tumor response."""
    rows: list[ScenarioComparisonRow] = []
    for scenario_name, schedule in scenarios.items():
        simulation = simulate_growth(
            sample_times=sample_times,
            parameters=parameters,
            reference=reference,
            schedule=schedule,
            scaling_model=scaling_model,
            family_parameters=family_parameters,
        )
        total_volume = np.asarray(simulation.total_volume, dtype=float)
        times = np.asarray(simulation.times, dtype=float)
        nadir_index = int(np.argmin(total_volume))
        rows.append(
            ScenarioComparisonRow(
                scenario=str(scenario_name),
                total_physical_dose=float(sum(item.dose for item in schedule)),
                family_sequence=tuple((item.family or "default") for item in schedule),
                min_total_volume=float(np.min(total_volume)),
                min_total_volume_day=float(times[nadir_index]),
                final_total_volume=float(total_volume[-1]),
                auc_total_volume=float(np.trapz(total_volume, times)),
            )
        )

    rows.sort(key=lambda row: (row.final_total_volume, row.auc_total_volume, row.min_total_volume))
    return ScenarioComparisonReport(rows=tuple(rows))
