# coding: utf-8
"""Secondary radiobiological analyses built on top of fitter and predictor backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
    AnalysisRunResult,
    LQFitResult,
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
