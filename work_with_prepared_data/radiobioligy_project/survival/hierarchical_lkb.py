"""Regularised hierarchical LKB model for clustered binary skin outcomes.

The implementation is deliberately separate from the legacy family-by-family
MLE.  It fits one common LKB ``m`` parameter, partially pooled family-specific
``TD50`` values, and a calendar-series random intercept.  Calendar effects are
integrated with Gauss-Hermite quadrature.  Weakly informative priors provide an
empirical-Bayes/MAP regularisation that keeps all-zero and all-one families
finite without claiming that they contain an independently estimable slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr


@dataclass(frozen=True)
class HierarchicalLKBPriors:
    """Weakly informative priors used by the regularised MAP fit."""

    log_td50_center: float
    log_td50_sd: float = 1.25
    log_m_center: float = float(np.log(0.25))
    log_m_sd: float = 0.90
    family_log_td50_sd_scale: float = 0.75
    calendar_sd_scale: float = 1.00


@dataclass(frozen=True)
class HierarchicalLKBResult:
    """Fitted hierarchical LKB parameters and audit metadata."""

    families: tuple[str, ...]
    blocks: tuple[str, ...]
    log_td50_mean: float
    common_m: float
    family_sd_log_td50: float
    calendar_sd: float
    family_td50: tuple[float, ...]
    family_z: tuple[float, ...]
    negative_log_likelihood: float
    negative_log_posterior: float
    converged: bool
    message: str
    iterations: int
    parameter_vector: tuple[float, ...]

    def td50_for(self, family: str) -> float:
        return float(self.family_td50[self.families.index(str(family))])


@dataclass(frozen=True)
class HierarchicalLKBData:
    """Numerical representation of individual outcomes grouped by calendar."""

    frame: pd.DataFrame
    families: tuple[str, ...]
    blocks: tuple[str, ...]
    doses: np.ndarray
    outcomes: np.ndarray
    family_codes: np.ndarray
    block_indices: tuple[np.ndarray, ...]


REQUIRED_COLUMNS = {
    "family",
    "calendar_series",
    "physical_total_dose",
    "complication",
}


def prepare_hierarchical_lkb_data(
    frame: pd.DataFrame,
    *,
    family_levels: Sequence[str] | None = None,
) -> HierarchicalLKBData:
    """Validate and encode one row per animal for the hierarchical fit."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing hierarchical LKB columns: {sorted(missing)}")
    clean = frame.copy()
    clean["family"] = clean["family"].astype(str)
    clean["calendar_series"] = clean["calendar_series"].astype(str)
    clean["physical_total_dose"] = pd.to_numeric(
        clean["physical_total_dose"], errors="coerce"
    )
    clean["complication"] = pd.to_numeric(clean["complication"], errors="coerce")
    clean = clean.loc[
        clean["physical_total_dose"].notna()
        & clean["complication"].isin([0, 1])
        & (clean["physical_total_dose"] > 0)
    ].reset_index(drop=True)
    if clean.empty:
        raise ValueError("No usable individual LKB observations.")

    if family_levels is None:
        families = tuple(sorted(clean["family"].unique()))
    else:
        families = tuple(str(value) for value in family_levels)
        unknown = sorted(set(clean["family"]) - set(families))
        if unknown:
            raise ValueError(f"Unknown families outside fixed levels: {unknown}")
    family_lookup = {family: index for index, family in enumerate(families)}
    family_codes = clean["family"].map(family_lookup).to_numpy(dtype=int)

    blocks = tuple(sorted(clean["calendar_series"].unique()))
    block_indices = tuple(
        np.flatnonzero(clean["calendar_series"].eq(block).to_numpy()) for block in blocks
    )
    return HierarchicalLKBData(
        frame=clean,
        families=families,
        blocks=blocks,
        doses=clean["physical_total_dose"].to_numpy(dtype=float),
        outcomes=clean["complication"].to_numpy(dtype=float),
        family_codes=family_codes,
        block_indices=block_indices,
    )


class HierarchicalLKBModel:
    """Regularised hierarchical probit/LKB model with calendar clustering."""

    def __init__(
        self,
        *,
        quadrature_points: int = 15,
        priors: HierarchicalLKBPriors | None = None,
    ) -> None:
        if quadrature_points < 7:
            raise ValueError("Use at least seven Gauss-Hermite points.")
        nodes, weights = hermgauss(int(quadrature_points))
        self.random_nodes = np.sqrt(2.0) * nodes
        self.log_random_weights = np.log(weights) - 0.5 * np.log(np.pi)
        self.priors = priors

    @staticmethod
    def default_priors(frame: pd.DataFrame) -> HierarchicalLKBPriors:
        doses = pd.to_numeric(frame["physical_total_dose"], errors="coerce")
        doses = doses[np.isfinite(doses) & (doses > 0)]
        if not len(doses):
            raise ValueError("Cannot construct priors without positive doses.")
        return HierarchicalLKBPriors(log_td50_center=float(np.log(np.median(doses))))

    @staticmethod
    def _unpack(
        parameters: np.ndarray,
        n_families: int,
    ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        mu = float(parameters[0])
        common_m = float(np.exp(parameters[1]))
        family_sd = float(parameters[2])
        calendar_sd = float(parameters[3])
        family_z_raw = np.asarray(parameters[4 : 4 + n_families], dtype=float)
        family_z = family_z_raw - float(np.mean(family_z_raw))
        family_log_td50 = mu + family_sd * family_z
        return mu, common_m, family_sd, calendar_sd, family_z, family_log_td50

    def _cluster_log_likelihood(
        self,
        data: HierarchicalLKBData,
        parameters: np.ndarray,
    ) -> float:
        _, common_m, _, calendar_sd, _, family_log_td50 = self._unpack(
            parameters, len(data.families)
        )
        td50 = np.exp(family_log_td50[data.family_codes])
        fixed_eta = (data.doses - td50) / (common_m * td50)
        total = 0.0
        for indices in data.block_indices:
            eta = (
                fixed_eta[indices, np.newaxis]
                + calendar_sd * self.random_nodes[np.newaxis, :]
            )
            outcomes = data.outcomes[indices, np.newaxis]
            conditional = np.sum(
                outcomes * log_ndtr(eta) + (1.0 - outcomes) * log_ndtr(-eta),
                axis=0,
            )
            total += float(logsumexp(self.log_random_weights + conditional))
        return total

    def _negative_log_posterior(
        self,
        data: HierarchicalLKBData,
        parameters: np.ndarray,
        priors: HierarchicalLKBPriors,
    ) -> float:
        mu, _, family_sd, calendar_sd, _, _ = self._unpack(
            parameters, len(data.families)
        )
        log_likelihood = self._cluster_log_likelihood(data, parameters)
        family_z_raw = parameters[4 : 4 + len(data.families)]
        penalty = 0.5 * ((mu - priors.log_td50_center) / priors.log_td50_sd) ** 2
        penalty += 0.5 * (
            (parameters[1] - priors.log_m_center) / priors.log_m_sd
        ) ** 2
        penalty += 0.5 * np.sum(np.square(family_z_raw))
        penalty += 0.5 * (family_sd / priors.family_log_td50_sd_scale) ** 2
        penalty += 0.5 * (calendar_sd / priors.calendar_sd_scale) ** 2
        return float(-log_likelihood + penalty)

    def _negative_log_posterior_and_gradient(
        self,
        data: HierarchicalLKBData,
        parameters: np.ndarray,
        priors: HierarchicalLKBPriors,
    ) -> tuple[float, np.ndarray]:
        """Return the penalised objective and its analytic gradient."""

        n_families = len(data.families)
        mu, common_m, family_sd, calendar_sd, family_z, family_log_td50 = self._unpack(
            parameters, n_families
        )
        td50 = np.exp(family_log_td50[data.family_codes])
        dose_ratio = data.doses / td50
        fixed_eta = (dose_ratio - 1.0) / common_m
        derivative_theta = -dose_ratio / common_m

        log_likelihood = 0.0
        theta_score = np.zeros(n_families, dtype=float)
        log_m_score = 0.0
        calendar_sd_score = 0.0
        log_sqrt_two_pi = 0.5 * np.log(2.0 * np.pi)
        for indices in data.block_indices:
            eta = (
                fixed_eta[indices, np.newaxis]
                + calendar_sd * self.random_nodes[np.newaxis, :]
            )
            outcomes = data.outcomes[indices, np.newaxis]
            log_pdf = -0.5 * np.square(eta) - log_sqrt_two_pi
            event_score = np.exp(np.clip(log_pdf - log_ndtr(eta), -50.0, 50.0))
            nonevent_score = -np.exp(
                np.clip(log_pdf - log_ndtr(-eta), -50.0, 50.0)
            )
            score_eta = outcomes * event_score + (1.0 - outcomes) * nonevent_score
            conditional = np.sum(
                outcomes * log_ndtr(eta) + (1.0 - outcomes) * log_ndtr(-eta),
                axis=0,
            )
            log_terms = self.log_random_weights + conditional
            cluster_log_likelihood = float(logsumexp(log_terms))
            posterior_weights = np.exp(log_terms - cluster_log_likelihood)
            log_likelihood += cluster_log_likelihood

            integrated_score = score_eta @ posterior_weights
            family_contribution = integrated_score * derivative_theta[indices]
            np.add.at(
                theta_score,
                data.family_codes[indices],
                family_contribution,
            )
            log_m_score += float(
                np.sum(integrated_score * (-fixed_eta[indices]))
            )
            calendar_sd_score += float(
                np.sum(
                    posterior_weights
                    * np.sum(
                        score_eta * self.random_nodes[np.newaxis, :],
                        axis=0,
                    )
                )
            )

        family_z_raw = np.asarray(parameters[4 : 4 + n_families], dtype=float)
        penalty = 0.5 * ((mu - priors.log_td50_center) / priors.log_td50_sd) ** 2
        penalty += 0.5 * (
            (parameters[1] - priors.log_m_center) / priors.log_m_sd
        ) ** 2
        penalty += 0.5 * np.sum(np.square(family_z_raw))
        penalty += 0.5 * (family_sd / priors.family_log_td50_sd_scale) ** 2
        penalty += 0.5 * (calendar_sd / priors.calendar_sd_scale) ** 2

        likelihood_gradient = np.zeros_like(parameters, dtype=float)
        likelihood_gradient[0] = float(np.sum(theta_score))
        likelihood_gradient[1] = log_m_score
        likelihood_gradient[2] = float(np.dot(theta_score, family_z))
        likelihood_gradient[3] = calendar_sd_score
        likelihood_gradient[4:] = family_sd * (
            theta_score - float(np.mean(theta_score))
        )

        penalty_gradient = np.zeros_like(parameters, dtype=float)
        penalty_gradient[0] = (
            mu - priors.log_td50_center
        ) / priors.log_td50_sd**2
        penalty_gradient[1] = (
            parameters[1] - priors.log_m_center
        ) / priors.log_m_sd**2
        penalty_gradient[2] = family_sd / priors.family_log_td50_sd_scale**2
        penalty_gradient[3] = calendar_sd / priors.calendar_sd_scale**2
        penalty_gradient[4:] = family_z_raw
        objective = float(-log_likelihood + penalty)
        gradient = -likelihood_gradient + penalty_gradient
        return objective, gradient

    def _initial_parameters(
        self,
        data: HierarchicalLKBData,
        priors: HierarchicalLKBPriors,
    ) -> np.ndarray:
        family_sd = 0.30
        common_m = float(np.exp(priors.log_m_center))
        family_log_td50: list[float] = []
        for family in data.families:
            subset = data.frame.loc[data.frame["family"].eq(family)]
            dose = float(np.median(subset["physical_total_dose"]))
            events = float(subset["complication"].sum())
            total = float(len(subset))
            rate = (events + 0.5) / (total + 1.0)
            latent = float(np.clip(ndtri_safe(rate), -3.0, 3.0))
            denominator = max(0.20, 1.0 + common_m * latent)
            family_log_td50.append(float(np.log(dose / denominator)))
        family_log_td50_array = np.asarray(family_log_td50)
        mu = float(np.mean(family_log_td50_array))
        z = (family_log_td50_array - mu) / family_sd
        z -= float(np.mean(z))
        return np.concatenate(
            [
                np.asarray([mu, np.log(common_m), family_sd, 0.30]),
                np.clip(z, -3.0, 3.0),
            ]
        )

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        family_levels: Sequence[str] | None = None,
        initial_parameters: Sequence[float] | None = None,
        priors: HierarchicalLKBPriors | None = None,
        maxiter: int = 1200,
        retry: bool = True,
    ) -> HierarchicalLKBResult:
        """Fit the regularised model and integrate calendar random effects."""

        data = prepare_hierarchical_lkb_data(frame, family_levels=family_levels)
        resolved_priors = priors or self.priors or self.default_priors(data.frame)
        n_families = len(data.families)
        if initial_parameters is None:
            initial = self._initial_parameters(data, resolved_priors)
        else:
            initial = np.asarray(initial_parameters, dtype=float)
            if len(initial) != 4 + n_families:
                raise ValueError("Initial parameter vector has the wrong length.")
        bounds = [
            (np.log(1.0), np.log(250.0)),
            (np.log(0.01), np.log(5.0)),
            (0.0, 1.75),
            (0.0, 3.5),
            *[(-5.0, 5.0) for _ in range(n_families)],
        ]

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            return self._negative_log_posterior_and_gradient(
                data, parameters, resolved_priors
            )

        starts = [initial]
        if retry:
            conservative = initial.copy()
            conservative[1] = np.log(0.30)
            conservative[2] = 0.15
            conservative[3] = 0.10
            starts.append(conservative)
        results = [
            minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": int(maxiter), "ftol": 1e-10, "gtol": 1e-6},
            )
            for start in starts
        ]
        result = min(results, key=lambda item: float(item.fun))
        mu, common_m, family_sd, calendar_sd, family_z, family_log_td50 = self._unpack(
            result.x, n_families
        )
        log_likelihood = self._cluster_log_likelihood(data, result.x)
        return HierarchicalLKBResult(
            families=data.families,
            blocks=data.blocks,
            log_td50_mean=mu,
            common_m=common_m,
            family_sd_log_td50=family_sd,
            calendar_sd=calendar_sd,
            family_td50=tuple(float(value) for value in np.exp(family_log_td50)),
            family_z=tuple(float(value) for value in family_z),
            negative_log_likelihood=float(-log_likelihood),
            negative_log_posterior=float(result.fun),
            converged=bool(result.success and np.isfinite(result.fun)),
            message=str(result.message),
            iterations=int(getattr(result, "nit", 0)),
            parameter_vector=tuple(float(value) for value in result.x),
        )

    @staticmethod
    def predict_population_probability(
        result: HierarchicalLKBResult,
        family: str,
        doses: Iterable[float] | float,
    ) -> np.ndarray:
        """Population-average probability after integrating the calendar effect."""

        values = np.atleast_1d(np.asarray(doses, dtype=float))
        td50 = result.td50_for(str(family))
        fixed_eta = (values - td50) / (result.common_m * td50)
        marginal_eta = fixed_eta / np.sqrt(1.0 + result.calendar_sd**2)
        return np.asarray(ndtr(marginal_eta), dtype=float)


def ndtri_safe(probability: float) -> float:
    """Numerically stable inverse standard-normal CDF for initialisation."""

    from scipy.special import ndtri

    return float(ndtri(np.clip(float(probability), 1e-6, 1.0 - 1e-6)))


def bootstrap_calendar_blocks(
    frame: pd.DataFrame,
    *,
    model: HierarchicalLKBModel,
    point_result: HierarchicalLKBResult,
    priors: HierarchicalLKBPriors,
    iterations: int,
    seed: int,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit after resampling entire calendar series with replacement."""

    rng = np.random.default_rng(int(seed))
    blocks = tuple(sorted(frame["calendar_series"].astype(str).unique()))
    families = point_result.families
    dose_ranges = (
        frame.groupby("family")["physical_total_dose"].agg(["min", "max"]).to_dict("index")
    )
    common_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    initial = point_result.parameter_vector
    for replicate in range(1, int(iterations) + 1):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        chunks: list[pd.DataFrame] = []
        present_families: set[str] = set()
        for draw_index, block in enumerate(sampled_blocks):
            chunk = frame.loc[frame["calendar_series"].astype(str).eq(str(block))].copy()
            chunk["calendar_series"] = f"{block}__bootstrap_{draw_index:03d}"
            present_families.update(chunk["family"].astype(str).unique())
            chunks.append(chunk)
        sampled = pd.concat(chunks, ignore_index=True)
        try:
            fitted = model.fit(
                sampled,
                family_levels=families,
                initial_parameters=initial,
                priors=priors,
                maxiter=700,
                retry=False,
            )
        except Exception as exc:  # pragma: no cover - audit output
            common_rows.append(
                {
                    "replicate": replicate,
                    "success": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        success = bool(fitted.converged)
        common_rows.append(
            {
                "replicate": replicate,
                "success": success,
                "common_m": fitted.common_m,
                "geometric_mean_td50_gy": float(np.exp(fitted.log_td50_mean)),
                "family_sd_log_td50": fitted.family_sd_log_td50,
                "calendar_sd": fitted.calendar_sd,
                "negative_log_likelihood": fitted.negative_log_likelihood,
                "message": fitted.message,
            }
        )
        if not success:
            continue
        for family in families:
            limits = dose_ranges[family]
            probabilities = model.predict_population_probability(
                fitted,
                family,
                [limits["min"], limits["max"]],
            )
            family_rows.append(
                {
                    "replicate": replicate,
                    "family": family,
                    "family_present": family in present_families,
                    "td50": fitted.td50_for(family),
                    "probability_min_observed_dose": float(probabilities[0]),
                    "probability_max_observed_dose": float(probabilities[1]),
                }
            )
        if progress_every and (
            replicate % int(progress_every) == 0 or replicate == int(iterations)
        ):
            completed = sum(bool(row.get("success", False)) for row in common_rows)
            print(
                f"Bootstrap: {replicate}/{iterations}; successful fits: {completed}",
                flush=True,
            )
    return pd.DataFrame(common_rows), pd.DataFrame(family_rows)


def leave_one_calendar_out(
    frame: pd.DataFrame,
    *,
    model: HierarchicalLKBModel,
    point_result: HierarchicalLKBResult,
    priors: HierarchicalLKBPriors,
) -> pd.DataFrame:
    """Calendar-block LOSO predictions versus a smoothed family constant."""

    output: list[dict[str, object]] = []
    blocks = tuple(sorted(frame["calendar_series"].astype(str).unique()))
    for block in blocks:
        held_out = frame.loc[frame["calendar_series"].astype(str).eq(block)].copy()
        training = frame.loc[~frame["calendar_series"].astype(str).eq(block)].copy()
        fitted = model.fit(
            training,
            family_levels=point_result.families,
            initial_parameters=point_result.parameter_vector,
            priors=priors,
            maxiter=900,
            retry=False,
        )
        global_events = float(training["complication"].sum())
        global_n = float(len(training))
        global_rate = (global_events + 0.5) / (global_n + 1.0)
        family_counts = training.groupby("family")["complication"].agg(["sum", "count"])
        for row in held_out.itertuples(index=False):
            probability = float(
                model.predict_population_probability(
                    fitted,
                    str(row.family),
                    [float(row.physical_total_dose)],
                )[0]
            )
            if str(row.family) in family_counts.index:
                counts = family_counts.loc[str(row.family)]
                baseline = (float(counts["sum"]) + 0.5) / (float(counts["count"]) + 1.0)
                family_seen = True
            else:
                baseline = global_rate
                family_seen = False
            outcome = int(row.complication)
            clipped_model = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
            clipped_baseline = float(np.clip(baseline, 1e-9, 1.0 - 1e-9))
            output.append(
                {
                    "held_out_calendar_series": block,
                    "family": str(row.family),
                    "dose_gy": float(row.physical_total_dose),
                    "outcome": outcome,
                    "family_seen_in_training": family_seen,
                    "hierarchical_probability": clipped_model,
                    "family_constant_probability": clipped_baseline,
                    "hierarchical_logloss": -(
                        outcome * np.log(clipped_model)
                        + (1 - outcome) * np.log(1.0 - clipped_model)
                    ),
                    "family_constant_logloss": -(
                        outcome * np.log(clipped_baseline)
                        + (1 - outcome) * np.log(1.0 - clipped_baseline)
                    ),
                    "hierarchical_brier": (outcome - clipped_model) ** 2,
                    "family_constant_brier": (outcome - clipped_baseline) ** 2,
                }
            )
    return pd.DataFrame(output)
