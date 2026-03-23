# coding: utf-8
"""LET-dependent alpha/beta parametrization built from per-family fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
        LQFitResult,
    )


@dataclass(frozen=True)
class LETDependentParams:
    """Linear LET -> alpha/beta model for one tissue class."""

    alpha_0: float
    lambda_alpha: float
    beta_0: float
    lambda_beta: float
    let_max: Optional[float] = None
    alpha_r_squared: Optional[float] = None
    beta_r_squared: Optional[float] = None
    family_order: Tuple[str, ...] = ()

    def alpha(self, let_kev_um: float) -> float:
        """Evaluate alpha(LET) with optional saturation at ``let_max``."""
        let_value = float(let_kev_um)
        alpha = self.alpha_0 + self.lambda_alpha * let_value
        if self.let_max is not None:
            alpha_max = self.alpha_0 + self.lambda_alpha * float(self.let_max)
            alpha = min(alpha, alpha_max)
        return float(alpha)

    def beta(self, let_kev_um: float) -> float:
        """Evaluate beta(LET)."""
        return float(self.beta_0 + self.lambda_beta * float(let_kev_um))

    def alpha_beta_ratio(self, let_kev_um: float) -> float:
        """Evaluate alpha/beta at the requested LET."""
        beta_value = self.beta(let_kev_um)
        if beta_value <= 0.0:
            return float("inf")
        return float(self.alpha(let_kev_um) / beta_value)


def fit_let_dependence(
    family_results: Mapping[str, "LQFitResult"],
    mean_lets: Mapping[str, float],
    *,
    let_max: Optional[float] = None,
) -> LETDependentParams:
    """Fit alpha(LET) and beta(LET) from already-fitted family-specific LQ results."""
    rows: list[tuple[str, float, float, float, float]] = []
    for raw_family, result in family_results.items():
        family = _normalize_family(raw_family)
        if family is None or family not in mean_lets:
            continue
        let_value = float(mean_lets[family])
        if not np.isfinite(let_value):
            continue
        alpha_value = _extract_alpha(result, let_value, family)
        beta_value = float(result.beta)
        if not np.isfinite(alpha_value) or not np.isfinite(beta_value):
            continue
        train_count = max(int(getattr(result, "train_count", 0)), 1)
        rows.append((family, let_value, alpha_value, beta_value, float(train_count)))

    if len(rows) < 3:
        raise ValueError("LET parametrization needs at least three family-specific fit results.")

    let_levels = {round(row[1], 8) for row in rows}
    if len(let_levels) < 2:
        raise ValueError("LET parametrization needs at least two distinct LET levels.")

    rows.sort(key=lambda row: (row[1], row[0]))
    let_values = np.asarray([row[1] for row in rows], dtype=float)
    alpha_values = np.asarray([row[2] for row in rows], dtype=float)
    beta_values = np.asarray([row[3] for row in rows], dtype=float)
    weights = np.asarray([row[4] for row in rows], dtype=float)

    alpha_0, lambda_alpha, alpha_r_squared = _fit_weighted_linear(let_values, alpha_values, weights)
    beta_0, lambda_beta, beta_r_squared = _fit_weighted_linear(let_values, beta_values, weights)
    return LETDependentParams(
        alpha_0=alpha_0,
        lambda_alpha=lambda_alpha,
        beta_0=beta_0,
        lambda_beta=lambda_beta,
        let_max=let_max,
        alpha_r_squared=alpha_r_squared,
        beta_r_squared=beta_r_squared,
        family_order=tuple(row[0] for row in rows),
    )


def _extract_alpha(result: "LQFitResult", let_value: float, family: str) -> float:
    effective_alpha = getattr(result, "effective_alpha", None)
    if callable(effective_alpha):
        return float(effective_alpha(let_value, family=family))
    return float(getattr(result, "alpha"))


def _fit_weighted_linear(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    design = np.column_stack((np.ones_like(x_values, dtype=float), x_values))
    sqrt_weights = np.sqrt(np.clip(weights, 1.0e-8, None))
    weighted_design = design * sqrt_weights[:, None]
    weighted_response = y_values * sqrt_weights
    params, *_ = np.linalg.lstsq(weighted_design, weighted_response, rcond=None)
    intercept = float(params[0])
    slope = float(params[1])
    predicted = intercept + slope * x_values
    r_squared = _r_squared(y_values, predicted, weights)
    return intercept, slope, r_squared


def _r_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float("nan")
    weighted_mean = float(np.sum(weights * observed) / weight_sum)
    ss_res = float(np.sum(weights * np.square(observed - predicted)))
    ss_tot = float(np.sum(weights * np.square(observed - weighted_mean)))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 1.0e-12 else float("nan")
    return float(1.0 - ss_res / ss_tot)


def _normalize_family(family: Optional[str]) -> Optional[str]:
    if family is None:
        return None
    normalized = family.strip().lower()
    return normalized or None
