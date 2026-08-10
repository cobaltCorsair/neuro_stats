"""Training primitives matching the frozen dissertation calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
HUBER_C = 1.5
ROBUST_ITERATIONS = 3
INNER_SPLITS = 4


@dataclass
class RobustRidgeFit:
    scaler: StandardScaler
    model: Ridge

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(matrix)), dtype=float)


def fit_robust_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    robust_iterations: int = ROBUST_ITERATIONS,
    huber_c: float = HUBER_C,
) -> RobustRidgeFit:
    scaler = StandardScaler(with_mean=False)
    scaled = scaler.fit_transform(np.asarray(matrix, dtype=float))
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    current_weights = weights.copy()
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    for _ in range(int(robust_iterations)):
        model.fit(scaled, target, sample_weight=current_weights)
        residual = target - model.predict(scaled)
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        if not np.isfinite(scale) or scale <= 1.0e-8:
            break
        threshold = float(huber_c) * scale
        robust_weights = np.ones_like(residual)
        large = np.abs(residual) > threshold
        robust_weights[large] = threshold / np.abs(residual[large])
        current_weights = weights * robust_weights
    return RobustRidgeFit(scaler=scaler, model=model)


def tune_alpha_by_group(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    groups: Sequence[str],
    *,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two calendar groups are required")
    splitter = GroupKFold(n_splits=min(INNER_SPLITS, len(unique)))
    folds = list(splitter.split(np.zeros(len(groups)), groups=groups))
    best_alpha = float(alpha_grid[0])
    best_error = float("inf")
    best_oof = np.full(len(target), np.nan)
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        oof = np.full(len(target), np.nan)
        for train_index, valid_index in folds:
            fitted = fit_robust_ridge(
                matrix[train_index], target[train_index], weights[train_index], float(alpha)
            )
            oof[valid_index] = fitted.predict(matrix[valid_index])
        error = float(np.sqrt(np.average(np.square(target - oof), weights=weights)))
        rows.append({"alpha": float(alpha), "weighted_log_rmse": error})
        if error < best_error - 1.0e-12:
            best_alpha, best_error, best_oof = float(alpha), error, oof.copy()
    return best_alpha, best_oof, rows
