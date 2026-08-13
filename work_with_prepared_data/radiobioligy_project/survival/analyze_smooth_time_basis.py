"""Проверка гладкого временного базиса без принудительного излома на 12-е сутки.

Скрипт сравнивает окончательную 133-признаковую модель с двумя семействами
гладких кандидатов:

1. compact_K: C1-гладкий насыщаемый множитель и ранние beta-компоненты,
   плавно затухающие к K суткам;
2. tail_tau: асимптотический экспоненциальный множитель и gamma-компоненты,
   которые не обнуляются в заранее заданный день.

Форма временного базиса и ridge-штраф выбираются только во внутренних
групповых складках обучающей части. Сутки фактического минимума не входят в
целевую функцию выбора и используются как независимая диагностическая метрика.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EARLY_SCRIPT = HERE / "analyze_early_response_extension.py"
OUTPUT_DIR = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Гладкий_временной_базис"
)

BASELINE_RAMP_DAY = 12.0
WINDOW_DAY = 21.0
COMPACT_HORIZONS = (12.0, 14.0, 16.0, 18.0, 21.0)
TAIL_TAUS = (4.0, 6.0, 8.0, 10.0)
DECOUPLED_EARLY_HORIZONS = (14.0, 16.0, 18.0)
HYBRID_GAMMA_SHAPES = (4.0, 6.0, 8.0)
ROUNDING_HALFWIDTHS = (1.0, 2.0, 3.0)
EARLY_PEAKS = (2.0, 4.0, 6.0)
GAMMA_SHAPE = 4.0
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
LATE_GUARDRAIL = 1.02
SEED = 20260812
BOOTSTRAP_ITERATIONS = 5000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EARLY = load_module("smooth_time_early", EARLY_SCRIPT)
BASE = EARLY.BASE


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    parameter: float


CANDIDATES = (
    Candidate("fixed_12", "fixed", 12.0),
    *(
        Candidate(f"compact_{int(horizon)}", "compact", horizon)
        for horizon in COMPACT_HORIZONS
    ),
    *(
        Candidate(f"tail_tau{int(tau)}", "tail", tau)
        for tau in TAIL_TAUS
    ),
    *(
        Candidate(f"decoupled_K{int(horizon)}", "decoupled", horizon)
        for horizon in DECOUPLED_EARLY_HORIZONS
    ),
    *(
        Candidate(f"hybrid_gamma_q{int(shape)}", "hybrid_gamma", shape)
        for shape in HYBRID_GAMMA_SHAPES
    ),
    *(
        Candidate(f"rounded_delta{int(delta)}", "rounded", delta)
        for delta in ROUNDING_HALFWIDTHS
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep_ramp(day: np.ndarray, horizon: float) -> np.ndarray:
    """C1-гладкое насыщение от 0 до 1 без угла в точке horizon."""
    u = np.clip(np.asarray(day, dtype=float) / float(horizon), 0.0, 1.0)
    return 3.0 * u**2 - 2.0 * u**3


def exponential_ramp(day: np.ndarray, tau: float) -> np.ndarray:
    """Гладкое асимптотическое нарастание, нормированное к единице на 21-е сутки."""
    day = np.maximum(np.asarray(day, dtype=float), 0.0)
    denominator = 1.0 - math.exp(-WINDOW_DAY / float(tau))
    return (1.0 - np.exp(-day / float(tau))) / denominator


def rounded_linear_ramp(day: np.ndarray, halfwidth: float) -> np.ndarray:
    """Линейный g12 с C1-скруглением только в окрестности 12-х суток."""
    day = np.asarray(day, dtype=float)
    left = BASELINE_RAMP_DAY - float(halfwidth)
    right = BASELINE_RAMP_DAY + float(halfwidth)
    values = np.empty_like(day)
    before = day <= left
    after = day >= right
    middle = ~(before | after)
    values[before] = np.maximum(day[before], 0.0) / BASELINE_RAMP_DAY
    values[after] = 1.0
    u = (day[middle] - left) / (right - left)
    start = left / BASELINE_RAMP_DAY
    difference = 1.0 - start
    # Hermite connection: slope 1/12 at left, slope 0 at right.
    values[middle] = start + difference * (2.0 * u - u**2)
    values[day <= 0.0] = 0.0
    return values


def compact_beta_bump(day: np.ndarray, peak: float, horizon: float) -> np.ndarray:
    """C1-гладкий transient с заданным максимумом и нулём после horizon."""
    day = np.asarray(day, dtype=float)
    u = np.clip(day / float(horizon), 0.0, 1.0)
    a = 2.0
    b = a * (float(horizon) - float(peak)) / float(peak)
    if b < 2.0 - 1.0e-12:
        raise ValueError("horizon must be at least twice the latest peak")
    raw = np.power(u, a) * np.power(1.0 - u, b)
    peak_u = a / (a + b)
    normalizer = peak_u**a * (1.0 - peak_u) ** b
    values = raw / normalizer
    values[(day <= 0.0) | (day >= horizon)] = 0.0
    return values


def gamma_bump(
    day: np.ndarray, peak: float, shape: float = GAMMA_SHAPE
) -> np.ndarray:
    """Гладкий transient с максимумом peak и экспоненциальным хвостом."""
    day = np.asarray(day, dtype=float)
    ratio = np.maximum(day, 0.0) / float(peak)
    values = np.power(ratio, float(shape)) * np.exp(float(shape) * (1.0 - ratio))
    values[day <= 0.0] = 0.0
    return values


def spliced_beta_tail_bump(
    day: np.ndarray, a: float, b: float, splice: float
) -> np.ndarray:
    """Исходный beta-bump до splice и C1-гладкий экспоненциальный хвост после."""
    day = np.asarray(day, dtype=float)
    u = np.clip(day / BASELINE_RAMP_DAY, 0.0, 1.0)
    peak_u = a / (a + b)
    normalizer = peak_u**a * (1.0 - peak_u) ** b
    values = np.power(u, a) * np.power(1.0 - u, b) / normalizer
    splice_u = float(splice) / BASELINE_RAMP_DAY
    splice_value = splice_u**a * (1.0 - splice_u) ** b / normalizer
    log_slope = a / float(splice) - b / (BASELINE_RAMP_DAY - float(splice))
    tail = day > float(splice)
    values[tail] = splice_value * np.exp(log_slope * (day[tail] - float(splice)))
    values[day <= 0.0] = 0.0
    return values


def retime_g12_columns(
    matrix: np.ndarray,
    names: Sequence[str],
    day: np.ndarray,
    new_ramp: np.ndarray,
) -> np.ndarray:
    """Заменить только множитель g12 во всех 49 содержащих его столбцах."""
    old_ramp = np.minimum(np.asarray(day, dtype=float) / BASELINE_RAMP_DAY, 1.0)
    scale = np.divide(
        np.asarray(new_ramp, dtype=float),
        old_ramp,
        out=np.zeros_like(old_ramp),
        where=old_ramp > 0.0,
    )
    output = matrix.copy()
    for index, name in enumerate(names):
        if "g12" in name:
            output[:, index] *= scale
    return output


def add_early_family_features(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    names: list[str],
    bumps: np.ndarray,
    bump_names: Sequence[str],
    categories: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, list[str]]:
    treated = frame["is_treated"].to_numpy(float)
    log_dose = np.log1p(frame["total_dose_gy"].to_numpy(float))
    additions: list[np.ndarray] = []
    added_names: list[str] = []
    for column, name in zip(bumps.T, bump_names):
        additions.extend((treated * column, log_dose * column))
        added_names.extend((f"treated_{name}", f"logdose_{name}"))
    family_values = frame["family"].astype(str).to_numpy()
    for family in categories["families"]:
        indicator = (family_values == family).astype(float)
        for column, name in zip(bumps.T, bump_names):
            additions.append(indicator * column)
            added_names.append(f"family_{family}_{name}")
    return np.column_stack([matrix, *additions]), [*names, *added_names]


def build_candidate_features(
    frame: pd.DataFrame,
    candidate: Candidate,
    categories: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, list[str]]:
    if candidate.family == "fixed":
        return EARLY.build_arm_features(frame, "early_family", categories)

    base, names = BASE.build_features(frame, "hierarchical_robust", categories)
    day = frame["day"].to_numpy(float)
    if candidate.family == "compact":
        horizon = candidate.parameter
        ramp = smoothstep_ramp(day, horizon)
        bumps = np.column_stack(
            [compact_beta_bump(day, peak, horizon) for peak in EARLY_PEAKS]
        )
        bump_names = [f"compact_peak{int(peak)}_K{int(horizon)}" for peak in EARLY_PEAKS]
    elif candidate.family == "tail":
        tau = candidate.parameter
        ramp = exponential_ramp(day, tau)
        bumps = np.column_stack([gamma_bump(day, peak) for peak in EARLY_PEAKS])
        bump_names = [f"gamma_peak{int(peak)}" for peak in EARLY_PEAKS]
    elif candidate.family == "decoupled":
        horizon = candidate.parameter
        ramp = smoothstep_ramp(day, BASELINE_RAMP_DAY)
        bumps = np.column_stack(
            [compact_beta_bump(day, peak, horizon) for peak in EARLY_PEAKS]
        )
        bump_names = [
            f"decoupled_peak{int(peak)}_K{int(horizon)}" for peak in EARLY_PEAKS
        ]
    elif candidate.family == "hybrid_gamma":
        shape = candidate.parameter
        ramp = smoothstep_ramp(day, BASELINE_RAMP_DAY)
        bumps = np.column_stack(
            [gamma_bump(day, peak, shape=shape) for peak in EARLY_PEAKS]
        )
        bump_names = [
            f"hybrid_gamma_peak{int(peak)}_q{int(shape)}" for peak in EARLY_PEAKS
        ]
    elif candidate.family == "rounded":
        halfwidth = candidate.parameter
        splice = BASELINE_RAMP_DAY - halfwidth
        ramp = rounded_linear_ramp(day, halfwidth)
        bumps = np.column_stack(
            [
                spliced_beta_tail_bump(day, 1.0, 5.0, splice),
                spliced_beta_tail_bump(day, 1.0, 2.0, splice),
                spliced_beta_tail_bump(day, 1.0, 1.0, splice),
            ]
        )
        bump_names = [
            f"rounded_peak{int(peak)}_delta{int(halfwidth)}" for peak in EARLY_PEAKS
        ]
    else:
        raise ValueError(candidate.family)

    base = retime_g12_columns(base, names, day, ramp)
    return add_early_family_features(
        frame, base, names, bumps, bump_names, categories
    )


def tune_candidates(
    train: pd.DataFrame,
    candidates: Sequence[Candidate],
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    target = train["log_relative_volume"].to_numpy(float)
    weights = BASE.series_balanced_weights(train)
    folds = BASE.inner_group_splits(train["date"].astype(str).to_numpy())
    rows: list[dict[str, float | str]] = []
    for candidate in candidates:
        matrix, _ = build_candidate_features(train, candidate, categories)
        for alpha in ALPHA_GRID:
            oof = np.full(len(train), np.nan)
            for fit_index, valid_index in folds:
                fitted = BASE.fit_linear(
                    matrix[fit_index],
                    target[fit_index],
                    weights[fit_index],
                    float(alpha),
                    robust=True,
                )
                oof[valid_index] = fitted.predict(matrix[valid_index])
            score, periods = EARLY.objective(train, oof)
            rows.append(
                {
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "parameter": candidate.parameter,
                    "alpha": float(alpha),
                    "objective": score,
                    **{f"{key}_log_rmse": value for key, value in periods.items()},
                }
            )
    return pd.DataFrame(rows)


def best_row(tuning: pd.DataFrame, *, smooth_only: bool) -> pd.Series:
    available = tuning.loc[tuning["family"] != "fixed"] if smooth_only else tuning
    return available.loc[available["objective"].idxmin()]


def guarded_row(tuning: pd.DataFrame) -> pd.Series:
    fixed = best_row(tuning.loc[tuning["family"] == "fixed"], smooth_only=False)
    smooth = best_row(tuning, smooth_only=True)
    improves_objective = float(smooth["objective"]) < float(fixed["objective"])
    respects_late = float(smooth["day13_21_log_rmse"]) <= (
        LATE_GUARDRAIL * float(fixed["day13_21_log_rmse"])
    )
    return smooth if improves_objective and respects_late else fixed


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: Candidate,
    alpha: float,
    categories: Mapping[str, Sequence[str]],
) -> np.ndarray:
    train_matrix, train_names = build_candidate_features(train, candidate, categories)
    test_matrix, test_names = build_candidate_features(test, candidate, categories)
    if train_names != test_names:
        raise RuntimeError("Feature schema changed between train and test")
    fitted = BASE.fit_linear(
        train_matrix,
        train["log_relative_volume"].to_numpy(float),
        BASE.series_balanced_weights(train),
        float(alpha),
        robust=True,
    )
    return fitted.predict(test_matrix)


def candidate_by_name(name: str) -> Candidate:
    return next(candidate for candidate in CANDIDATES if candidate.name == name)


def run_validation(
    data: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    tuning_parts: list[pd.DataFrame] = []

    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        for holdout, train, test in EARLY.outer_folds(data, scheme):
            tuning = tune_candidates(train, CANDIDATES, categories)
            tuning.insert(0, "holdout", holdout)
            tuning.insert(0, "cv_kind", scheme)
            tuning_parts.append(tuning)

            fixed = best_row(tuning.loc[tuning["family"] == "fixed"], smooth_only=False)
            smooth = best_row(tuning, smooth_only=True)
            guarded = guarded_row(tuning)
            candidate_predictions: dict[str, np.ndarray] = {}
            for candidate in CANDIDATES:
                row = tuning.loc[tuning["candidate"] == candidate.name]
                row = row.loc[row["objective"].idxmin()]
                candidate_predictions[candidate.name] = fit_predict(
                    train, test, candidate, float(row["alpha"]), categories
                )
                prediction_parts.append(
                    EARLY.prediction_frame(
                        test,
                        candidate_predictions[candidate.name],
                        model=candidate.name,
                        scheme=scheme,
                        holdout=holdout,
                        selected_arm=candidate.name,
                    )
                )

            choices = {"smooth_best": smooth, "guarded_choice": guarded}
            for model, row in choices.items():
                candidate = candidate_by_name(str(row["candidate"]))
                prediction_parts.append(
                    EARLY.prediction_frame(
                        test,
                        candidate_predictions[candidate.name],
                        model=model,
                        scheme=scheme,
                        holdout=holdout,
                        selected_arm=candidate.name,
                    )
                )
                selection_rows.append(
                    {
                        "cv_kind": scheme,
                        "holdout": holdout,
                        "reported_model": model,
                        "candidate": candidate.name,
                        "family": candidate.family,
                        "parameter": candidate.parameter,
                        "alpha": float(row["alpha"]),
                        "inner_objective": float(row["objective"]),
                        "inner_day1_7_log_rmse": float(row["day1_7_log_rmse"]),
                        "inner_day8_12_log_rmse": float(row["day8_12_log_rmse"]),
                        "inner_day13_21_log_rmse": float(row["day13_21_log_rmse"]),
                    }
                )
            selection_rows.append(
                {
                    "cv_kind": scheme,
                    "holdout": holdout,
                    "reported_model": "fixed_12",
                    "candidate": "fixed_12",
                    "family": "fixed",
                    "parameter": 12.0,
                    "alpha": float(fixed["alpha"]),
                    "inner_objective": float(fixed["objective"]),
                    "inner_day1_7_log_rmse": float(fixed["day1_7_log_rmse"]),
                    "inner_day8_12_log_rmse": float(fixed["day8_12_log_rmse"]),
                    "inner_day13_21_log_rmse": float(fixed["day13_21_log_rmse"]),
                }
            )

            compact_predictions = [
                candidate_predictions[candidate.name]
                for candidate in CANDIDATES
                if candidate.family == "compact"
            ]
            ensemble = np.mean(np.column_stack(compact_predictions), axis=1)
            prediction_parts.append(
                EARLY.prediction_frame(
                    test,
                    ensemble,
                    model="compact_ensemble",
                    scheme=scheme,
                    holdout=holdout,
                    selected_arm="mean(compact_12,14,16,18,21)",
                )
            )
            print(
                f"{scheme} / {holdout}: smooth={smooth['candidate']}, "
                f"guarded={guarded['candidate']}"
            )

    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.DataFrame(selection_rows),
        pd.concat(tuning_parts, ignore_index=True),
    )


def ordinary_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    if denominator <= 0.0:
        return math.nan
    return 1.0 - float(np.sum(np.square(actual - predicted))) / denominator


def minimum_table(part: pd.DataFrame) -> pd.DataFrame:
    part = part.loc[(part["is_treated"].astype(int) == 1) & (part["day"] > 0)].copy()
    rows: list[dict[str, float | str]] = []
    for series_key, series in part.groupby("series_key", sort=True):
        observed = series.loc[series["actual_relative_volume"].idxmin()]
        predicted = series.loc[series["predicted_relative_volume"].idxmin()]
        rows.append(
            {
                "series_key": str(series_key),
                "observed_day": float(observed["day"]),
                "predicted_day": float(predicted["day"]),
            }
        )
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (scheme, model), part in predictions.groupby(["cv_kind", "model"], sort=False):
        _, periods = EARLY.objective(
            part, part["predicted_log_relative"].to_numpy(float)
        )
        treated = part.loc[part["is_treated"].astype(int) == 1]
        day21 = treated.loc[np.isclose(treated["day"], 21.0)]
        minima = minimum_table(part)
        rows.append(
            {
                "cv_kind": scheme,
                "model": model,
                **{f"{key}_log_rmse": value for key, value in periods.items()},
                "day21_r2": ordinary_r2(
                    day21["actual_log_relative"].to_numpy(float),
                    day21["predicted_log_relative"].to_numpy(float),
                ),
                "n_series": len(minima),
                "minimum_exact_day12_percent": 100.0
                * float(np.isclose(minima["predicted_day"], 12.0).mean()),
                "minimum_after_day12_percent": 100.0
                * float((minima["predicted_day"] > 12.0).mean()),
                "minimum_within_2_days_percent": 100.0
                * float(
                    (np.abs(minima["predicted_day"] - minima["observed_day"]) <= 2.0).mean()
                ),
                "minimum_spearman": float(
                    minima["observed_day"].corr(minima["predicted_day"], method="spearman")
                ),
                "minimum_median_absolute_error_days": float(
                    np.median(np.abs(minima["predicted_day"] - minima["observed_day"]))
                ),
                "minimum_mean_bias_days": float(
                    np.mean(minima["predicted_day"] - minima["observed_day"])
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_series_table(
    predictions: pd.DataFrame, scheme: str, comparison: str
) -> pd.DataFrame:
    part = predictions.loc[
        (predictions["cv_kind"] == scheme)
        & (predictions["model"].isin(["fixed_12", comparison]))
        & (predictions["is_treated"].astype(int) == 1)
        & (predictions["day"] > 0)
    ].copy()
    rows: list[dict[str, float | str]] = []
    for series_key, series in part.groupby("series_key", sort=True):
        fixed = series.loc[series["model"] == "fixed_12"]
        alternative = series.loc[series["model"] == comparison]
        if fixed.empty or alternative.empty:
            continue
        observed_day = float(fixed.loc[fixed["actual_relative_volume"].idxmin(), "day"])
        row: dict[str, float | str] = {
            "series_key": str(series_key),
            "observed_day": observed_day,
        }
        for label, frame in (("fixed", fixed), ("alternative", alternative)):
            row[f"predicted_day_{label}"] = float(
                frame.loc[frame["predicted_relative_volume"].idxmin(), "day"]
            )
            for period, (start, end) in EARLY.PERIODS.items():
                interval = frame.loc[frame["day"].between(start, end)]
                row[f"{period}_rmse_{label}"] = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                interval["actual_log_relative"].to_numpy(float)
                                - interval["predicted_log_relative"].to_numpy(float)
                            )
                        )
                    )
                )
            day21 = frame.loc[np.isclose(frame["day"], 21.0)].iloc[0]
            row["actual_day21"] = float(day21["actual_log_relative"])
            row[f"predicted_day21_{label}"] = float(day21["predicted_log_relative"])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_metrics(table: pd.DataFrame) -> dict[str, float]:
    observed = table["observed_day"].to_numpy(float)
    fixed_day = table["predicted_day_fixed"].to_numpy(float)
    alternative_day = table["predicted_day_alternative"].to_numpy(float)
    result = {
        f"{period}_rmse_difference": float(
            np.mean(
                table[f"{period}_rmse_alternative"].to_numpy(float)
                - table[f"{period}_rmse_fixed"].to_numpy(float)
            )
        )
        for period in EARLY.PERIODS
    }
    result.update(
        {
            "minimum_spearman_difference": float(
                pd.Series(observed).corr(pd.Series(alternative_day), method="spearman")
                - pd.Series(observed).corr(pd.Series(fixed_day), method="spearman")
            ),
            "minimum_exact_day12_difference": float(
                np.mean(np.isclose(alternative_day, 12.0))
                - np.mean(np.isclose(fixed_day, 12.0))
            ),
            "minimum_within_2_days_difference": float(
                np.mean(np.abs(alternative_day - observed) <= 2.0)
                - np.mean(np.abs(fixed_day - observed) <= 2.0)
            ),
            "day21_r2_difference": ordinary_r2(
                table["actual_day21"].to_numpy(float),
                table["predicted_day21_alternative"].to_numpy(float),
            )
            - ordinary_r2(
                table["actual_day21"].to_numpy(float),
                table["predicted_day21_fixed"].to_numpy(float),
            ),
        }
    )
    return result


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        for comparison in (
            "compact_12",
            "decoupled_K16",
            "hybrid_gamma_q4",
            "smooth_best",
            "guarded_choice",
            "compact_ensemble",
        ):
            table = paired_series_table(predictions, scheme, comparison)
            point = paired_metrics(table)
            samples = {metric: [] for metric in point}
            for _ in range(BOOTSTRAP_ITERATIONS):
                indices = rng.integers(0, len(table), len(table))
                values = paired_metrics(table.iloc[indices])
                for metric, value in values.items():
                    if np.isfinite(value):
                        samples[metric].append(value)
            for metric, estimate in point.items():
                values = np.asarray(samples[metric], dtype=float)
                rows.append(
                    {
                        "cv_kind": scheme,
                        "comparison": comparison,
                        "metric": metric,
                        "estimate_alternative_minus_fixed": estimate,
                        "ci_2_5": float(np.quantile(values, 0.025)),
                        "ci_97_5": float(np.quantile(values, 0.975)),
                        "probability_above_zero": float(np.mean(values > 0.0)),
                        "iterations": BOOTSTRAP_ITERATIONS,
                        "n_series": len(table),
                    }
                )
    return pd.DataFrame(rows)


def plot_basis() -> None:
    days = np.linspace(0.0, WINDOW_DAY, 421)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(days, np.minimum(days / 12.0, 1.0), color="black", lw=2.2, label="исходный g12")
    axes[0].plot(days, smoothstep_ramp(days, 16.0), lw=2.2, label="compact, K=16")
    axes[0].plot(days, exponential_ramp(days, 6.0), lw=2.2, label="tail, τ=6")
    axes[0].axvline(12.0, color="0.6", ls="--", lw=1.0)
    axes[0].set(xlabel="Сутки", ylabel="Временной множитель", title="Нарастание ответа")
    axes[0].legend(frameon=False)
    for peak in EARLY_PEAKS:
        axes[1].plot(days, compact_beta_bump(days, peak, 16.0), lw=2.0, label=f"compact, пик {peak:g}")
        axes[1].plot(days, gamma_bump(days, peak), lw=1.5, ls="--", label=f"tail, пик {peak:g}")
    axes[1].axvline(12.0, color="0.6", ls="--", lw=1.0)
    axes[1].set(xlabel="Сутки", ylabel="Амплитуда", title="Ранние переходные компоненты")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0.0, WINDOW_DAY)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "smooth_time_basis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_minimum_distributions(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=True, sharey="row")
    models = ("fixed_12", "compact_12", "decoupled_K16")
    titles = (
        "Исходный излом на 12-х сутках",
        "C1-гладкий базис K=12",
        "Развязанный базис 12/16 суток",
    )
    for row, scheme in enumerate(("leave_one_year_out", "rolling_origin_year")):
        for column, (model, title) in enumerate(zip(models, titles)):
            part = predictions.loc[
                (predictions["cv_kind"] == scheme) & (predictions["model"] == model)
            ]
            minima = minimum_table(part)
            counts = minima["predicted_day"].value_counts().sort_index()
            axes[row, column].bar(counts.index, counts.values, width=0.8, color="#4C78A8")
            axes[row, column].axvline(12.0, color="#D62728", ls="--", lw=1.2)
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].grid(axis="y", alpha=0.2)
            if column == 0:
                axes[row, column].set_ylabel(
                    "Число серий\nLOYO" if row == 0 else "Число серий\nrolling-origin"
                )
            if row == 1:
                axes[row, column].set_xlabel("Прогнозируемые сутки минимума")
    fig.suptitle("Распределение прогнозируемых суток минимума", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "minimum_day_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Минимальный Markdown-вывод без необязательной зависимости tabulate."""
    if frame.empty:
        return "_Нет данных._"

    def format_value(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return "—"
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    bootstrap: pd.DataFrame,
    data: pd.DataFrame,
) -> None:
    lines = [
        "# Проверка гладкого временного базиса",
        "",
        "Проверка устраняет оба источника структурной привязки к 12-м суткам: "
        "кусочно-линейный множитель `g12` в 49 признаках и принудительное "
        "обнуление 36 ранних признаков на 12-х сутках.",
        "",
        f"Проанализировано {data['series_key'].nunique()} серий и {len(data)} продольных строк. "
        "Форма базиса и ridge-штраф подбирались только во внутренних групповых складках; "
        "сутки минимума в выборе не участвовали.",
        "",
        "## Внешняя проверка",
        "",
        markdown_table(summary),
        "",
        "## Частота выбора",
        "",
        markdown_table(
            selection.groupby(["cv_kind", "reported_model", "candidate"])
            .size()
            .rename("n")
            .reset_index(),
            float_digits=0,
        ),
        "",
        "## Парный bootstrap",
        "",
        markdown_table(bootstrap),
        "",
        "## Правило интерпретации",
        "",
        "Гладкий вариант может заменить исходную модель только при воспроизводимом выигрыше "
        "во внешней ошибке без ухудшения позднего интервала и без потери R² на 21-е сутки. "
        "Смещение распределения суток минимума само по себе недостаточно, поскольку эта "
        "метрика не использовалась для обучения и имеет разведочный статус.",
        "",
        "## Зафиксированное решение",
        "",
        "Полная гладкая замена устраняет искусственную концентрацию прогнозов "
        "на 12-х сутках, но не обеспечивает воспроизводимого улучшения точности "
        "фактических суток минимума. Вложенный выбор между гладкими формами "
        "ухудшает внешнюю ошибку. Развязанный вариант с гладким насыщением "
        "дозового блока к 12-м суткам и ранними функциями до 16-х суток является "
        "наиболее сбалансированной апостериорной чувствительностью: поздняя ошибка "
        "и R² конечной точки не ухудшаются, но ранняя ошибка будущих лет возрастает. "
        "Поэтому основная замороженная 133-признаковая модель не заменена, а сутки "
        "минимума сохраняют статус невалидированного выхода.",
        "",
    ]
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Загрузка продольных данных...")
    _, evaluation, _, _ = BASE.build_longitudinal_tables()
    data = BASE.collapse_animal_daily_to_series(evaluation)
    categories = BASE.fixed_categories(data)
    print(f"Серий: {data['series_key'].nunique()}, строк: {len(data)}")
    print("Кандидаты:", ", ".join(candidate.name for candidate in CANDIDATES))

    predictions, selection, tuning = run_validation(data, categories)
    summary = summarize(predictions)
    bootstrap = paired_bootstrap(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        OUTPUT_DIR / "smooth_predictions.csv", sep=";", index=False, encoding="utf-8-sig"
    )
    selection.to_csv(OUTPUT_DIR / "smooth_selection.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(OUTPUT_DIR / "smooth_inner_tuning.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "smooth_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUTPUT_DIR / "smooth_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "early_script": str(EARLY_SCRIPT.resolve()),
        "early_script_sha256": sha256(EARLY_SCRIPT.resolve()),
        "base_script": str(EARLY.BASE_SCRIPT.resolve()),
        "base_script_sha256": sha256(EARLY.BASE_SCRIPT.resolve()),
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "candidates": [candidate.__dict__ for candidate in CANDIDATES],
    }
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_basis()
    plot_minimum_distributions(predictions)
    write_report(summary, selection, bootstrap, data)
    print("\nИтог:")
    print(summary.to_string(index=False))
    print(f"\nРезультаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
