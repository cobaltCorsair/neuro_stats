from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_SCRIPT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Ковариаты_pole_gamma"
    r"\run_growth_prediction.py"
)
OUTPUT_DIR = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Ранний_переходный_ответ"
)

SEED = 20260811
WINDOW_DAY = 21.0
EARLY_END = 7.0
TRANSITION_END = 12.0
PATTERN_THRESHOLD = math.log(1.20)
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
ARM_ORDER = ("base_97", "early_shared", "early_family")
PERIODS = {
    "day1_7": (1.0, 7.0),
    "day8_12": (8.0, 12.0),
    "day13_21": (13.0, 21.0),
    "day1_21": (1.0, 21.0),
}
OBJECTIVE_WEIGHTS = {
    "day1_7": 0.50,
    "day8_12": 0.20,
    "day13_21": 0.30,
}
ACCEPT_MIN_EARLY_IMPROVEMENT = 0.05
ACCEPT_MAX_LATE_WORSENING = 0.02
BOOTSTRAP_ITERATIONS = 5000

MODEL_LABELS = {
    "base_97": "исходная модель",
    "early_shared": "общий ранний компонент",
    "early_family": "семейственно-зависимый ранний компонент",
    "nested_selected": "выбранный внутри обучения кандидат",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("growth_prediction_early_extension_base", BASE_SCRIPT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def outer_folds(
    data: pd.DataFrame, scheme: str
) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]:
    years = sorted(data["year"].astype(int).unique())
    for year in years:
        if scheme == "leave_one_year_out":
            train = data.loc[data["year"].astype(int) != year]
        elif scheme == "rolling_origin_year":
            train = data.loc[data["year"].astype(int) < year]
            if train["year"].nunique() < 3 or train["date"].nunique() < 8:
                continue
        else:
            raise ValueError(scheme)
        test = data.loc[data["year"].astype(int) == year]
        if not train.empty and not test.empty:
            yield str(year), train.reset_index(drop=True), test.reset_index(drop=True)


def beta_bump(day: np.ndarray, a: float, b: float) -> np.ndarray:
    """A smooth transient that is exactly zero at days 0 and >=12."""
    day = np.asarray(day, dtype=float)
    u = np.clip(day / TRANSITION_END, 0.0, 1.0)
    raw = np.power(u, a) * np.power(1.0 - u, b)
    peak_u = a / (a + b)
    peak = peak_u**a * (1.0 - peak_u) ** b
    values = raw / peak
    values[(day <= 0.0) | (day >= TRANSITION_END)] = 0.0
    return values


def early_basis(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    day = frame["day"].to_numpy(float)
    # Analytical maxima are at days 2, 4 and 6.
    columns = [
        beta_bump(day, 1.0, 5.0),
        beta_bump(day, 1.0, 2.0),
        beta_bump(day, 1.0, 1.0),
    ]
    return np.column_stack(columns), ["bump_day2", "bump_day4", "bump_day6"]


def build_arm_features(
    frame: pd.DataFrame,
    arm: str,
    categories: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, list[str]]:
    base, names = BASE.build_features(frame, "hierarchical_robust", categories)
    if arm == "base_97":
        return base, names
    bump, bump_names = early_basis(frame)
    treated = frame["is_treated"].to_numpy(float)
    log_dose = np.log1p(frame["total_dose_gy"].to_numpy(float))
    additions: list[np.ndarray] = []
    added_names: list[str] = []
    for column, name in zip(bump.T, bump_names):
        additions.append(treated * column)
        added_names.append(f"treated_{name}")
        additions.append(log_dose * column)
        added_names.append(f"logdose_{name}")
    if arm == "early_family":
        family_values = frame["family"].astype(str).to_numpy()
        for family in categories["families"]:
            indicator = (family_values == family).astype(float)
            for column, name in zip(bump.T, bump_names):
                additions.append(indicator * column)
                added_names.append(f"family_{family}_{name}")
    elif arm != "early_shared":
        raise ValueError(arm)
    return np.column_stack([base, *additions]), [*names, *added_names]


def mean_series_rmse(
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    start: float,
    end: float,
) -> float:
    work = frame.assign(
        actual=np.asarray(actual, dtype=float),
        predicted=np.asarray(predicted, dtype=float),
    )
    work = work.loc[
        (work["is_treated"].astype(int) == 1)
        & (work["day"].to_numpy(float) >= start)
        & (work["day"].to_numpy(float) <= end)
    ].copy()
    if work.empty:
        return math.nan
    work["sq_error"] = np.square(work["actual"] - work["predicted"])
    per_series = work.groupby("series_key")["sq_error"].mean().pow(0.5)
    return float(per_series.mean())


def objective(frame: pd.DataFrame, predicted: np.ndarray) -> tuple[float, dict[str, float]]:
    actual = frame["log_relative_volume"].to_numpy(float)
    metrics = {
        label: mean_series_rmse(frame, actual, predicted, *PERIODS[label])
        for label in OBJECTIVE_WEIGHTS
    }
    score = float(sum(OBJECTIVE_WEIGHTS[key] * metrics[key] for key in metrics))
    return score, metrics


def tune_arm(
    arm: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame, list[str]]:
    train_matrix, names = build_arm_features(train, arm, categories)
    test_matrix, test_names = build_arm_features(test, arm, categories)
    if names != test_names:
        raise RuntimeError("Feature schema changed between training and test data")
    target = train["log_relative_volume"].to_numpy(float)
    weights = BASE.series_balanced_weights(train)
    folds = BASE.inner_group_splits(train["date"].astype(str).to_numpy())
    rows: list[dict[str, float | int | str]] = []
    best = (math.inf, float(ALPHA_GRID[0]), np.full(len(train), np.nan))
    for alpha in ALPHA_GRID:
        oof = np.full(len(train), np.nan)
        for fit_index, valid_index in folds:
            fitted = BASE.fit_linear(
                train_matrix[fit_index],
                target[fit_index],
                weights[fit_index],
                float(alpha),
                robust=True,
            )
            oof[valid_index] = fitted.predict(train_matrix[valid_index])
        score, periods = objective(train, oof)
        rows.append(
            {
                "arm": arm,
                "alpha": float(alpha),
                "objective": score,
                **{f"{key}_log_rmse": value for key, value in periods.items()},
                "selected_alpha": 0,
            }
        )
        if score < best[0] - 1.0e-12:
            best = (score, float(alpha), oof.copy())
    fitted = BASE.fit_linear(
        train_matrix,
        target,
        weights,
        best[1],
        robust=True,
    )
    tuning = pd.DataFrame(rows)
    tuning.loc[np.isclose(tuning["alpha"], best[1]), "selected_alpha"] = 1
    return fitted.predict(test_matrix), best[2], best[1], tuning, names


def prediction_frame(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    scheme: str,
    holdout: str,
    selected_arm: str,
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "holdout", holdout)
    output.insert(0, "cv_kind", scheme)
    output.insert(0, "model", model)
    output["selected_arm"] = selected_arm
    output["actual_log_relative"] = output["log_relative_volume"].to_numpy(float)
    output["predicted_log_relative"] = np.asarray(predicted, dtype=float)
    output["actual_relative_volume"] = np.exp(output["actual_log_relative"])
    output["predicted_relative_volume"] = np.exp(output["predicted_log_relative"])
    return output


def run_outer_validation(
    data: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        folds = list(outer_folds(data, scheme))
        for fold_number, (holdout, train, test) in enumerate(folds, start=1):
            predictions: dict[str, np.ndarray] = {}
            oof_predictions: dict[str, np.ndarray] = {}
            alphas: dict[str, float] = {}
            scores: dict[str, float] = {}
            for arm in ARM_ORDER:
                if arm == "base_97":
                    test_pred, oof, alpha, tuning = BASE.fit_candidate(
                        "hierarchical_robust", train, test, categories
                    )
                    feature_names = build_arm_features(train, arm, categories)[1]
                    selected_score, selected_periods = objective(train, oof)
                    tuning = tuning.copy()
                    tuning.insert(0, "arm", arm)
                    tuning["selected_alpha"] = np.isclose(
                        tuning["alpha"].to_numpy(float), float(alpha)
                    ).astype(int)
                    tuning["objective"] = np.where(
                        tuning["selected_alpha"].astype(bool), selected_score, np.nan
                    )
                    for period, value in selected_periods.items():
                        tuning[f"{period}_log_rmse"] = np.where(
                            tuning["selected_alpha"].astype(bool), value, np.nan
                        )
                else:
                    test_pred, oof, alpha, tuning, feature_names = tune_arm(
                        arm, train, test, categories
                    )
                predictions[arm] = test_pred
                oof_predictions[arm] = oof
                alphas[arm] = alpha
                scores[arm] = objective(train, oof)[0]
                tuning.insert(0, "holdout", holdout)
                tuning.insert(0, "cv_kind", scheme)
                tuning_parts.append(tuning)
                if fold_number == 1:
                    feature_rows.extend(
                        {
                            "cv_kind": scheme,
                            "arm": arm,
                            "feature_index": index,
                            "feature": name,
                        }
                        for index, name in enumerate(feature_names, start=1)
                    )
            # This arm choice is fully internal to each outer-training set.
            selected = min(ARM_ORDER, key=lambda name: (scores[name], ARM_ORDER.index(name)))
            for model in ARM_ORDER:
                prediction_parts.append(
                    prediction_frame(
                        test,
                        predictions[model],
                        model=model,
                        scheme=scheme,
                        holdout=holdout,
                        selected_arm=model,
                    )
                )
            prediction_parts.append(
                prediction_frame(
                    test,
                    predictions[selected],
                    model="nested_selected",
                    scheme=scheme,
                    holdout=holdout,
                    selected_arm=selected,
                )
            )
            fold_rows.append(
                {
                    "cv_kind": scheme,
                    "fold_number": fold_number,
                    "holdout": holdout,
                    "min_training_year": int(train["year"].astype(int).min()),
                    "max_training_year": int(train["year"].astype(int).max()),
                    "n_training_years": int(train["year"].nunique()),
                    "n_training_dates": int(train["date"].nunique()),
                    "n_training_series": int(train["series_key"].nunique()),
                    "n_test_series": int(test["series_key"].nunique()),
                    "selected_arm": selected,
                    **{f"alpha_{arm}": alphas[arm] for arm in ARM_ORDER},
                    **{f"inner_objective_{arm}": scores[arm] for arm in ARM_ORDER},
                }
            )
            print(
                f"{scheme}: {fold_number}/{len(folds)} {holdout}; selected={selected}",
                flush=True,
            )
    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(tuning_parts, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(feature_rows).drop_duplicates().reset_index(drop=True),
    )


def pattern_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series_key, series in group.groupby("series_key", sort=False):
        series = series.sort_values("day")
        early = series.loc[(series["day"] >= 1.0) & (series["day"] <= EARLY_END)]
        day0 = series.loc[np.isclose(series["day"], 0.0)]
        day12 = series.loc[np.isclose(series["day"], TRANSITION_END)]
        if early.empty or day0.empty or day12.empty:
            continue
        actual_peak_index = early["actual_log_relative"].idxmax()
        predicted_peak_index = early["predicted_log_relative"].idxmax()
        actual_peak = float(series.loc[actual_peak_index, "actual_log_relative"])
        predicted_peak = float(series.loc[predicted_peak_index, "predicted_log_relative"])
        actual_day0 = float(day0.iloc[0]["actual_log_relative"])
        predicted_day0 = float(day0.iloc[0]["predicted_log_relative"])
        actual_day12 = float(day12.iloc[0]["actual_log_relative"])
        predicted_day12 = float(day12.iloc[0]["predicted_log_relative"])
        actual_rise = actual_peak - actual_day0
        actual_fall = actual_peak - actual_day12
        predicted_rise = predicted_peak - predicted_day0
        predicted_fall = predicted_peak - predicted_day12
        rows.append(
            {
                "series_key": series_key,
                "holdout": str(series.iloc[0]["holdout"]),
                "family": str(series.iloc[0]["family"]),
                "actual_pattern": int(
                    actual_rise >= PATTERN_THRESHOLD and actual_fall >= PATTERN_THRESHOLD
                ),
                "predicted_pattern": int(
                    predicted_rise >= PATTERN_THRESHOLD
                    and predicted_fall >= PATTERN_THRESHOLD
                ),
                "actual_peak_day": float(series.loc[actual_peak_index, "day"]),
                "predicted_peak_day": float(series.loc[predicted_peak_index, "day"]),
                "actual_rise_log": actual_rise,
                "predicted_rise_log": predicted_rise,
                "actual_fall_log": actual_fall,
                "predicted_fall_log": predicted_fall,
            }
        )
    return rows


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = predictions.loc[predictions["is_treated"].astype(int) == 1].copy()
    metric_rows: list[dict[str, object]] = []
    all_pattern_rows: list[dict[str, object]] = []
    for (scheme, model), group in treated.groupby(["cv_kind", "model"], sort=False):
        actual = group["actual_log_relative"].to_numpy(float)
        predicted = group["predicted_log_relative"].to_numpy(float)
        periods = {
            label: mean_series_rmse(group, actual, predicted, *bounds)
            for label, bounds in PERIODS.items()
        }
        endpoint = group.loc[np.isclose(group["day"], WINDOW_DAY)]
        shapes = pd.DataFrame(pattern_rows(group))
        if shapes.empty:
            recall = precision = f1 = math.nan
            actual_patterns = predicted_patterns = 0
            peak_mae = rise_mae = fall_mae = math.nan
        else:
            actual_pattern = shapes["actual_pattern"].astype(bool)
            predicted_pattern = shapes["predicted_pattern"].astype(bool)
            true_positive = int((actual_pattern & predicted_pattern).sum())
            actual_patterns = int(actual_pattern.sum())
            predicted_patterns = int(predicted_pattern.sum())
            recall = true_positive / actual_patterns if actual_patterns else math.nan
            precision = true_positive / predicted_patterns if predicted_patterns else math.nan
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if np.isfinite(precision + recall) and precision + recall > 0.0
                else math.nan
            )
            patterned = shapes.loc[actual_pattern]
            peak_mae = float(
                np.mean(np.abs(patterned["actual_peak_day"] - patterned["predicted_peak_day"]))
            )
            rise_mae = float(
                np.mean(np.abs(patterned["actual_rise_log"] - patterned["predicted_rise_log"]))
            )
            fall_mae = float(
                np.mean(np.abs(patterned["actual_fall_log"] - patterned["predicted_fall_log"]))
            )
            shapes.insert(0, "model", model)
            shapes.insert(0, "cv_kind", scheme)
            all_pattern_rows.extend(shapes.to_dict("records"))
        metric_rows.append(
            {
                "cv_kind": scheme,
                "model": model,
                "n_series": int(group["series_key"].nunique()),
                **{f"{label}_mean_series_log_rmse": value for label, value in periods.items()},
                "day21_log_r2": float(
                    BASE.weighted_r2(
                        endpoint["actual_log_relative"].to_numpy(float),
                        endpoint["predicted_log_relative"].to_numpy(float),
                        BASE.series_balanced_weights(endpoint),
                    )
                ),
                "n_actual_early_patterns": actual_patterns,
                "n_predicted_early_patterns": predicted_patterns,
                "early_pattern_recall": recall,
                "early_pattern_precision": precision,
                "early_pattern_f1": f1,
                "peak_day_mae_among_actual_patterns": peak_mae,
                "rise_log_amplitude_mae": rise_mae,
                "fall_log_amplitude_mae": fall_mae,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(all_pattern_rows)


def acceptance_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = ("early_shared", "early_family", "nested_selected")
    for candidate_name in candidates:
        for scheme in ("leave_one_year_out", "rolling_origin_year"):
            indexed = summary.loc[summary["cv_kind"] == scheme].set_index("model")
            base = indexed.loc["base_97"]
            candidate = indexed.loc[candidate_name]
            early_improvement = 1.0 - (
                candidate["day1_7_mean_series_log_rmse"]
                / base["day1_7_mean_series_log_rmse"]
            )
            late_worsening = (
                candidate["day13_21_mean_series_log_rmse"]
                / base["day13_21_mean_series_log_rmse"]
                - 1.0
            )
            recall_gain = candidate["early_pattern_recall"] - base["early_pattern_recall"]
            accepted = bool(
                early_improvement >= ACCEPT_MIN_EARLY_IMPROVEMENT
                and late_worsening <= ACCEPT_MAX_LATE_WORSENING
                and recall_gain > 0.0
            )
            rows.append(
                {
                    "model": candidate_name,
                    "cv_kind": scheme,
                    "early_rmse_relative_improvement": float(early_improvement),
                    "late_rmse_relative_worsening": float(late_worsening),
                    "early_pattern_recall_gain": float(recall_gain),
                    "passes_scheme_rule": int(accepted),
                }
            )
    output = pd.DataFrame(rows)
    passed = output.groupby("model")["passes_scheme_rule"].all().astype(int)
    output["overall_acceptance"] = output["model"].map(passed)
    return output


def cluster_bootstrap_early_improvement(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for candidate_name in ("early_shared", "early_family", "nested_selected"):
        for scheme in ("leave_one_year_out", "rolling_origin_year"):
            subset = predictions.loc[
                (predictions["cv_kind"] == scheme)
                & (predictions["model"].isin(["base_97", candidate_name]))
                & (predictions["is_treated"].astype(int) == 1)
                & (predictions["day"].between(1.0, EARLY_END))
            ].copy()
            table = (
                subset.assign(
                    sq=np.square(
                        subset["actual_log_relative"] - subset["predicted_log_relative"]
                    )
                )
                .groupby(["holdout", "series_key", "model"])["sq"]
                .mean()
                .pow(0.5)
                .unstack("model")
                .dropna()
                .reset_index()
            )
            table["improvement"] = table["base_97"] - table[candidate_name]
            clusters = table["holdout"].astype(str).unique()
            samples: list[float] = []
            for _ in range(BOOTSTRAP_ITERATIONS):
                sampled = rng.choice(clusters, len(clusters), replace=True)
                values = [
                    table.loc[
                        table["holdout"].astype(str) == cluster, "improvement"
                    ].to_numpy(float)
                    for cluster in sampled
                ]
                samples.append(float(np.concatenate(values).mean()))
            values = np.asarray(samples)
            rows.append(
                {
                    "model": candidate_name,
                    "cv_kind": scheme,
                    "n_series": int(len(table)),
                    "n_year_clusters": int(len(clusters)),
                    "base_minus_candidate_mean_series_early_log_rmse": float(
                        table["improvement"].mean()
                    ),
                    "cluster_bootstrap_ci95_low": float(np.quantile(values, 0.025)),
                    "cluster_bootstrap_ci95_high": float(np.quantile(values, 0.975)),
                    "cluster_bootstrap_probability_improvement": float(np.mean(values > 0.0)),
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                }
            )
    return pd.DataFrame(rows)


def choose_candidate(summary: pd.DataFrame, acceptance: pd.DataFrame) -> dict[str, object]:
    accepted = sorted(
        acceptance.loc[acceptance["overall_acceptance"].astype(int) == 1, "model"].unique()
    )
    fixed_arms = [name for name in accepted if name in {"early_shared", "early_family"}]
    if not fixed_arms:
        return {
            "selected_model": "base_97",
            "status": "candidate_not_accepted",
            "selection_metric": "rolling_origin_year/day1_21_mean_series_log_rmse",
        }
    rolling = summary.loc[
        (summary["cv_kind"] == "rolling_origin_year")
        & summary["model"].isin(fixed_arms)
    ].copy()
    selected = str(
        rolling.sort_values("day1_21_mean_series_log_rmse").iloc[0]["model"]
    )
    return {
        "selected_model": selected,
        "status": "internally_supported_exploratory_extension",
        "selection_metric": "rolling_origin_year/day1_21_mean_series_log_rmse",
        "accepted_fixed_candidates": fixed_arms,
        "external_validation": False,
    }


def daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    treated = predictions.loc[predictions["is_treated"].astype(int) == 1]
    for (scheme, model, day), group in treated.groupby(
        ["cv_kind", "model", "day"], sort=False
    ):
        actual = group["actual_log_relative"].to_numpy(float)
        predicted = group["predicted_log_relative"].to_numpy(float)
        rows.append(
            {
                "cv_kind": scheme,
                "model": model,
                "day": float(day),
                "n_series": int(group["series_key"].nunique()),
                "log_rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
                "log_r2": float(BASE.weighted_r2(actual, predicted)),
            }
        )
    return pd.DataFrame(rows)


def fit_and_export_final_model(
    data: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
    selected_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if selected_model == "base_97":
        _, oof, alpha, tuning = BASE.fit_candidate(
            "hierarchical_robust", data, data, categories
        )
        matrix, names = build_arm_features(data, selected_model, categories)
    else:
        _, oof, alpha, tuning, names = tune_arm(
            selected_model, data, data, categories
        )
        matrix, check_names = build_arm_features(data, selected_model, categories)
        if names != check_names:
            raise RuntimeError("Final feature schema is inconsistent")
    target = data["log_relative_volume"].to_numpy(float)
    fitted = BASE.fit_linear(
        matrix,
        target,
        BASE.series_balanced_weights(data),
        float(alpha),
        robust=True,
    )
    scale = np.asarray(fitted.scaler.scale_, dtype=float)
    scaled_coefficient = np.asarray(fitted.model.coef_, dtype=float)
    coefficient = scaled_coefficient / scale
    parameters = pd.DataFrame(
        {
            "feature_index": np.arange(1, len(names) + 1),
            "feature": names,
            "training_scale": scale,
            "ridge_coefficient_scaled": scaled_coefficient,
            "coefficient_original_feature_scale": coefficient,
        }
    )
    reconstructed = matrix @ coefficient
    direct = fitted.predict(matrix)
    if not np.allclose(reconstructed, direct, atol=1.0e-10, rtol=1.0e-10):
        raise RuntimeError("Exported coefficients do not reproduce the fitted model")
    oof_frame = data[
        [
            "series_key",
            "date",
            "year",
            "family",
            "day",
            "log_relative_volume",
        ]
    ].copy()
    oof_frame["oof_predicted_log_relative"] = oof
    oof_frame["oof_predicted_relative_volume"] = np.exp(oof)
    metadata = {
        "selected_model": selected_model,
        "n_features": len(names),
        "ridge_alpha": float(alpha),
        "fit_intercept": False,
        "feature_centering": False,
        "feature_scaling": "training_standard_deviation",
        "robust_reweighting_iterations": int(BASE.ROBUST_ITERATIONS),
        "n_series": int(data["series_key"].nunique()),
        "n_treated_series": int(
            data.loc[data["is_treated"].astype(int) == 1, "series_key"].nunique()
        ),
        "n_control_series": int(
            data.loc[data["is_treated"].astype(int) == 0, "series_key"].nunique()
        ),
        "families": list(categories["families"]),
        "regimens": list(categories["regimens"]),
        "orders": list(categories["orders"]),
        "export_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - direct))
        ),
    }
    return parameters, oof_frame, metadata


def make_figures(
    predictions: pd.DataFrame, summary: pd.DataFrame, selected_model: str
) -> None:
    order = ["base_97", "early_shared", "early_family", "nested_selected"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    rolling = summary.loc[summary["cv_kind"] == "rolling_origin_year"].set_index("model")
    x = np.arange(len(order))
    width = 0.25
    for offset, period, label in (
        (-width, "day1_7", "1–7-е сутки"),
        (0.0, "day8_12", "8–12-е сутки"),
        (width, "day13_21", "13–21-е сутки"),
    ):
        axes[0].bar(
            x + offset,
            [rolling.loc[model, f"{period}_mean_series_log_rmse"] for model in order],
            width,
            label=label,
        )
    axes[0].set_xticks(x, [MODEL_LABELS[model] for model in order], rotation=18, ha="right")
    axes[0].set_ylabel("среднее СКО серии, log(V/V₀)")
    axes[0].set_title("Хронологический перенос на будущий год")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x,
        [rolling.loc[model, "early_pattern_recall"] * 100.0 for model in order],
        color=["#8c8c8c", "#5b8db8", "#d58a3c", "#467a4b"],
    )
    axes[1].set_xticks(x, [MODEL_LABELS[model] for model in order], rotation=18, ha="right")
    axes[1].set_ylabel("воспроизведённые ранние паттерны, %")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_title("Подъём ≥20% и спад к 12-м суткам ≥20%")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"figure_1_early_validation.{suffix}", dpi=240)
    plt.close(fig)

    rolling_predictions = predictions.loc[
        (predictions["cv_kind"] == "rolling_origin_year")
        & predictions["model"].isin(["base_97", selected_model])
        & (predictions["is_treated"].astype(int) == 1)
    ].copy()
    if selected_model == "base_97":
        return
    base = rolling_predictions.loc[rolling_predictions["model"] == "base_97"]
    nested = rolling_predictions.loc[rolling_predictions["model"] == selected_model]
    keys = ["holdout", "series_key", "day"]
    paired = base.merge(
        nested[keys + ["predicted_log_relative", "selected_arm"]],
        on=keys,
        suffixes=("_base", "_nested"),
        validate="one_to_one",
    )
    early = paired.loc[paired["day"].between(1.0, TRANSITION_END)].copy()
    scores = (
        early.assign(
            base_sq=np.square(
                early["actual_log_relative"] - early["predicted_log_relative_base"]
            ),
            nested_sq=np.square(
                early["actual_log_relative"] - early["predicted_log_relative_nested"]
            ),
        )
        .groupby("series_key")[["base_sq", "nested_sq"]]
        .mean()
        .pow(0.5)
    )
    scores["improvement"] = scores["base_sq"] - scores["nested_sq"]
    positive = scores.loc[scores["improvement"] > 0.0].sort_values("improvement")
    if positive.empty:
        selected_key = str(scores.index[0])
    else:
        median = float(positive["improvement"].median())
        selected_key = str((positive["improvement"] - median).abs().idxmin())
    example = paired.loc[paired["series_key"] == selected_key].sort_values("day")
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.plot(
        example["day"],
        example["actual_relative_volume"],
        "o-",
        color="black",
        label="наблюдаемая групповая траектория",
    )
    axis.plot(
        example["day"],
        np.exp(example["predicted_log_relative_base"]),
        "--",
        linewidth=2.2,
        label="исходная модель",
    )
    axis.plot(
        example["day"],
        np.exp(example["predicted_log_relative_nested"]),
        "-",
        linewidth=2.2,
        label=f"ранний кандидат ({MODEL_LABELS[selected_model]})",
    )
    axis.axvspan(0.0, TRANSITION_END, color="#d95f02", alpha=0.08)
    axis.axvline(TRANSITION_END, color="#d95f02", linewidth=1.2, linestyle=":")
    axis.set_xlabel("сутки после облучения")
    axis.set_ylabel("относительный объём V/V₀")
    axis.set_title(
        "Иллюстративная серия с медианным положительным изменением\n"
        + selected_key.replace("|", "; ")
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"figure_2_early_example.{suffix}", dpi=240)
    plt.close(fig)


def write_report(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    acceptance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    folds: pd.DataFrame,
    choice: Mapping[str, object],
) -> None:
    selected_model = str(choice["selected_model"])
    lines = [
        "# Ранний переходный компонент прогностической модели",
        "",
        "## Статус анализа",
        "",
        "Это отдельный анализ-кандидат. Исходная 97-признаковая модель и ранее "
        "полученные результаты не заменялись. Ранний компонент предложен после "
        "обнаружения систематического невоспроизведения подъёма и последующего "
        "спада объёма на 1–12-е сутки, поэтому результат является внутренним "
        "разведочным уточнением и требует особенно строгой проверки переноса.",
        "",
        "## Изменение модели",
        "",
        "К исходным признакам добавлены три гладкие beta-базисные функции времени "
        "с максимумами приблизительно на 2, 4 и 6-е сутки. Все они равны нулю "
        "в день 0 и начиная с 12-х суток. Проверены общий для облучённых серий "
        "и семейственно-зависимый варианты. Коэффициенты и ridge-параметр "
        "определялись только на обучающей части каждого внешнего цикла.",
        "",
        "## Валидация и правило принятия",
        "",
        f"Проанализировано {data.loc[data.is_treated.eq(1), 'series_key'].nunique()} "
        "облучённых серий. Использованы исключение целого календарного года и "
        "хронологический перенос на следующий год. Выбор варианта выполнялся во "
        "внутренней grouped-CV по дате. Целевая функция: 50% ошибки 1–7-х суток, "
        "20% ошибки 8–12-х суток и 30% ошибки 13–21-х суток. Ранний паттерн "
        "определён заранее как подъём не менее 20% относительно дня 0 и спад от "
        "раннего максимума к 12-м суткам не менее 20%.",
        "",
        "Кандидат принимается только если в обеих внешних схемах раннее СКО "
        "снижается не менее чем на 5%, полнота распознавания раннего паттерна "
        "увеличивается, а СКО 13–21-х суток не ухудшается более чем на 2%.",
        "",
        "## Результаты",
        "",
    ]
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        subset = summary.loc[
            (summary["cv_kind"] == scheme)
            & summary["model"].isin(["base_97", selected_model])
        ].set_index("model")
        base = subset.loc["base_97"]
        candidate = subset.loc[selected_model]
        accepted = acceptance.loc[
            (acceptance["model"] == selected_model)
            & (acceptance["cv_kind"] == scheme)
        ].iloc[0]
        boot = bootstrap.loc[
            (bootstrap["model"] == selected_model)
            & (bootstrap["cv_kind"] == scheme)
        ].iloc[0]
        lines.extend(
            [
                f"### {scheme}",
                "",
                f"- СКО 1–7-х суток: {base['day1_7_mean_series_log_rmse']:.3f} → "
                f"{candidate['day1_7_mean_series_log_rmse']:.3f} "
                f"({accepted['early_rmse_relative_improvement']:+.1%} относительного улучшения).",
                f"- СКО 13–21-х суток: {base['day13_21_mean_series_log_rmse']:.3f} → "
                f"{candidate['day13_21_mean_series_log_rmse']:.3f} "
                f"({accepted['late_rmse_relative_worsening']:+.1%} изменения).",
                f"- Полнота раннего паттерна: {base['early_pattern_recall']:.1%} → "
                f"{candidate['early_pattern_recall']:.1%}.",
                f"- Bootstrap-разность раннего СКО (исходная минус кандидат): "
                f"{boot['base_minus_candidate_mean_series_early_log_rmse']:.3f}; "
                f"95% ДИ {boot['cluster_bootstrap_ci95_low']:.3f}…"
                f"{boot['cluster_bootstrap_ci95_high']:.3f}.",
                f"- Правило для этой схемы: {'выполнено' if accepted['passes_scheme_rule'] else 'не выполнено'}.",
                "",
            ]
        )
    selected_counts = (
        folds.groupby(["cv_kind", "selected_arm"]).size().rename("n_folds").reset_index()
    )
    lines.extend(
        [
            "## Решение",
            "",
            (
                f"Предварительное правило принятия выполнено в обеих схемах. Из "
                f"фиксированных кандидатов выбран `{selected_model}`; его можно "
                "переносить в итоговую модель как внутренне подтверждённое "
                "разведочное уточнение."
                if selected_model != "base_97"
                else "Предварительное правило принятия не выполнено в обеих схемах. "
                "Ранний компонент следует оставить анализом чувствительности и не "
                "заменять им основную модель."
            ),
            "",
            "Частота выбора вариантов во внутренних циклах:",
            "",
        ]
    )
    for row in selected_counts.itertuples(index=False):
        lines.append(f"- {row.cv_kind}: {row.selected_arm} — {row.n_folds} внешних циклов.")
    lines.extend(
        [
            "",
            "Даже при принятии это не доказывает отдельный биологический механизм "
            "раннего набухания или отёка: базис описывает форму переходного ответа, "
            "а его физиологическая интерпретация требует независимых данных.",
            "",
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, evaluation, _, sources = BASE.build_longitudinal_tables()
    data = BASE.collapse_animal_daily_to_series(evaluation)
    categories = BASE.fixed_categories(data)
    predictions, tuning, folds, features = run_outer_validation(data, categories)
    summary, patterns = summarize_predictions(predictions)
    acceptance = acceptance_table(summary)
    bootstrap = cluster_bootstrap_early_improvement(predictions)
    choice = choose_candidate(summary, acceptance)
    by_day = daily_metrics(predictions)
    final_parameters, final_oof, final_metadata = fit_and_export_final_model(
        data, categories, str(choice["selected_model"])
    )

    write_csv(predictions, OUTPUT_DIR / "outer_predictions.csv")
    write_csv(tuning, OUTPUT_DIR / "inner_tuning.csv")
    write_csv(folds, OUTPUT_DIR / "outer_fold_selection.csv")
    write_csv(features, OUTPUT_DIR / "feature_dictionary.csv")
    write_csv(summary, OUTPUT_DIR / "validation_summary.csv")
    write_csv(patterns, OUTPUT_DIR / "early_pattern_by_series.csv")
    write_csv(acceptance, OUTPUT_DIR / "acceptance_rule.csv")
    write_csv(bootstrap, OUTPUT_DIR / "early_rmse_cluster_bootstrap.csv")
    write_csv(by_day, OUTPUT_DIR / "daily_metrics.csv")
    write_csv(final_parameters, OUTPUT_DIR / "final_model_parameters.csv")
    write_csv(final_oof, OUTPUT_DIR / "final_model_oof_predictions.csv")
    make_figures(predictions, summary, str(choice["selected_model"]))
    write_report(data, summary, acceptance, bootstrap, folds, choice)

    config = {
        "analysis": "early_response_extension",
        "status": "post_hoc_exploratory_candidate",
        "base_model": "hierarchical_robust_97_features",
        "outer_validation": ["leave_one_year_out", "rolling_origin_year"],
        "inner_grouping": "calendar_date",
        "arms": list(ARM_ORDER),
        "early_basis_peak_days": [2, 4, 6],
        "early_basis_zero_at_day0_and_from_day12": True,
        "pattern_threshold_relative": 0.20,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "alpha_grid": list(ALPHA_GRID),
        "acceptance": {
            "min_early_relative_improvement": ACCEPT_MIN_EARLY_IMPROVEMENT,
            "max_late_relative_worsening": ACCEPT_MAX_LATE_WORSENING,
            "require_positive_pattern_recall_gain": True,
            "require_both_outer_schemes": True,
        },
        "random_seed": SEED,
    }
    (OUTPUT_DIR / "analysis_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "model_choice.json").write_text(
        json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "final_model_metadata.json").write_text(
        json.dumps(final_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_rows = [
        {"path": str(BASE_SCRIPT.resolve()), "sha256": sha256(BASE_SCRIPT)},
        {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
    ]
    source_rows.extend(
        {"path": str(path.resolve()), "sha256": sha256(path)}
        for path in sources
        if path.exists()
    )
    write_csv(pd.DataFrame(source_rows).drop_duplicates(), OUTPUT_DIR / "source_manifest.csv")
    print(summary.to_string(index=False), flush=True)
    print(acceptance.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
