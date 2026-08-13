"""Проверка настройки прогностической модели непосредственно по ТРО14.

Основная 133-признаковая модель предсказывает полную траекторию ln(V/V0) и
подбирает регуляризацию по ошибке траектории. Здесь тот же самый априорный
признаковый базис используется для отдельного endpoint-кандидата:

    z14 = ln(R_treated(14) / R_control(14)) = ln(1 - TPO14 / 100).

Для каждой пары строк строится контраст признаков X_treated - X_control.
Коэффициенты ridge и параметр регуляризации оцениваются только внутри
обучающей части. Внешняя проверка выполняется исключением календарного года и
хронологическим переносом на будущий год. Исходная модель и её артефакт не
изменяются.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
EARLY_SCRIPT = HERE / "analyze_early_response_extension.py"
RESULTS_ROOT = Path(r"D:\Диссертация\Результаты")
MODEL_DIR = RESULTS_ROOT / "Задача_4_Модель" / "4.6_Ранний_переходный_ответ"
BASE_DATA = (
    RESULTS_ROOT
    / "Задача_4_Модель"
    / "4.6_Ковариаты_pole_gamma"
    / "series_daily_modelling_table.csv"
)
ENDPOINTS = (
    RESULTS_ROOT
    / "Задача_3_Анализ"
    / "3.1_3.2_Итоговый_протокол_после_аудита_дублей"
    / "matched_control_experiment_endpoints.csv"
)
BASELINE_PREDICTIONS = MODEL_DIR / "outer_predictions.csv"
OUTPUT_DIR = RESULTS_ROOT / "Задача_4_Модель" / "4.6_Настройка_по_ТРО14"

DAY = 14.0
SEED = 20260812
ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
HUBER_C = 1.5
ROBUST_ITERATIONS = 3
BOOTSTRAP_ITERATIONS = 10000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EARLY = load_module("tpo14_early_basis", EARLY_SCRIPT)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def control_date(control_kind: str, control_file: str, treated_date: str) -> str | None:
    if str(control_kind) == "same_date":
        return str(treated_date)
    match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", str(control_file))
    if match is None:
        return None
    return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"


def build_contrast_cohort(data: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    day14 = data.loc[np.isclose(data["day"].to_numpy(float), DAY)].copy()
    if day14["series_key"].duplicated().any():
        raise RuntimeError("Ожидалась одна групповая строка на серию в сутки 14")

    selector_rows: list[dict[str, object]] = []
    selector_columns = ["series_id", "family", "dose_group_physical_gy", "regimen_class"]
    for selector, group in endpoints.groupby(selector_columns, dropna=False, sort=False):
        treated_date, family, dose, regimen = selector
        controls = {
            control_date(row.control_kind, row.control_file, row.series_id)
            for row in group.itertuples(index=False)
        }
        controls.discard(None)
        if len(controls) != 1:
            raise RuntimeError(f"Неоднозначный контроль для {selector}: {controls}")
        selector_rows.append(
            {
                "treated_date": str(treated_date),
                "family": str(family),
                "total_dose_gy": float(dose),
                "regimen_class": str(regimen),
                "control_date": str(next(iter(controls))),
                "control_kind": "+".join(sorted(set(group["control_kind"].astype(str)))),
                "reported_tpo14_mean": float(group["tpo14_relative_pct"].mean()),
            }
        )

    rows: list[dict[str, object]] = []
    for selector in selector_rows:
        treated = day14.loc[
            day14["date"].astype(str).eq(selector["treated_date"])
            & day14["family"].astype(str).eq(selector["family"])
            & np.isclose(
                day14["total_dose_gy"].to_numpy(float),
                float(selector["total_dose_gy"]),
                atol=0.011,
            )
            & day14["regimen_class"].astype(str).eq(selector["regimen_class"])
        ]
        control = day14.loc[
            day14["date"].astype(str).eq(selector["control_date"])
            & day14["family"].astype(str).eq("control")
        ]
        if treated.empty or len(control) != 1:
            continue
        control_row = control.iloc[0]
        for treated_row in treated.itertuples(index=False):
            log_ratio = float(treated_row.log_relative_volume - control_row.log_relative_volume)
            rows.append(
                {
                    "series_key": str(treated_row.series_key),
                    "control_series_key": str(control_row.series_key),
                    "date": str(treated_row.date),
                    "year": int(treated_row.year),
                    "family": str(treated_row.family),
                    "regimen_class": str(treated_row.regimen_class),
                    "order": str(treated_row.order),
                    "total_dose_gy": float(treated_row.total_dose_gy),
                    "control_kind": selector["control_kind"],
                    "actual_log_ratio": log_ratio,
                    "actual_tpo14_pct": 100.0 * (1.0 - math.exp(log_ratio)),
                    "reported_tpo14_mean": selector["reported_tpo14_mean"],
                    "treated_log_relative_day14": float(treated_row.log_relative_volume),
                    "control_log_relative_day14": float(control_row.log_relative_volume),
                }
            )
    cohort = pd.DataFrame(rows).drop_duplicates("series_key").sort_values(
        ["year", "date", "family", "total_dose_gy", "order"]
    )
    if cohort.empty:
        raise RuntimeError("Не удалось сформировать ТРО-когорту")
    return cohort.reset_index(drop=True)


def build_contrast_matrix(
    data: pd.DataFrame, cohort: pd.DataFrame
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    day14 = data.loc[np.isclose(data["day"].to_numpy(float), DAY)].copy()
    categories = EARLY.BASE.fixed_categories(data)
    matrix, names = EARLY.build_arm_features(day14, "early_family", categories)
    by_key = {key: index for index, key in enumerate(day14["series_key"].astype(str))}
    contrasts = []
    for row in cohort.itertuples(index=False):
        contrasts.append(matrix[by_key[row.series_key]] - matrix[by_key[row.control_series_key]])
    return np.vstack(contrasts), names, categories


@dataclass
class RobustRidge:
    scaler: StandardScaler
    model: Ridge

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(matrix)), dtype=float)


def fit_robust_ridge(matrix: np.ndarray, target: np.ndarray, alpha: float) -> RobustRidge:
    scaler = StandardScaler(with_mean=False)
    scaled = scaler.fit_transform(matrix)
    weights = np.ones(len(target), dtype=float)
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    for _ in range(ROBUST_ITERATIONS):
        model.fit(scaled, target, sample_weight=weights)
        residual = target - model.predict(scaled)
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        if not np.isfinite(scale) or scale <= 1.0e-8:
            break
        threshold = HUBER_C * scale
        weights = np.ones_like(residual)
        large = np.abs(residual) > threshold
        weights[large] = threshold / np.abs(residual[large])
    return RobustRidge(scaler=scaler, model=model)


def mean_year_rmse(frame: pd.DataFrame, predicted: np.ndarray) -> float:
    work = frame.assign(predicted=np.asarray(predicted, dtype=float))
    work["sq"] = np.square(work["actual_log_ratio"] - work["predicted"])
    return float(work.groupby("year")["sq"].mean().pow(0.5).mean())


def tune_alpha(
    matrix: np.ndarray, cohort: pd.DataFrame, train_indices: np.ndarray
) -> tuple[float, pd.DataFrame]:
    train_years = sorted(cohort.iloc[train_indices]["year"].astype(int).unique())
    rows: list[dict[str, float | int]] = []
    if len(train_years) < 2:
        return float(ALPHA_GRID[len(ALPHA_GRID) // 2]), pd.DataFrame()
    best = (math.inf, float(ALPHA_GRID[0]))
    for alpha in ALPHA_GRID:
        predictions = np.full(len(train_indices), np.nan)
        train_frame = cohort.iloc[train_indices].reset_index(drop=True)
        for year in train_years:
            fit_local = np.flatnonzero(train_frame["year"].to_numpy(int) != year)
            valid_local = np.flatnonzero(train_frame["year"].to_numpy(int) == year)
            fitted = fit_robust_ridge(
                matrix[train_indices[fit_local]],
                train_frame.iloc[fit_local]["actual_log_ratio"].to_numpy(float),
                float(alpha),
            )
            predictions[valid_local] = fitted.predict(matrix[train_indices[valid_local]])
        score = mean_year_rmse(train_frame, predictions)
        rows.append({"alpha": float(alpha), "inner_year_balanced_log_ratio_rmse": score})
        if score < best[0] - 1.0e-12:
            best = (score, float(alpha))
    tuning = pd.DataFrame(rows)
    tuning["selected"] = np.isclose(tuning["alpha"], best[1]).astype(int)
    return best[1], tuning


def outer_splits(cohort: pd.DataFrame, scheme: str) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    years = sorted(cohort["year"].astype(int).unique())
    for year in years:
        if scheme == "leave_one_year_out":
            train = np.flatnonzero(cohort["year"].to_numpy(int) != year)
        elif scheme == "rolling_origin_year":
            train = np.flatnonzero(cohort["year"].to_numpy(int) < year)
            if cohort.iloc[train]["year"].nunique() < 2:
                continue
        else:
            raise ValueError(scheme)
        test = np.flatnonzero(cohort["year"].to_numpy(int) == year)
        if len(train) and len(test):
            yield year, train, test


def run_direct_validation(
    matrix: np.ndarray, cohort: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    target = cohort["actual_log_ratio"].to_numpy(float)
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        for fold_number, (year, train, test) in enumerate(outer_splits(cohort, scheme), start=1):
            alpha, tuning = tune_alpha(matrix, cohort, train)
            fitted = fit_robust_ridge(matrix[train], target[train], alpha)
            predicted = fitted.predict(matrix[test])
            part = cohort.iloc[test].copy()
            part.insert(0, "holdout_year", year)
            part.insert(0, "cv_kind", scheme)
            part["model"] = "direct_tpo14_ridge"
            part["predicted_log_ratio"] = predicted
            part["predicted_tpo14_pct"] = 100.0 * (1.0 - np.exp(predicted))
            predictions.append(part)
            if not tuning.empty:
                tuning.insert(0, "holdout_year", year)
                tuning.insert(0, "cv_kind", scheme)
                tuning_parts.append(tuning)
            fold_rows.append(
                {
                    "cv_kind": scheme,
                    "fold_number": fold_number,
                    "holdout_year": year,
                    "n_training_years": int(cohort.iloc[train]["year"].nunique()),
                    "n_training_series": int(len(train)),
                    "n_test_series": int(len(test)),
                    "selected_alpha": alpha,
                }
            )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(tuning_parts, ignore_index=True) if tuning_parts else pd.DataFrame(),
        pd.DataFrame(fold_rows),
    )


def baseline_predictions(cohort: pd.DataFrame) -> pd.DataFrame:
    raw = read_csv(BASELINE_PREDICTIONS)
    raw = raw.loc[
        raw["model"].astype(str).eq("early_family")
        & np.isclose(raw["day"].to_numpy(float), DAY)
    ].copy()
    indexed = raw.set_index(["cv_kind", "series_key"])
    rows: list[dict[str, object]] = []
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        for row in cohort.itertuples(index=False):
            treated_key = (scheme, row.series_key)
            control_key = (scheme, row.control_series_key)
            if treated_key not in indexed.index or control_key not in indexed.index:
                continue
            treated = indexed.loc[treated_key]
            control = indexed.loc[control_key]
            if isinstance(treated, pd.DataFrame):
                treated = treated.iloc[0]
            if isinstance(control, pd.DataFrame):
                control = control.iloc[0]
            predicted = float(treated.predicted_log_relative - control.predicted_log_relative)
            item = row._asdict()
            item.update(
                {
                    "cv_kind": scheme,
                    "holdout_year": int(row.year),
                    "model": "trajectory_tuned_133",
                    "predicted_log_ratio": predicted,
                    "predicted_tpo14_pct": 100.0 * (1.0 - math.exp(predicted)),
                }
            )
            rows.append(item)
    return pd.DataFrame(rows)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, model), group in predictions.groupby(["cv_kind", "model"], sort=False):
        log_error = group["predicted_log_ratio"] - group["actual_log_ratio"]
        tpo_error = group["predicted_tpo14_pct"] - group["actual_tpo14_pct"]
        denominator = float(
            np.sum(np.square(group["actual_tpo14_pct"] - group["actual_tpo14_pct"].mean()))
        )
        rows.append(
            {
                "cv_kind": scheme,
                "model": model,
                "n_series": int(len(group)),
                "n_years": int(group["year"].nunique()),
                "log_ratio_rmse": float(np.sqrt(np.mean(np.square(log_error)))),
                "tpo14_rmse_pp": float(np.sqrt(np.mean(np.square(tpo_error)))),
                "tpo14_mae_pp": float(np.mean(np.abs(tpo_error))),
                "tpo14_bias_pp": float(np.mean(tpo_error)),
                "tpo14_r2": (
                    float(1.0 - np.sum(np.square(tpo_error)) / denominator)
                    if denominator > 0.0
                    else math.nan
                ),
                "tpo14_spearman": float(
                    group["actual_tpo14_pct"].corr(
                        group["predicted_tpo14_pct"], method="spearman"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for scheme, group in predictions.groupby("cv_kind"):
        pivot = group.pivot(
            index="series_key", columns="model", values=["actual_tpo14_pct", "predicted_tpo14_pct"]
        )
        models = set(pivot["predicted_tpo14_pct"].columns)
        if {"trajectory_tuned_133", "direct_tpo14_ridge"} - models:
            continue
        actual = pivot["actual_tpo14_pct"].iloc[:, 0].to_numpy(float)
        base = pivot["predicted_tpo14_pct"]["trajectory_tuned_133"].to_numpy(float)
        direct = pivot["predicted_tpo14_pct"]["direct_tpo14_ridge"].to_numpy(float)
        delta = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
        for index in range(BOOTSTRAP_ITERATIONS):
            selected = rng.integers(0, len(actual), len(actual))
            base_rmse = np.sqrt(np.mean(np.square(base[selected] - actual[selected])))
            direct_rmse = np.sqrt(np.mean(np.square(direct[selected] - actual[selected])))
            delta[index] = base_rmse - direct_rmse
        rows.append(
            {
                "cv_kind": scheme,
                "n_series": int(len(actual)),
                "baseline_minus_direct_rmse_pp": float(
                    np.sqrt(np.mean(np.square(base - actual)))
                    - np.sqrt(np.mean(np.square(direct - actual)))
                ),
                "bootstrap_p2_5": float(np.quantile(delta, 0.025)),
                "bootstrap_median": float(np.median(delta)),
                "bootstrap_p97_5": float(np.quantile(delta, 0.975)),
                "probability_direct_better": float(np.mean(delta > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(predictions: pd.DataFrame, path: Path) -> None:
    labels = {
        "trajectory_tuned_133": "основная 133-признаковая",
        "direct_tpo14_ridge": "прямая настройка по ТРО₁₄",
    }
    colors = {"trajectory_tuned_133": "#d95f02", "direct_tpo14_ridge": "#2171b5"}
    schemes = ["leave_one_year_out", "rolling_origin_year"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), constrained_layout=True)
    for row_index, scheme in enumerate(schemes):
        part = predictions.loc[predictions["cv_kind"].eq(scheme)]
        axis = axes[row_index, 0]
        limits = [
            float(min(part["actual_tpo14_pct"].min(), part["predicted_tpo14_pct"].min()) - 2),
            float(max(part["actual_tpo14_pct"].max(), part["predicted_tpo14_pct"].max()) + 2),
        ]
        axis.plot(limits, limits, color="#777777", ls="--", lw=1.2)
        for model, group in part.groupby("model"):
            axis.scatter(
                group["actual_tpo14_pct"],
                group["predicted_tpo14_pct"],
                s=45,
                alpha=0.8,
                color=colors[model],
                label=labels[model],
            )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("Наблюдаемое ТРО₁₄, %")
        axis.set_ylabel("Прогнозируемое ТРО₁₄, %")
        axis.set_title("Исключение года" if row_index == 0 else "Прогноз будущих лет")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)

        axis = axes[row_index, 1]
        paired = part.pivot(
            index="series_key", columns="model", values=["actual_tpo14_pct", "predicted_tpo14_pct"]
        )
        actual = paired["actual_tpo14_pct"].iloc[:, 0]
        base_error = np.abs(paired["predicted_tpo14_pct"]["trajectory_tuned_133"] - actual)
        direct_error = np.abs(paired["predicted_tpo14_pct"]["direct_tpo14_ridge"] - actual)
        for index, key in enumerate(paired.index):
            axis.plot([0, 1], [base_error.loc[key], direct_error.loc[key]], color="#bdbdbd", lw=0.8)
        axis.scatter(np.zeros(len(paired)), base_error, color=colors["trajectory_tuned_133"], s=34)
        axis.scatter(np.ones(len(paired)), direct_error, color=colors["direct_tpo14_ridge"], s=34)
        axis.set_xticks([0, 1], ["траектория", "ТРО₁₄"])
        axis.set_ylabel("Абсолютная ошибка ТРО₁₄, п.п.")
        axis.set_title("Посерийное изменение ошибки")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Изменяет ли прямая настройка по ТРО₁₄ перенос на новые эпохи", fontsize=15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    folds: pd.DataFrame,
    feature_count: int,
) -> None:
    lines = [
        "# Настройка прогностической модели по ТРО₁₄",
        "",
        "## Постановка",
        "",
        "Основная модель не изменена. Проверен отдельный endpoint-кандидат: тот же "
        f"{feature_count}-признаковый базис использован для прямой ridge-регрессии "
        "$\\log[R_{опыт}(14)/R_{контроль}(14)]$. Регуляризацию выбирали только внутри "
        "обучающих календарных лет; внешняя проверка исключала год целиком либо обучалась "
        "только на предшествующих годах.",
        "",
        f"В строгую ТРО-когорту вошли {len(cohort)} режимные серии из "
        f"{cohort['year'].nunique()} календарных лет. Семейства: "
        + ", ".join(f"{key} (n={value})" for key, value in cohort['family'].value_counts().items())
        + ".",
        "",
        "## Результаты",
        "",
        "| Проверка | Модель | n | RMSE ТРО₁₄, п.п. | MAE, п.п. | bias, п.п. | R² | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    scheme_labels = {
        "leave_one_year_out": "исключение года",
        "rolling_origin_year": "будущие годы",
    }
    model_labels = {
        "trajectory_tuned_133": "основная траекторная",
        "direct_tpo14_ridge": "прямая ТРО₁₄",
    }
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {scheme_labels[row.cv_kind]} | {model_labels[row.model]} | {row.n_series} | "
            f"{row.tpo14_rmse_pp:.2f} | {row.tpo14_mae_pp:.2f} | {row.tpo14_bias_pp:+.2f} | "
            f"{row.tpo14_r2:.3f} | {row.tpo14_spearman:.3f} |"
        )
    lines.extend(["", "Парный bootstrap разности RMSE (положительное значение — преимущество прямой ТРО-модели):", ""])
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"- {scheme_labels[row.cv_kind]}: {row.baseline_minus_direct_rmse_pp:+.2f} п.п.; "
            f"95% интервал {row.bootstrap_p2_5:+.2f}…{row.bootstrap_p97_5:+.2f}; "
            f"P(прямая модель лучше)={row.probability_direct_better:.3f}."
        )
    lines.extend(
        [
            "",
            "## Решение",
            "",
        ]
    )
    verdicts = []
    for scheme in scheme_labels:
        subset = metrics.loc[metrics["cv_kind"].eq(scheme)].set_index("model")
        if {"trajectory_tuned_133", "direct_tpo14_ridge"}.issubset(subset.index):
            verdicts.append(
                float(subset.loc["direct_tpo14_ridge", "tpo14_rmse_pp"])
                < float(subset.loc["trajectory_tuned_133", "tpo14_rmse_pp"])
            )
    if verdicts and all(verdicts):
        lines.append(
            "Прямая настройка снизила ошибку ТРО₁₄ в обеих внешних схемах. Её можно "
            "сохранить как отдельный endpoint-контур для опытов с contemporaneous control, "
            "но не как замену модели полной траектории."
        )
    else:
        lines.append(
            "Прямая настройка по ТРО₁₄ не дала воспроизводимого выигрыша в обеих внешних "
            "схемах. Основную траекторную модель заменять нельзя; результат сохраняется как "
            "отрицательная проверка чувствительности."
        )
    lines.extend(
        [
            "",
            "Ограничения принципиальны: малая выборка, насыщение большинства значений около "
            "90–99%, отсутствие нейтронов и необходимость иметь сопоставимую контрольную группу. "
            "Поэтому этот анализ не может улучшить априорный прогноз для серии без контроля и не "
            "идентифицирует отдельный биологический механизм.",
            "",
            "## Воспроизводимость",
            "",
            f"Проверено {len(folds)} внешних циклов; сетка ridge alpha: "
            + ", ".join(str(value) for value in ALPHA_GRID)
            + ". Исходный артефакт и прежние результаты не перезаписывались.",
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = read_csv(BASE_DATA)
    endpoints = read_csv(ENDPOINTS)
    cohort = build_contrast_cohort(data, endpoints)
    matrix, names, categories = build_contrast_matrix(data, cohort)
    direct, tuning, folds = run_direct_validation(matrix, cohort)
    baseline = baseline_predictions(cohort)
    combined = pd.concat([baseline, direct], ignore_index=True, sort=False)
    metrics = metric_rows(combined)
    bootstrap = paired_bootstrap(combined)

    write_csv(cohort, OUTPUT_DIR / "tpo14_contrast_cohort.csv")
    write_csv(pd.DataFrame({"feature_index": range(1, len(names) + 1), "feature": names}), OUTPUT_DIR / "feature_dictionary.csv")
    write_csv(combined, OUTPUT_DIR / "outer_tpo14_predictions.csv")
    write_csv(metrics, OUTPUT_DIR / "model_comparison.csv")
    write_csv(bootstrap, OUTPUT_DIR / "paired_rmse_bootstrap.csv")
    write_csv(tuning, OUTPUT_DIR / "inner_tuning.csv")
    write_csv(folds, OUTPUT_DIR / "outer_fold_parameters.csv")
    plot_comparison(combined, OUTPUT_DIR / "tpo14_tuning_comparison.png")
    write_report(cohort, metrics, bootstrap, folds, len(names))
    provenance = {
        "status": "sensitivity_analysis",
        "endpoint": "TPO14",
        "target": "log(R_treated_day14 / R_control_day14)",
        "n_contrast_series": int(len(cohort)),
        "n_years": int(cohort["year"].nunique()),
        "feature_basis": "early_family_133",
        "feature_count": int(len(names)),
        "categories": categories,
        "alpha_grid": list(ALPHA_GRID),
        "outer_validation": ["leave_one_year_out", "rolling_origin_year"],
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "seed": SEED,
        "source_data": str(BASE_DATA),
        "source_endpoints": str(ENDPOINTS),
        "baseline_predictions": str(BASELINE_PREDICTIONS),
        "main_model_replaced": False,
    }
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
