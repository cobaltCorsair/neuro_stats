"""Подбор узла насыщающегося временного множителя вместо его назначения.

Слагаемые воздействия, режима и документированной конфигурации входят в
модель через g_K(t) = min(t/K, 1) с K = 12, что совпадает с рубежом
обновления адаптивного контура. Излом при t = K создаёт структурное
предпочтение минимума в этой точке. Здесь узел развязан с рубежом и включён
в подбор наравне с гребневым штрафом.

Подбор ведётся ТОЛЬКО на обучающей части каждой внешней складки по той же
целевой функции, что и штраф. Согласие по суткам минимума в выборе не
участвует и служит независимой проверкой.
"""

from __future__ import annotations

import importlib.util
import io
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
EARLY_SCRIPT = HERE / "analyze_early_response_extension.py"
OUTPUT_DIR = Path(r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Узел_пандуса")
FIGURE_COPY = Path(r"C:\dev\dissertation_text\figures\figure_ramp_knot.png")

BASELINE_KNOT = 12.0
KNOT_GRID = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0)
ARM = "early_family"          # окончательная 133-признаковая спецификация
SEED = 20260812
BOOTSTRAP_ITERATIONS = 10_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EARLY = load_module("early_ext", EARLY_SCRIPT)
BASE = EARLY.BASE


def retune_knot(
    matrix: np.ndarray, names: Sequence[str], day: np.ndarray, knot: float
) -> np.ndarray:
    """Пересчитать g12-столбцы под другой узел.

    Все такие столбцы имеют вид (нечто) * g12, поэтому точное преобразование —
    деление на старый пандус и умножение на новый. При g12 = 0 (нулевые сутки)
    столбец равен нулю и таким остаётся.
    """
    old = np.minimum(day / BASELINE_KNOT, 1.0)
    new = np.minimum(day / knot, 1.0)
    scale = np.divide(new, old, out=np.zeros_like(old), where=old > 0.0)
    out = matrix.copy()
    for index, name in enumerate(names):
        if "g12" in name:
            out[:, index] = out[:, index] * scale
    return out


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
    knot: float,
    alpha: float,
) -> np.ndarray:
    train_matrix, names = EARLY.build_arm_features(train, ARM, categories)
    test_matrix, _ = EARLY.build_arm_features(test, ARM, categories)
    train_matrix = retune_knot(train_matrix, names, train["day"].to_numpy(float), knot)
    test_matrix = retune_knot(test_matrix, names, test["day"].to_numpy(float), knot)
    fitted = BASE.fit_linear(
        train_matrix,
        train["log_relative_volume"].to_numpy(float),
        BASE.series_balanced_weights(train),
        float(alpha),
        robust=True,
    )
    return fitted.predict(test_matrix)


def inner_select(
    train: pd.DataFrame, categories: Mapping[str, Sequence[str]]
) -> tuple[float, float, pd.DataFrame]:
    """Совместный выбор узла и штрафа по внутренним складкам обучающей части."""
    target = train["log_relative_volume"].to_numpy(float)
    weights = BASE.series_balanced_weights(train)
    folds = BASE.inner_group_splits(train["date"].astype(str).to_numpy())
    base_matrix, names = EARLY.build_arm_features(train, ARM, categories)
    day = train["day"].to_numpy(float)

    rows: list[dict[str, float]] = []
    best = (math.inf, BASELINE_KNOT, float(EARLY.ALPHA_GRID[0]))
    for knot in KNOT_GRID:
        matrix = retune_knot(base_matrix, names, day, knot)
        for alpha in EARLY.ALPHA_GRID:
            oof = np.full(len(train), np.nan)
            for fit_index, valid_index in folds:
                fitted = BASE.fit_linear(
                    matrix[fit_index], target[fit_index], weights[fit_index],
                    float(alpha), robust=True,
                )
                oof[valid_index] = fitted.predict(matrix[valid_index])
            score, periods = EARLY.objective(train, oof)
            rows.append({"knot": knot, "alpha": float(alpha), "objective": score,
                         **{f"{k}_log_rmse": v for k, v in periods.items()}})
            if score < best[0] - 1.0e-12:
                best = (score, knot, float(alpha))
    return best[1], best[2], pd.DataFrame(rows)


def minimum_day_agreement(predictions: pd.DataFrame) -> dict[str, float]:
    part = predictions.loc[(predictions["is_treated"].astype(int) == 1) & (predictions["day"] > 0)]
    observed = part.loc[part.groupby("series_key")["actual_relative_volume"].idxmin()]
    predicted = part.loc[part.groupby("series_key")["predicted_relative_volume"].idxmin()]
    joined = (
        observed.set_index("series_key")["day"].rename("obs")
        .to_frame()
        .join(predicted.set_index("series_key")["day"].rename("prd"))
    )
    return {
        "серий": float(len(joined)),
        "доля на 12-х сутках, %": 100.0 * float((joined["prd"] == 12.0).mean()),
        "доля позже 12-х, %": 100.0 * float((joined["prd"] > 12.0).mean()),
        "Спирмен": float(joined["obs"].corr(joined["prd"], method="spearman")),
        "медиана |набл-прогн|": float((joined["obs"] - joined["prd"]).abs().median()),
        "размах прогноза": f"{joined['prd'].min():.0f}-{joined['prd'].max():.0f}",
    }


def paired_series_table(predictions: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Одна строка на серию для парного сравнения fixed_12 и tuned_knot."""
    part = predictions.loc[
        (predictions["cv_kind"] == scheme)
        & (predictions["is_treated"].astype(int) == 1)
        & (predictions["day"] > 0)
    ]
    rows: list[dict[str, float | str]] = []
    for series_key, series in part.groupby("series_key", sort=True):
        fixed = series.loc[series["model"] == "fixed_12"]
        tuned = series.loc[series["model"] == "tuned_knot"]
        if fixed.empty or tuned.empty:
            continue
        observed_day = float(
            fixed.loc[fixed["actual_relative_volume"].idxmin(), "day"]
        )
        row: dict[str, float | str] = {
            "series_key": str(series_key),
            "observed_day": observed_day,
        }
        for label, frame in (("fixed_12", fixed), ("tuned_knot", tuned)):
            row[f"predicted_day_{label}"] = float(
                frame.loc[frame["predicted_relative_volume"].idxmin(), "day"]
            )
            late = frame.loc[frame["day"].between(13.0, 21.0)]
            row[f"late_rmse_{label}"] = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            late["actual_log_relative"].to_numpy(float)
                            - late["predicted_log_relative"].to_numpy(float)
                        )
                    )
                )
            )
            day21 = frame.loc[np.isclose(frame["day"], 21.0)].iloc[0]
            row["actual_day21"] = float(day21["actual_log_relative"])
            row[f"predicted_day21_{label}"] = float(
                day21["predicted_log_relative"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def ordinary_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    if denominator <= 0.0:
        return math.nan
    return 1.0 - float(np.sum(np.square(actual - predicted))) / denominator


def paired_metrics(table: pd.DataFrame) -> dict[str, float]:
    observed = table["observed_day"].to_numpy(float)
    fixed_day = table["predicted_day_fixed_12"].to_numpy(float)
    tuned_day = table["predicted_day_tuned_knot"].to_numpy(float)
    actual_day21 = table["actual_day21"].to_numpy(float)
    return {
        "spearman_difference": float(
            pd.Series(observed).corr(pd.Series(tuned_day), method="spearman")
            - pd.Series(observed).corr(pd.Series(fixed_day), method="spearman")
        ),
        "late_rmse_difference": float(
            np.mean(
                table["late_rmse_tuned_knot"].to_numpy(float)
                - table["late_rmse_fixed_12"].to_numpy(float)
            )
        ),
        "day21_r2_difference": float(
            ordinary_r2(
                actual_day21,
                table["predicted_day21_tuned_knot"].to_numpy(float),
            )
            - ordinary_r2(
                actual_day21,
                table["predicted_day21_fixed_12"].to_numpy(float),
            )
        ),
    }


def paired_cluster_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    """Парный bootstrap серий; положительный знак всегда означает tuned - fixed."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        table = paired_series_table(predictions, scheme)
        point = paired_metrics(table)
        samples = {metric: [] for metric in point}
        for _ in range(BOOTSTRAP_ITERATIONS):
            indices = rng.integers(0, len(table), len(table))
            metrics = paired_metrics(table.iloc[indices])
            for metric, value in metrics.items():
                if np.isfinite(value):
                    samples[metric].append(value)
        for metric, estimate in point.items():
            values = np.asarray(samples[metric], dtype=float)
            rows.append(
                {
                    "cv_kind": scheme,
                    "metric": metric,
                    "estimate_tuned_minus_fixed": estimate,
                    "ci_2_5": float(np.quantile(values, 0.025)),
                    "ci_97_5": float(np.quantile(values, 0.975)),
                    "probability_above_zero": float(np.mean(values > 0.0)),
                    "iterations": BOOTSTRAP_ITERATIONS,
                    "n_series": len(table),
                }
            )
    return pd.DataFrame(rows)


def reproduction_check(
    data: pd.DataFrame, categories: Mapping[str, Sequence[str]]
) -> dict[str, float | int]:
    matrix, names = EARLY.build_arm_features(data, ARM, categories)
    reproduced = retune_knot(
        matrix, names, data["day"].to_numpy(float), BASELINE_KNOT
    )
    difference = np.abs(reproduced - matrix)
    return {
        "rows": int(len(data)),
        "features": int(matrix.shape[1]),
        "g12_features": int(sum("g12" in name for name in names)),
        "max_absolute_difference": float(np.max(difference)),
        "exact_array_equal": int(np.array_equal(reproduced, matrix)),
    }


def run(data: pd.DataFrame, categories: Mapping[str, Sequence[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    tuning_parts: list[pd.DataFrame] = []

    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        for holdout, train, test in EARLY.outer_folds(data, scheme):
            knot, alpha, tuning = inner_select(train, categories)
            tuning.insert(0, "holdout", holdout)
            tuning.insert(0, "cv_kind", scheme)
            tuning_parts.append(tuning)
            selection_rows.append(
                {"cv_kind": scheme, "holdout": holdout, "knot": knot, "alpha": alpha}
            )
            for label, use_knot, use_alpha in (
                ("fixed_12", BASELINE_KNOT, None),
                ("tuned_knot", knot, alpha),
            ):
                if use_alpha is None:
                    # штраф подбираем при закреплённом узле — как в исходной работе
                    sub = tuning.loc[np.isclose(tuning["knot"], BASELINE_KNOT)]
                    use_alpha = float(sub.loc[sub["objective"].idxmin(), "alpha"])
                predicted = fit_predict(train, test, categories, use_knot, use_alpha)
                prediction_parts.append(
                    EARLY.prediction_frame(
                        test, predicted, model=label, scheme=scheme,
                        holdout=holdout, selected_arm=f"knot={use_knot:g}",
                    )
                )
            print(f"  {scheme} / {holdout}: узел {knot:g}, штраф {alpha:g}")

    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.DataFrame(selection_rows),
        pd.concat(tuning_parts, ignore_index=True),
    )


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, model), part in predictions.groupby(["cv_kind", "model"], sort=False):
        _, periods = EARLY.objective(part, part["predicted_log_relative"].to_numpy(float))
        treated = part.loc[part["is_treated"].astype(int) == 1]
        day21 = treated.loc[treated["day"] == 21.0]
        r2 = BASE.weighted_r2(
            day21["actual_log_relative"].to_numpy(float),
            day21["predicted_log_relative"].to_numpy(float),
        )
        rows.append(
            {
                "схема": scheme,
                "модель": model,
                **{f"СКО {k}": v for k, v in periods.items()},
                "R2 21 сут": r2,
                **minimum_day_agreement(part),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("Загрузка продольных таблиц…")
    _, evaluation, _, _ = BASE.build_longitudinal_tables()
    data = BASE.collapse_animal_daily_to_series(evaluation)
    categories = BASE.fixed_categories(data)
    print(f"серий: {data['series_key'].nunique()}, строк: {len(data)}")
    print(f"сетка узлов: {KNOT_GRID}\n")

    predictions, selection, tuning = run(data, categories)
    summary = summarize(predictions)
    bootstrap = paired_cluster_bootstrap(predictions)
    reproduction = reproduction_check(data, categories)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "knot_predictions.csv", sep=";", index=False, encoding="utf-8-sig")
    selection.to_csv(OUTPUT_DIR / "knot_selection.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(OUTPUT_DIR / "knot_inner_tuning.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "knot_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(
        OUTPUT_DIR / "knot_paired_cluster_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "knot_reproduction_check.json").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== выбранные узлы по складкам ===")
    print(selection.groupby(["cv_kind", "knot"]).size().to_string())
    print("\n=== сводка ===")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== парный кластерный bootstrap: tuned - fixed ===")
    print(bootstrap.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== точное воспроизведение K=12 ===")
    print(json.dumps(reproduction, ensure_ascii=False, indent=2))
    print(f"\nРезультаты: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
