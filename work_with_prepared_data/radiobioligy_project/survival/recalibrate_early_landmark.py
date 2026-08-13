from __future__ import annotations

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


EARLY_SCRIPT = Path(__file__).resolve().with_name("analyze_early_response_extension.py")
V5_DIR = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\prediction_v5_landmark_trend"
)
V5_SCRIPT = V5_DIR / "run_prediction_v5.py"
OUTPUT_DIR = V5_DIR.parent / "prediction_v6_early_landmark"

LANDMARKS = (7, 10, 12, 14)
PRIMARY_LANDMARK = 12
SEED = 20260811
BOOTSTRAP_ITERATIONS = 10000
MAX_ALLOWED_RELATIVE_WORSENING = 0.02


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EARLY = load_module("early_response_extension_v6", EARLY_SCRIPT)
V5 = load_module("landmark_trend_v5_for_v6", V5_SCRIPT)
BASE = EARLY.BASE


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def outer_folds(
    data: pd.DataFrame, scheme: str
) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]:
    yield from EARLY.outer_folds(data, scheme)


def run_internal_validation(
    data: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        folds = list(outer_folds(data, scheme))
        for fold_number, (holdout, train, test) in enumerate(folds, start=1):
            base_test, base_oof, alpha, base_tuning, _ = EARLY.tune_arm(
                "early_family", train, test, categories
            )
            for landmark in LANDMARKS:
                level_params, level_oof, level_tuning = V5.tune_level(
                    train, base_oof, landmark
                )
                trend_params, trend_oof, trend_tuning = V5.tune_trend(
                    train, base_oof, landmark
                )
                level_test = V5.apply_level_update(
                    test,
                    base_test,
                    landmark,
                    float(level_params["shrinkage"]),
                    level_params["bounds"],
                )
                trend_test = V5.apply_trend_update(
                    test,
                    base_test,
                    landmark,
                    float(trend_params["level_weight"]),
                    float(trend_params["slope_weight"]),
                    float(trend_params["horizon_cap"]),
                    trend_params["bounds"],
                )
                for model, test_prediction, oof_prediction in (
                    ("apriori_early_family", base_test, base_oof),
                    ("adaptive_level_early_family", level_test, level_oof),
                    ("adaptive_trend_early_family", trend_test, trend_oof),
                ):
                    q90, q95 = V5.interval_quantiles(train, oof_prediction, landmark)
                    prediction_parts.append(
                        V5.prediction_frame(
                            test,
                            test_prediction,
                            model=model,
                            scheme=scheme,
                            holdout=holdout,
                            landmark=landmark,
                            q90=q90,
                            q95=q95,
                        )
                    )
                for tuning, stage in (
                    (base_tuning, "early_family_base"),
                    (level_tuning, "adaptive_level"),
                    (trend_tuning, "adaptive_trend"),
                ):
                    item = tuning.copy()
                    if "landmark_day" in item.columns:
                        item["landmark_day"] = landmark
                    else:
                        item.insert(0, "landmark_day", landmark)
                    item.insert(0, "tuning_stage", stage)
                    item.insert(0, "holdout", holdout)
                    item.insert(0, "cv_kind", scheme)
                    tuning_parts.append(item)
                fold_rows.append(
                    {
                        "cv_kind": scheme,
                        "fold_number": fold_number,
                        "holdout": holdout,
                        "min_training_year": int(train["year"].astype(int).min()),
                        "max_training_year": int(train["year"].astype(int).max()),
                        "n_training_years": int(train["year"].nunique()),
                        "n_training_series": int(train["series_key"].nunique()),
                        "n_test_series": int(test["series_key"].nunique()),
                        "landmark_day": landmark,
                        "early_family_alpha": float(alpha),
                        "level_weight": float(level_params["shrinkage"]),
                        "trend_level_weight": float(trend_params["level_weight"]),
                        "trend_slope_weight": float(trend_params["slope_weight"]),
                        "trend_horizon_cap": float(trend_params["horizon_cap"]),
                    }
                )
            print(
                f"{scheme}: {fold_number}/{len(folds)} {holdout}", flush=True
            )
    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(tuning_parts, ignore_index=True),
        pd.DataFrame(fold_rows),
    )


def v6_metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    renamed = predictions.copy()
    renamed["model"] = renamed["model"].replace(
        {
            "apriori_early_family": "apriori_pole_gamma",
            "adaptive_level_early_family": "adaptive_level",
            "adaptive_trend_early_family": "adaptive_trend",
        }
    )
    summary = V5.metric_summary(renamed)
    summary["model"] = summary["model"].replace(
        {
            "apriori_pole_gamma": "apriori_early_family",
            "adaptive_level": "adaptive_level_early_family",
            "adaptive_trend": "adaptive_trend_early_family",
        }
    )
    return summary


def compare_with_v5(v6_summary: pd.DataFrame) -> pd.DataFrame:
    v5 = pd.read_csv(V5_DIR / "model_comparison_summary.csv", sep=";")
    v5 = v5.loc[v5["landmark_day"].astype(int) == PRIMARY_LANDMARK].copy()
    v5["version"] = "v5_base97"
    v6 = v6_summary.loc[
        v6_summary["landmark_day"].astype(int) == PRIMARY_LANDMARK
    ].copy()
    v6["model"] = v6["model"].replace(
        {
            "apriori_early_family": "apriori_pole_gamma",
            "adaptive_level_early_family": "adaptive_level",
            "adaptive_trend_early_family": "adaptive_trend",
        }
    )
    v6["version"] = "v6_early133"
    columns = [
        "version",
        "cv_kind",
        "model",
        "landmark_day",
        "n_series",
        "mean_series_log_rmse",
        "day21_log_rmse",
        "day21_log_r2",
        "coverage90_day21",
        "coverage95_day21",
    ]
    return pd.concat([v5[columns], v6[columns]], ignore_index=True)


def paired_version_bootstrap(
    v6_predictions: pd.DataFrame,
    *,
    model_v5: str = "adaptive_trend",
    model_v6: str = "adaptive_trend_early_family",
) -> pd.DataFrame:
    v5 = pd.read_csv(V5_DIR / "outer_predictions.csv", sep=";")
    v5 = v5.loc[
        (v5["landmark_day"].astype(int) == PRIMARY_LANDMARK)
        & (v5["model"] == model_v5)
        & (v5["is_treated"].astype(int) == 1)
        & (v5["day"].to_numpy(float) > PRIMARY_LANDMARK)
    ].copy()
    v6 = v6_predictions.loc[
        (v6_predictions["landmark_day"].astype(int) == PRIMARY_LANDMARK)
        & (v6_predictions["model"] == model_v6)
        & (v6_predictions["is_treated"].astype(int) == 1)
        & (v6_predictions["day"].to_numpy(float) > PRIMARY_LANDMARK)
    ].copy()
    keys = ["cv_kind", "holdout", "series_key", "day"]
    v5["holdout"] = v5["holdout"].astype(str)
    v6["holdout"] = v6["holdout"].astype(str)
    paired = v5[
        keys + ["actual_log_relative", "predicted_log_relative"]
    ].merge(
        v6[keys + ["predicted_log_relative"]],
        on=keys,
        suffixes=("_v5", "_v6"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for scheme, group in paired.groupby("cv_kind", sort=False):
        per_series_rows: list[dict[str, object]] = []
        for (holdout, series_key), series in group.groupby(
            ["holdout", "series_key"], sort=False
        ):
            actual = series["actual_log_relative"].to_numpy(float)
            rmse_v5 = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            actual - series["predicted_log_relative_v5"].to_numpy(float)
                        )
                    )
                )
            )
            rmse_v6 = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            actual - series["predicted_log_relative_v6"].to_numpy(float)
                        )
                    )
                )
            )
            per_series_rows.append(
                {
                    "holdout": str(holdout),
                    "series_key": str(series_key),
                    "difference": rmse_v5 - rmse_v6,
                }
            )
        differences = pd.DataFrame(per_series_rows)
        clusters = differences["holdout"].unique()
        samples: list[float] = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            sampled = rng.choice(clusters, len(clusters), replace=True)
            values = [
                differences.loc[
                    differences["holdout"] == cluster, "difference"
                ].to_numpy(float)
                for cluster in sampled
            ]
            samples.append(float(np.concatenate(values).mean()))
        samples_array = np.asarray(samples)
        rows.append(
            {
                "cv_kind": scheme,
                "n_series": int(len(differences)),
                "n_year_clusters": int(len(clusters)),
                "v5_minus_v6_mean_series_future_log_rmse": float(
                    differences["difference"].mean()
                ),
                "cluster_bootstrap_ci95_low": float(
                    np.quantile(samples_array, 0.025)
                ),
                "cluster_bootstrap_ci95_high": float(
                    np.quantile(samples_array, 0.975)
                ),
                "cluster_bootstrap_probability_v6_improves": float(
                    np.mean(samples_array > 0.0)
                ),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            }
        )
    return pd.DataFrame(rows)


def migration_decision(comparison: pd.DataFrame) -> dict[str, object]:
    target = comparison.loc[comparison["model"] == "adaptive_trend"].copy()
    pivot = target.pivot(index="cv_kind", columns="version")
    scheme_rows: list[dict[str, object]] = []
    for scheme in target["cv_kind"].unique():
        old = target.loc[
            (target["cv_kind"] == scheme) & (target["version"] == "v5_base97")
        ].iloc[0]
        new = target.loc[
            (target["cv_kind"] == scheme) & (target["version"] == "v6_early133")
        ].iloc[0]
        future_change = (
            new["mean_series_log_rmse"] / old["mean_series_log_rmse"] - 1.0
        )
        endpoint_change = new["day21_log_rmse"] / old["day21_log_rmse"] - 1.0
        passes = bool(
            future_change <= MAX_ALLOWED_RELATIVE_WORSENING
            and endpoint_change <= MAX_ALLOWED_RELATIVE_WORSENING
        )
        scheme_rows.append(
            {
                "cv_kind": scheme,
                "future_log_rmse_relative_change": float(future_change),
                "day21_log_rmse_relative_change": float(endpoint_change),
                "passes_noninferiority_rule": passes,
            }
        )
    accepted = all(row["passes_noninferiority_rule"] for row in scheme_rows)
    return {
        "selected_apriori_base": "early_family_133",
        "selected_adaptive_model": (
            "adaptive_trend_early_family" if accepted else "adaptive_trend_base97"
        ),
        "landmark_day": PRIMARY_LANDMARK,
        "migration_accepted": accepted,
        "maximum_allowed_relative_worsening": MAX_ALLOWED_RELATIVE_WORSENING,
        "scheme_checks": scheme_rows,
        "external_validation": False,
    }


def score_external(
    data: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    external_animals, sources = V5.external_animal_grid()
    external = BASE.collapse_animal_daily_to_series(external_animals)
    base_test, base_oof, alpha, _, _ = EARLY.tune_arm(
        "early_family", data, external, categories
    )
    level_params, level_oof, _ = V5.tune_level(data, base_oof, PRIMARY_LANDMARK)
    trend_params, trend_oof, _ = V5.tune_trend(data, base_oof, PRIMARY_LANDMARK)
    level_test = V5.apply_level_update(
        external,
        base_test,
        PRIMARY_LANDMARK,
        float(level_params["shrinkage"]),
        level_params["bounds"],
    )
    trend_test = V5.apply_trend_update(
        external,
        base_test,
        PRIMARY_LANDMARK,
        float(trend_params["level_weight"]),
        float(trend_params["slope_weight"]),
        float(trend_params["horizon_cap"]),
        trend_params["bounds"],
    )
    parts: list[pd.DataFrame] = []
    for model, predicted, oof in (
        ("apriori_early_family", base_test, base_oof),
        ("adaptive_level_early_family", level_test, level_oof),
        ("adaptive_trend_early_family", trend_test, trend_oof),
    ):
        q90, q95 = V5.interval_quantiles(data, oof, PRIMARY_LANDMARK)
        parts.append(
            V5.prediction_frame(
                external,
                predicted,
                model=model,
                scheme="external_june_2026_diagnostic",
                holdout="2026-06",
                landmark=PRIMARY_LANDMARK,
                q90=q90,
                q95=q95,
            )
        )
    parameters = pd.DataFrame(
        [
            {"model": "early_family", "alpha": alpha},
            {
                "model": "adaptive_level_early_family",
                "level_weight": level_params["shrinkage"],
                "level_lower": level_params["bounds"][0],
                "level_upper": level_params["bounds"][1],
            },
            {
                "model": "adaptive_trend_early_family",
                "level_weight": trend_params["level_weight"],
                "slope_weight": trend_params["slope_weight"],
                "horizon_cap": trend_params["horizon_cap"],
                "level_lower": trend_params["bounds"]["level"][0],
                "level_upper": trend_params["bounds"]["level"][1],
                "slope_lower": trend_params["bounds"]["slope"][0],
                "slope_upper": trend_params["bounds"]["slope"][1],
            },
        ]
    )
    return pd.concat(parts, ignore_index=True), parameters, pd.DataFrame(sources)


def external_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    renamed = predictions.copy()
    renamed["model"] = renamed["model"].replace(
        {
            "apriori_early_family": "apriori_pole_gamma",
            "adaptive_level_early_family": "adaptive_level",
            "adaptive_trend_early_family": "adaptive_trend",
        }
    )
    summary = V5.external_summary(renamed)
    summary["model"] = summary["model"].replace(
        {
            "apriori_pole_gamma": "apriori_early_family",
            "adaptive_level": "adaptive_level_early_family",
            "adaptive_trend": "adaptive_trend_early_family",
        }
    )
    return summary


def final_adaptive_parameters(
    data: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    _, base_oof, alpha, _, names = EARLY.tune_arm(
        "early_family", data, data, categories
    )
    trend_params, _, tuning = V5.tune_trend(data, base_oof, PRIMARY_LANDMARK)
    row = {
        "base_model": "early_family",
        "n_features": len(names),
        "ridge_alpha": float(alpha),
        "landmark_day": PRIMARY_LANDMARK,
        "recent_window_start_day": PRIMARY_LANDMARK - V5.RECENT_WINDOW_DAYS,
        "recent_window_end_day": PRIMARY_LANDMARK,
        "level_weight": float(trend_params["level_weight"]),
        "slope_weight": float(trend_params["slope_weight"]),
        "horizon_cap_days": float(trend_params["horizon_cap"]),
        "level_lower": float(trend_params["bounds"]["level"][0]),
        "level_upper": float(trend_params["bounds"]["level"][1]),
        "slope_lower": float(trend_params["bounds"]["slope"][0]),
        "slope_upper": float(trend_params["bounds"]["slope"][1]),
    }
    return tuning, row


def make_figures(comparison: pd.DataFrame, external: pd.DataFrame) -> None:
    models = ("apriori_pole_gamma", "adaptive_level", "adaptive_trend")
    labels = {
        "apriori_pole_gamma": "априорный",
        "adaptive_level": "уровень",
        "adaptive_trend": "уровень + наклон",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for axis, scheme in zip(axes, ("leave_one_year_out", "rolling_origin_year")):
        subset = comparison.loc[comparison["cv_kind"] == scheme]
        x = np.arange(len(models))
        width = 0.36
        for offset, version, label, color in (
            (-width / 2, "v5_base97", "основа 97", "#8c8c8c"),
            (width / 2, "v6_early133", "основа 133", "#d95f02"),
        ):
            rows = subset.loc[subset["version"] == version].set_index("model")
            axis.bar(
                x + offset,
                [rows.loc[model, "mean_series_log_rmse"] for model in models],
                width,
                label=label,
                color=color,
            )
        axis.set_xticks(x, [labels[model] for model in models], rotation=15)
        axis.set_title(
            "Исключение года"
            if scheme == "leave_one_year_out"
            else "Только предшествующие годы"
        )
        axis.set_ylabel("средняя посерийная log-СКО после 12-х суток")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"figure_1_v5_v6_comparison.{suffix}", dpi=240)
    plt.close(fig)

    series = sorted(external["series_key"].unique())
    fig, axes = plt.subplots(1, len(series), figsize=(12.0, 4.8))
    if len(series) == 1:
        axes = [axes]
    colors = {
        "apriori_early_family": "#2878b5",
        "adaptive_level_early_family": "#d95f02",
        "adaptive_trend_early_family": "#1b9e77",
    }
    for axis, series_key in zip(axes, series):
        group = external.loc[external["series_key"] == series_key]
        observed = group.loc[group["model"] == "apriori_early_family"].sort_values("day")
        axis.plot(
            observed["day"],
            observed["actual_relative_volume"],
            "o-",
            color="black",
            label="наблюдение",
        )
        for model, color in colors.items():
            rows = group.loc[group["model"] == model].sort_values("day")
            axis.plot(
                rows["day"],
                rows["predicted_relative_volume"],
                linewidth=2.0,
                color=color,
                label=model,
            )
        axis.axvline(PRIMARY_LANDMARK, color="#555555", linestyle=":")
        axis.set_title(series_key.split("|")[0])
        axis.set_xlabel("сутки")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("относительный объём V/V₀")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"figure_2_external_v6.{suffix}", dpi=240)
    plt.close(fig)


def build_report(
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decision: Mapping[str, object],
    external: pd.DataFrame,
) -> str:
    lines = [
        "# Адаптивный прогноз на 133-признаковой ранней основе",
        "",
        "## Назначение",
        "",
        "Версия 6 переносит прежний алгоритм обновления уровня и раннего наклона "
        "на выбранную 133-признаковую априорную модель. Версия 5 на 97 признаках "
        "сохранена без изменений и использована как непосредственный контроль.",
        "",
        "Рубеж 12 суток сохранён как ранее установленный компромисс между точностью "
        "и девятисуточным упреждением; рубежи 7, 10, 12 и 14 суток повторно "
        "проверены на новой основе. Все коэффициенты обновления выбирались только "
        "на обучающей части каждого внешнего цикла.",
        "",
        "## Сравнение при рубеже 12 суток",
        "",
    ]
    for scheme in ("leave_one_year_out", "rolling_origin_year"):
        rows = comparison.loc[
            (comparison["cv_kind"] == scheme)
            & (comparison["model"] == "adaptive_trend")
        ].set_index("version")
        old = rows.loc["v5_base97"]
        new = rows.loc["v6_early133"]
        boot = bootstrap.loc[bootstrap["cv_kind"] == scheme].iloc[0]
        lines.extend(
            [
                f"### {scheme}",
                "",
                f"- Средняя посерийная log-СКО после рубежа: "
                f"{old['mean_series_log_rmse']:.3f} → {new['mean_series_log_rmse']:.3f}.",
                f"- Log-СКО 21-х суток: {old['day21_log_rmse']:.3f} → "
                f"{new['day21_log_rmse']:.3f}.",
                f"- $R^2$ 21-х суток: {old['day21_log_r2']:.3f} → "
                f"{new['day21_log_r2']:.3f}.",
                f"- Bootstrap-разность ошибки v5−v6: "
                f"{boot['v5_minus_v6_mean_series_future_log_rmse']:.3f}; 95% ДИ "
                f"{boot['cluster_bootstrap_ci95_low']:.3f}…"
                f"{boot['cluster_bootstrap_ci95_high']:.3f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Решение",
            "",
            f"Миграция адаптивного контура: "
            f"{'принята' if decision['migration_accepted'] else 'не принята'}.",
            f"Основная априорная основа: `{decision['selected_apriori_base']}`.",
            f"Основной адаптивный вариант: `{decision['selected_adaptive_model']}`.",
            "",
            "## Июнь 2026 года",
            "",
            "Две серии июня 2026 года были просмотрены до разработки версии 6, "
            "поэтому их повторный расчёт является диагностикой, а не новой "
            "независимой проверкой.",
            "",
            V5.markdown_table(external),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, evaluation, _, _ = BASE.build_longitudinal_tables()
    data = BASE.collapse_animal_daily_to_series(evaluation)
    categories = BASE.fixed_categories(data)

    predictions, tuning, folds = run_internal_validation(data, categories)
    summary = v6_metric_summary(predictions)
    comparison = compare_with_v5(summary)
    bootstrap = paired_version_bootstrap(predictions)
    decision = migration_decision(comparison)
    external_predictions, external_parameters, external_sources = score_external(
        data, categories
    )
    external_metrics = external_summary(external_predictions)
    final_tuning, final_parameters = final_adaptive_parameters(data, categories)

    write_csv(predictions, OUTPUT_DIR / "outer_predictions.csv")
    write_csv(tuning, OUTPUT_DIR / "inner_tuning.csv")
    write_csv(folds, OUTPUT_DIR / "outer_fold_parameters.csv")
    write_csv(summary, OUTPUT_DIR / "model_comparison_summary.csv")
    write_csv(comparison, OUTPUT_DIR / "v5_v6_comparison.csv")
    write_csv(bootstrap, OUTPUT_DIR / "v5_v6_cluster_bootstrap.csv")
    write_csv(external_predictions, OUTPUT_DIR / "external_predictions.csv")
    write_csv(external_metrics, OUTPUT_DIR / "external_summary.csv")
    write_csv(external_parameters, OUTPUT_DIR / "external_fit_parameters.csv")
    write_csv(external_sources, OUTPUT_DIR / "external_source_manifest.csv")
    write_csv(final_tuning, OUTPUT_DIR / "final_adaptive_tuning.csv")
    write_csv(pd.DataFrame([final_parameters]), OUTPUT_DIR / "final_adaptive_parameters.csv")
    (OUTPUT_DIR / "deployment_choice.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = {
        "version": "prediction_v6_early_landmark",
        "apriori_model": "early_family_133",
        "apriori_parameter_file": str(
            EARLY.OUTPUT_DIR / "final_model_parameters.csv"
        ),
        "adaptive_parameter_file": str(
            OUTPUT_DIR / "final_adaptive_parameters.csv"
        ),
        "landmark_observations": "days_8_to_12_inclusive",
        "prediction_horizon": "days_13_to_21",
        "validation": ["leave_one_year_out", "rolling_origin_year"],
        "external_validation": False,
        "random_seed": SEED,
    }
    (OUTPUT_DIR / "model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figures(comparison, external_predictions)
    (OUTPUT_DIR / "REPORT.md").write_text(
        build_report(comparison, bootstrap, decision, external_metrics),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(external_metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
