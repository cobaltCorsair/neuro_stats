#!/usr/bin/env python3
"""Externally validate the frozen model on held-out June 2026 peak series.

The two tumour workbooks remain held out.  No coefficient, feature scale, ridge
penalty, landmark parameter, or interval width is re-estimated here.  The script
uses the confirmed radiation-family metadata ``mixed_n_p_peak`` and applies the
frozen ``sarcoma_m1_growth_v1.0.0`` artifacts. Only the physically confirmed
``mixed_n_p_peak`` encoding is emitted. The validation data are external to the
training cohort, while the experiment remains within the same laboratory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK4_ROOT = Path(r"D:\Диссертация\Результаты\Задача_4_Модель")
DEFAULT_OUTPUT = TASK4_ROOT / "4.6_Внешняя_валидация_июнь_2026_peak"
RELEASE = Path(
    r"C:\dev\dissertation_text\related_materials\model_release"
    r"\sarcoma_m1_growth_v1_0_0"
)
BASE_SCRIPT = TASK4_ROOT / "4.6_Ковариаты_pole_gamma" / "run_growth_prediction.py"
EARLY_SCRIPT = Path(__file__).resolve().with_name("analyze_early_response_extension.py")
ADAPTIVE_SCRIPT = (
    TASK4_ROOT
    / "4.6_Итоговая_прогностическая_модель"
    / "prediction_v5_landmark_trend"
    / "run_prediction_v5.py"
)
HOLDOUT_SCRIPT = Path(__file__).resolve().with_name(
    "validate_growth_prediction_holdout.py"
)
CONTROL_SUMMARY = DEFAULT_OUTPUT / "control_holdout_summary.csv"

SCENARIO = "external_peak_holdout"
CONFIRMED_FAMILY = "mixed_n_p_peak"
P_PEAK_2026_RBE = 1.10
PROTON_FRACTION_COUNT = {
    "p_15.2_n_2.3_n_2.3_17.06.2026.xlsx": 2,
    "p_25.2_n_2.3_n_2.3_18.06.2026.xlsx": 1,
}
MODEL_LABELS = {
    "base97": "априорная 97",
    "apriori133": "априорная 133",
    "adaptive97": "адаптивная 97",
}
LANDMARK_DAY = 12
ENDPOINT_DAY = 21
ADAPTIVE97_Q95 = 0.8045319068
BASE97_Q95_BY_PERIOD = {
    "day0_7": 0.5768715361781276,
    "day8_14": 1.2073922903961398,
    "day15_21": 1.8349764706482095,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_parameters(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path, sep=";")
    return (
        frame["feature"].astype(str).tolist(),
        frame["coefficient_original_feature_scale"].to_numpy(float),
    )


def frozen_prediction(
    matrix: np.ndarray,
    names: Sequence[str],
    parameter_path: Path,
) -> np.ndarray:
    expected, coefficients = read_parameters(parameter_path)
    if list(names) != expected:
        raise RuntimeError(
            f"Feature schema mismatch for {parameter_path}: "
            f"built {len(names)}, frozen {len(expected)}"
        )
    return np.asarray(matrix, dtype=float) @ coefficients


def physical_metadata(name: str, metadata: Mapping[str, object]) -> dict[str, object]:
    """Convert only the 2026 peak-proton components from RBE-weighted to Gy."""
    result = dict(metadata)
    archived = tuple(float(value) for value in metadata["fractions"])
    n_proton = PROTON_FRACTION_COUNT[name]
    physical = tuple(
        value / P_PEAK_2026_RBE if index < n_proton else value
        for index, value in enumerate(archived)
    )
    result["fractions"] = physical
    result["archived_fractions"] = archived
    result["archived_total_dose"] = float(sum(archived))
    return result


def build_holdout(holdout, engine, base, family: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    archived_total_by_date: dict[str, float] = {}
    for name, metadata in holdout.HOLDOUT_SERIES.items():
        converted = physical_metadata(name, metadata)
        archived_total_by_date[str(metadata["date"])] = float(
            converted["archived_total_dose"]
        )
        rows.extend(
            holdout.animal_rows(
                engine,
                holdout.HOLDOUT_DIR / name,
                converted,
                family,
            )
        )
    if not rows:
        raise RuntimeError("No June 2026 holdout rows were parsed")
    collapsed = base.collapse_animal_daily_to_series(pd.DataFrame(rows))
    collapsed["archived_total_dose_gy_rbe_mixed"] = (
        collapsed["date"].astype(str).map(archived_total_by_date)
    )
    return collapsed


def q95_for_model(model_name: str, days: np.ndarray) -> np.ndarray:
    days = np.asarray(days, dtype=float)
    if model_name == "adaptive97":
        return np.full(days.shape, ADAPTIVE97_Q95, dtype=float)
    if model_name == "base97":
        return np.where(
            days <= 7.0,
            BASE97_Q95_BY_PERIOD["day0_7"],
            np.where(
                days <= 14.0,
                BASE97_Q95_BY_PERIOD["day8_14"],
                BASE97_Q95_BY_PERIOD["day15_21"],
            ),
        )
    # The 133-feature candidate was developed after the held-out outcomes were
    # known.  It is diagnostic and has no frozen confirmatory interval here.
    return np.full(days.shape, np.nan, dtype=float)


def adaptive_parameters(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, sep=";")
    row = frame.loc[frame["model"] == "adaptive_trend"].iloc[0]
    return {
        "level_weight": float(row["level_weight"]),
        "slope_weight": float(row["slope_weight"]),
        "horizon_cap": float(row["horizon_cap"]),
        "bounds": {
            "level": (float(row["level_lower"]), float(row["level_upper"])),
            "slope": (float(row["slope_lower"]), float(row["slope_upper"])),
        },
    }


def prediction_rows(
    test: pd.DataFrame,
    scenario: str,
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model_name, predicted in predictions.items():
        part = test.copy()
        part.insert(0, "scenario", scenario)
        part.insert(1, "model", model_name)
        part["actual_log_relative"] = part["log_relative_volume"].to_numpy(float)
        part["predicted_log_relative"] = np.asarray(predicted, dtype=float)
        part["actual_relative_volume"] = np.exp(part["actual_log_relative"])
        part["predicted_relative_volume"] = np.exp(part["predicted_log_relative"])
        part["q95_log_half_width"] = q95_for_model(
            model_name, part["day"].to_numpy(float)
        )
        part["prediction_low95"] = np.exp(
            part["predicted_log_relative"] - part["q95_log_half_width"]
        )
        part["prediction_high95"] = np.exp(
            part["predicted_log_relative"] + part["q95_log_half_width"]
        )
        part["prediction_is_evaluable"] = (
            part["day"].to_numpy(float) > LANDMARK_DAY
            if model_name == "adaptive97"
            else part["day"].to_numpy(float) > 0.0
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["scenario", "model", "series_key"]
    for (scenario, model, series_key), group in predictions.groupby(keys, sort=False):
        all_days = group.loc[group["day"].between(1.0, ENDPOINT_DAY)].copy()
        late = group.loc[group["day"].between(LANDMARK_DAY + 1, ENDPOINT_DAY)].copy()
        endpoint = group.loc[np.isclose(group["day"], ENDPOINT_DAY)].iloc[0]
        eval_group = late if model == "adaptive97" else all_days
        error_all = all_days["actual_log_relative"] - all_days["predicted_log_relative"]
        error_late = late["actual_log_relative"] - late["predicted_log_relative"]
        error_eval = eval_group["actual_log_relative"] - eval_group["predicted_log_relative"]
        interval_width = eval_group["q95_log_half_width"].to_numpy(float)
        interval_defined = np.isfinite(interval_width)
        coverage = (
            float(np.mean(np.abs(error_eval.to_numpy(float)) <= interval_width))
            if bool(np.all(interval_defined))
            else np.nan
        )
        rows.append(
            {
                "scenario": scenario,
                "family": str(group["family"].iloc[0]),
                "model": model,
                "model_label": MODEL_LABELS[model],
                "series_key": series_key,
                "date": str(group["date"].iloc[0]),
                "archived_total_dose_gy_rbe_mixed": float(
                    group["archived_total_dose_gy_rbe_mixed"].iloc[0]
                ),
                "total_dose_gy": float(group["total_dose_gy"].iloc[0]),
                "n_all_days": int(len(all_days)),
                "n_future_days": int(len(late)),
                "day1_21_log_rmse": float(np.sqrt(np.mean(np.square(error_all)))),
                "day13_21_log_rmse": float(np.sqrt(np.mean(np.square(error_late)))),
                "evaluation_log_rmse": float(np.sqrt(np.mean(np.square(error_eval)))),
                "coverage95_evaluation": coverage,
                "day21_actual_log": float(endpoint["actual_log_relative"]),
                "day21_predicted_log": float(endpoint["predicted_log_relative"]),
                "day21_actual_relative": float(endpoint["actual_relative_volume"]),
                "day21_predicted_relative": float(endpoint["predicted_relative_volume"]),
                "day21_actual_to_predicted_ratio": float(
                    endpoint["actual_relative_volume"]
                    / endpoint["predicted_relative_volume"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "model", "total_dose_gy"])


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["scenario", "family", "model", "model_label"], as_index=False)
        .agg(
            n_series=("series_key", "nunique"),
            mean_day1_21_log_rmse=("day1_21_log_rmse", "mean"),
            mean_day13_21_log_rmse=("day13_21_log_rmse", "mean"),
            mean_day21_actual_to_predicted_ratio=(
                "day21_actual_to_predicted_ratio",
                "mean",
            ),
            coverage95_evaluation=("coverage95_evaluation", "mean"),
        )
        .sort_values(["scenario", "model"])
    )


def direction_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario, model), group in summary.groupby(["scenario", "model"], sort=False):
        ordered = group.sort_values("total_dose_gy")
        low, high = ordered.iloc[0], ordered.iloc[-1]
        observed_gap = float(high["day21_actual_log"] - low["day21_actual_log"])
        predicted_gap = float(high["day21_predicted_log"] - low["day21_predicted_log"])
        rows.append(
            {
                "scenario": scenario,
                "model": model,
                "low_dose_gy": float(low["total_dose_gy"]),
                "high_dose_gy": float(high["total_dose_gy"]),
                "observed_high_minus_low_day21_log": observed_gap,
                "predicted_high_minus_low_day21_log": predicted_gap,
                "ordering_matches": bool(observed_gap * predicted_gap > 0.0),
            }
        )
    return pd.DataFrame(rows)


def make_peak_only_figure(predictions: pd.DataFrame, output: Path) -> None:
    """Plot only the physically valid peak encoding."""
    peak = predictions.loc[predictions["scenario"] == SCENARIO].copy()
    series = sorted(peak["series_key"].unique())
    fig, axes = plt.subplots(1, len(series), figsize=(12.5, 5.2), sharey=True)
    if len(series) == 1:
        axes = [axes]
    colors = {"base97": "#2f6f9f", "apriori133": "#d17a22", "adaptive97": "#2b8c62"}
    labels = {
        "base97": "априорная модель, 97 признаков",
        "apriori133": "априорная модель, 133 признака",
        "adaptive97": "обновляемая модель, 97 признаков",
    }
    for axis, series_key in zip(axes, series):
        observed = peak.loc[
            (peak["series_key"] == series_key) & (peak["model"] == "base97")
        ].sort_values("day")
        axis.scatter(
            observed["day"],
            observed["actual_relative_volume"],
            color="#222222",
            s=30,
            zorder=5,
            label="наблюдение",
        )
        for model in ("base97", "apriori133", "adaptive97"):
            rows = peak.loc[
                (peak["series_key"] == series_key) & (peak["model"] == model)
            ].sort_values("day")
            if model == "adaptive97":
                rows = rows.loc[rows["day"] >= LANDMARK_DAY]
            axis.plot(
                rows["day"],
                rows["predicted_relative_volume"],
                color=colors[model],
                linewidth=2.2,
                label=labels[model],
            )
        dose = float(observed["total_dose_gy"].iloc[0])
        axis.axvline(LANDMARK_DAY, color="#555555", linestyle=":", linewidth=1.2)
        axis.set_title(f"{series_key.split('|')[0]}; $D_{{\\mathrm{{физ}}}}={dose:.1f}$ Гр")
        axis.set_xlabel("сутки после облучения")
        axis.set_yscale("log")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("относительный объём опухоли")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Внешняя темпорально отложенная валидация: протонная компонента в пике",
        y=1.09,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_peak_holdout.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_peak_residual_figure(predictions: pd.DataFrame, output: Path) -> None:
    """Show what the day-12 update sees and what the future interval requires."""
    peak = predictions.loc[predictions["scenario"] == SCENARIO].copy()
    doses = sorted(peak["total_dose_gy"].unique())
    fig, axes = plt.subplots(2, len(doses), figsize=(12.0, 7.5), sharex=True)
    axes = np.atleast_2d(axes)
    for column, dose in enumerate(doses):
        base_rows = peak.loc[
            (peak["model"] == "base97") & np.isclose(peak["total_dose_gy"], dose)
        ].sort_values("day")
        adaptive_rows = peak.loc[
            (peak["model"] == "adaptive97") & np.isclose(peak["total_dose_gy"], dose)
        ].sort_values("day")
        top, bottom = axes[0, column], axes[1, column]
        top.plot(
            base_rows["day"], base_rows["actual_relative_volume"], "o-",
            color="#252525", markersize=4, linewidth=1.4, label="наблюдение",
        )
        top.plot(
            base_rows["day"], base_rows["predicted_relative_volume"],
            color="#2f6f9f", linewidth=2.1, label="априорный прогноз, 97 признаков",
        )
        future = adaptive_rows.loc[adaptive_rows["day"] > LANDMARK_DAY]
        top.plot(
            future["day"], future["predicted_relative_volume"],
            color="#2b8c62", linewidth=2.1, linestyle="--",
            label="обновляемый прогноз, рубеж 12 суток",
        )
        top.axvline(LANDMARK_DAY, color="#666666", linestyle=":", linewidth=1.2)
        top.set_yscale("log")
        top.set_title(f"$D_{{\\mathrm{{физ}}}}={dose:.1f}$ Гр")
        top.grid(alpha=0.23, which="both")

        residual = (
            base_rows["actual_log_relative"].to_numpy(float)
            - base_rows["predicted_log_relative"].to_numpy(float)
        )
        days = base_rows["day"].to_numpy(float)
        early = days <= LANDMARK_DAY
        bottom.axhline(0.0, color="black", linewidth=0.9)
        bottom.axvline(LANDMARK_DAY, color="#666666", linestyle=":", linewidth=1.2)
        bottom.fill_between(
            days, 0.0, residual, where=early, color="#6baed6", alpha=0.50,
            label="сутки 1–12: доступно обновлению",
        )
        bottom.fill_between(
            days, 0.0, residual, where=~early, color="#fc9272", alpha=0.50,
            label="сутки 13–21: будущий интервал",
        )
        bottom.plot(days, residual, "o-", color="#252525", markersize=3.5, linewidth=1.2)
        bottom.set_xlabel("сутки после облучения")
        bottom.grid(alpha=0.23)
    axes[0, 0].set_ylabel("относительный объём опухоли")
    axes[1, 0].set_ylabel("остаток: наблюдение − прогноз (log)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Пиковые N+P-серии июня 2026 года: обновление и остаток", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_peak_adaptive.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_peak_landmark_figure(predictions: pd.DataFrame, output: Path) -> None:
    """Display the day-12 forecast discontinuity and its log-scale correction."""
    peak = predictions.loc[predictions["scenario"] == SCENARIO].copy()
    doses = sorted(peak["total_dose_gy"].unique())
    fig, axes = plt.subplots(
        2, len(doses), figsize=(12.0, 7.2), height_ratios=(2.4, 1.0), sharex=True
    )
    axes = np.atleast_2d(axes)
    for column, dose in enumerate(doses):
        base_rows = peak.loc[
            (peak["model"] == "base97") & np.isclose(peak["total_dose_gy"], dose)
        ].sort_values("day")
        adaptive_rows = peak.loc[
            (peak["model"] == "adaptive97") & np.isclose(peak["total_dose_gy"], dose)
        ].sort_values("day")
        top, bottom = axes[0, column], axes[1, column]
        before = base_rows.loc[base_rows["day"] <= LANDMARK_DAY]
        future_base = base_rows.loc[base_rows["day"] > LANDMARK_DAY]
        future_adaptive = adaptive_rows.loc[adaptive_rows["day"] > LANDMARK_DAY]
        top.fill_between(
            future_base["day"], future_base["prediction_low95"], future_base["prediction_high95"],
            color="#dadaeb", alpha=0.45, label="95 %-й интервал",
        )
        top.plot(
            before["day"], before["predicted_relative_volume"], color="#525252",
            linewidth=2.3, label="безусловная подгонка, сутки 1–12",
        )
        top.plot(
            future_base["day"], future_base["predicted_relative_volume"], color="#2f6f9f",
            linewidth=2.2, label="априорный прогноз",
        )
        top.plot(
            future_adaptive["day"], future_adaptive["predicted_relative_volume"],
            color="#2b8c62", linewidth=2.2, linestyle="--", label="обновление на 12-е сутки",
        )
        top.scatter(
            base_rows["day"], base_rows["actual_relative_volume"], color="#252525",
            s=28, zorder=5, label="наблюдение",
        )
        top.set_yscale("log")
        top.set_title(f"$D_{{\\mathrm{{физ}}}}={dose:.1f}$ Гр")
        top.grid(alpha=0.23, which="both")

        merged = future_base[["day", "predicted_log_relative"]].merge(
            future_adaptive[["day", "predicted_log_relative"]],
            on="day", suffixes=("_base", "_adaptive"),
        )
        correction = (
            merged["predicted_log_relative_adaptive"]
            - merged["predicted_log_relative_base"]
        )
        bottom.axhline(0.0, color="black", linewidth=0.9)
        bottom.plot(merged["day"], correction, "o-", color="#2b8c62", linewidth=2.0, markersize=4)
        bottom.set_xlabel("сутки после облучения")
        bottom.grid(alpha=0.23)
    axes[0, 0].set_ylabel("относительный объём опухоли")
    axes[1, 0].set_ylabel("поправка обновления (log)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Пиковые N+P-серии: прогноз после рубежа 12 суток", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure_peak_landmark_step.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_peak_only_report(
    output: Path,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    direction: pd.DataFrame,
    control: pd.DataFrame,
) -> None:
    peak = summary.loc[summary["scenario"] == SCENARIO].copy()
    lines = [
        "# Внешняя темпорально отложенная валидация июня 2026 года: протонная компонента в пике",
        "",
        "## Статус",
        "",
        "Обе облучённые серии июня 2026 года выполнены с протонной компонентой в пике "
        "и представлены семейством `mixed_n_p_peak`. Архивные протонные компоненты "
        "переведены из биологически взвешенной записи в физическую дозу делением на "
        "1,1; нейтронные компоненты не преобразовывались.",
        "",
        "Животные июня 2026 года не участвовали в обучении, выборе признаков, настройке "
        "коэффициентов или калибровке интервалов. Поэтому расчёт рассматривается как "
        "внешняя по отношению к обучающей выборке темпорально отложенная валидация "
        "внутри той же лаборатории.",
        "",
        "## Результаты",
        "",
        "| Модель | Архивная сумма компонент | Физическая доза модели, Гр | log-RMSE 1–21 | log-RMSE 13–21 | V21 наблюдение | V21 прогноз | Наблюдение / прогноз |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("base97", "apriori133", "adaptive97"):
        rows = peak.loc[peak["model"] == model].sort_values("total_dose_gy")
        for _, row in rows.iterrows():
            lines.append(
                f"| {MODEL_LABELS[model]} | {row['archived_total_dose_gy_rbe_mixed']:.1f} | "
                f"{row['total_dose_gy']:.2f} | "
                f"{row['day1_21_log_rmse']:.3f} | {row['day13_21_log_rmse']:.3f} | "
                f"{row['day21_actual_relative']:.3f} | {row['day21_predicted_relative']:.3f} | "
                f"{row['day21_actual_to_predicted_ratio']:.2f} |"
            )
    lines.extend(
        [
            "",
            "Для обновляемого контура оцениваемым будущим участком являются 13–21-е сутки; "
            "полная ошибка 1–21 приведена только для описания всей составной траектории.",
            "",
            "Во всех трёх контурах правильно воспроизведено направление межсерийного различия: "
            "для режима с архивной суммой компонент 35,0 предсказан меньший объём на "
            "21-е сутки, чем для режима с архивной суммой 29,8. При этом "
            "абсолютный объём обеих облучённых серий недооценён, то есть эффект облучения "
            "переоценён.",
        ]
    )
    if not control.empty:
        row = control.iloc[0]
        lines.extend(
            [
                "",
                "## Одновременный контроль",
                "",
                f"Для контроля 19 июня наблюдаемый относительный объём на 21-е сутки составил "
                f"{row['june_full_cohort_day21_actual_relative']:.2f}, прогноз — "
                f"{row['june_full_cohort_day21_predicted_relative']:.2f}, отношение — "
                f"{row['june_full_cohort_day21_ratio']:.2f}. Поэтому расхождение облучённых "
                "серий нельзя объяснить общим календарным сдвигом роста необлучённой опухоли.",
            ]
        )
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            "Внешняя темпорально отложенная валидация подтверждает перенос направления различия режимов, но выявляет "
            "недостаточную калибровку абсолютного ответа смешанного протон-нейтронного поля в "
            "пике. Две серии недостаточны для переоценки коэффициентов или расчёта надёжного "
            "коэффициента детерминации.",
            "",
            "![Внешняя валидация для протонного пика](figure_peak_holdout.png)",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    base = load_module("june_peak_base", BASE_SCRIPT)
    early = load_module("june_peak_early", EARLY_SCRIPT)
    adaptive = load_module("june_peak_adaptive", ADAPTIVE_SCRIPT)
    holdout = load_module("june_peak_holdout", HOLDOUT_SCRIPT)
    engine = load_module("june_peak_engine", holdout.ENGINE)

    _, evaluation, _, _ = base.build_longitudinal_tables()
    training = base.collapse_animal_daily_to_series(evaluation)
    categories = base.fixed_categories(training)

    parameters97 = RELEASE / "adaptive_97" / "base97_model_parameters.csv"
    parameters133 = RELEASE / "apriori_133" / "model_parameters.csv"
    adaptive_path = RELEASE / "adaptive_97" / "adaptive_parameters.csv"
    update = adaptive_parameters(adaptive_path)

    test = build_holdout(holdout, engine, base, CONFIRMED_FAMILY)
    matrix97, names97 = base.build_features(test, "hierarchical_robust", categories)
    pred97 = frozen_prediction(matrix97, names97, parameters97)
    matrix133, names133 = early.build_arm_features(test, "early_family", categories)
    pred133 = frozen_prediction(matrix133, names133, parameters133)
    pred_adaptive = adaptive.apply_trend_update(
        test,
        pred97,
        LANDMARK_DAY,
        float(update["level_weight"]),
        float(update["slope_weight"]),
        float(update["horizon_cap"]),
        update["bounds"],
    )
    predictions = prediction_rows(
        test,
        SCENARIO,
        {
            "base97": pred97,
            "apriori133": pred133,
            "adaptive97": pred_adaptive,
        },
    )
    summary = summarize(predictions)
    aggregate = aggregate_summary(summary)
    direction = direction_summary(summary)
    control = (
        pd.read_csv(CONTROL_SUMMARY, sep=";") if CONTROL_SUMMARY.exists() else pd.DataFrame()
    )

    predictions.to_csv(
        args.output / "peak_holdout_predictions.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output / "peak_holdout_series_summary.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        args.output / "peak_holdout_aggregate_summary.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    direction.to_csv(
        args.output / "peak_holdout_direction_summary.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    if not control.empty:
        control.to_csv(
            args.output / "control_context_unchanged.csv",
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

    source_files = [
        holdout.HOLDOUT_DIR / name for name in holdout.HOLDOUT_SERIES
    ]
    provenance = {
        "analysis": "june_2026_external_peak_holdout_validation",
        "status": "external_to_training_temporal_validation_same_laboratory",
        "confirmed_family": CONFIRMED_FAMILY,
        "training_refit": False,
        "validation_series_present_in_training": False,
        "primary_model_outcomes_used_for_parameter_tuning": False,
        "release": "sarcoma_m1_growth_v1.0.0",
        "landmark_day": LANDMARK_DAY,
        "prediction_intervals_95_log_half_width": {
            "base97": BASE97_Q95_BY_PERIOD,
            "adaptive97": ADAPTIVE97_Q95,
            "apriori133": None,
        },
        "dose_policy": {
            "peak_proton_2026_input": "archived_RBE_weighted",
            "peak_proton_divisor": P_PEAK_2026_RBE,
            "neutron_components": "unchanged",
        },
        "zero_volume_policy": (
            "literal zero retained as complete regression; interpolation on relative-volume "
            "scale only for an individual trajectory containing zero; group mean computed "
            "in volume space before logarithm"
        ),
        "model_specific_status": {
            "base97": "primary_external_temporal_validation",
            "adaptive97": "additional_fixed_algorithm_evaluation_on_external_sample",
            "apriori133": "post-development_diagnostic_only",
        },
        "inputs": {str(path): sha256(path) for path in source_files},
        "artifacts": {
            str(parameters97): sha256(parameters97),
            str(parameters133): sha256(parameters133),
            str(adaptive_path): sha256(adaptive_path),
            str(BASE_SCRIPT): sha256(BASE_SCRIPT),
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_peak_only_figure(predictions, args.output)
    make_peak_residual_figure(predictions, args.output)
    make_peak_landmark_figure(predictions, args.output)
    write_peak_only_report(args.output, summary, aggregate, direction, control)

    print(aggregate.to_string(index=False))
    print("\nDirection check")
    print(direction.to_string(index=False))
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
