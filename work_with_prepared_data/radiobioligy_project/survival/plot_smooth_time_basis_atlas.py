from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


FIXED = "fixed_12"
SMOOTH = "decoupled_K16"
CV_LABELS = {
    "leave_one_year_out": "Исключение года (LOYO)",
    "rolling_origin_year": "Прогноз будущих лет",
}
FAMILY_ORDER = [
    "y",
    "e",
    "p_through",
    "p_peak",
    "n",
    "c_through",
    "c_modified_peak",
    "c_peak_no_filter",
    "mixed_n_p_through",
    "mixed_n_p_peak",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an atlas comparing the fixed day-12 and smooth 12/16 time bases."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )


def load_pair(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=";")
    pair = data[data["model"].isin([FIXED, SMOOTH])].copy()
    pair["is_treated"] = pair["is_treated"].astype(int)
    pair["day"] = pair["day"].astype(float)
    return pair


def series_metrics(pair: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "cv_kind",
        "series_key",
        "date",
        "year",
        "family",
        "family_label",
        "regimen_class",
        "dose_signature",
        "total_dose_gy",
        "is_treated",
    ]
    rows: list[dict[str, object]] = []
    for key, block in pair.groupby(keys, dropna=False, sort=False):
        meta = dict(zip(keys, key))
        wide = block.pivot_table(
            index="day",
            columns="model",
            values="predicted_log_relative",
            aggfunc="first",
        ).sort_index()
        actual = (
            block.groupby("day", sort=True)["actual_log_relative"].first().reindex(wide.index)
        )
        if FIXED not in wide or SMOOTH not in wide:
            continue
        row: dict[str, object] = dict(meta)
        for label, lo, hi in [
            ("early", 1, 7),
            ("middle", 8, 12),
            ("late", 13, 21),
            ("all", 1, 21),
        ]:
            mask = (wide.index >= lo) & (wide.index <= hi)
            for model in [FIXED, SMOOTH]:
                error = wide.loc[mask, model].to_numpy(float) - actual.loc[mask].to_numpy(float)
                row[f"rmse_{label}_{model}"] = float(np.sqrt(np.mean(np.square(error))))
            row[f"delta_rmse_{label}"] = (
                row[f"rmse_{label}_{SMOOTH}"] - row[f"rmse_{label}_{FIXED}"]
            )
        row["observed_min_day"] = float(actual.idxmin())
        row["fixed_min_day"] = float(wide[FIXED].idxmin())
        row["smooth_min_day"] = float(wide[SMOOTH].idxmin())
        row["fixed_min_error_days"] = abs(row["fixed_min_day"] - row["observed_min_day"])
        row["smooth_min_error_days"] = abs(row["smooth_min_day"] - row["observed_min_day"])
        row["delta_min_error_days"] = (
            row["smooth_min_error_days"] - row["fixed_min_error_days"]
        )
        row["maximum_curve_change"] = float(np.max(np.abs(wide[SMOOTH] - wide[FIXED])))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_curve(pair: pd.DataFrame, cv_kind: str, series_key: str) -> pd.DataFrame:
    block = pair[(pair["cv_kind"] == cv_kind) & (pair["series_key"] == series_key)]
    meta = block.iloc[0]
    actual = block.groupby("day", sort=True)["actual_log_relative"].first()
    fixed = block[block["model"] == FIXED].set_index("day")["predicted_log_relative"]
    smooth = block[block["model"] == SMOOTH].set_index("day")["predicted_log_relative"]
    days = sorted(set(actual.index) & set(fixed.index) & set(smooth.index))
    return pd.DataFrame(
        {
            "day": days,
            "actual": actual.reindex(days).to_numpy(float),
            "fixed": fixed.reindex(days).to_numpy(float),
            "smooth": smooth.reindex(days).to_numpy(float),
            "family_label": meta["family_label"],
            "date": meta["date"],
            "dose_signature": meta["dose_signature"],
        }
    )


def save_figure(fig: plt.Figure, path: Path, pdf: PdfPages) -> None:
    fig.savefig(path, dpi=240, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_single_curve(ax: plt.Axes, curve: pd.DataFrame, metric: pd.Series) -> None:
    ax.plot(curve["day"], curve["actual"], "o-", color="#252525", ms=3.2, lw=1.1)
    ax.plot(curve["day"], curve["fixed"], "--", color="#d97706", lw=1.8)
    ax.plot(curve["day"], curve["smooth"], "-", color="#2563a6", lw=2.0)
    ax.axvline(12, color="#d97706", lw=0.9, ls=":", alpha=0.8)
    ax.axvline(16, color="#2563a6", lw=0.9, ls=":", alpha=0.65)
    ax.scatter(
        [metric["observed_min_day"]],
        [curve.loc[curve["day"] == metric["observed_min_day"], "actual"].iloc[0]],
        marker="v",
        s=34,
        color="#252525",
        zorder=5,
    )
    ax.scatter(
        [metric["fixed_min_day"]],
        [curve.loc[curve["day"] == metric["fixed_min_day"], "fixed"].iloc[0]],
        marker="v",
        s=32,
        color="#d97706",
        zorder=5,
    )
    ax.scatter(
        [metric["smooth_min_day"]],
        [curve.loc[curve["day"] == metric["smooth_min_day"], "smooth"].iloc[0]],
        marker="v",
        s=32,
        color="#2563a6",
        zorder=5,
    )
    family = str(metric["family_label"])
    dose = str(metric["dose_signature"])
    ax.set_title(f"{family}; {metric['date']}; {dose}", loc="left")
    ax.text(
        0.02,
        0.97,
        (
            f"min: факт {metric['observed_min_day']:.0f}, "
            f"12-сут {metric['fixed_min_day']:.0f}, 12/16 {metric['smooth_min_day']:.0f}\n"
            f"ΔRMSE: ран. {metric['delta_rmse_early']:+.3f}; "
            f"поздн. {metric['delta_rmse_late']:+.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )
    ax.set_xlim(0, 21)
    ax.set_xticks([0, 3, 7, 12, 16, 21])


def choose_unique(
    metrics: pd.DataFrame,
    score: str,
    n: int,
    ascending: bool,
    excluded: set[str] | None = None,
) -> pd.DataFrame:
    excluded = excluded or set()
    chosen: list[pd.Series] = []
    used = set(excluded)
    for _, row in metrics.sort_values(score, ascending=ascending).iterrows():
        key = str(row["series_key"])
        if key in used:
            continue
        chosen.append(row)
        used.add(key)
        if len(chosen) >= n:
            break
    return pd.DataFrame(chosen)


def example_panel(
    pair: pd.DataFrame,
    selected: pd.DataFrame,
    title: str,
    path: Path,
    pdf: PdfPages,
) -> None:
    n = len(selected)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.75 * rows), squeeze=False)
    for ax, (_, metric) in zip(axes.flat, selected.iterrows()):
        curve = extract_curve(pair, str(metric["cv_kind"]), str(metric["series_key"]))
        plot_single_curve(ax, curve, metric)
    for ax in axes.flat[n:]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="#252525", lw=1.1, label="наблюдение"),
        Line2D([0], [0], color="#d97706", lw=1.8, ls="--", label="исходный базис: узел 12"),
        Line2D([0], [0], color="#2563a6", lw=2.0, label="гладкий базис 12/16"),
        Line2D([0], [0], marker="v", color="#252525", lw=0, label="минимум"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, y=0.995)
    fig.supxlabel("Сутки после облучения")
    fig.supylabel("ln(V/V₀)")
    fig.tight_layout(rect=(0.025, 0.035, 1, 0.935))
    save_figure(fig, path, pdf)


def family_medians(
    pair: pd.DataFrame, cv_kind: str, metrics: pd.DataFrame, path: Path, pdf: PdfPages
) -> None:
    treated = pair[(pair["cv_kind"] == cv_kind) & (pair["is_treated"] == 1)].copy()
    families = [f for f in FAMILY_ORDER if f in set(treated["family"])]
    cols = 3
    rows = math.ceil(len(families) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.7 * rows), squeeze=False, sharex=True)
    for ax, family in zip(axes.flat, families):
        block = treated[treated["family"] == family]
        actual = block.groupby(["series_key", "day"])["actual_log_relative"].first().reset_index()
        actual_summary = actual.groupby("day")["actual_log_relative"].agg(
            median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
        )
        ax.fill_between(
            actual_summary.index,
            actual_summary["q25"],
            actual_summary["q75"],
            color="#737373",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(actual_summary.index, actual_summary["median"], "o-", color="#252525", ms=2.6, lw=1.0)
        for model, color, style in [
            (FIXED, "#d97706", "--"),
            (SMOOTH, "#2563a6", "-"),
        ]:
            pred = block[block["model"] == model].groupby("day")["predicted_log_relative"].median()
            ax.plot(pred.index, pred, style, color=color, lw=2.0)
        family_label = str(block["family_label"].iloc[0])
        n = block["series_key"].nunique()
        m = metrics[(metrics["cv_kind"] == cv_kind) & (metrics["family"] == family)]
        de = m["delta_rmse_early"].median()
        dl = m["delta_rmse_late"].median()
        ax.set_title(f"{family_label}; n={n}", loc="left", pad=18)
        ax.text(
            0.0,
            1.015,
            f"медиана ΔRMSE: ран. {de:+.3f}; поздн. {dl:+.3f}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8.3,
        )
        ax.axvline(12, color="#d97706", lw=0.8, ls=":")
        ax.axvline(16, color="#2563a6", lw=0.8, ls=":")
        ax.set_xlim(0, 21)
        ax.set_xticks([0, 3, 7, 12, 16, 21])
    for ax in axes.flat[len(families) :]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="#252525", lw=1, label="медиана наблюдений (полоса: IQR)"),
        Line2D([0], [0], color="#d97706", lw=2, ls="--", label="исходный базис"),
        Line2D([0], [0], color="#2563a6", lw=2, label="гладкий базис 12/16"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Медианные формы траекторий по семействам — {CV_LABELS[cv_kind]}", y=0.995)
    fig.supxlabel("Сутки после облучения")
    fig.supylabel("ln(V/V₀)")
    fig.tight_layout(rect=(0.025, 0.035, 1, 0.92), h_pad=2.0)
    save_figure(fig, path, pdf)


def minimum_overview(metrics: pd.DataFrame, cv_kind: str, path: Path, pdf: PdfPages) -> None:
    data = metrics[(metrics["cv_kind"] == cv_kind) & (metrics["is_treated"] == 1)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    bins = np.arange(-0.5, 22.5, 1)
    axes[0].hist(data["observed_min_day"], bins=bins, histtype="step", lw=2.0, color="#252525", label="наблюдение")
    axes[0].hist(data["fixed_min_day"], bins=bins, histtype="step", lw=2.0, color="#d97706", label="исходный базис")
    axes[0].hist(data["smooth_min_day"], bins=bins, histtype="step", lw=2.0, color="#2563a6", label="гладкий 12/16")
    axes[0].set(xlabel="Сутки минимума", ylabel="Число серий", title="Распределение минимумов")
    axes[0].legend(frameon=False)
    for ax, column, title, color in [
        (axes[1], "fixed_min_day", "Исходный базис", "#d97706"),
        (axes[2], "smooth_min_day", "Гладкий базис 12/16", "#2563a6"),
    ]:
        jitter = np.linspace(-0.16, 0.16, len(data))
        ax.scatter(data["observed_min_day"] + jitter, data[column] - jitter, s=20, alpha=0.6, color=color, edgecolor="none")
        ax.plot([0, 21], [0, 21], color="#252525", lw=1, ls=":")
        ax.set_xlim(-0.5, 21.5)
        ax.set_ylim(-0.5, 21.5)
        ax.set_xticks([0, 3, 7, 12, 16, 21])
        ax.set_yticks([0, 3, 7, 12, 16, 21])
        ax.set(xlabel="Наблюдаемые сутки", ylabel="Прогнозируемые сутки", title=title)
    fig.suptitle(f"Сутки минимума — {CV_LABELS[cv_kind]}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path, pdf)


def minimum_overview_combined(metrics: pd.DataFrame, path: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    bins = np.arange(-0.5, 22.5, 1)
    for row, cv_kind in enumerate(["leave_one_year_out", "rolling_origin_year"]):
        data = metrics[(metrics["cv_kind"] == cv_kind) & (metrics["is_treated"] == 1)].copy()
        axes[row, 0].hist(data["observed_min_day"], bins=bins, histtype="step", lw=2.0, color="#252525", label="наблюдение")
        axes[row, 0].hist(data["fixed_min_day"], bins=bins, histtype="step", lw=2.0, color="#d97706", label="исходный базис")
        axes[row, 0].hist(data["smooth_min_day"], bins=bins, histtype="step", lw=2.0, color="#2563a6", label="гладкий 12/16")
        axes[row, 0].set(xlabel="Сутки минимума", ylabel="Число серий")
        axes[row, 0].set_title(f"{CV_LABELS[cv_kind]}: распределение", loc="left")
        axes[row, 0].legend(frameon=False)
        for column, model_column, title, color in [
            (1, "fixed_min_day", "исходный базис", "#d97706"),
            (2, "smooth_min_day", "гладкий базис 12/16", "#2563a6"),
        ]:
            ax = axes[row, column]
            jitter = np.linspace(-0.16, 0.16, len(data))
            ax.scatter(data["observed_min_day"] + jitter, data[model_column] - jitter, s=20, alpha=0.6, color=color, edgecolor="none")
            ax.plot([0, 21], [0, 21], color="#252525", lw=1, ls=":")
            ax.set_xlim(-0.5, 21.5)
            ax.set_ylim(-0.5, 21.5)
            ax.set_xticks([0, 3, 7, 12, 16, 21])
            ax.set_yticks([0, 3, 7, 12, 16, 21])
            ax.set(xlabel="Наблюдаемые сутки", ylabel="Прогнозируемые сутки")
            ax.set_title(f"{CV_LABELS[cv_kind]}: {title}", loc="left")
    fig.suptitle("Как гладкая поправка меняет распределение прогнозируемых суток минимума")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2)
    save_figure(fig, path, pdf)


def error_tradeoff(metrics: pd.DataFrame, cv_kind: str, path: Path, pdf: PdfPages) -> None:
    data = metrics[(metrics["cv_kind"] == cv_kind) & (metrics["is_treated"] == 1)].copy()
    families = [f for f in FAMILY_ORDER if f in set(data["family"])]
    cmap = plt.get_cmap("tab10")
    colors = {family: cmap(i % 10) for i, family in enumerate(families)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for family in families:
        block = data[data["family"] == family]
        label = str(block["family_label"].iloc[0])
        axes[0].scatter(
            block["delta_rmse_early"],
            block["delta_rmse_late"],
            s=30,
            alpha=0.72,
            color=colors[family],
            label=label,
        )
        axes[1].scatter(
            block["maximum_curve_change"],
            block["delta_min_error_days"],
            s=30,
            alpha=0.72,
            color=colors[family],
            label=label,
        )
    axes[0].axhline(0, color="#252525", lw=1)
    axes[0].axvline(0, color="#252525", lw=1)
    axes[0].set(
        xlabel="ΔRMSE 1–7-е сутки (гладкий − исходный)",
        ylabel="ΔRMSE 13–21-е сутки",
        title="Компромисс ранней и поздней ошибки",
    )
    axes[1].axhline(0, color="#252525", lw=1)
    axes[1].set(
        xlabel="Максимальное изменение прогноза, |Δ ln(V/V₀)|",
        ylabel="Δ абсолютной ошибки суток минимума",
        title="Изменение формы и точности минимума",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Посерийный эффект поправки — {CV_LABELS[cv_kind]}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, path, pdf)


def family_effects(metrics: pd.DataFrame, path: Path, pdf: PdfPages) -> None:
    data = metrics[metrics["is_treated"] == 1].copy()
    summary = (
        data.groupby(["cv_kind", "family", "family_label"], as_index=False)
        .agg(
            n=("series_key", "nunique"),
            early=("delta_rmse_early", "median"),
            middle=("delta_rmse_middle", "median"),
            late=("delta_rmse_late", "median"),
            min_error=("delta_min_error_days", "median"),
        )
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    for ax, cv_kind in zip(axes, ["leave_one_year_out", "rolling_origin_year"]):
        block = summary[summary["cv_kind"] == cv_kind].copy()
        block["order"] = block["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
        block = block.sort_values("order")
        x = np.arange(len(block))
        width = 0.24
        ax.bar(x - width, block["early"], width, color="#7f8c8d", label="1–7-е сутки")
        ax.bar(x, block["middle"], width, color="#d97706", label="8–12-е сутки")
        ax.bar(x + width, block["late"], width, color="#2563a6", label="13–21-е сутки")
        ax.axhline(0, color="#252525", lw=1)
        ax.set_xticks(x, [f"{lab}\nn={n}" for lab, n in zip(block["family_label"], block["n"])], rotation=20, ha="right")
        ax.set_ylabel("Медиана ΔRMSE")
        ax.set_title(CV_LABELS[cv_kind], loc="left")
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("По каким семействам гладкая поправка улучшает или ухудшает ошибку")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path, pdf)


def dissertation_examples(
    pair: pd.DataFrame, metrics: pd.DataFrame, path: Path, pdf: PdfPages
) -> None:
    data = metrics[(metrics["cv_kind"] == "leave_one_year_out") & (metrics["is_treated"] == 1)].copy()
    selected_keys = [
        "2017-02-15|p_through|29|single_fraction|none",
        "2017-11-02|p_through|32|single_fraction|none",
        "2017-05-04|p_through|32|single_fraction|none",
        "2026-03-24|mixed_n_p_peak|17.0909+2.36+17.0909+2.36|mixed_sequence|P_first",
        "2026-03-24|mixed_n_p_peak|2.36+17.0909+2.36+17.0909|mixed_sequence|N_first",
        "2017-03-27|c_peak_no_filter|12|single_fraction|none",
    ]
    selected = data.set_index("series_key").loc[selected_keys].reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), squeeze=False)
    for index, (ax, (_, metric)) in enumerate(zip(axes.flat, selected.iterrows())):
        curve = extract_curve(pair, str(metric["cv_kind"]), str(metric["series_key"]))
        plot_single_curve(ax, curve, metric)
        ax.text(
            0.98,
            0.04,
            "улучшение" if index < 3 else "ухудшение",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
        )
    handles = [
        Line2D([0], [0], marker="o", color="#252525", lw=1.1, label="наблюдение"),
        Line2D([0], [0], color="#d97706", lw=1.8, ls="--", label="исходный базис: узел 12"),
        Line2D([0], [0], color="#2563a6", lw=2.0, label="гладкий базис 12/16"),
        Line2D([0], [0], marker="v", color="#252525", lw=0, label="минимум"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Показательные улучшения и ухудшения формы траектории — LOYO", y=0.995)
    fig.supxlabel("Сутки после облучения")
    fig.supylabel("ln(V/V₀)")
    fig.tight_layout(rect=(0.025, 0.04, 1, 0.92), h_pad=2.0)
    save_figure(fig, path, pdf)


def write_readme(output: Path, selected: pd.DataFrame) -> None:
    lines = [
        "# Атлас сравнения временного базиса",
        "",
        "Сравниваются исходный жёсткий узел на 12-х сутках и гладкий развязанный вариант 12/16.",
        "Во всех графиках отрицательное ΔRMSE означает улучшение гладкой модели.",
        "Треугольниками на индивидуальных кривых отмечены сутки минимума.",
        "",
        "## Файлы",
        "",
        "- `00_curve_shape_atlas.pdf` — все рисунки одним многостраничным файлом;",
        "- `01`–`04` — индивидуальные траектории с разными типами эффекта;",
        "- `05`–`06` — медианные траектории по семействам;",
        "- `07`–`08` — распределение и согласие суток минимума;",
        "- `09`–`10` — посерийный компромисс ошибок;",
        "- `11` — медианный эффект по семействам;",
        "- `series_comparison_metrics.csv` — численные показатели всех серий;",
        "- `selected_examples.csv` — серии, вошедшие в индивидуальные панели.",
        "",
        "Индивидуальные панели отобраны алгоритмически: наибольшее изменение формы,",
        "наибольшее улучшение поздней ошибки, ухудшение ранней ошибки и изменение суток минимума.",
        "Они предназначены для визуальной диагностики, а не для повторного отбора модели.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected.to_csv(output / "selected_examples.csv", sep=";", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()
    pair = load_pair(args.input)
    metrics = series_metrics(pair)
    metrics.to_csv(
        args.output / "series_comparison_metrics.csv", sep=";", index=False, encoding="utf-8-sig"
    )

    loyo = metrics[(metrics["cv_kind"] == "leave_one_year_out") & (metrics["is_treated"] == 1)]
    rolling = metrics[(metrics["cv_kind"] == "rolling_origin_year") & (metrics["is_treated"] == 1)]
    selected_parts: list[pd.DataFrame] = []

    changed = choose_unique(loyo, "maximum_curve_change", 9, ascending=False)
    changed["panel"] = "largest_shape_change"
    selected_parts.append(changed)

    best_late = choose_unique(loyo, "delta_rmse_late", 9, ascending=True, excluded=set(changed["series_key"]))
    best_late["panel"] = "best_late_improvement"
    selected_parts.append(best_late)

    worse_early = choose_unique(loyo, "delta_rmse_early", 9, ascending=False)
    worse_early["panel"] = "largest_early_worsening"
    selected_parts.append(worse_early)

    shifted = rolling.copy()
    shifted["minimum_shift"] = (shifted["smooth_min_day"] - shifted["fixed_min_day"]).abs()
    shifted = choose_unique(shifted, "minimum_shift", 9, ascending=False)
    shifted["panel"] = "future_year_minimum_shift"
    selected_parts.append(shifted)

    selected = pd.concat(selected_parts, ignore_index=True)
    pdf_path = args.output / "00_curve_shape_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        example_panel(
            pair,
            changed,
            "Где гладкая поправка сильнее всего меняет форму — LOYO",
            args.output / "01_largest_shape_changes_loyo.png",
            pdf,
        )
        example_panel(
            pair,
            best_late,
            "Серии с наибольшим улучшением поздней ошибки — LOYO",
            args.output / "02_best_late_improvements_loyo.png",
            pdf,
        )
        example_panel(
            pair,
            worse_early,
            "Серии с наибольшим ухудшением ранней ошибки — LOYO",
            args.output / "03_largest_early_worsening_loyo.png",
            pdf,
        )
        example_panel(
            pair,
            shifted,
            "Наибольшее смещение прогнозируемого минимума — будущие годы",
            args.output / "04_future_year_minimum_shifts.png",
            pdf,
        )
        family_medians(
            pair,
            "leave_one_year_out",
            metrics,
            args.output / "05_family_median_curves_loyo.png",
            pdf,
        )
        family_medians(
            pair,
            "rolling_origin_year",
            metrics,
            args.output / "06_family_median_curves_future_years.png",
            pdf,
        )
        minimum_overview(
            metrics,
            "leave_one_year_out",
            args.output / "07_minimum_days_loyo.png",
            pdf,
        )
        minimum_overview(
            metrics,
            "rolling_origin_year",
            args.output / "08_minimum_days_future_years.png",
            pdf,
        )
        minimum_overview_combined(
            metrics,
            args.output / "08a_minimum_days_both_schemes.png",
            pdf,
        )
        error_tradeoff(
            metrics,
            "leave_one_year_out",
            args.output / "09_series_error_tradeoff_loyo.png",
            pdf,
        )
        error_tradeoff(
            metrics,
            "rolling_origin_year",
            args.output / "10_series_error_tradeoff_future_years.png",
            pdf,
        )
        family_effects(metrics, args.output / "11_family_error_changes.png", pdf)
        dissertation_examples(
            pair,
            metrics,
            args.output / "12_dissertation_examples.png",
            pdf,
        )

    write_readme(args.output, selected)
    print(f"Wrote {len(list(args.output.glob('*.png')))} PNG files and {pdf_path}")


if __name__ == "__main__":
    main()
