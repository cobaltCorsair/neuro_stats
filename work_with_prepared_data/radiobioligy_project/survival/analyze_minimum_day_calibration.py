"""Диагностика: сутки минимального относительного объёма — данные против прогноза.

Проверяет, оценивает ли модель положение минимума траектории или воспроизводит
положение узла временного базиса. Многочисленные слагаемые воздействия, режима
и документированной конфигурации входят через насыщающийся множитель
g12(t) = min(t/12, 1), производная которого обрывается на 12-х сутках. Если
этот излом существенно влияет на положение минимума, предсказанные сутки
должны образовать выраженное скопление вблизи 12-х суток.

Вход:  outer_predictions.csv из расчёта раннего переходного ответа.
Выход: рисунок, таблица по сериям и сводка в OUTPUT_DIR.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SOURCE = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Ранний_переходный_ответ\outer_predictions.csv"
)
OUTPUT_DIR = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Сутки_минимума"
)
FIGURE_COPY = Path(r"C:\dev\dissertation_text\figures\figure_minimum_day.png")

KNOT_DAY = 12.0
SEED = 20260811

CV_LABELS = {
    "leave_one_year_out": "исключение календарного года",
    "rolling_origin_year": "обучение только на предшествующих годах",
}
ARM_LABELS = {
    "base_97": "базовая 97-признаковая",
    "early_family": "окончательная 133-признаковая",
}
# в тексте диссертации принят термин «сквозной пучок»
FAMILY_RENAME = {"прострел": "сквозной пучок"}


def load() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE, sep=";")
    frame["family_label"] = (
        frame["family_label"].str.split().str.join(" ").replace(FAMILY_RENAME, regex=True)
    )
    treated = frame.loc[(frame["is_treated"].astype(int) == 1) & (frame["day"] > 0.0)]
    return treated.copy()


def minimum_days(frame: pd.DataFrame, cv_kind: str, arm: str) -> pd.DataFrame:
    part = frame.loc[(frame["cv_kind"] == cv_kind) & (frame["model"] == arm)]
    observed = part.loc[part.groupby("series_key")["actual_relative_volume"].idxmin()]
    predicted = part.loc[part.groupby("series_key")["predicted_relative_volume"].idxmin()]
    joined = (
        observed.set_index("series_key")[
            ["day", "family_label", "total_dose_gy", "year"]
        ]
        .rename(columns={"day": "observed_day"})
        .join(
            predicted.set_index("series_key")["day"].rename("predicted_day"),
            how="inner",
        )
    )
    joined["cv_kind"] = cv_kind
    joined["model"] = arm
    return joined.reset_index()


def summarize(table: pd.DataFrame) -> dict[str, float]:
    return {
        "серий": len(table),
        "набл. медиана": table["observed_day"].median(),
        "набл. размах": f"{table['observed_day'].min():.0f}–{table['observed_day'].max():.0f}",
        "прогн. медиана": table["predicted_day"].median(),
        "прогн. размах": f"{table['predicted_day'].min():.0f}–{table['predicted_day'].max():.0f}",
        "доля ровно на узле, %": 100.0 * (table["predicted_day"] == KNOT_DAY).mean(),
        "доля позже узла, %": 100.0 * (table["predicted_day"] > KNOT_DAY).mean(),
        "набл. позже узла, %": 100.0 * (table["observed_day"] > KNOT_DAY).mean(),
        "Спирмен": table["observed_day"].corr(table["predicted_day"], method="spearman"),
        "медиана |набл-прогн|": (table["observed_day"] - table["predicted_day"]).abs().median(),
    }


def make_figure(tables: dict[tuple[str, str], pd.DataFrame]) -> None:
    rng = np.random.default_rng(SEED)
    cv = "leave_one_year_out"
    main = tables[(cv, "early_family")]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # --- слева: наблюдение против прогноза
    ax = axes[0]
    families = sorted(main["family_label"].unique())
    palette = plt.get_cmap("tab10")
    ax.axvspan(KNOT_DAY - 0.35, KNOT_DAY + 0.35, color="#d9534f", alpha=0.10, zorder=0)
    ax.plot([0, 22], [0, 22], color="#444444", lw=1.0, ls="--", zorder=1,
            label="совпадение")
    for index, family in enumerate(families):
        subset = main.loc[main["family_label"] == family]
        ax.scatter(
            subset["predicted_day"] + rng.uniform(-0.22, 0.22, len(subset)),
            subset["observed_day"] + rng.uniform(-0.22, 0.22, len(subset)),
            s=30, alpha=0.85, linewidths=0.4, edgecolors="white",
            color=palette(index % 10), label=family, zorder=3,
        )
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 22)
    ax.set_xlabel("предсказанные сутки минимума")
    ax.set_ylabel("наблюдаемые сутки минимума")
    ax.set_title(
        f"Сутки минимального относительного объёма\n{CV_LABELS[cv]}, "
        f"{ARM_LABELS['early_family']} модель"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    share = 100.0 * (main["predicted_day"] == KNOT_DAY).mean()
    later = 100.0 * (main["observed_day"] > KNOT_DAY).mean()
    ax.annotate(
        f"{share:.0f} % прогнозов —\nровно 12-е сутки",
        xy=(KNOT_DAY, 20.6), xytext=(14.6, 19.6), fontsize=8.5, color="#a3312c",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#a3312c", lw=1.0),
    )
    ax.axhspan(KNOT_DAY, 22, xmin=0.545, color="#5b8db8", alpha=0.07, zorder=0)
    ax.annotate(
        f"{later:.0f} % наблюдений лежат позже 12-х суток;\nпрогнозов в этой области нет ни одного",
        xy=(17.6, 16.8), fontsize=8.5, color="#2f5d80", ha="center", va="center",
    )

    # --- справа: распределения
    ax = axes[1]
    bins = np.arange(0.5, 22.5, 1.0)
    ax.hist(main["observed_day"], bins=bins, color="#5b8db8", alpha=0.85,
            label="наблюдение", zorder=2)
    ax.hist(main["predicted_day"], bins=bins, histtype="step", lw=2.0,
            color="#d58a3c", label="прогноз", zorder=3)
    ax.axvline(KNOT_DAY, color="#d9534f", lw=1.2, ls=":", zorder=1)
    ax.set_xlim(0, 22)
    ax.set_xlabel("сутки минимума")
    ax.set_ylabel("число серий")
    ax.set_title(
        "Узел базиса $g_{12}(t)=\\min(t/12,\\,1)$\n"
        "формирует скопление прогнозов на 12-х сутках"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"figure_minimum_day.{suffix}", dpi=240)
    fig.savefig(FIGURE_COPY, dpi=240)
    plt.close(fig)


def main() -> None:
    frame = load()
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for cv_kind in ("leave_one_year_out", "rolling_origin_year"):
        for arm in ("base_97", "early_family"):
            table = minimum_days(frame, cv_kind, arm)
            tables[(cv_kind, arm)] = table
            rows.append({"схема": CV_LABELS[cv_kind], "модель": ARM_LABELS[arm], **summarize(table)})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "minimum_day_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(tables.values()).to_csv(
        OUTPUT_DIR / "minimum_day_by_series.csv", index=False, encoding="utf-8-sig"
    )
    make_figure(tables)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nРисунок и таблицы: {OUTPUT_DIR}")
    print(f"Копия рисунка:      {FIGURE_COPY}")


if __name__ == "__main__":
    main()
