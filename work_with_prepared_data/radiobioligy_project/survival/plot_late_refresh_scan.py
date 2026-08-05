"""Figure: what a later last update actually buys.

The left panel is the finding. The model's day-21 coefficient of determination
rises with the landmark, but so does that of a forecast built from the observed
trajectory alone, and the two cross between days 15 and 16. Past that crossing
the model is worse than holding a ruler to the last few measurements, which means
the late-landmark gain is the shrinking horizon and not better description.

The right panel is the corroborating tell. The gap between holding out a year at
random and forecasting a strictly later year is the calendar-epoch problem that
runs through this work; it collapses to zero at day 18. A forecast that no longer
cares which year it is has stopped using the year, the dose and the field -- it
is reading the trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCAN = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\prediction_v8_late_refresh"
)
DEFAULT_OUTPUT = Path(r"C:\dev\dissertation_text\figures\figure_late_refresh_scan")


def figure(table: pd.DataFrame, output: Path) -> None:
    pivot = table.pivot(index="landmark_day", columns="source", values="day21_log_r2")
    days = pivot.index.to_numpy(float)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.9))

    series = (
        ("model_leave_one_year_out", "модель, исключение года", "#2171b5", "o-", 2.2),
        ("model_rolling_origin_year", "модель, скользящее начало", "#6baed6", "s-", 2.2),
        ("model_free_projection", "без модели: перенос наклона", "#cb181d", "^--", 1.8),
        ("model_free_persistence", "без модели: последнее значение", "#fb6a4a", "v--", 1.8),
    )
    for column, label, colour, style, width in series:
        left.plot(
            days, pivot[column], style, color=colour, linewidth=width,
            markersize=5.5, label=label,
        )
    left.axhline(0.0, color="#525252", linewidth=0.8)
    left.axvspan(15.0, 16.0, color="#969696", alpha=0.18)
    left.annotate(
        "модель перестаёт\nдавать преимущество",
        xy=(15.5, 0.72), xytext=(15.5, 1.18), fontsize=8.5, ha="center",
        color="#252525",
        arrowprops={"arrowstyle": "-|>", "color": "#525252", "linewidth": 0.9},
    )
    left.set_ylim(-1.35, 1.45)
    left.set_xlabel("рубеж последнего обновления, сутки")
    left.set_ylabel("коэффициент детерминации на 21-е сутки")
    left.set_title("Модель против прогноза без модели", fontsize=10.5)
    left.legend(fontsize=8, loc="lower right")
    left.grid(alpha=0.25)

    gap = pivot["model_leave_one_year_out"] - pivot["model_rolling_origin_year"]
    right.bar(days, gap, width=0.62, color="#54278f", alpha=0.85)
    right.axhline(0.0, color="black", linewidth=0.9)
    for day, value in zip(days, gap):
        right.annotate(
            f"{value:+.3f}", xy=(day, value), xytext=(0, 4 if value >= 0 else -12),
            textcoords="offset points", ha="center", fontsize=8, color="#3f007d",
        )
    right.set_ylim(-0.03, 0.30)
    right.set_xlabel("рубеж последнего обновления, сутки")
    right.set_ylabel("разность коэффициентов детерминации")
    right.set_title(
        "Календарная составляющая ошибки:\nисключение года минус скользящее начало",
        fontsize=10.5,
    )
    right.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=SCAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    table = pd.read_csv(args.scan / "model_free_benchmark.csv", sep=";", encoding="utf-8-sig")
    figure(table, args.output)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
