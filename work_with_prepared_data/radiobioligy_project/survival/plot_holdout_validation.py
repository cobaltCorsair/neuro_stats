"""Figures for the held-out June 2026 check.

Two panels per series: the observed group trajectory against the frozen model's
prediction with its 95 per cent band, and the landmark curve that motivated the
recommended observation boundary. The point of the first is that the band is
read in relative-volume units -- comparing the logarithm against those bounds
puts the prediction outside its own interval, which is how a units error in the
evaluation was caught -- so the panels are drawn on the volume scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOLDOUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Внешняя_проверка_июнь_2026"
)
LANDMARK = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\prediction_v2_landmark_scan"
)
LABELS = {29.8: "29,8 Гр, 3 фракции", 35.0: "35,0 Гр, 4 фракции"}


def trajectory_figure(frame: pd.DataFrame, output: Path) -> None:
    doses = sorted(frame["total_dose_gy"].unique())
    figure, axes = plt.subplots(1, len(doses), figsize=(11.0, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, dose in zip(axes, doses):
        group = frame.loc[np.isclose(frame["total_dose_gy"], dose)].sort_values("day")
        axis.fill_between(
            group["day"],
            group["prediction_low95"],
            group["prediction_high95"],
            color="#c6dbef",
            label="95 %-й интервал прогноза",
        )
        axis.plot(
            group["day"],
            group["predicted_relative_volume"],
            color="#2171b5",
            linewidth=2.0,
            label="прогноз",
        )
        outside = ~(
            (group["actual_relative_volume"] >= group["prediction_low95"])
            & (group["actual_relative_volume"] <= group["prediction_high95"])
        )
        axis.plot(
            group["day"],
            group["actual_relative_volume"],
            "o-",
            color="#252525",
            markersize=4.0,
            linewidth=1.4,
            label="наблюдение",
        )
        if outside.any():
            axis.plot(
                group.loc[outside, "day"],
                group.loc[outside, "actual_relative_volume"],
                "o",
                color="#cb181d",
                markersize=7.0,
                markerfacecolor="none",
                markeredgewidth=1.6,
                label="вне интервала",
            )
        coverage = float((~outside).mean())
        axis.set_title(f"{LABELS.get(dose, dose)}\nпокрытие {coverage:.3f}", fontsize=10)
        axis.set_xlabel("сутки после облучения")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("относительный объём опухоли")
    axes[0].legend(fontsize=8, loc="lower left")
    figure.suptitle(
        "Проверка на независимой отложенной выборке: серии июня 2026 года",
        fontsize=11,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=200)
    plt.close(figure)


def landmark_figure(output: Path) -> None:
    summary = pd.read_csv(
        LANDMARK / "model_comparison_summary.csv", sep=";", encoding="utf-8-sig"
    )
    adaptive = summary.loc[summary["model"] == "adaptive_landmark"]
    prior = summary.loc[summary["model"] == "v1_stacked"]
    schemes = {
        "leave_one_year_out": ("исключение года", "#2171b5"),
        "leave_one_calendar_series_out": ("исключение даты", "#238b45"),
        "rolling_origin_year": ("скользящее начало", "#cb181d"),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for scheme, (label, colour) in schemes.items():
        rows = adaptive.loc[adaptive["cv_kind"] == scheme].sort_values("landmark_day")
        axis.plot(
            rows["landmark_day"],
            rows["day21_raw_r2"],
            "o-",
            color=colour,
            label=label,
            linewidth=1.8,
            markersize=5,
        )
        base = prior.loc[prior["cv_kind"] == scheme]
        if len(base):
            axis.axhline(
                float(base.iloc[0]["day21_raw_r2"]),
                color=colour,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )
    axis.axvline(12, color="#525252", linestyle="--", linewidth=1.2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("рубеж наблюдения, сутки")
    axis.set_ylabel("$R^2$ объёма на 21-е сутки")
    axis.set_ylim(-3.0, 1.0)
    # Placed against the axis top so it clears the dotted no-update references,
    # which sit near the bottom of the range for the date-holdout scheme.
    axis.annotate(
        "рекомендуемый рубеж",
        xy=(12, 0.95),
        xytext=(11.7, 0.95),
        fontsize=8,
        color="#525252",
        ha="right",
        va="top",
    )
    axis.set_title(
        "Качество обновляемого прогноза в зависимости от рубежа\n"
        "пунктиром — тот же показатель без обновления",
        fontsize=10,
    )
    axis.set_ylim(-3.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.holdout / "holdout_predictions.csv", sep=";")
    trajectory_figure(frame, args.holdout / "figure_holdout_trajectories")
    landmark_figure(args.holdout / "figure_landmark_curve")
    print(f"written to {args.holdout}")


if __name__ == "__main__":
    main()
