"""Figures for the held-out check: trajectories and the residual over time.

The second panel is the point of the figure. The updating contour assumes the
residual observed up to the landmark is a level offset that persists, and on this
pair it is not: the residual swings from positive through negative and back, so
what the landmark sees at day 12 has little to do with what days 13 to 21 need.
For the heavier series the two even carry opposite signs, which is why the update
moves the forecast the wrong way.
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
LANDMARK_DAY = 12
LABELS = {29.8: "29,8 Гр, 3 фракции", 35.0: "35,0 Гр, 4 фракции"}


def figure(frame: pd.DataFrame, output: Path) -> None:
    doses = sorted(frame["total_dose_gy"].unique())
    fig, axes = plt.subplots(2, len(doses), figsize=(11.4, 7.6), sharex=True)
    axes = np.atleast_2d(axes)
    for column, dose in enumerate(doses):
        group = frame.loc[np.isclose(frame["total_dose_gy"], dose)].sort_values("day")
        top, bottom = axes[0, column], axes[1, column]

        top.plot(
            group["day"], np.exp(group["actual_log_relative"]),
            "o-", color="#252525", markersize=4, linewidth=1.5, label="наблюдение",
        )
        top.plot(
            group["day"], np.exp(group["predicted_log_relative"]),
            color="#2171b5", linewidth=2.0, label="априорный прогноз",
        )
        future = group.loc[group["day"] > LANDMARK_DAY]
        top.plot(
            future["day"], np.exp(future["adaptive_log_relative"]),
            color="#cb181d", linewidth=2.0, linestyle="--",
            label="адаптивный, рубеж 12 сут",
        )
        top.axvline(LANDMARK_DAY, color="#969696", linestyle=":", linewidth=1.2)
        top.set_yscale("log")
        top.set_title(LABELS.get(dose, dose), fontsize=10)
        top.grid(alpha=0.25, which="both")

        residual = group["actual_log_relative"] - group["predicted_log_relative"]
        bottom.axhline(0.0, color="black", linewidth=0.9)
        bottom.axvline(LANDMARK_DAY, color="#969696", linestyle=":", linewidth=1.2)
        early = group["day"] <= LANDMARK_DAY
        bottom.fill_between(
            group["day"], 0.0, residual, where=early,
            color="#6baed6", alpha=0.55, label="сутки 1–12: что видит адаптация",
        )
        bottom.fill_between(
            group["day"], 0.0, residual, where=~early,
            color="#fc9272", alpha=0.55, label="сутки 13–21: что требуется",
        )
        bottom.plot(group["day"], residual, "o-", color="#252525", markersize=3.5, linewidth=1.3)
        mean_early = float(residual[early].mean())
        mean_late = float(residual[~early].mean())
        bottom.annotate(
            f"среднее {mean_early:+.3f}", xy=(5.5, mean_early), fontsize=8,
            color="#08519c", ha="center",
        )
        bottom.annotate(
            f"среднее {mean_late:+.3f}", xy=(17.0, mean_late), fontsize=8,
            color="#a50f15", ha="center",
        )
        bottom.set_xlabel("сутки после облучения")
        bottom.grid(alpha=0.25)

    axes[0, 0].set_ylabel("относительный объём опухоли")
    axes[1, 0].set_ylabel("остаток, факт − прогноз (log)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Отложенная выборка июня 2026 года: априорный и адаптивный прогноз\n"
        "нижний ряд — остаток по суткам; вертикальный пунктир — рубеж обновления",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.holdout / "holdout_adaptive_predictions.csv", sep=";")
    figure(frame, args.holdout / "figure_holdout_adaptive")
    print(f"written to {args.holdout}")


if __name__ == "__main__":
    main()
