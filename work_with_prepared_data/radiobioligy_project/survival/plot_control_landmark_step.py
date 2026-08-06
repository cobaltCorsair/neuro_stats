"""The landmark step drawn honestly: log axis, and the line broken at the rubicon.

The usual rendering joins the pre-landmark curve to the post-landmark one and
plots relative volume on a linear axis. Both choices mislead. Before the landmark
the curve is the unconditional fit; after it, a forecast conditioned on the
observations up to that day. Joining them draws a jump that belongs to the
splice, not to the model. And on a linear axis a constant log offset opens into a
widening wedge, so a correction that is in fact frozen from day 15 looks like one
that keeps growing.

So: a logarithmic ordinate, on which the correction after saturation is a
parallel displacement, and a gap at the landmark instead of a connecting segment.
The lower panel carries the correction itself, which is what the upper panel can
only imply -- a step at the first forecast day, a short ramp while the slope
carry is still accumulating, and a flat line once it hits its three-day cap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\control_zero_dose_validation"
)
DEFAULT_OUTPUT = Path(r"C:\dev\dissertation_text\figures\figure_control_landmark_step")
LANDMARK = 12.0
REFRESH = 14.0
GAP = 0.28  # half-width of the break, in days
BASE = "apriori_pole_gamma"
UPDATED = "adaptive_trend"
REFRESHED = "dynamic_refresh_12_14"


def split(frame: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unconditional fit up to the landmark, conditional forecast after it."""
    frame = frame.sort_values("day")
    return (
        frame.loc[frame["day"].to_numpy(float) <= LANDMARK],
        frame.loc[frame["day"].to_numpy(float) > LANDMARK],
    )


def figure(
    daily: pd.DataFrame,
    observed: pd.DataFrame,
    animals: pd.DataFrame,
    output: Path,
) -> None:
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.6, 8.2), height_ratios=(2.5, 1.0), sharex=True
    )
    curves = {name: group for name, group in daily.groupby("model")}
    base = curves[BASE].sort_values("day")

    for _, animal in animals.groupby("animal_code"):
        animal = animal.sort_values("day")
        top.plot(
            animal["day"], animal["relative_volume"],
            color="#bdbdbd", linewidth=0.7, alpha=0.7, zorder=1,
        )

    band = curves[UPDATED].sort_values("day")
    band_future = band.loc[band["day"].to_numpy(float) > LANDMARK]
    top.fill_between(
        band_future["day"], band_future["prediction_low90"], band_future["prediction_high90"],
        color="#dadaeb", alpha=0.55, zorder=0, label="90 %-й интервал обновлённого",
    )

    # The unconditional fit stops at the landmark and the forecasts start after
    # it, with nothing drawn across the gap: they are different objects and a
    # connecting segment would assert a continuity that does not exist.
    fit, forecast = split(base, "predicted_relative_volume")
    top.plot(
        fit["day"], fit["predicted_relative_volume"],
        color="#525252", linewidth=2.4, zorder=4,
        label="безусловная подгонка, сутки 0–12",
    )
    for model, colour, label in (
        (BASE, "#2171b5", "априорный прогноз"),
        (UPDATED, "#e6550d", "обновление на 12-е сутки"),
        (REFRESHED, "#238b45", "обновление на 12-е и 14-е"),
    ):
        _, future = split(curves[model].sort_values("day"), "predicted_relative_volume")
        top.plot(
            future["day"] , future["predicted_relative_volume"],
            color=colour, linewidth=2.2, zorder=4, label=label,
        )

    points = observed.loc[observed["model"] == BASE].sort_values("day")
    spread = (
        animals.groupby("day")["relative_volume"].agg(["mean", "std"]).reset_index()
    )
    top.errorbar(
        spread["day"], spread["mean"], yerr=spread["std"],
        fmt="o", color="#252525", markersize=5, capsize=3, linewidth=1.1,
        zorder=5, label="наблюдение, среднее ± СКО",
    )

    # No masking band: the break belongs to the model curves, which are already
    # drawn as two separate objects. The animal trajectories run through the
    # landmark because observation is continuous there -- it is only the meaning
    # of the drawn line that changes.
    for axis in (top, bottom):
        axis.axvline(LANDMARK, color="#252525", linestyle="--", linewidth=1.2, zorder=2)
        axis.axvline(REFRESH, color="#969696", linestyle=":", linewidth=1.1, zorder=2)
        axis.grid(alpha=0.25, which="both")
    top.set_yscale("log")
    top.set_ylabel("относительный объём опухоли, логарифмическая ось")
    top.set_title(
        "Контроль 25.05.2026, доза 0: разрыв на рубеже и логарифмическая ось\n"
        "после насыщения переноса наклона поправка — параллельный сдвиг",
        fontsize=11,
    )
    top.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    top.annotate(
        "рубеж 12 сут: безусловная подгонка\nсменяется прогнозом, линия разорвана",
        xy=(LANDMARK, 3.4), xytext=(6.6, 1.5), fontsize=8.5, ha="center",
        color="#252525",
        arrowprops={"arrowstyle": "-|>", "color": "#525252", "linewidth": 0.9},
    )

    base_log = base.set_index("day")["predicted_log_relative"]
    for model, colour, label in (
        (UPDATED, "#e6550d", "обновление на 12-е сутки"),
        (REFRESHED, "#238b45", "обновление на 12-е и 14-е"),
    ):
        curve = curves[model].sort_values("day").set_index("day")
        delta = (curve["predicted_log_relative"] - base_log).dropna()
        delta = delta.loc[delta.index > LANDMARK]
        # The landmark day itself is carried at zero: the update multiplies the
        # whole correction by (day > landmark), so nothing is applied there. That
        # zero is the step, and omitting it would hide the very thing drawn.
        days = np.concatenate(([LANDMARK], delta.index.to_numpy(float)))
        values = np.concatenate(([0.0], delta.to_numpy()))
        # Stepped, not sloped: there is no day 12.5 at which the correction is
        # half applied. It is zero on the landmark and whole the next day.
        bottom.plot(days, values, color=colour, linewidth=2.0,
                    drawstyle="steps-post", label=label)
        bottom.plot(days[1:], values[1:], "o", color=colour, markersize=4)
    bottom.axhline(0.0, color="black", linewidth=0.9)
    bottom.annotate(
        "скачок $+0{,}289$\nна первые сутки прогноза",
        xy=(13.0, 0.145), xytext=(9.4, 0.205), fontsize=8.5, ha="center",
        color="#a63603",
        arrowprops={"arrowstyle": "-|>", "color": "#a63603", "linewidth": 0.9},
    )
    bottom.annotate(
        "перенос наклона исчерпан к 15-м суткам,\nдалее поправка строго постоянна",
        xy=(18.0, 0.2846), xytext=(17.4, 0.135), fontsize=8.5, ha="center",
        color="#252525",
        arrowprops={"arrowstyle": "-|>", "color": "#525252", "linewidth": 0.9},
    )
    bottom.set_ylim(-0.02, 0.34)
    bottom.set_xlim(-0.4, 21.6)
    bottom.set_xlabel("сутки после перевивки")
    bottom.set_ylabel("поправка, log")
    bottom.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    read = lambda name: pd.read_csv(args.source / name, sep=";", encoding="utf-8-sig")
    figure(
        read("last_control_daily_predictions.csv"),
        read("last_control_observed_predictions.csv"),
        read("last_control_animal_observations.csv"),
        args.output,
    )
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
