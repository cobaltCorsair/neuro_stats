"""The landmark step on the treated June series, drawn as figure 3.15 draws it.

Same two corrections as the zero-dose figure: a logarithmic ordinate, on which a
constant log offset is a parallel displacement rather than a widening wedge, and
a break at the landmark, because the line before it is the unconditional fit and
the line after it a forecast conditioned on days 1 to 12.

The contrast with the control is the point. There the correction was +0.289 and
carried the forecast towards the observations for most of the window. Here it is
-0.052 and -0.344 -- downward, while both series lie above the forecast. The
update is not merely insufficient on the heavier series, it moves the wrong way,
which is what the residual traces in figure 3.13 already show and what the
control figure could not show, having no such reversal.

These are the held-out series and the updating contour applied to them is
post-hoc under amendment 11. Nothing here is new analysis: the quantities drawn
are those already reported in table 3.46, redrawn.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOLDOUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Внешняя_проверка_июнь_2026"
)
HOLDOUT_SCRIPT = Path(__file__).resolve().with_name(
    "validate_growth_prediction_holdout.py"
)
DEFAULT_OUTPUT = Path(r"C:\dev\dissertation_text\figures\figure_holdout_landmark_step")
LANDMARK = 12.0
LABELS = {29.8: "29,8 Гр, 3 фракции", 35.0: "35,0 Гр, 4 фракции"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def animal_frame() -> pd.DataFrame:
    """Per-animal relative volumes, rebuilt from the held-out books."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    holdout = load_module("holdout_for_plot", HOLDOUT_SCRIPT)
    engine = load_module("engine_for_plot", holdout.ENGINE)
    rows: list[dict] = []
    for name, meta in holdout.HOLDOUT_SERIES.items():
        for row in holdout.animal_rows(
            engine, holdout.HOLDOUT_DIR / name, meta, "mixed_n_p_through"
        ):
            rows.append(
                {
                    "total_dose_gy": row["total_dose_gy"],
                    "animal_id": row["animal_id"],
                    "day": row["day"],
                    "relative_volume": float(np.exp(row["log_relative_volume"])),
                }
            )
    return pd.DataFrame(rows)


def figure(frame: pd.DataFrame, animals: pd.DataFrame, output: Path) -> None:
    doses = sorted(frame["total_dose_gy"].unique())
    fig, axes = plt.subplots(
        2, len(doses), figsize=(11.8, 8.0), height_ratios=(2.4, 1.0), sharex=True
    )
    axes = np.atleast_2d(axes)

    for column, dose in enumerate(doses):
        group = frame.loc[np.isclose(frame["total_dose_gy"], dose)].sort_values("day")
        herd = animals.loc[np.isclose(animals["total_dose_gy"], dose)]
        top, bottom = axes[0, column], axes[1, column]

        # A logarithmic axis has no room for a complete regression. Blanking the
        # zeros breaks the line there instead of pinning it to the axis floor,
        # where it would read as a small volume rather than none at all; the
        # days concerned are marked below. They stay in the mean and the spread,
        # which are computed on volumes and to which zero is a legitimate value.
        vanished = herd.loc[herd["relative_volume"] <= 0.0]
        for _, animal in herd.groupby("animal_id"):
            animal = animal.sort_values("day")
            drawable = animal["relative_volume"].where(animal["relative_volume"] > 0.0)
            top.plot(
                animal["day"], drawable,
                color="#bdbdbd", linewidth=0.7, alpha=0.75, zorder=1,
            )

        future = group.loc[group["day"].to_numpy(float) > LANDMARK]
        top.fill_between(
            future["day"], future["prediction_low90"], future["prediction_high90"],
            color="#dadaeb", alpha=0.5, zorder=0, label="90 %-й интервал",
        )

        fit = group.loc[group["day"].to_numpy(float) <= LANDMARK]
        top.plot(
            fit["day"], fit["predicted_relative_volume"],
            color="#525252", linewidth=2.4, zorder=4,
            label="безусловная подгонка, сутки 1–12",
        )
        top.plot(
            future["day"], future["predicted_relative_volume"],
            color="#2171b5", linewidth=2.2, zorder=4, label="априорный прогноз",
        )
        top.plot(
            future["day"], np.exp(future["adaptive_log_relative"]),
            color="#cb181d", linewidth=2.2, linestyle="--", zorder=4,
            label="обновление на 12-е сутки",
        )

        spread = herd.groupby("day")["relative_volume"].agg(["mean", "std"]).reset_index()
        top.errorbar(
            spread["day"], spread["mean"], yerr=spread["std"],
            fmt="o", color="#252525", markersize=4.5, capsize=3, linewidth=1.0,
            zorder=5, label="наблюдение, среднее ± СКО",
        )

        top.set_yscale("log")
        top.set_title(LABELS.get(dose, f"{dose} Гр"), fontsize=10.5)
        if not vanished.empty:
            floor = top.get_ylim()[0]
            top.plot(
                vanished["day"], np.full(len(vanished), floor * 1.06),
                "v", color="#6a51a3", markersize=6, zorder=6, clip_on=False,
                label="полная регрессия, объём 0",
            )

        correction = (
            group["adaptive_log_relative"] - group["predicted_log_relative"]
        ).to_numpy(float)
        days = group["day"].to_numpy(float)
        keep = days >= LANDMARK
        bottom.plot(
            days[keep], correction[keep], color="#cb181d", linewidth=2.0,
            drawstyle="steps-post",
        )
        bottom.plot(days[keep & (days > LANDMARK)], correction[keep & (days > LANDMARK)],
                    "o", color="#cb181d", markersize=4)
        bottom.axhline(0.0, color="black", linewidth=0.9)
        step = float(correction[days > LANDMARK][0])
        bottom.annotate(
            f"поправка ${step:+.3f}$".replace(".", "{,}"),
            xy=(15.5, step), xytext=(16.0, step - 0.16 if step > -0.2 else step + 0.13),
            fontsize=9, ha="center", color="#a50f15",
            arrowprops={"arrowstyle": "-|>", "color": "#a50f15", "linewidth": 0.9},
        )

        for axis in (top, bottom):
            axis.axvline(LANDMARK, color="#252525", linestyle="--", linewidth=1.2, zorder=2)
            axis.grid(alpha=0.25, which="both")
        bottom.set_ylim(-0.62, 0.30)
        bottom.set_xlim(0.4, 21.6)
        bottom.set_xlabel("сутки после облучения")

    axes[0, 0].set_ylabel("относительный объём, логарифмическая ось")
    axes[1, 0].set_ylabel("поправка, log")
    for axis in axes[0]:
        handles, names = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, names, fontsize=8, loc="lower left", framealpha=0.92)
    fig.suptitle(
        "Облучённые серии июня 2026 года: разрыв на рубеже и логарифмическая ось\n"
        "поправка направлена вниз, тогда как наблюдение лежит выше прогноза",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.holdout / "holdout_adaptive_predictions.csv", sep=";")
    figure(frame, animal_frame(), args.output)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
