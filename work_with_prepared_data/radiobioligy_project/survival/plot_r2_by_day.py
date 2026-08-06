"""Up to which day does the a priori forecast beat the series mean?

The endpoint coefficient of determination is negative, which invites the reading
that the model predicts nothing. Taken day by day the statement is finer and more
useful, and it needs one caveat stated before any of the numbers.

Early in the window the coefficient is uninformative rather than bad. It compares
the prediction error against the spread between series, and on day 1 that spread
is 0.073 in log while by day 21 it is 0.943: there is almost nothing to predict at
the start, so any error at all drives the ratio far below zero. Days before the
spread has developed are therefore plotted but not read.

What the curves then show is that the a priori forecast carries the shape of the
trajectory and not the level of a new epoch, and that one measurement supplies
exactly what it lacks -- the updated forecast leaps from 0.17 to 0.91 on the day
after the landmark and is still at 0.59 nine days later.
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


BASE_SCRIPT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Ковариаты_pole_gamma\run_growth_prediction.py"
)
PREDICTIONS = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Итоговая_прогностическая_модель"
    r"\prediction_v5_landmark_trend\outer_predictions.csv"
)
OUTPUT_CSV = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\r2_by_day.csv"
)
DEFAULT_FIGURE = Path(r"C:\dev\dissertation_text\figures\figure_r2_by_day")
LANDMARK = 12.0
SCHEMES = {
    "leave_one_year_out": "исключение года",
    "rolling_origin_year": "скользящее начало",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(base) -> pd.DataFrame:
    frame = pd.read_csv(PREDICTIONS, sep=";", encoding="utf-8-sig")
    frame = frame.loc[
        (frame["is_treated"].astype(int) == 1)
        & (frame["landmark_day"].astype(int) == int(LANDMARK))
        & frame["model"].isin(["apriori_pole_gamma", "adaptive_trend"])
    ]
    rows: list[dict] = []
    for (scheme, model, day), group in frame.groupby(["cv_kind", "model", "day"]):
        if day <= 0:
            continue
        weights = base.series_balanced_weights(group)
        actual = group["actual_log_relative"].to_numpy(float)
        predicted = group["predicted_log_relative"].to_numpy(float)
        mean = np.average(actual, weights=weights)
        rows.append(
            {
                "cv_kind": scheme,
                "model": model,
                "day": float(day),
                "r2": base.weighted_r2(actual, predicted, weights),
                "between_series_sd": float(
                    np.sqrt(np.average(np.square(actual - mean), weights=weights))
                ),
                "prediction_rmse": float(
                    np.sqrt(np.average(np.square(actual - predicted), weights=weights))
                ),
            }
        )
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6), height_ratios=(2.2, 1.0),
                             sharex=True)
    for column, (scheme, label) in enumerate(SCHEMES.items()):
        top, bottom = axes[0, column], axes[1, column]
        part = table.loc[table["cv_kind"] == scheme]
        prior = part.loc[part["model"] == "apriori_pole_gamma"].sort_values("day")
        updated = part.loc[part["model"] == "adaptive_trend"].sort_values("day")

        # Before the spread between series has developed there is nothing to
        # predict, so the ratio is uninformative rather than damning. Shaded, and
        # excluded from any claim.
        top.axvspan(0.5, 8.5, color="#f0f0f0", zorder=0)
        top.annotate("предсказывать\nещё нечего", xy=(4.4, -1.55), fontsize=8,
                     ha="center", color="#737373")
        top.axhline(0.0, color="#252525", linewidth=1.0)
        top.plot(prior["day"], prior["r2"], "o-", color="#2171b5", linewidth=2.0,
                 markersize=4, label="априорный прогноз")
        future = updated.loc[updated["day"].to_numpy(float) > LANDMARK]
        top.plot(future["day"], future["r2"], "s-", color="#cb181d", linewidth=2.0,
                 markersize=4, label="после обновления на 12-е сутки")
        top.axvline(LANDMARK, color="#969696", linestyle="--", linewidth=1.1)
        top.set_ylim(-2.0, 1.0)
        top.set_title(label, fontsize=10.5)
        top.grid(alpha=0.25)
        if column == 0:
            top.set_ylabel("коэффициент детерминации, log")
            top.legend(fontsize=8.5, loc="lower left")

        bottom.plot(prior["day"], prior["between_series_sd"], color="#252525",
                    linewidth=1.8, label="разброс между сериями")
        bottom.plot(prior["day"], prior["prediction_rmse"], color="#2171b5",
                    linewidth=1.8, linestyle="--", label="ошибка прогноза")
        bottom.axvline(LANDMARK, color="#969696", linestyle="--", linewidth=1.1)
        bottom.grid(alpha=0.25)
        bottom.set_xlabel("сутки после облучения")
        if column == 0:
            bottom.set_ylabel("логарифмическая шкала")
            bottom.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "До каких суток прогноз превосходит среднее по сериям\n"
        "внизу — что предсказывается и с какой ошибкой",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    base = load_module("r2_base", BASE_SCRIPT)
    table = build(base)
    table.to_csv(OUTPUT_CSV, sep=";", index=False, encoding="utf-8-sig")
    figure(table, args.output)

    pd.set_option("display.width", 200)
    for scheme, label in SCHEMES.items():
        prior = table.loc[
            (table["cv_kind"] == scheme) & (table["model"] == "apriori_pole_gamma")
        ].sort_values("day")
        positive = prior.loc[prior["r2"] > 0]
        informative = prior.loc[prior["day"] >= 9]
        print(f"\n=== {label} ===")
        if positive.empty:
            print("  априорный прогноз не превосходит среднее ни на одних сутках")
        else:
            print(f"  априорный превосходит среднее с {positive['day'].min():.0f} "
                  f"по {positive['day'].max():.0f} сутки, "
                  f"наибольшее значение {positive['r2'].max():.3f} "
                  f"на {positive.loc[positive['r2'].idxmax(), 'day']:.0f}-е")
        print(f"  на 21-е сутки: {prior.loc[np.isclose(prior['day'], 21.0), 'r2'].iloc[0]:+.3f}")
        upd = table.loc[
            (table["cv_kind"] == scheme) & (table["model"] == "adaptive_trend")
        ].sort_values("day")
        for day in (13.0, 21.0):
            print(f"  после обновления, {day:.0f}-е сутки: "
                  f"{upd.loc[np.isclose(upd['day'], day), 'r2'].iloc[0]:+.3f}")
    print(f"\nwritten to {OUTPUT_CSV} and {args.output}")


if __name__ == "__main__":
    main()
