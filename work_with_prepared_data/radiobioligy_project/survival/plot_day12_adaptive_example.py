"""Plot a reproducible example of the day-12 adaptive tumour forecast.

The example is selected without looking for the most favourable trajectory: among
treated series in the rolling-origin validation it is the series whose log-RMSE
over days 13--21 is closest to the cohort median.  Observations through day 12
are available to the adaptive update; later observations are displayed only as
held-out verification points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель"
    r"\prediction_v5_landmark_trend\outer_predictions.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\dev\dissertation_text\figures\figure_day12_adaptive_example"
)
CV_KIND = "rolling_origin_year"
LANDMARK_DAY = 12
APRIORI_MODEL = "apriori_pole_gamma"
ADAPTIVE_MODEL = "adaptive_trend"


def select_representative(frame: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    adaptive = frame.loc[
        (frame["cv_kind"] == CV_KIND)
        & (frame["model"] == ADAPTIVE_MODEL)
        & (frame["landmark_day"].astype(int) == LANDMARK_DAY)
        & (frame["is_treated"].astype(int) == 1)
        & (frame["day"].astype(float) > LANDMARK_DAY)
    ].copy()
    adaptive["squared_error"] = np.square(
        adaptive["actual_log_relative"] - adaptive["predicted_log_relative"]
    )
    scores = (
        adaptive.groupby("series_key", sort=False)
        .agg(
            future_log_rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            n_future=("squared_error", "size"),
        )
        .reset_index()
    )
    scores = scores.loc[scores["n_future"] >= 5].copy()
    median_rmse = float(scores["future_log_rmse"].median())
    scores["distance_to_median"] = np.abs(scores["future_log_rmse"] - median_rmse)
    selected = scores.sort_values(
        ["distance_to_median", "series_key"], kind="mergesort"
    ).iloc[0]
    selected_frame = frame.loc[
        (frame["cv_kind"] == CV_KIND)
        & (frame["landmark_day"].astype(int) == LANDMARK_DAY)
        & (frame["series_key"] == selected["series_key"])
        & frame["model"].isin([APRIORI_MODEL, ADAPTIVE_MODEL])
    ].copy()
    selected_frame.attrs["median_future_log_rmse"] = median_rmse
    selected_frame.attrs["selected_future_log_rmse"] = float(
        selected["future_log_rmse"]
    )
    return str(selected["series_key"]), selected_frame


def plot_example(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    apriori = frame.loc[frame["model"] == APRIORI_MODEL].sort_values("day")
    adaptive = frame.loc[frame["model"] == ADAPTIVE_MODEL].sort_values("day")
    if apriori.empty or adaptive.empty:
        raise ValueError("The selected series lacks apriori or adaptive predictions")

    day = adaptive["day"].to_numpy(float)
    observed = adaptive["actual_relative_volume"].to_numpy(float)
    predicted = adaptive["predicted_relative_volume"].to_numpy(float)
    future = day > LANDMARK_DAY
    known = day <= LANDMARK_DAY

    fig, axis = plt.subplots(figsize=(8.7, 5.5))
    axis.fill_between(
        day[future],
        adaptive.loc[future, "prediction_low95"].to_numpy(float),
        adaptive.loc[future, "prediction_high95"].to_numpy(float),
        color="#fcbba1",
        alpha=0.35,
        label="95 %-й прогнозный интервал",
        zorder=0,
    )
    axis.plot(
        apriori["day"],
        apriori["predicted_relative_volume"],
        color="#3182bd",
        linewidth=2.2,
        label="априорный прогноз",
        zorder=2,
    )
    axis.plot(
        day[future],
        predicted[future],
        color="#cb181d",
        linewidth=2.5,
        label="адаптивный прогноз после 12-х суток",
        zorder=4,
    )
    axis.plot(
        day[known],
        observed[known],
        "o-",
        color="#252525",
        linewidth=1.3,
        markersize=4.7,
        label="наблюдения, доступные при обновлении",
        zorder=5,
    )
    axis.plot(
        day[future],
        observed[future],
        "o",
        markerfacecolor="white",
        markeredgecolor="#252525",
        markeredgewidth=1.3,
        markersize=5.2,
        label="скрытые точки 13–21-х суток",
        zorder=6,
    )
    day12 = adaptive.loc[np.isclose(adaptive["day"], LANDMARK_DAY)].iloc[0]
    axis.scatter(
        [LANDMARK_DAY],
        [day12["actual_relative_volume"]],
        s=92,
        facecolor="#fed976",
        edgecolor="#252525",
        linewidth=1.2,
        zorder=7,
    )
    axis.axvline(
        LANDMARK_DAY, color="#636363", linestyle="--", linewidth=1.2, zorder=1
    )
    axis.text(
        LANDMARK_DAY + 0.18,
        axis.get_ylim()[1] * 0.96,
        "рубеж обновления",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#525252",
    )

    family = str(adaptive["family_label"].iloc[0])
    dose = float(adaptive["total_dose_gy"].iloc[0])
    date = str(adaptive["date"].iloc[0])
    selected_rmse = float(frame.attrs["selected_future_log_rmse"])
    median_rmse = float(frame.attrs["median_future_log_rmse"])
    axis.set_title(
        "Пример адаптивного прогноза групповой траектории\n"
        f"{family}, {dose:.1f} Гр; серия {date}",
        fontsize=12,
    )
    axis.text(
        0.02,
        0.97,
        "Серия выбрана по заранее заданному правилу:\n"
        "ошибка продолжения ближе всего к медиане когорты\n"
        f"лог-СКО {selected_rmse:.3f}; медиана {median_rmse:.3f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "#bdbdbd"},
    )
    axis.set_xlabel("сутки после облучения")
    axis.set_ylabel("относительный объём опухоли")
    axis.set_xlim(0, 21.5)
    axis.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    axis.grid(alpha=0.22)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8.4,
        framealpha=0.94,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)

    metadata = pd.DataFrame(
        [
            {
                "selection_rule": "closest_to_median_future_log_rmse",
                "cv_kind": CV_KIND,
                "landmark_day": LANDMARK_DAY,
                "series_key": str(adaptive["series_key"].iloc[0]),
                "date": date,
                "family_label": family,
                "total_dose_gy": dose,
                "selected_future_log_rmse": selected_rmse,
                "cohort_median_future_log_rmse": median_rmse,
                "n_future_days": int(np.sum(future)),
            }
        ]
    )
    metadata.to_csv(output.with_name(output.name + "_metadata.csv"), sep=";", index=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, sep=";")
    _, selected = select_representative(frame)
    metadata = plot_example(selected, args.output)
    print(metadata.to_string(index=False))
    print(f"written to {args.output.parent}")


if __name__ == "__main__":
    main()
