"""Is the housing season confounded, and what does it actually do to the outcome?

The seasonal covariate is the only epoch proxy in this work that satisfies the
transferability criterion by construction: two harmonics of the day of the year
vary inside every calendar year, which is exactly what calibration drift and the
source-decay term could not do. That makes it worth asking what it correlates
with before asking what it explains.

Two questions, two panels. Whether the harmonics are entangled with dose, family
or year -- if they were, the seasonal coefficient would be something else wearing
a seasonal label, the trap this work has fallen into twice. And what the seasonal
shape in the outcome looks like once the calendar year is removed, since a
between-year effect read as seasonal would be the same trap again.

The outcome is the day-21 log relative volume of treated series, demeaned within
year. Demeaning first is the whole point: it strips the epoch level and leaves
only what varies inside the year, which is the only part a seasonal term may
legitimately claim.
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
from matplotlib.colors import LinearSegmentedColormap


BASE_SCRIPT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Ковариаты_pole_gamma\run_growth_prediction.py"
)
DEFAULT_OUTPUT = Path(r"C:\dev\dissertation_text\figures\figure_seasonality")

DIVERGING = LinearSegmentedColormap.from_list(
    "corr", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a09f", "#e34948", "#8f2020"]
)
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
WINDOW_DAY = 21.0

OTHER = {
    "total_dose_gy": "суммарная доза",
    "sum_d2_gy2": "сумма квадратов доз",
    "n_events": "число воздействий",
    "duration_hours": "длительность курса",
    "log_v0": "исходный объём, log",
    "is_mixed": "смешанный режим",
    "year_num": "календарный год",
}
HARMONICS = {
    "season_sin1": "sin φ",
    "season_cos1": "cos φ",
    "season_sin2": "sin 2φ",
    "season_cos2": "cos 2φ",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seasonal_terms(dates: pd.Series) -> pd.DataFrame:
    """The same phase convention the covariate uses: day of year shifted by ten."""
    day_of_year = pd.to_datetime(dates).dt.dayofyear.to_numpy(float)
    phase = 2.0 * np.pi * ((day_of_year + 10.0) % 365.0) / 365.0
    return pd.DataFrame({
        "season_sin1": np.sin(phase),
        "season_cos1": np.cos(phase),
        "season_sin2": np.sin(2.0 * phase),
        "season_cos2": np.cos(2.0 * phase),
        "phase": phase,
        "day_of_year": day_of_year,
    })


def figure(series: pd.DataFrame, output: Path) -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": (1.45, 1.0)}
    )
    fig.patch.set_facecolor(SURFACE)
    for axis in (left, right):
        axis.set_facecolor(SURFACE)

    # --- seasonal shape in the year-demeaned outcome ---
    order = np.argsort(series["day_of_year"].to_numpy())
    day = series["day_of_year"].to_numpy()[order]
    residual = series["demeaned"].to_numpy()[order]
    design = np.column_stack([
        np.ones_like(day),
        series["season_sin1"].to_numpy()[order], series["season_cos1"].to_numpy()[order],
        series["season_sin2"].to_numpy()[order], series["season_cos2"].to_numpy()[order],
    ])
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    grid = np.linspace(1, 365, 365)
    grid_phase = 2.0 * np.pi * ((grid + 10.0) % 365.0) / 365.0
    curve = (coefficients[0]
             + coefficients[1] * np.sin(grid_phase) + coefficients[2] * np.cos(grid_phase)
             + coefficients[3] * np.sin(2 * grid_phase) + coefficients[4] * np.cos(2 * grid_phase))
    explained = 1.0 - ((residual - design @ coefficients) ** 2).sum() / (residual ** 2).sum()

    left.axhline(0.0, color=INK_MUTED, linewidth=0.9)
    left.plot(day, residual, "o", color=SERIES, markersize=7, alpha=0.75,
              markeredgecolor=SURFACE, markeredgewidth=1.5)
    left.plot(grid, curve, color="#8f2020", linewidth=2.2)
    left.set_xticks([15, 105, 197, 288], ["январь", "апрель", "июль", "октябрь"],
                    fontsize=9, color=INK)
    left.set_xlim(0, 366)
    left.set_ylabel("отклонение от среднего своего года, log", fontsize=9.5, color=INK)
    left.set_title(
        f"Сезонная форма в отклике на 21-е сутки\nдве гармоники объясняют {explained:.3f} "
        "внутригодовой дисперсии", fontsize=10.5, color=INK, pad=10)
    left.grid(alpha=0.22)
    for side in left.spines.values():
        side.set_color("#d8d7d2")

    # --- what the harmonics are entangled with ---
    matrix = np.array([
        [series[h].corr(series[o]) for o in OTHER] for h in HARMONICS
    ], dtype=float)
    image = right.imshow(matrix, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    right.set_xticks(range(len(OTHER)), list(OTHER.values()), rotation=40, ha="right",
                     fontsize=9, color=INK)
    right.set_yticks(range(len(HARMONICS)), list(HARMONICS.values()), fontsize=10, color=INK)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            right.text(j, i, f"{value:+.2f}".replace("0.", ".").replace("-", "\u2212"),
                       ha="center", va="center", fontsize=8.5,
                       color="#ffffff" if abs(value) > 0.62 else INK)
    right.set_xticks(np.arange(-0.5, len(OTHER), 1), minor=True)
    right.set_yticks(np.arange(-0.5, len(HARMONICS), 1), minor=True)
    right.grid(which="minor", color=SURFACE, linewidth=2.0)
    right.tick_params(which="minor", length=0)
    right.tick_params(length=0)
    for side in right.spines.values():
        side.set_visible(False)
    right.set_title("С чем сцеплены сами гармоники\nкоэффициент корреляции Пирсона",
                    fontsize=10.5, color=INK, pad=10)
    bar = fig.colorbar(image, ax=right, fraction=0.045, pad=0.03, ticks=[-1, 0, 1])
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    bar.outline.set_visible(False)

    fig.suptitle(
        "Сезонность: единственный признак эпохи, меняющийся внутри года",
        fontsize=11.5, color=INK,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return explained, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = load_module("season_base", BASE_SCRIPT)
    _, evaluation, _, _ = base.build_longitudinal_tables()
    data = base.collapse_animal_daily_to_series(evaluation)

    endpoint = data.loc[
        (data["is_treated"].astype(int) == 1)
        & np.isclose(data["day"].to_numpy(float), WINDOW_DAY)
    ].drop_duplicates("series_key").reset_index(drop=True)
    series = pd.concat([endpoint, seasonal_terms(endpoint["date"])], axis=1)
    series["year_num"] = series["year"].astype(int)
    # Within-year demeaning: the epoch level is exactly what a seasonal term may
    # not claim, so it is removed before the seasonal shape is looked at.
    series["demeaned"] = (
        series["log_relative_volume"]
        - series.groupby("year_num")["log_relative_volume"].transform("mean")
    )

    print(f"облучённых серий с точкой 21 сут: {len(series)}")
    coverage = series.groupby("year_num")["day_of_year"].agg(
        серий="size", мин="min", макс="max"
    )
    coverage["охват_года"] = (coverage["макс"] - coverage["мин"]) / 365.0
    print("\n=== покрытие года по годам ===")
    print(coverage.round(2).to_string())

    explained, matrix = figure(series, args.output)
    print(f"\nдве гармоники объясняют {explained:.3f} внутригодовой дисперсии отклика")
    print("\n=== корреляция гармоник с прочими предикторами ===")
    # ASCII row names for the console: the Windows code page cannot encode phi.
    print(pd.DataFrame(matrix, index=["sin1", "cos1", "sin2", "cos2"],
                       columns=list(OTHER.values())).round(3).to_string())
    print(f"\nнаибольшая по модулю: {np.abs(matrix).max():.3f}")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
