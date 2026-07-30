"""Decompose between-series variance of the group response and bound predictable R^2.

The prospective contour of task 4.6 reports a low coefficient of determination for
the day-21 endpoint, which invites two very different readings: either the group
mean is too noisy to be predicted at all, or the response is reproducible but
driven by factors the regimen predictors do not carry.

This script separates the two. It estimates how much of the observed variance of
series means is measurement noise of the group mean itself (giving a ceiling on
any model's R^2) and then attributes the remaining variance to candidate factors.
The calendar year is included deliberately: the external validation removes whole
years, so a large year share explains a weak prospective forecast without implying
that the model is misspecified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\animal_daily_interpolated.csv"
)
DEFAULT_OUTPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\variance_decomposition"
)

RESPONSE = "log_relative_volume"
FACTORS = ("year", "family", "control_kind")
GROUP_SIZES = (6, 8, 10, 12, 16, 20)


def load_treated(path: Path, max_day: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    treated = frame["is_treated"] == 1
    window = (frame["day"] > 0) & (frame["day"] <= max_day)
    return frame.loc[treated & window].copy()


def cell_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (series, day): animal count, group mean and within-series variance."""
    cells = (
        frame.groupby(["series_key", "day"])[RESPONSE]
        .agg(["count", "mean", "var"])
        .reset_index()
    )
    # A single animal gives no variance estimate and cannot bound the noise.
    return cells.dropna(subset=["var"]).loc[lambda d: d["count"] >= 2]


def ceiling(cells: pd.DataFrame) -> dict[str, float]:
    """Upper bound on R^2 when the model predicts the true group mean exactly.

    The observed series mean carries a sampling error of the animals it averages,
    so its variance is inflated by ``within / n``. That inflation is irreducible
    for any predictor built on regimen descriptors.
    """
    noise = float((cells["var"] / cells["count"]).mean())
    total = float(cells["mean"].var(ddof=1))
    return {
        "cells": int(len(cells)),
        "median_animals": float(cells["count"].median()),
        "var_noise": noise,
        "var_total": total,
        "r2_max": 1.0 - noise / total,
    }


def eta_squared(frame: pd.DataFrame, column: str, response: str) -> float:
    """Share of variance lying between levels of a categorical factor."""
    grand = frame[response].mean()
    between = sum(
        len(group) * (group.mean() - grand) ** 2
        for _, group in frame.groupby(column)[response]
    )
    total = float(((frame[response] - grand) ** 2).sum())
    return float(between / total)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint-day", type=int, default=21)
    args = parser.parse_args()

    frame = load_treated(args.input, args.endpoint_day)
    cells = cell_statistics(frame)

    ceiling_rows = []
    for label, subset in (
        (f"day_{args.endpoint_day}", cells[cells["day"] == args.endpoint_day]),
        ("day_14", cells[cells["day"] == 14]),
        (f"pooled_1_{args.endpoint_day}", cells),
    ):
        if len(subset) >= 3:
            ceiling_rows.append({"scope": label, **ceiling(subset)})

    endpoint = cells[cells["day"] == args.endpoint_day]
    endpoint_total = float(endpoint["mean"].var(ddof=1))
    mean_within = float(endpoint["var"].mean())
    size_rows = [
        {
            "animals_per_series": size,
            "r2_max": 1.0 - (mean_within / size) / endpoint_total,
        }
        for size in GROUP_SIZES
    ]

    # Attribution uses one row per series so that long series do not dominate.
    series = (
        frame[frame["day"] == args.endpoint_day]
        .groupby(["series_key", *FACTORS, "total_dose_gy"])[RESPONSE]
        .mean()
        .reset_index()
        .rename(columns={RESPONSE: "series_mean"})
    )
    factor_rows = [
        {
            "factor": name,
            "levels": int(series[name].nunique()),
            "eta_squared": eta_squared(series, name, "series_mean"),
        }
        for name in FACTORS
    ]
    series["year_family"] = series["year"].astype(str) + "|" + series["family"]
    factor_rows.append(
        {
            "factor": "year_family",
            "levels": int(series["year_family"].nunique()),
            "eta_squared": eta_squared(series, "year_family", "series_mean"),
        }
    )
    dose_r = float(np.corrcoef(series["total_dose_gy"], series["series_mean"])[0, 1])
    factor_rows.append(
        {"factor": "total_dose_gy", "levels": 0, "eta_squared": dose_r**2}
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output / "r2_ceiling.csv",
        ceiling_rows,
        ["scope", "cells", "median_animals", "var_noise", "var_total", "r2_max"],
    )
    write_csv(
        args.output / "r2_ceiling_by_group_size.csv",
        size_rows,
        ["animals_per_series", "r2_max"],
    )
    write_csv(
        args.output / "variance_attribution.csv",
        factor_rows,
        ["factor", "levels", "eta_squared"],
    )

    print(f"series at day {args.endpoint_day}: {len(series)}")
    for row in ceiling_rows:
        print(
            f"  {row['scope']:<16} n_med={row['median_animals']:.1f} "
            f"var_noise={row['var_noise']:.4f} var_total={row['var_total']:.4f} "
            f"R2_max={row['r2_max']:.3f}"
        )
    for row in factor_rows:
        print(f"  {row['factor']:<16} eta2={row['eta_squared']:.3f}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
