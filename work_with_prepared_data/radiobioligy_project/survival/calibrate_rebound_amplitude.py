"""Calibrate the amplitude of the predicted rebound after the nadir.

The forecast reproduces about four fifths of the observed rise from the trajectory
minimum to day 21 -- median +0.994 against +1.238 across the archive -- which is
what ridge shrinkage does to an amplitude. The day-21 endpoint sits on that rising
limb, so the compression shows up there directly and is the largest identified
systematic error left in the model.

This applies the calibration-slope correction standard in prediction-model work:
regress the observed rise on the predicted rise and rescale. Two choices keep it
honest. The nadir is taken from the *predicted* trajectory, since that is what is
available when a forecast is issued; using the observed minimum would leak the
outcome. And the slope is fitted for each held-out year on the out-of-fold
predictions of the other years only, never on the year being corrected.

Development is internal by necessity: the held-out June 2026 series are spent, so
whatever this produces is cross-validated evidence and must be reported as such.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Ковариаты_pole_gamma"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "rebound_calibration"
ENDPOINT_DAY = 21
MODEL = "stacked_ensemble"


def rise_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per row: rise since the predicted nadir, observed and predicted."""
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(["cv_kind", "series_key"], sort=False):
        group = group.sort_values("day").copy()
        nadir = group["predicted_log_relative"].idxmin()
        nadir_day = float(group.loc[nadir, "day"])
        base_pred = float(group.loc[nadir, "predicted_log_relative"])
        base_obs = float(group.loc[nadir, "actual_log_relative"])
        group["nadir_day"] = nadir_day
        group["predicted_rise"] = group["predicted_log_relative"] - base_pred
        group["observed_rise"] = group["actual_log_relative"] - base_obs
        group["on_rising_limb"] = group["day"] > nadir_day
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def fit_slope(frame: pd.DataFrame) -> float:
    """Least squares through the origin: at the nadir both rises are zero."""
    limb = frame.loc[frame["on_rising_limb"]]
    x = limb["predicted_rise"].to_numpy(float)
    y = limb["observed_rise"].to_numpy(float)
    denominator = float((x * x).sum())
    return float((x * y).sum() / denominator) if denominator > 0 else 1.0


def metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    endpoint = frame.loc[np.isclose(frame["day"], ENDPOINT_DAY)]
    actual = endpoint["actual_log_relative"].to_numpy(float)
    predicted = endpoint[column].to_numpy(float)
    total = float(((actual - actual.mean()) ** 2).sum())
    residual = float(((actual - predicted) ** 2).sum())
    per_series = (
        frame.assign(error=(frame["actual_log_relative"] - frame[column]) ** 2)
        .groupby("series_key")["error"]
        .mean()
        .pow(0.5)
    )
    return {
        "day21_log_r2": 1.0 - residual / total,
        "day21_log_rmse": float(np.sqrt(residual / len(actual))),
        "mean_series_log_rmse": float(per_series.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.input / "rebound_calibration")

    frame = pd.read_csv(
        args.input / "outer_cv_series_day_predictions.csv", sep=";", encoding="utf-8-sig"
    )
    frame = frame.loc[(frame["model"] == MODEL) & (frame["family"] != "control")]

    rows: list[dict] = []
    calibrated_parts: list[pd.DataFrame] = []
    for cv_kind, scheme in frame.groupby("cv_kind"):
        table = rise_table(scheme)
        pooled_slope = fit_slope(table)
        corrected = np.empty(len(table))
        slopes: list[dict] = []
        for holdout, group in table.groupby("holdout"):
            others = table.loc[table["holdout"] != holdout]
            slope = fit_slope(others) if len(others) else 1.0
            slopes.append({"cv_kind": cv_kind, "holdout": holdout, "slope": slope})
            index = table.index.get_indexer(group.index)
            adjusted = group["predicted_log_relative"].to_numpy(float).copy()
            limb = group["on_rising_limb"].to_numpy(bool)
            adjusted[limb] = (
                group["predicted_log_relative"].to_numpy(float)[limb]
                - group["predicted_rise"].to_numpy(float)[limb]
                + slope * group["predicted_rise"].to_numpy(float)[limb]
            )
            corrected[index] = adjusted
        table["calibrated_log_relative"] = corrected
        calibrated_parts.append(table)

        before = metrics(table, "predicted_log_relative")
        after = metrics(table, "calibrated_log_relative")
        rows.append(
            {
                "cv_kind": cv_kind,
                "pooled_slope": pooled_slope,
                "fold_slope_min": min(s["slope"] for s in slopes),
                "fold_slope_max": max(s["slope"] for s in slopes),
                **{f"{key}_before": value for key, value in before.items()},
                **{f"{key}_after": value for key, value in after.items()},
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    pd.concat(calibrated_parts, ignore_index=True).to_csv(
        output / "calibrated_predictions.csv", sep=";", index=False
    )
    with (output / "calibration_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"\n{row['cv_kind']}  slope={row['pooled_slope']:.3f} "
              f"(folds {row['fold_slope_min']:.3f}–{row['fold_slope_max']:.3f})")
        for key in ("day21_log_r2", "day21_log_rmse", "mean_series_log_rmse"):
            print(f"    {key:<22} {row[key + '_before']:+.4f} -> {row[key + '_after']:+.4f}")
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
