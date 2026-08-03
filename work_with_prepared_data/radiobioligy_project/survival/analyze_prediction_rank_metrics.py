"""Rank agreement between predicted and observed day-21 response.

The prospective contour of task 4.6 is reported through the coefficient of
determination, which penalises any shift of the absolute level.  Because the
calendar epoch moves that level and external validation removes the epoch by
construction, a low ``R^2`` conflates two different failures: not knowing where
the new series sits, and not knowing how its regimens order among themselves.

Rank agreement separates them.  It is invariant to a monotone shift of the whole
epoch, so it measures only the second question -- whether the model puts the
regimens of an unseen period in the right order.  That is the question a
regimen comparison actually asks.

Two summaries are produced.  The pooled coefficient mixes within- and
between-epoch ordering and is therefore optimistic for a forecasting claim.  The
within-year coefficient, averaged over held-out years weighted by series count,
is the honest one: every pair it scores comes from the same epoch.

The electron subset is reported separately.  Its exclusion is decided on
documentary grounds established before this analysis -- the applicator-dependent
calibration of the electron unit spans a factor of 2.5 and its delivered dose
carries a systematic of up to 2.3 -- and not on the metric computed here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_INPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Ковариаты_тубус_v2"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "rank_metrics"

ENDPOINT_DAY = 21
MIN_SERIES_PER_YEAR = 4
ACTUAL = "actual_log_relative"
PREDICTED = "predicted_log_relative"


def load_endpoint(run_dir: Path) -> pd.DataFrame:
    """Day-21 rows for treated series only.

    Control series enter the cohort as dose-zero rows and are ordered trivially --
    an untreated tumour outgrows every irradiated one -- so including them inflates
    the coefficient in whatever years happen to have a digitised control. Adding
    the 2020 control moved that year from -1.000 to -0.086 without any change in
    how the treated series were predicted. The question the measure is meant to
    answer concerns the ordering of irradiated regimens, so controls are dropped.
    """
    frame = pd.read_csv(
        run_dir / "outer_cv_series_day_predictions.csv", sep=";", encoding="utf-8-sig"
    )
    endpoint = frame.loc[frame["day"] == ENDPOINT_DAY].copy()
    return endpoint.loc[endpoint["family"].astype(str) != "control"]


def within_year_rho(frame: pd.DataFrame) -> tuple[float, int, int]:
    """Series-weighted mean of the per-year rank coefficients."""
    weights: list[int] = []
    values: list[float] = []
    for _, group in frame.groupby("year"):
        if len(group) < MIN_SERIES_PER_YEAR:
            continue
        rho = spearmanr(group[PREDICTED], group[ACTUAL])[0]
        if np.isnan(rho):
            continue
        weights.append(len(group))
        values.append(float(rho))
    if not weights:
        return float("nan"), 0, 0
    w = np.asarray(weights, dtype=float)
    return float((w * np.asarray(values)).sum() / w.sum()), len(values), int(w.sum())


def bootstrap_interval(
    frame: pd.DataFrame, draws: int, seed: int
) -> tuple[float, float]:
    """Cluster bootstrap over held-out years.

    Series of one year share an epoch and are not independent, so the resampling
    unit is the year rather than the series.
    """
    rng = np.random.default_rng(seed)
    years = frame["year"].unique()
    if len(years) < 3:
        return float("nan"), float("nan")
    sample = np.empty(draws)
    for index in range(draws):
        drawn = rng.choice(years, size=len(years), replace=True)
        parts = [frame.loc[frame["year"] == year] for year in drawn]
        sample[index] = within_year_rho(
            pd.concat(parts).assign(year=np.repeat(np.arange(len(drawn)), [len(p) for p in parts]))
        )[0]
    sample = sample[~np.isnan(sample)]
    if len(sample) < draws // 10:
        return float("nan"), float("nan")
    return float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or (args.input / "rank_metrics")

    endpoint = load_endpoint(args.input)
    summary_rows: list[dict] = []
    year_rows: list[dict] = []

    subsets = {
        "all": endpoint,
        "non_electron": endpoint.loc[endpoint["family"] != "e"],
    }
    for cv_kind in sorted(endpoint["cv_kind"].unique()):
        for model in sorted(endpoint["model"].unique()):
            for label, subset in subsets.items():
                frame = subset.loc[
                    (subset["cv_kind"] == cv_kind) & (subset["model"] == model)
                ]
                if len(frame) < MIN_SERIES_PER_YEAR:
                    continue
                pooled = spearmanr(frame[PREDICTED], frame[ACTUAL])[0]
                mean_rho, n_years, n_scored = within_year_rho(frame)
                low, high = bootstrap_interval(frame, args.bootstrap, args.seed)
                summary_rows.append(
                    {
                        "cv_kind": cv_kind,
                        "model": model,
                        "subset": label,
                        "n_series": int(len(frame)),
                        "n_series_scored": n_scored,
                        "n_years": n_years,
                        "spearman_pooled": float(pooled),
                        "spearman_within_year": mean_rho,
                        "within_year_ci_low": low,
                        "within_year_ci_high": high,
                    }
                )
                for year, group in frame.groupby("year"):
                    if len(group) < MIN_SERIES_PER_YEAR:
                        continue
                    year_rows.append(
                        {
                            "cv_kind": cv_kind,
                            "model": model,
                            "subset": label,
                            "year": int(year),
                            "n_series": int(len(group)),
                            "spearman": float(
                                spearmanr(group[PREDICTED], group[ACTUAL])[0]
                            ),
                        }
                    )

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("rank_metrics_by_model.csv", summary_rows),
        ("rank_metrics_by_year.csv", year_rows),
    ):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

    for row in summary_rows:
        if row["model"] != "stacked_ensemble":
            continue
        print(
            f"{row['cv_kind']:<30} {row['subset']:<13} "
            f"n={row['n_series']:<4} pooled={row['spearman_pooled']:.3f} "
            f"within_year={row['spearman_within_year']:.3f} "
            f"[{row['within_year_ci_low']:.3f}; {row['within_year_ci_high']:.3f}]"
        )
    print(f"written to {output}")


if __name__ == "__main__":
    main()
