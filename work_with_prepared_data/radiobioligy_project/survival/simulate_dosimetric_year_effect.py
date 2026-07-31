"""Is the documented dosimetric drift enough to produce the observed year effect?

The calendar year absorbs the largest share of between-series variance and is the
reason the prospective forecast is weak.  Section 3.15 established that the
electron unit carries a real dose systematic, so the natural question is whether
that systematic alone can generate the year share, or whether a second source
has to be present.

The simulation answers it by construction.  Series means are generated from a
dose response with no year term whatsoever, the recorded dose of the electron
series is then perturbed by factors taken from the calibration protocols, and the
same variance decomposition is applied.  Whatever year share appears is caused by
the dose error and by nothing else.

Two choices keep the answer honest.  The perturbation factors come only from
measured monitor-unit tables and applicator ratios, never from the biologically
back-estimated multipliers -- those were derived from the response and would make
the argument circular.  And the dose slope is estimated within calendar year: the
marginal slope is confounded by the epoch badly enough to carry the wrong sign
for electrons and gamma, so using it would understate the effect of a dose error.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\animal_daily_interpolated.csv"
)
DEFAULT_OUTPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\dosimetric_simulation"
)

ENDPOINT_DAY = 21
OBSERVED_YEAR_OMEGA2 = 0.371
PERMUTATION_NULL = 0.106

# Monitor units per gray at 10 MeV, applicator 100 mm, from the absolute
# dosimetry protocols.  These are measured quantities, independent of any
# tumour response.
CALIBRATION = {
    "pre_2022_03": 20.4077,
    "2022_03": 17.4755,
    "2023_02": 15.5407,
    "2024_02": 18.8230,
}
# Ratio between successive tables: the dose error incurred if the table in force
# lagged the true output by one revision.
LAG_RATIOS = (20.4077 / 17.4755, 17.4755 / 15.5407, 15.5407 / 18.8230)
# Applicator mismatch at 10 MeV from the February 2023 table: the factor by which
# the dose is overshot when the wide-field number is applied to a narrow tube.
APPLICATOR_MISMATCH = (15.5407 / 6.277, 15.5407 / 6.647, 15.5407 / 7.267)


def epoch_of(date: str) -> str:
    if date < "2022-03-22":
        return "pre_2022_03"
    if date < "2023-02-20":
        return "2022_03"
    if date < "2024-02-22":
        return "2023_02"
    return "2024_02"


def load_series(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    treated = frame.loc[(frame["is_treated"] == 1) & (frame["day"] == ENDPOINT_DAY)]
    series = (
        treated.groupby(["series_key", "date", "year", "family", "total_dose_gy"])[
            "log_relative_volume"
        ]
        .mean()
        .reset_index(name="observed")
    )
    series["epoch"] = series["date"].astype(str).map(epoch_of)
    return series


def within_year_slope(series: pd.DataFrame, family: str) -> float:
    """Dose slope estimated inside calendar years, free of the epoch confound."""
    group = series.loc[series["family"] == family]
    numerator = 0.0
    denominator = 0.0
    for _, year_group in group.groupby("year"):
        if len(year_group) < 2 or year_group["total_dose_gy"].nunique() < 2:
            continue
        x = year_group["total_dose_gy"] - year_group["total_dose_gy"].mean()
        y = year_group["observed"] - year_group["observed"].mean()
        numerator += float((x * y).sum())
        denominator += float((x**2).sum())
    return numerator / denominator if denominator else 0.0


def omega_squared(frame: pd.DataFrame, column: str, response: str) -> float:
    grand = frame[response].mean()
    total = float(((frame[response] - grand) ** 2).sum())
    between = sum(
        len(g) * (g.mean() - grand) ** 2 for _, g in frame.groupby(column)[response]
    )
    levels = int(frame[column].nunique())
    within_ms = (total - between) / (len(frame) - levels)
    return float((between - (levels - 1) * within_ms) / (total + within_ms))


def draw_multipliers(scenario: str, years: np.ndarray, rng) -> dict[int, float]:
    """One dose multiplier per calendar year, from documented dosimetry only."""
    if scenario == "none":
        return {int(y): 1.0 for y in years}
    if scenario == "drift":
        return {int(y): float(rng.choice(LAG_RATIOS)) for y in years}
    if scenario == "drift_applicator":
        out = {}
        for y in years:
            factor = float(rng.choice(LAG_RATIOS))
            # The wide-field number is assumed misapplied in half of the years,
            # which is the most favourable assumption for the hypothesis.
            if rng.random() < 0.5:
                factor *= float(rng.choice(APPLICATOR_MISMATCH))
            out[int(y)] = factor
        return out
    if scenario == "full_range":
        low, high = 0.86, max(APPLICATOR_MISMATCH)
        return {int(y): float(rng.uniform(low, high)) for y in years}
    raise ValueError(scenario)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    series = load_series(args.input)
    families = sorted(series["family"].unique())
    slopes = {f: within_year_slope(series, f) for f in families}
    intercepts = {
        f: float(
            series.loc[series["family"] == f, "observed"].mean()
            - slopes[f] * series.loc[series["family"] == f, "total_dose_gy"].mean()
        )
        for f in families
    }

    dose = series["total_dose_gy"].to_numpy(float)
    family = series["family"].to_numpy()
    is_electron = family == "e"
    base = np.array([intercepts[f] + slopes[f] * d for f, d in zip(family, dose)])
    observed_total = float(series["observed"].var(ddof=1))
    # Noise is set so the simulated spread matches the observed one; otherwise the
    # share attributed to the year would depend on an arbitrary scale.
    noise_sd = float(np.sqrt(max(observed_total - base.var(ddof=1), 1e-6)))

    years = series["year"].to_numpy()
    unique_years = np.unique(years)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for scenario in ("none", "drift", "drift_applicator", "full_range"):
        sample = np.empty(args.draws)
        for index in range(args.draws):
            multiplier = draw_multipliers(scenario, unique_years, rng)
            factor = np.where(
                is_electron, np.array([multiplier[int(y)] for y in years]), 1.0
            )
            shift = np.array(
                [slopes[f] * (k - 1.0) * d for f, k, d in zip(family, factor, dose)]
            )
            simulated = base + shift + rng.normal(0.0, noise_sd, len(series))
            work = pd.DataFrame({"year": years, "value": simulated})
            sample[index] = omega_squared(work, "year", "value")
        rows.append(
            {
                "scenario": scenario,
                "median": float(np.median(sample)),
                "p2_5": float(np.percentile(sample, 2.5)),
                "p97_5": float(np.percentile(sample, 97.5)),
                "fraction_reaching_observed": float(
                    (sample >= OBSERVED_YEAR_OMEGA2).mean()
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "year_effect_simulation.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "within_year_dose_slopes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["family", "within_year_slope", "n_series"])
        for f in families:
            writer.writerow(
                [f, slopes[f], int((series["family"] == f).sum())]
            )

    print(f"series={len(series)} electron={int(is_electron.sum())} noise_sd={noise_sd:.3f}")
    print(f"observed year omega^2 = {OBSERVED_YEAR_OMEGA2}; permutation null = {PERMUTATION_NULL}")
    for row in rows:
        print(
            f"  {row['scenario']:<18} median={row['median']:.3f} "
            f"[{row['p2_5']:.3f}; {row['p97_5']:.3f}] "
            f"reaches observed in {100 * row['fraction_reaching_observed']:.1f}%"
        )
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
