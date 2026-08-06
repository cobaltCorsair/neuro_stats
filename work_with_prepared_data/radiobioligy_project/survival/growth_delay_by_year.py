"""Does the calendar epoch move the growth delay, or only whether there is one?

The calendar year accounts for the largest single share of variance in the day-21
volume, and the question is where in the response it sits. The delay decomposes
that response into two parts that can move independently: whether the tumour
regresses at all, and how long it stays down once it has. They are tested apart.

Neither is tested raw. Year is confounded with both dose and radiation type --
later years used heavier doses and different beams -- so a year effect read off
the margins would largely be a dose effect wearing a date. For the delay the
share of variance is compared against a permutation null and against what family
and dose take. For the response the test is a likelihood ratio: year is added to
a logistic model that already contains dose and family, and asked whether it buys
anything.

Only series that regressed enter the delay analysis, and only those whose return
was observed inside the window: a censored return has no value to decompose. That
drops the sample to 58 and the caveat belongs with the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


DEFAULT_INPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Задержка_роста"
)
PERMUTATIONS = 10000
SEED = 20260720


def eta_omega(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Share of variance taken by a factor, raw and bias-corrected."""
    groups = [values[labels == level] for level in np.unique(labels)]
    groups = [g for g in groups if len(g) > 0]
    grand = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((values - grand) ** 2).sum())
    k, n = len(groups), len(values)
    if n <= k or ss_total <= 0:
        return float("nan"), float("nan")
    ss_within = ss_total - ss_between
    ms_within = ss_within / (n - k)
    eta = ss_between / ss_total
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return float(eta), float(omega)


def permutation_null(values: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observed_eta, observed_omega = eta_omega(values, labels)
    draws = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        draws[index] = eta_omega(values, rng.permutation(labels))[0]
    return {
        "eta2": observed_eta,
        "omega2": observed_omega,
        "null_eta2_median": float(np.median(draws)),
        "null_eta2_p95": float(np.quantile(draws, 0.95)),
        "p_value": float((np.sum(draws >= observed_eta) + 1) / (PERMUTATIONS + 1)),
    }


def logistic_deviance(design: np.ndarray, outcome: np.ndarray) -> float:
    """Deviance of a ridge-free logistic fit; used only for a likelihood ratio."""
    def negative_log_likelihood(beta: np.ndarray) -> float:
        linear = np.clip(design @ beta, -30.0, 30.0)
        return float(np.sum(np.logaddexp(0.0, linear) - outcome * linear))

    start = np.zeros(design.shape[1])
    fit = optimize.minimize(negative_log_likelihood, start, method="BFGS",
                            options={"maxiter": 2000})
    return 2.0 * float(fit.fun)


def indicator_block(labels: pd.Series) -> np.ndarray:
    levels = sorted(labels.unique())[1:]  # first level is the reference
    return np.column_stack(
        [(labels.to_numpy() == level).astype(float) for level in levels]
    ) if levels else np.empty((len(labels), 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input / "growth_delay_by_series.csv", sep=";")
    frame["regressed"] = frame["observed_nadir_log"] < 0.0

    delay = frame.loc[frame["regressed"] & frame["observed_delay_days"].notna()].copy()
    values = delay["observed_delay_days"].to_numpy(float)
    print(f"delay analysed on {len(delay)} regressing, uncensored series "
          f"across {delay['year'].nunique()} years")
    print("\n=== share of delay variance ===")
    rows = []
    for name, labels in (
        ("year", delay["year"].astype(str).to_numpy()),
        ("family", delay["family"].astype(str).to_numpy()),
    ):
        result = permutation_null(values, labels, SEED)
        rows.append({"factor": name, **result})
    # Dose is continuous, so its share is the squared correlation rather than a
    # group variance; reported beside the factors for scale, not as the same test.
    dose_r = float(np.corrcoef(delay["total_dose_gy"], values)[0, 1])
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    print(f"  dose (continuous): r = {dose_r:+.3f}, r2 = {dose_r ** 2:.4f}")

    print("\n=== does year survive adjustment for dose and family? ===")
    outcome = frame["regressed"].to_numpy(float)
    intercept = np.ones((len(frame), 1))
    dose = frame["total_dose_gy"].to_numpy(float).reshape(-1, 1)
    dose = (dose - dose.mean()) / dose.std()
    family = indicator_block(frame["family"].astype(str))
    year = indicator_block(frame["year"].astype(str))
    base = np.hstack([intercept, dose, family])
    full = np.hstack([base, year])
    deviance_base = logistic_deviance(base, outcome)
    deviance_full = logistic_deviance(full, outcome)
    statistic = deviance_base - deviance_full
    degrees = year.shape[1]
    # Five years contain no non-responder at all, so the year indicators separate
    # those cells perfectly and the unpenalised fit drives their coefficients to
    # the clip. The chi-square reference distribution does not hold under that,
    # and would overstate the evidence. The null is therefore built by permuting
    # the year labels and recomputing the same statistic, which inherits the
    # separation and so prices it in rather than assuming it away.
    rng = np.random.default_rng(SEED)
    draws = np.empty(1000)
    years = frame["year"].astype(str).to_numpy()
    for index in range(len(draws)):
        shuffled = indicator_block(pd.Series(rng.permutation(years)))
        draws[index] = deviance_base - logistic_deviance(
            np.hstack([base, shuffled]), outcome
        )
    permuted_p = float((np.sum(draws >= statistic) + 1) / (len(draws) + 1))
    print(f"  response: statistic = {statistic:.2f} on {degrees} df")
    print(f"            chi-square p = {stats.chi2.sf(statistic, degrees):.4f} "
          f"(unreliable: perfect separation in five years)")
    print(f"            permutation p = {permuted_p:.4f}, "
          f"null median {np.median(draws):.2f}, p95 {np.quantile(draws, 0.95):.2f}")

    # The same question for the delay, by adding year to a dose-plus-family
    # linear fit on the uncensored series.
    d_dose = delay["total_dose_gy"].to_numpy(float).reshape(-1, 1)
    d_dose = (d_dose - d_dose.mean()) / d_dose.std()
    d_base = np.hstack([np.ones((len(delay), 1)), d_dose,
                        indicator_block(delay["family"].astype(str))])
    d_year = indicator_block(delay["year"].astype(str))
    d_full = np.hstack([d_base, d_year])

    def rss(design: np.ndarray) -> float:
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        return float(((values - design @ coefficients) ** 2).sum())

    rss_base, rss_full = rss(d_base), rss(d_full)
    df_num = d_year.shape[1]
    df_den = len(delay) - d_full.shape[1]
    f_statistic = ((rss_base - rss_full) / df_num) / (rss_full / df_den) if df_den > 0 else np.nan
    print(f"  delay:    F = {f_statistic:.2f} on {df_num} and {df_den} df, "
          f"p = {stats.f.sf(f_statistic, df_num, df_den):.3f}")

    summary = frame.groupby("year").apply(
        lambda g: pd.Series(
            {
                "n_series": len(g),
                "response_fraction": g["regressed"].mean(),
                "median_dose_gy": g["total_dose_gy"].median(),
                "n_families": g["family"].nunique(),
                "median_delay_days": g.loc[
                    g["regressed"], "observed_delay_days"
                ].median(),
            }
        ),
        include_groups=False,
    )
    summary.to_csv(args.input / "growth_delay_by_year.csv", sep=";")
    print("\n=== by year ===")
    print(summary.round(2).to_string())
    print(f"\nwritten to {args.input / 'growth_delay_by_year.csv'}")


if __name__ == "__main__":
    main()
