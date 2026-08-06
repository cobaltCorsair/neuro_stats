"""Growth delay against dose within each family, with the censoring handled.

The family medians hide the quantity that matters radiobiologically: how many
days of delay a gray buys. Reading that off the uncensored series alone would
understate it badly, because the series that never regain their starting volume
inside the window are exactly the heavily irradiated ones -- seven of twelve in
the proton Bragg peak, eight of twenty-three among electrons. Dropping them
truncates the response at its steep end.

So the slope is fitted by censored normal regression: the uncensored series enter
through the density, the censored ones through the probability of exceeding the
observation window. The naive fit on uncensored series only is reported beside it,
not as an alternative but to show the size of the bias it carries.

Both the observed and the predicted delays are fitted, which turns the comparison
into a statement about the model rather than about the tumour: whether it recovers
the right number of days per gray, not merely the right ranking of families.
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
WINDOW_DAY = 21.0
MIN_SERIES = 8
MIN_DOSES = 3


def tobit_slope(dose: np.ndarray, delay: np.ndarray) -> dict[str, float]:
    """Censored normal regression of delay on dose, right-censored at day 21.

    Censored series carry no delay value; what is known of them is only that the
    return time exceeds the window, which is exactly the survival term below.
    """
    censored = np.isnan(delay)
    observed = ~censored
    if observed.sum() < 3 or len(np.unique(dose[observed])) < 2:
        return {"intercept": np.nan, "slope": np.nan, "sigma": np.nan, "slope_se": np.nan}

    def negative_log_likelihood(theta: np.ndarray) -> float:
        intercept, slope, log_sigma = theta
        sigma = np.exp(log_sigma)
        mean = intercept + slope * dose
        total = 0.0
        if observed.any():
            total -= np.sum(
                stats.norm.logpdf(delay[observed], mean[observed], sigma)
            )
        if censored.any():
            survival = stats.norm.sf(WINDOW_DAY, mean[censored], sigma)
            total -= np.sum(np.log(np.clip(survival, 1e-12, None)))
        return float(total)

    start = np.polyfit(dose[observed], delay[observed], 1)
    guess = np.array([start[1], start[0], np.log(max(delay[observed].std(), 1.0))])
    fit = optimize.minimize(negative_log_likelihood, guess, method="Nelder-Mead",
                            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
    intercept, slope, log_sigma = fit.x
    # Standard error from a numerical Hessian; a failure to invert means the
    # likelihood is flat here, which is reported as absent rather than as zero.
    step = 1e-4 * np.maximum(np.abs(fit.x), 1.0)
    hessian = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            up, down = fit.x.copy(), fit.x.copy()
            up[i] += step[i]; up[j] += step[j]
            down[i] += step[i]; down[j] -= step[j]
            up2, down2 = fit.x.copy(), fit.x.copy()
            up2[i] -= step[i]; up2[j] += step[j]
            down2[i] -= step[i]; down2[j] -= step[j]
            hessian[i, j] = (
                negative_log_likelihood(up) - negative_log_likelihood(down)
                - negative_log_likelihood(up2) + negative_log_likelihood(down2)
            ) / (4 * step[i] * step[j])
    try:
        slope_se = float(np.sqrt(np.linalg.inv(hessian)[1, 1]))
    except (np.linalg.LinAlgError, ValueError):
        slope_se = np.nan
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "sigma": float(np.exp(log_sigma)),
        "slope_se": slope_se if np.isfinite(slope_se) else np.nan,
    }


def naive_slope(dose: np.ndarray, delay: np.ndarray) -> float:
    observed = ~np.isnan(delay)
    if observed.sum() < 3 or len(np.unique(dose[observed])) < 2:
        return np.nan
    return float(np.polyfit(dose[observed], delay[observed], 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input / "growth_delay_by_series.csv", sep=";")

    # A delay of zero was recorded for every series whose tumour never fell below
    # the volume it had at irradiation -- all twenty-four of them. That is not a
    # short delay, it is no regression at all, and averaging the two together is
    # what made the per-dose medians run backwards. The two are separated here:
    # whether the tumour responded, and given that it did, for how long it was
    # held down.
    frame["regressed"] = frame["observed_nadir_log"] < 0.0
    frame["predicted_regressed"] = frame["predicted_nadir_log"] < 0.0

    per_dose: list[dict] = []
    for (family, dose), group in frame.groupby(["family", "total_dose_gy"]):
        returners = group.loc[group["regressed"]]
        per_dose.append(
            {
                "family": family,
                "total_dose_gy": round(float(dose), 2),
                "n_series": len(group),
                "n_no_regression": int((~group["regressed"]).sum()),
                "n_censored": int(
                    returners["observed_delay_days"].isna().sum()
                ),
                "observed_delay_median": float(
                    returners["observed_delay_days"].median()
                ) if len(returners) else np.nan,
                "predicted_delay_median": float(
                    group.loc[group["predicted_regressed"], "predicted_delay_days"].median()
                ) if group["predicted_regressed"].any() else np.nan,
            }
        )
    dose_table = pd.DataFrame(per_dose).sort_values(["family", "total_dose_gy"])

    slopes: list[dict] = []
    for family, whole in frame.groupby("family"):
        group = whole.loc[whole["regressed"]]
        if len(group) < MIN_SERIES or group["total_dose_gy"].nunique() < MIN_DOSES:
            continue
        dose = group["total_dose_gy"].to_numpy(float)
        responded = whole["regressed"].to_numpy(bool)
        whole_dose = whole["total_dose_gy"].to_numpy(float)
        row: dict[str, object] = {
            "family": family,
            "n_series": len(whole),
            "n_regressed": int(len(group)),
            "n_doses": int(group["total_dose_gy"].nunique()),
            "n_censored": int(group["observed_delay_days"].isna().sum()),
            "dose_min": float(dose.min()),
            "dose_max": float(dose.max()),
            # Whether responding at all rises with dose is its own question and
            # is answered on all series, not only those that responded.
            "response_dose_corr": float(
                np.corrcoef(whole_dose, responded.astype(float))[0, 1]
            ) if responded.std() > 0 else np.nan,
        }
        for label, column in (("observed", "observed_delay_days"),
                              ("predicted", "predicted_delay_days")):
            delay = group[column].to_numpy(float)
            fit = tobit_slope(dose, delay)
            row[f"{label}_slope_days_per_gy"] = fit["slope"]
            row[f"{label}_slope_se"] = fit["slope_se"]
            row[f"{label}_naive_slope"] = naive_slope(dose, delay)
        slopes.append(row)
    slope_table = pd.DataFrame(slopes).sort_values("observed_slope_days_per_gy")

    dose_table.to_csv(args.input / "growth_delay_by_dose.csv", sep=";", index=False)
    slope_table.to_csv(args.input / "growth_delay_dose_slopes.csv", sep=";", index=False)

    pd.set_option("display.width", 220)
    print(f"series whose tumour never fell below its starting volume: "
          f"{int((~frame['regressed']).sum())} of {len(frame)}")
    print("\n=== delay by dose within family, regressing series only ===")
    print(dose_table.to_string(index=False))
    print("\n=== days of delay per gray, censored fit ===")
    print(
        slope_table[
            ["family", "n_series", "n_regressed", "n_censored", "dose_min", "dose_max",
             "response_dose_corr", "observed_slope_days_per_gy", "observed_slope_se",
             "predicted_slope_days_per_gy"]
        ].round(3).to_string(index=False)
    )
    print("\ncensoring bias in the naive fit, days per gray:")
    for row in slope_table.to_dict("records"):
        gap = row["observed_naive_slope"] - row["observed_slope_days_per_gy"]
        print(f"  {row['family']:<20} censored {row['n_censored']:>2}/{row['n_series']:<3}"
              f" naive {row['observed_naive_slope']:+.3f} vs censored-fit "
              f"{row['observed_slope_days_per_gy']:+.3f}  ({gap:+.3f})")
    print(f"\nwritten to {args.input}")


if __name__ == "__main__":
    main()
