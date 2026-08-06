"""Growth delay in days: the arrest expressed as a time shift, not a coefficient.

Cell-cycle arrest after irradiation shows up macroscopically as a delay before the
tumour resumes growing. Inside a log-volume model that is an additive term -k*tau,
the product of the regrowth rate and the arrest. The fitted model carries the
product but neither factor, so the coefficient on dose is not interpretable as a
delay and no duration can be read off it.

The delay is recoverable anyway, by the definition radiobiology already uses:
the time for the tumour to return to the volume it had at irradiation. The control
is at that volume on day zero, so for a treated series the return time is itself
the delay -- no counterfactual required, and nothing depends on the model being
able to tell a fast-growing tumour from a slow one, which it largely cannot.

Both the observed and the predicted trajectory are crossed, so the comparison also
says whether the model reproduces the delay it was never asked to represent.
Series that never return within the window are reported as censored rather than
extrapolated: the crossing would be invented, and the count of them is itself a
result -- those are the doses that hold the tumour down for longer than the
experiment lasts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FROZEN_MODEL = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Ковариаты_pole_gamma\run_growth_prediction.py"
)
DEFAULT_OUTPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Задержка_роста"
)
RETURN_LEVEL = 0.0  # log relative volume, i.e. back to the volume at irradiation
WINDOW_DAY = 21.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def return_time(days: np.ndarray, values: np.ndarray) -> float:
    """First day after the nadir at which the trajectory regains its start.

    Taken after the nadir so that a series which has not yet dipped does not
    register a spurious crossing on day one, and by linear interpolation between
    the bracketing days rather than by rounding to the grid.
    """
    order = np.argsort(days)
    days, values = days[order], values[order]
    nadir = int(np.argmin(values))
    if values[nadir] >= RETURN_LEVEL:
        return 0.0
    for index in range(nadir + 1, len(days)):
        if values[index] >= RETURN_LEVEL:
            previous, current = values[index - 1], values[index]
            if current == previous:
                return float(days[index])
            fraction = (RETURN_LEVEL - previous) / (current - previous)
            return float(days[index - 1] + fraction * (days[index] - days[index - 1]))
    return float("nan")  # censored: still below its starting volume at day 21


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=FROZEN_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = load_module("delay_model", args.model)
    _, evaluation, _, _ = model.build_longitudinal_tables()
    data = model.collapse_animal_daily_to_series(evaluation)
    categories = model.fixed_categories(data)
    selected = json.loads(
        (args.model.parent / "validation_summary.json").read_text(encoding="utf-8")
    )["selected_model"]
    fitted, _, _ = model.full_fit_selected(data, data, categories, selected)

    rows: list[dict] = []
    for key, group in fitted.groupby("series_key"):
        if int(group["is_treated"].iloc[0]) == 0:
            continue
        days = group["day"].to_numpy(float)
        rows.append(
            {
                "series_key": key,
                "date": group["date"].iloc[0],
                "year": int(group["year"].iloc[0]),
                "family": str(group["family"].iloc[0]),
                "total_dose_gy": float(group["total_dose_gy"].iloc[0]),
                "n_events": int(group["n_events"].iloc[0]),
                "observed_delay_days": return_time(
                    days, group["actual_log_relative"].to_numpy(float)
                ),
                "predicted_delay_days": return_time(
                    days, group["predicted_log_relative"].to_numpy(float)
                ),
                "observed_nadir_log": float(group["actual_log_relative"].min()),
                "predicted_nadir_log": float(group["predicted_log_relative"].min()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["both_returned"] = (
        frame["observed_delay_days"].notna() & frame["predicted_delay_days"].notna()
    )
    frame["delay_error_days"] = (
        frame["observed_delay_days"] - frame["predicted_delay_days"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "growth_delay_by_series.csv", sep=";", index=False)

    paired = frame.loc[frame["both_returned"]]
    summary = (
        frame.assign(
            censored_observed=frame["observed_delay_days"].isna().astype(int),
            censored_predicted=frame["predicted_delay_days"].isna().astype(int),
        )
        .groupby("family")
        .agg(
            n_series=("series_key", "size"),
            median_dose_gy=("total_dose_gy", "median"),
            n_censored_observed=("censored_observed", "sum"),
            n_censored_predicted=("censored_predicted", "sum"),
            median_observed_delay=("observed_delay_days", "median"),
            median_predicted_delay=("predicted_delay_days", "median"),
        )
        .reset_index()
        .sort_values("median_observed_delay")
    )
    summary.to_csv(args.output / "growth_delay_by_family.csv", sep=";", index=False)

    pd.set_option("display.width", 200)
    print(f"treated series: {len(frame)}")
    print(
        f"never regained the starting volume by day 21 -- "
        f"observed {int(frame['observed_delay_days'].isna().sum())}, "
        f"predicted {int(frame['predicted_delay_days'].isna().sum())}"
    )
    print("\n=== growth delay by family, days to regain the volume at irradiation ===")
    print(summary.round(2).to_string(index=False))
    print("\n=== agreement on the series where both return ===")
    print(f"  n = {len(paired)}")
    print(f"  observed median {paired['observed_delay_days'].median():.2f} d, "
          f"predicted {paired['predicted_delay_days'].median():.2f} d")
    print(f"  error median {paired['delay_error_days'].median():+.2f} d, "
          f"mean {paired['delay_error_days'].mean():+.2f} d, "
          f"rmse {np.sqrt((paired['delay_error_days'] ** 2).mean()):.2f} d")
    print(f"  correlation {paired['observed_delay_days'].corr(paired['predicted_delay_days']):+.3f}")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
