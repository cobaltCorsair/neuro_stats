"""Predict the June 2026 control: was it the epoch, or was it the response?

The registered check under-predicted both treated series by about a factor of
two, and section 3.19 could not say whether the June epoch simply grew fast or
those two series recovered unusually strongly. A dose-zero series from the same
week separates the two, because no irradiation response enters it at all.

The series is a real target, not a tautology. Relative volume is built per animal
against that animal's own day-zero volume, so an untreated tumour has a genuine
growth curve to predict. The training manifest lists 125 files and none from the
held-out directory, and the control was never used to normalise the treated
series -- their outcomes come from their own baselines, and `control_kind` only
records that a same-date control existed.

One thing has to be established first. The fourteen training controls are in the
fit but excluded from every reported metric by the is_treated filter, so the
model was never tuned for them and has never been scored on them. Without that
reference distribution the June number would have nothing to sit against, and a
poor result could not be told apart from the model simply not working at zero
dose. So the internal controls are scored first, and the June series is reported
as a percentile of them.

Amendment 13 of the validation protocol fixes the interpretation rules and was
committed before this script read the file.
"""

from __future__ import annotations

import argparse
import csv
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
HOLDOUT_SCRIPT = Path(__file__).resolve().with_name(
    "validate_growth_prediction_holdout.py"
)
DEFAULT_OUTPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Внешняя_проверка_июнь_2026"
)
CONTROL_FILE = "control_19.06.2026.xlsx"
CONTROL_DATE = "2026-06-19"
ENDPOINT_DAY = 21.0
FIRST_DAY = 1.0
# A control is the model's baseline case: zero dose, is_treated off, and every
# family, regimen and order indicator off, because `control` and `none` are not
# among the fitted levels. Reproduced from a training control row rather than
# adapted from the treated builder, which hard-codes mixed_sequence and P_first.
CONTROL_FIELDS = {
    "family": "control",
    "family_label": "контроль",
    "regimen_class": "control",
    "dose_signature": "0",
    "total_dose_gy": 0.0,
    "sum_d2_gy2": 0.0,
    "n_events": 0,
    "duration_hours": 0.0,
    "mean_interval_hours": 0.0,
    "timing_known": 1,
    "order": "none",
    "is_mixed": 0,
    "is_treated": 0,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def control_rows(engine, path: Path, date: str) -> list[dict]:
    """Per-animal relative volumes for an untreated series, on the daily grid."""
    from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (
        process_tumor_data_excel,
    )

    _, time_labels, _, volumes = process_tumor_data_excel(str(path))
    times = np.asarray(engine.parse_time_days(time_labels), dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    key = f"{date}|control|0|control|none"
    rows: list[dict] = []
    for index in range(volumes.shape[0]):
        series = volumes[index]
        finite = np.isfinite(series) & np.isfinite(times)
        if finite.sum() < 3 or not finite[0]:
            continue
        baseline = float(series[finite][0])
        if not np.isfinite(baseline) or baseline <= 0.0:
            continue
        animal_times = times[finite]
        log_relative = np.log(series[finite] / baseline)
        for day in range(int(FIRST_DAY), int(ENDPOINT_DAY) + 1):
            if day > animal_times.max():
                continue
            rows.append(
                {
                    "series_key": key,
                    "date": date,
                    "year": date[:4],
                    **CONTROL_FIELDS,
                    "animal_id": f"{path.stem}::{index}",
                    "day": float(day),
                    "log_v0": float(np.log(baseline)),
                    "log_relative_volume": float(
                        np.interp(day, animal_times, log_relative)
                    ),
                }
            )
    return rows


def check_encoding(test: pd.DataFrame, controls: pd.DataFrame) -> None:
    """The held-out control must be encoded exactly as the training ones.

    The usual guard -- reject values absent from the fitted levels -- is wrong
    here, because `control` and `none` are absent by design and being absent is
    what makes the indicators zero. So compare against the training controls
    instead: any field where the two disagree would silently move the held-out
    series onto a different row of the design matrix.
    """
    for field, expected in CONTROL_FIELDS.items():
        seen = set(test[field].astype(str))
        training = set(controls[field].astype(str))
        if seen != {str(expected)} or seen != training:
            raise RuntimeError(
                f"{field}: held-out control has {sorted(seen)}, "
                f"training controls have {sorted(training)}"
            )


def series_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame["actual_log_relative"] - frame["predicted_log_relative"]
    endpoint = frame.loc[np.isclose(frame["day"].to_numpy(float), ENDPOINT_DAY)]
    row = {
        "n_days": int(len(frame)),
        "log_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_signed_error": float(error.mean()),
    }
    if not endpoint.empty:
        actual = float(endpoint["actual_log_relative"].iloc[0])
        predicted = float(endpoint["predicted_log_relative"].iloc[0])
        row.update(
            day21_actual_log=actual,
            day21_predicted_log=predicted,
            day21_error_log=actual - predicted,
            day21_actual_relative=float(np.exp(actual)),
            day21_predicted_relative=float(np.exp(predicted)),
            day21_ratio=float(np.exp(actual - predicted)),
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=FROZEN_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    holdout = load_module("holdout_builder", HOLDOUT_SCRIPT)
    engine = load_module("control_engine", holdout.ENGINE)
    model = load_module("control_model", args.model)

    _, evaluation, _, _ = model.build_longitudinal_tables()
    modelling = model.collapse_animal_daily_to_series(evaluation)
    categories = model.fixed_categories(modelling)
    selected = json.loads(
        (args.model.parent / "validation_summary.json").read_text(encoding="utf-8")
    )["selected_model"]

    # Day 0 is zero by construction for every series, so scoring it would
    # flatter the reference and not the held-out series, which starts at day 1.
    controls = modelling.loc[
        (modelling["family"].astype(str) == "control")
        & (modelling["day"].to_numpy(float) >= FIRST_DAY)
    ]
    print(
        f"training controls: {controls['series_key'].nunique()} series, "
        f"model {selected}"
    )

    # Step 1 of the amendment: the reference distribution. Each control year is
    # held out in turn so the control being scored never trained the model that
    # scores it -- the same discipline the treated series get, applied to a
    # stratum the pipeline has never reported on.
    reference: list[dict] = []
    for year, group in controls.groupby(controls["year"].astype(int)):
        train = modelling.loc[modelling["year"].astype(int) != year]
        fitted, _, _ = model.full_fit_selected(train, group, categories, selected)
        for key, series in fitted.groupby("series_key"):
            reference.append(
                {"series_key": key, "year": int(year), **series_metrics(series)}
            )
    reference_frame = pd.DataFrame(reference).sort_values("year")

    # Step 2: the held-out control, predicted by the frozen model exactly as the
    # treated series were -- full training cohort, same call, nothing tuned here.
    rows = control_rows(engine, holdout.HOLDOUT_DIR / CONTROL_FILE, CONTROL_DATE)
    test = model.collapse_animal_daily_to_series(pd.DataFrame(rows))
    check_encoding(test, controls)
    print(
        f"held-out control: {int(test['n_available_animals'].max())} animals, "
        f"{len(test)} days"
    )
    predicted, _, _ = model.full_fit_selected(modelling, test, categories, selected)
    june = series_metrics(predicted)

    # The reference above holds out each control's own year, but the registered
    # call trains on the whole archive -- which for June 2026 already contains
    # two controls of the same year. That advantage is real and would flatter the
    # percentile, so the same series is predicted again with 2026 removed. The
    # first number is comparable to the treated series, the second to the
    # reference distribution; both are reported rather than one chosen.
    without_year = modelling.loc[modelling["year"].astype(int) != 2026]
    strict_frame, _, _ = model.full_fit_selected(
        without_year, test, categories, selected
    )
    strict = series_metrics(strict_frame)

    def percentile(column: str, value: float) -> float:
        sample = reference_frame[column].to_numpy(float)
        return float(100.0 * np.mean(sample <= value))

    verdict = {"june_full_cohort_" + key: value for key, value in june.items()}
    verdict.update({"june_year_excluded_" + key: value for key, value in strict.items()})
    verdict.update(
        {
            "reference_n_controls": int(len(reference_frame)),
            "reference_log_rmse_median": float(reference_frame["log_rmse"].median()),
            "reference_log_rmse_p90": float(reference_frame["log_rmse"].quantile(0.90)),
            "reference_day21_error_median": float(
                reference_frame["day21_error_log"].median()
            ),
            "june_full_cohort_log_rmse_percentile": percentile(
                "log_rmse", june["log_rmse"]
            ),
            "june_year_excluded_log_rmse_percentile": percentile(
                "log_rmse", strict["log_rmse"]
            ),
            "june_full_cohort_day21_error_percentile": percentile(
                "day21_error_log", june["day21_error_log"]
            ),
            "june_year_excluded_day21_error_percentile": percentile(
                "day21_error_log", strict["day21_error_log"]
            ),
        }
    )

    args.output.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(args.output / "control_holdout_predictions.csv", sep=";", index=False)
    reference_frame.to_csv(
        args.output / "control_reference_distribution.csv", sep=";", index=False
    )
    with (args.output / "control_holdout_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verdict), delimiter=";")
        writer.writeheader()
        writer.writerow(verdict)

    print("\n=== reference: training controls, own year held out ===")
    print(
        reference_frame[
            ["year", "n_days", "log_rmse", "day21_error_log", "day21_ratio"]
        ].round(4).to_string(index=False)
    )
    for label, row, tag in (
        ("full cohort (as registered, comparable to the treated series)", june, "full_cohort"),
        ("2026 excluded (comparable to the reference distribution)", strict, "year_excluded"),
    ):
        print(f"\n=== June 2026 control, {label} ===")
        print(f"  log rmse             {row['log_rmse']:.4f}"
              f"   (percentile {verdict[f'june_{tag}_log_rmse_percentile']:.0f} of controls)")
        print(f"  day21 observed       {row['day21_actual_relative']:.3f}")
        print(f"  day21 predicted      {row['day21_predicted_relative']:.3f}")
        print(f"  day21 ratio          {row['day21_ratio']:.2f}")
        print(f"  day21 error (log)    {row['day21_error_log']:+.4f}"
              f"   (percentile {verdict[f'june_{tag}_day21_error_percentile']:.0f};"
              f" control median {verdict['reference_day21_error_median']:+.4f})")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
