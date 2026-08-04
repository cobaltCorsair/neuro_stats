"""Predict the held-out June 2026 series with the frozen growth model.

The forecast has only ever been checked by cross-validation inside the archive
that also chose its model, its features and its landmark day. The June 2026
experiments were digitised separately, sit later than every training series, and
carry their own contemporaneous control, so they can be predicted once by a model
that has never seen them.

The protocol fixing the criteria is committed before this script runs; nothing
here selects a metric or a subset. The model is loaded from the frozen directory
and used exactly as it stands: fitted on the training cohort, then asked for the
held-out series. Interval coverage, the sign of the predicted difference between
the two series, and the per-series logarithmic error are reported whatever they
come to.

Two series cannot support a coefficient of determination or a rank statistic.
This is a calibration and direction check, and is reported as one.
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
HOLDOUT_DIR = Path(__file__).resolve().with_name("exps") / "Для внешней валидации"
DEFAULT_OUTPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Внешняя_проверка_июнь_2026"
)
ENGINE = Path(__file__).resolve().with_name("recalculate_alpha_beta_eff.py")

ENDPOINT_DAY = 21
WINDOW = 21.0
# Composition read from the parameter row of each book, not from the file name.
HOLDOUT_SERIES = {
    "p_15.2_n_2.3_n_2.3_17.06.2026.xlsx": {
        "date": "2026-06-17",
        "fractions": (15.2, 15.2, 2.3, 2.3),
        "gaps_hours": (2.0, 24.0, 2.0),
    },
    "p_25.2_n_2.3_n_2.3_18.06.2026.xlsx": {
        "date": "2026-06-18",
        "fractions": (25.2, 2.3, 2.3),
        "gaps_hours": (24.0, 2.0),
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def animal_rows(engine, path: Path, meta: dict, family: str) -> list[dict]:
    """Per-animal relative volumes on the model's daily grid."""
    from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (
        process_tumor_data_excel,
    )

    _, time_labels, _, volumes = process_tumor_data_excel(str(path))
    times = np.asarray(engine.parse_time_days(time_labels), dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    total = float(sum(meta["fractions"]))
    signature = "+".join(f"{value:g}" for value in meta["fractions"])
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
        for day in range(1, ENDPOINT_DAY + 1):
            if day > animal_times.max():
                continue
            rows.append(
                {
                    "series_key": f"{meta['date']}|{family}|{signature}|mixed_sequence|P_first",
                    "date": meta["date"],
                    "year": meta["date"][:4],
                    "family": family,
                    "family_label": family,
                    "regimen_class": "mixed_sequence",
                    "dose_signature": signature,
                    "total_dose_gy": total,
                    "sum_d2_gy2": float(sum(v * v for v in meta["fractions"])),
                    "n_events": len(meta["fractions"]),
                    "duration_hours": float(sum(meta["gaps_hours"])),
                    "mean_interval_hours": float(np.mean(meta["gaps_hours"])),
                    "timing_known": 1,
                    "order": "P_first",
                    "is_mixed": 1,
                    "control_kind": "same_date",
                    "animal_id": f"{path.stem}::{index}",
                    "day": float(day),
                    "log_v0": float(np.log(baseline)),
                    "log_relative_volume": float(
                        np.interp(day, animal_times, log_relative)
                    ),
                    "is_treated": 1,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=FROZEN_MODEL)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    engine = load_module("holdout_engine", ENGINE)
    model = load_module("holdout_model", args.model)

    observed, evaluation, series, _ = model.build_longitudinal_tables()
    # The pipeline models series means, not animal rows: main() collapses the
    # interpolated animal grid before fitting and before deriving the category
    # levels. Training on the animal frame instead fits a different model, so the
    # collapse is reproduced here exactly as main() performs it.
    modelling = model.collapse_animal_daily_to_series(evaluation)
    categories = model.fixed_categories(modelling)
    print(f"training series rows={len(modelling)} series={modelling['series_key'].nunique()}")

    # The model to fit is the one the frozen run selected for itself, read from
    # its own validation summary rather than assumed.
    summary_path = args.model.parent / "validation_summary.json"
    selected = json.loads(summary_path.read_text(encoding="utf-8"))["selected_model"]
    print(f"selected model recorded by the frozen run: {selected}")

    families = set(modelling["family"].astype(str))
    family = "mixed_n_p_through"
    if family not in families:
        raise RuntimeError(f"{family} absent from training families {sorted(families)}")

    rows: list[dict] = []
    for name, meta in HOLDOUT_SERIES.items():
        path = args.holdout / name
        if not path.exists():
            raise RuntimeError(f"Missing holdout book {path}")
        produced = animal_rows(engine, path, meta, family)
        if not produced:
            raise RuntimeError(f"No usable animals parsed from {name}")
        rows.extend(produced)
        print(f"{name}: animals={len({r['animal_id'] for r in produced})}")
    test = model.collapse_animal_daily_to_series(pd.DataFrame(rows))

    # Categorical features enter as exact string matches against the training
    # levels, so a label that merely looks right silently zeroes the indicator
    # instead of raising. The first version of this script wrote "mixed" and
    # "p_first" where the cohort uses "mixed_sequence" and "P_first", which
    # switched off the mixed-schedule and component-order features for the whole
    # held-out set and made the forecast look better than it is.
    for column, key in (("family", "families"), ("regimen_class", "regimens"), ("order", "orders")):
        unseen = set(test[column].astype(str)) - set(categories[key])
        if unseen:
            raise RuntimeError(
                f"{column} values {sorted(unseen)} are absent from the training "
                f"levels {sorted(categories[key])}; their indicators would be zero"
            )

    aggregated, _, _ = model.full_fit_selected(modelling, test, categories, selected)

    args.output.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output / "holdout_predictions.csv", sep=";", index=False)

    summary: list[dict] = []
    for key, group in aggregated.groupby("series_key"):
        # The interval bounds are returned in relative-volume units while the
        # point columns carry logarithms, so the observation must be compared in
        # volume space. Comparing the log against the bounds puts the prediction
        # itself outside its own interval, which is how this was caught.
        inside = (
            (group["actual_relative_volume"] >= group["prediction_low95"])
            & (group["actual_relative_volume"] <= group["prediction_high95"])
        )
        endpoint = group.loc[np.isclose(group["day"], ENDPOINT_DAY)]
        summary.append(
            {
                "series_key": key,
                "total_dose_gy": float(group["total_dose_gy"].iloc[0]),
                "n_days": int(len(group)),
                "coverage95": float(inside.mean()),
                "log_rmse": float(
                    np.sqrt(
                        np.mean(
                            (group["actual_log_relative"] - group["predicted_log_relative"]) ** 2
                        )
                    )
                ),
                "day21_actual": float(endpoint["actual_log_relative"].iloc[0])
                if len(endpoint)
                else float("nan"),
                "day21_predicted": float(endpoint["predicted_log_relative"].iloc[0])
                if len(endpoint)
                else float("nan"),
            }
        )
    frame = pd.DataFrame(summary).sort_values("total_dose_gy")
    with (args.output / "holdout_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame.columns), delimiter=";")
        writer.writeheader()
        writer.writerows(frame.to_dict("records"))

    print("\nHELD-OUT SERIES")
    for row in frame.to_dict("records"):
        print(
            f"  {row['total_dose_gy']:5.1f} Gy  days={row['n_days']:3d}  "
            f"coverage95={row['coverage95']:.3f}  log_rmse={row['log_rmse']:.3f}  "
            f"day21 actual={row['day21_actual']:+.3f} predicted={row['day21_predicted']:+.3f}"
        )
    low, high = frame.iloc[0], frame.iloc[-1]
    predicted_gap = high["day21_predicted"] - low["day21_predicted"]
    observed_gap = high["day21_actual"] - low["day21_actual"]
    print(
        f"\n  ordering: predicted {predicted_gap:+.3f}, observed {observed_gap:+.3f}, "
        f"sign {'MATCHES' if predicted_gap * observed_gap > 0 else 'DIFFERS'}"
    )
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
