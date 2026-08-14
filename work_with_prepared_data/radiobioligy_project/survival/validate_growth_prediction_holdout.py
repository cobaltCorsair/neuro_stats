"""Shared input helpers for the June 2026 external validation.

The former standalone entry point used an obsolete irradiation-family encoding
and is deliberately disabled.  The only active analysis is
``validate_june2026_peak_holdout.py``, which uses the confirmed
``mixed_n_p_peak`` geometry and the documented dose conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


HOLDOUT_DIR = Path(__file__).resolve().with_name("exps") / "Для внешней валидации"
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
        finite = np.isfinite(series) & np.isfinite(times) & (series >= 0.0)
        if finite.sum() < 3 or not finite[0]:
            continue
        baseline = float(series[finite][0])
        if not np.isfinite(baseline) or baseline <= 0.0:
            continue
        animal_times = times[finite]
        relative = series[finite] / baseline
        contains_zero = bool(np.any(relative == 0.0))
        if not contains_zero:
            log_relative = np.log(relative)
        for day in range(1, ENDPOINT_DAY + 1):
            if day > animal_times.max():
                continue
            if contains_zero:
                # A literal zero denotes a measured complete regression, not a
                # missing animal.  Log interpolation is undefined across such a
                # point, so only the affected trajectory is interpolated on the
                # relative-volume scale.  The group mean is taken in volume
                # space downstream and is logarithmised only after aggregation.
                relative_at_day = float(np.interp(day, animal_times, relative))
                log_relative_at_day = (
                    -np.inf
                    if relative_at_day <= 0.0
                    else float(np.log(relative_at_day))
                )
            else:
                log_relative_at_day = float(
                    np.interp(day, animal_times, log_relative)
                )
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
                    "log_relative_volume": log_relative_at_day,
                    "is_treated": 1,
                }
            )
    return rows


def main() -> None:
    current = Path(__file__).resolve().with_name("validate_june2026_peak_holdout.py")
    raise SystemExit(
        "This obsolete standalone entry point is disabled because it used the "
        "wrong irradiation family. Run the confirmed peak validation instead: "
        f"python {current}"
    )


if __name__ == "__main__":
    main()
