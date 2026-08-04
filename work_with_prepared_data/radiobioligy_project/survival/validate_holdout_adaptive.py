"""Post-hoc: apply the updating contour to the held-out June 2026 series.

This is NOT part of the pre-registration. The a priori check under-predicted both
series by about a factor of two in the same direction, which is an offset of the
epoch rather than an error of shape, and the updating contour exists to remove
exactly that. Running it after seeing the a priori result is legitimate only if
labelled, and the protocol amendment of 3 August records it as post-hoc with no
thresholds -- setting one now would mean setting it against a known result.

Nothing is tuned on the held-out series. The landmark of day 12 was chosen from
training data before this check existed; the shrinkage coefficient and the delta
bounds are fitted on the training cohort and applied unchanged. Only the observed
days 1 to 12 of each held-out series enter, and only days 13 to 21 are scored.
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
LANDMARK_MODULE = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель"
    r"\4.6_Итоговая_прогностическая_модель\prediction_v2_landmark_scan\run_prediction_v2.py"
)
HOLDOUT_SCRIPT = Path(__file__).resolve().with_name(
    "validate_growth_prediction_holdout.py"
)
DEFAULT_OUTPUT = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Внешняя_проверка_июнь_2026"
)
LANDMARK_DAY = 12
ENDPOINT_DAY = 21


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=FROZEN_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    holdout = load_module("holdout_builder", HOLDOUT_SCRIPT)
    engine = load_module("adaptive_engine", holdout.ENGINE)
    model = load_module("adaptive_model", args.model)
    landmark_module = load_module("adaptive_landmark", LANDMARK_MODULE)

    observed, evaluation, _, _ = model.build_longitudinal_tables()
    modelling = model.collapse_animal_daily_to_series(evaluation)
    categories = model.fixed_categories(modelling)
    selected = json.loads(
        (args.model.parent / "validation_summary.json").read_text(encoding="utf-8")
    )["selected_model"]

    rows: list[dict] = []
    for name, meta in holdout.HOLDOUT_SERIES.items():
        rows.extend(
            holdout.animal_rows(
                engine, holdout.HOLDOUT_DIR / name, meta, "mixed_n_p_through"
            )
        )
    test = model.collapse_animal_daily_to_series(pd.DataFrame(rows))

    # Prior contour: the same fit reported by the registered check.
    prior_frame, _, _ = model.full_fit_selected(modelling, test, categories, selected)
    prior = prior_frame["predicted_log_relative"].to_numpy(float)

    # Shrinkage and bounds come from the training cohort alone. The out-of-fold
    # baseline needed for that tuning is the model's own fit on training rows.
    train_fit, _, _ = model.full_fit_selected(
        modelling, modelling, categories, selected
    )
    # The landmark helpers read the target under the modelling frame's name and
    # weight by animal count, so the aggregated fit is relabelled rather than
    # re-derived; using the aggregate keeps prediction and target aligned row by
    # row, which a separate re-derivation would not guarantee.
    train_work = train_fit.assign(
        log_relative_volume=train_fit["actual_log_relative"],
        n_available_animals=train_fit.get("n_animals", 1),
        is_treated=1,
    )
    shrinkage, bounds, _, _ = landmark_module.tune_landmark(
        train_work,
        train_fit["predicted_log_relative"].to_numpy(float),
        LANDMARK_DAY,
    )
    print(f"shrinkage tuned on training: {shrinkage:.3f}, bounds {bounds}")

    work = prior_frame.assign(
        log_relative_volume=prior_frame["actual_log_relative"],
        n_available_animals=1,
    )
    updated = landmark_module.apply_landmark_update(
        work, prior, LANDMARK_DAY, shrinkage, bounds
    )
    prior_frame = prior_frame.assign(adaptive_log_relative=updated)
    prior_frame["adaptive_relative_volume"] = np.exp(updated)

    args.output.mkdir(parents=True, exist_ok=True)
    prior_frame.to_csv(args.output / "holdout_adaptive_predictions.csv", sep=";", index=False)

    summary: list[dict] = []
    for key, group in prior_frame.groupby("series_key"):
        future = group.loc[group["day"] > LANDMARK_DAY]
        endpoint = group.loc[np.isclose(group["day"], ENDPOINT_DAY)]
        summary.append(
            {
                "series_key": key,
                "total_dose_gy": float(group["total_dose_gy"].iloc[0]),
                "n_future_days": int(len(future)),
                "prior_future_log_rmse": float(
                    np.sqrt(
                        np.mean(
                            (future["actual_log_relative"] - future["predicted_log_relative"]) ** 2
                        )
                    )
                ),
                "adaptive_future_log_rmse": float(
                    np.sqrt(
                        np.mean(
                            (future["actual_log_relative"] - future["adaptive_log_relative"]) ** 2
                        )
                    )
                ),
                "day21_actual": float(endpoint["actual_log_relative"].iloc[0]),
                "day21_prior": float(endpoint["predicted_log_relative"].iloc[0]),
                "day21_adaptive": float(endpoint["adaptive_log_relative"].iloc[0]),
            }
        )
    frame = pd.DataFrame(summary).sort_values("total_dose_gy")
    with (args.output / "holdout_adaptive_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame.columns), delimiter=";")
        writer.writeheader()
        writer.writerows(frame.to_dict("records"))

    print("\nADAPTIVE CONTOUR, days 13-21 (post-hoc, not pre-registered)")
    for row in frame.to_dict("records"):
        print(
            f"  {row['total_dose_gy']:5.1f} Gy  future log_rmse "
            f"{row['prior_future_log_rmse']:.3f} -> {row['adaptive_future_log_rmse']:.3f}   "
            f"day21 actual={np.exp(row['day21_actual']):.3f} "
            f"prior={np.exp(row['day21_prior']):.3f} "
            f"adaptive={np.exp(row['day21_adaptive']):.3f}"
        )
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
