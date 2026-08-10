"""Rebuild and verify the canonical model artifact from the audited archive."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    from .model import CanonicalGrowthModel, SeriesInputs, build_feature_vector, feature_names
    from .training import (
        ALPHA_GRID,
        HUBER_C,
        ROBUST_ITERATIONS,
        fit_robust_ridge,
        tune_alpha_by_group,
    )
except ImportError:  # Direct execution from this directory.
    from model import CanonicalGrowthModel, SeriesInputs, build_feature_vector, feature_names
    from training import (
        ALPHA_GRID,
        HUBER_C,
        ROBUST_ITERATIONS,
        fit_robust_ridge,
        tune_alpha_by_group,
    )


SOURCE = Path(
    r"D:\Диссертация\Результаты\Задача_4_Модель\4.6_Ковариаты_pole_gamma\run_growth_prediction.py"
)
OUTPUT = Path(__file__).with_name("canonical_model_v1.json")


def load_source():
    spec = importlib.util.spec_from_file_location("growth_prediction_archive_v97", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_from_row(row, source) -> SeriesInputs:
    one = row.to_frame().T
    _, mismatch, applicator_known = source.configuration_covariates(one)
    field, field_known = source.gamma_field_covariates(one)
    return SeriesInputs(
        family=str(row.family),
        regimen_class=str(row.regimen_class),
        order=str(row.order),
        log_v0=float(row.log_v0),
        total_dose_gy=float(row.total_dose_gy),
        sum_d2_gy2=float(row.sum_d2_gy2),
        n_events=int(row.n_events),
        duration_hours=float(row.duration_hours),
        mean_interval_hours=float(row.mean_interval_hours),
        timing_known=bool(row.timing_known),
        is_mixed=bool(row.is_mixed),
        e_applicator_mismatch_log=float(mismatch[0]),
        e_applicator_known=bool(applicator_known[0]),
        gamma_field_enlargement_log=float(field[0]),
        gamma_field_known=bool(field_known[0]),
    )


def main() -> None:
    source = load_source()
    observed, evaluation, series, sources = source.build_longitudinal_tables()
    modelling = source.collapse_animal_daily_to_series(evaluation)
    categories = source.fixed_categories(modelling)
    source_matrix, source_names = source.build_features(
        modelling, "hierarchical_robust", categories
    )
    canonical_matrix = np.vstack(
        [
            build_feature_vector(float(row.day), inputs_from_row(row, source), categories)
            for _, row in modelling.iterrows()
        ]
    )
    names = feature_names(categories)
    if names != source_names:
        raise AssertionError("feature name/order mismatch")
    max_feature_difference = float(np.max(np.abs(source_matrix - canonical_matrix)))
    if max_feature_difference > 1.0e-12:
        raise AssertionError(f"feature matrix mismatch: {max_feature_difference}")

    target = modelling["log_relative_volume"].to_numpy(float)
    weights = source.series_balanced_weights(modelling)
    groups = modelling["date"].astype(str).to_numpy()
    alpha, oof, tuning = tune_alpha_by_group(canonical_matrix, target, weights, groups)
    fitted = fit_robust_ridge(canonical_matrix, target, weights, alpha)
    source_fit = source.fit_linear(
        source_matrix, target, weights, alpha, robust=True
    )
    prediction_difference = float(
        np.max(np.abs(fitted.predict(canonical_matrix) - source_fit.predict(source_matrix)))
    )
    coefficient_difference = float(np.max(np.abs(fitted.model.coef_ - source_fit.model.coef_)))
    scale_difference = float(np.max(np.abs(fitted.scaler.scale_ - source_fit.scaler.scale_)))
    if max(prediction_difference, coefficient_difference, scale_difference) > 1.0e-10:
        raise AssertionError("canonical training does not reproduce the archived implementation")

    q90, q95 = source.calibration_quantiles(modelling, target, oof)
    verification_rows = [0, len(modelling) // 2, len(modelling) - 1]
    verification_cases = []
    for index in verification_rows:
        row = modelling.iloc[index]
        inputs = inputs_from_row(row, source)
        raw = build_feature_vector(float(row.day), inputs, categories)
        predicted_log = float((raw / fitted.scaler.scale_) @ fitted.model.coef_)
        verification_cases.append(
            {
                "series_key": str(row.series_key),
                "day": float(row.day),
                "inputs": asdict(inputs),
                "predicted_log_relative": predicted_log,
                "predicted_relative": float(np.exp(predicted_log)),
            }
        )

    manifest = source.source_manifest(sources)
    artifact = {
        "schema_version": 1,
        "model_name": "sarcoma_m1_group_growth_97_feature_robust_ridge",
        "status": "frozen_internal_model; not externally validated",
        "target": "log(mean_i(V_i(t)/V_i(0))) within a calendar-regimen series",
        "prediction_unit": "group mean trajectory, not an individual animal",
        "prediction_window_days": 21,
        "categories": categories,
        "feature_names": names,
        "scaler_scale": fitted.scaler.scale_.astype(float).tolist(),
        "scaled_coefficients": fitted.model.coef_.astype(float).tolist(),
        "raw_coefficients": (fitted.model.coef_ / fitted.scaler.scale_).astype(float).tolist(),
        "fit_intercept": False,
        "selected_alpha": float(alpha),
        "alpha_grid": list(ALPHA_GRID),
        "robust_iterations": ROBUST_ITERATIONS,
        "huber_c": HUBER_C,
        "row_weight": "1 / number of available days in the series",
        "inner_grouping": "calendar date",
        "prediction_interval_log_half_width": {"q90": q90, "q95": q95},
        "training_inventory": {
            "treatment_files": int(series.loc[series.family != "control", "n_experiments"].sum()),
            "treatment_series": int(series.loc[series.family != "control", "series_key"].nunique()),
            "control_files": int(series.loc[series.family == "control", "n_experiments"].sum()),
            "control_series": int(series.loc[series.family == "control", "series_key"].nunique()),
            "treatment_animals": int(series.loc[series.family != "control", "n_animals"].sum()),
            "control_animals": int(series.loc[series.family == "control", "n_animals"].sum()),
            "series_day_rows": int(len(modelling)),
            "source_files": int(len(manifest)),
        },
        "source_provenance": {
            "archived_training_script": str(SOURCE),
            "archived_training_script_sha256": sha256(SOURCE),
            "source_manifest_sha256": hashlib.sha256(
                manifest.to_csv(index=False).encode("utf-8")
            ).hexdigest(),
        },
        "parity_check": {
            "max_abs_feature_difference": max_feature_difference,
            "max_abs_prediction_difference": prediction_difference,
            "max_abs_scaled_coefficient_difference": coefficient_difference,
            "max_abs_scaler_scale_difference": scale_difference,
        },
        "full_fit_tuning": tuning,
        "verification_cases": verification_cases,
    }
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    loaded = CanonicalGrowthModel.load(OUTPUT)
    for case in verification_cases:
        inputs = SeriesInputs(**case["inputs"])
        reproduced = loaded.predict_log_relative([case["day"]], inputs)[0]
        if abs(reproduced - case["predicted_log_relative"]) > 1.0e-12:
            raise AssertionError("serialized artifact verification failed")
    print(json.dumps({"artifact": str(OUTPUT), "features": len(names), "alpha": alpha,
                      "parity": artifact["parity_check"], "inventory": artifact["training_inventory"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
