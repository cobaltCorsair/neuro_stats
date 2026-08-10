"""Standalone inference code for the frozen 97-feature growth model.

The model predicts the group mean relative tumour volume.  It does not predict
an individual animal and does not reconstruct an unknown delivered dose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_ARTIFACT = Path(__file__).with_name("canonical_model_v1.json")


@dataclass(frozen=True)
class SeriesInputs:
    """Information known before irradiation for one calendar-regimen series."""

    family: str
    regimen_class: str
    order: str
    log_v0: float
    total_dose_gy: float
    sum_d2_gy2: float
    n_events: int
    duration_hours: float = 0.0
    mean_interval_hours: float = 0.0
    timing_known: bool = False
    is_mixed: bool = False
    e_applicator_mismatch_log: float = 0.0
    e_applicator_known: bool = False
    gamma_field_enlargement_log: float = 0.0
    gamma_field_known: bool = False

    @classmethod
    def from_fractions(
        cls,
        *,
        family: str,
        regimen_class: str,
        order: str,
        log_v0: float,
        fraction_doses_gy: Sequence[float],
        duration_hours: float = 0.0,
        mean_interval_hours: float = 0.0,
        timing_known: bool = False,
        is_mixed: bool = False,
        e_applicator_mismatch_log: float = 0.0,
        e_applicator_known: bool = False,
        gamma_field_enlargement_log: float = 0.0,
        gamma_field_known: bool = False,
    ) -> "SeriesInputs":
        doses = np.asarray(fraction_doses_gy, dtype=float)
        if doses.ndim != 1 or not len(doses) or not np.all(np.isfinite(doses)):
            raise ValueError("fraction_doses_gy must be a non-empty finite sequence")
        return cls(
            family=family,
            regimen_class=regimen_class,
            order=order,
            log_v0=float(log_v0),
            total_dose_gy=float(np.sum(doses)),
            sum_d2_gy2=float(np.sum(np.square(doses))),
            n_events=int(len(doses)),
            duration_hours=float(duration_hours),
            mean_interval_hours=float(mean_interval_hours),
            timing_known=bool(timing_known),
            is_mixed=bool(is_mixed),
            e_applicator_mismatch_log=float(e_applicator_mismatch_log),
            e_applicator_known=bool(e_applicator_known),
            gamma_field_enlargement_log=float(gamma_field_enlargement_log),
            gamma_field_known=bool(gamma_field_known),
        )


def feature_names(categories: Mapping[str, Sequence[str]]) -> list[str]:
    names = [
        "time",
        "time2",
        "time3",
        "hinge_day3_sq",
        "hinge_day7_sq",
        "hinge_day14_sq",
        "log_v0_x_time",
        "log_v0_x_time2",
        "treated_x_g12",
        "dose_x_g12",
        "sum_d2_x_g12",
        "e_applicator_mismatch_dose_g12",
        "y_field_enlargement_g12",
        "y_field_known_g12",
        "e_applicator_known_g12",
    ]
    for family in categories["families"]:
        names.extend((f"family_{family}_dose_g12", f"family_{family}_sumd2_g12"))
    names.extend(
        (
            "sqrt_dose_x_g12",
            "dose2_x_g12",
            "n_events_x_g12",
            "duration_x_g12",
            "mean_interval_x_g12",
            "timing_known_x_g12",
            "mixed_x_g12",
        )
    )
    for family in categories["families"]:
        names.extend(
            (
                f"family_{family}_g7",
                f"family_{family}_g12",
                f"family_{family}_g21",
                f"family_{family}_dose_g7",
                f"family_{family}_dose_g21",
            )
        )
    names.extend(f"regimen_{regimen}_g12" for regimen in categories["regimens"])
    names.extend(f"order_{order}_g12" for order in categories["orders"])
    return names


def build_feature_vector(
    day: float,
    inputs: SeriesInputs,
    categories: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Build one raw predictor row in the exact order used for training."""

    day = float(day)
    if not np.isfinite(day) or day < 0.0 or day > 21.0:
        raise ValueError("day must be finite and between 0 and 21")
    if inputs.family != "control" and inputs.family not in categories["families"]:
        raise ValueError(f"unknown radiation family: {inputs.family!r}")
    if inputs.regimen_class != "control" and inputs.regimen_class not in categories["regimens"]:
        raise ValueError(f"unknown regimen class: {inputs.regimen_class!r}")
    if inputs.order != "none" and inputs.order not in categories["orders"]:
        raise ValueError(f"unknown component order: {inputs.order!r}")

    time = day / 21.0
    g7 = min(day / 7.0, 1.0)
    g12 = min(day / 12.0, 1.0)
    g21 = time
    dose = float(inputs.total_dose_gy)
    sum_d2 = float(inputs.sum_d2_gy2)
    treated = float(inputs.family != "control")
    values = [
        time,
        time**2,
        time**3,
        max(time - 3.0 / 21.0, 0.0) ** 2,
        max(time - 7.0 / 21.0, 0.0) ** 2,
        max(time - 14.0 / 21.0, 0.0) ** 2,
        float(inputs.log_v0) * time,
        float(inputs.log_v0) * time**2,
        treated * g12,
        dose * g12,
        sum_d2 * g12,
        float(inputs.e_applicator_mismatch_log) * dose * g12,
        float(inputs.gamma_field_enlargement_log) * g12,
        float(inputs.gamma_field_known) * g12,
        float(inputs.e_applicator_known) * g12,
    ]
    for family in categories["families"]:
        indicator = float(inputs.family == family)
        values.extend((indicator * dose * g12, indicator * sum_d2 * g12))
    values.extend(
        (
            np.sqrt(max(dose, 0.0)) * g12,
            dose**2 * g12,
            float(inputs.n_events) * g12,
            np.log1p(float(inputs.duration_hours)) * g12,
            np.log1p(float(inputs.mean_interval_hours)) * g12,
            float(inputs.timing_known) * g12,
            float(inputs.is_mixed) * g12,
        )
    )
    for family in categories["families"]:
        indicator = float(inputs.family == family)
        values.extend(
            (
                indicator * g7,
                indicator * g12,
                indicator * g21,
                indicator * dose * g7,
                indicator * dose * g21,
            )
        )
    values.extend(float(inputs.regimen_class == item) * g12 for item in categories["regimens"])
    values.extend(float(inputs.order == item) * g12 for item in categories["orders"])
    return np.asarray(values, dtype=float)


def build_feature_matrix(
    days: Iterable[float],
    inputs: SeriesInputs,
    categories: Mapping[str, Sequence[str]],
) -> np.ndarray:
    return np.vstack([build_feature_vector(day, inputs, categories) for day in days])


class CanonicalGrowthModel:
    """Frozen predictor with preprocessing and coefficients stored together."""

    def __init__(self, artifact: Mapping[str, object]):
        self.artifact = dict(artifact)
        self.categories = self.artifact["categories"]
        self.names = list(self.artifact["feature_names"])
        self.scales = np.asarray(self.artifact["scaler_scale"], dtype=float)
        self.coefficients = np.asarray(self.artifact["scaled_coefficients"], dtype=float)
        expected = feature_names(self.categories)
        if self.names != expected:
            raise ValueError("artifact feature order does not match the canonical builder")
        if len(self.scales) != len(self.names) or len(self.coefficients) != len(self.names):
            raise ValueError("artifact vectors have inconsistent lengths")
        if np.any(~np.isfinite(self.scales)) or np.any(self.scales <= 0.0):
            raise ValueError("artifact contains invalid feature scales")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT) -> "CanonicalGrowthModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def design_matrix(self, days: Iterable[float], inputs: SeriesInputs) -> np.ndarray:
        return build_feature_matrix(days, inputs, self.categories)

    def predict_log_relative(self, days: Iterable[float], inputs: SeriesInputs) -> np.ndarray:
        raw = self.design_matrix(days, inputs)
        return (raw / self.scales) @ self.coefficients

    def predict_relative(self, days: Iterable[float], inputs: SeriesInputs) -> np.ndarray:
        return np.exp(self.predict_log_relative(days, inputs))

    @staticmethod
    def _interval_bin(day: float) -> str:
        if day <= 7.0:
            return "day0_7"
        if day <= 14.0:
            return "day8_14"
        return "day15_21"

    def predict_interval(
        self,
        days: Iterable[float],
        inputs: SeriesInputs,
        *,
        level: int = 90,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        days_array = np.asarray(list(days), dtype=float)
        predicted_log = self.predict_log_relative(days_array, inputs)
        key = f"q{int(level)}"
        if key not in self.artifact["prediction_interval_log_half_width"]:
            raise ValueError("level must be one of the calibrated artifact levels")
        quantiles = self.artifact["prediction_interval_log_half_width"][key]
        half_width = np.asarray([quantiles[self._interval_bin(day)] for day in days_array])
        return (
            np.exp(predicted_log),
            np.exp(predicted_log - half_width),
            np.exp(predicted_log + half_width),
        )

    def explain(self, day: float, inputs: SeriesInputs) -> list[dict[str, float | str]]:
        raw = build_feature_vector(day, inputs, self.categories)
        contributions = raw / self.scales * self.coefficients
        return [
            {"feature": name, "raw_value": float(value), "log_contribution": float(part)}
            for name, value, part in zip(self.names, raw, contributions)
        ]

    def verification_payload(self, day: float, inputs: SeriesInputs) -> dict[str, object]:
        return {
            "day": float(day),
            "inputs": asdict(inputs),
            "predicted_log_relative": float(self.predict_log_relative([day], inputs)[0]),
            "predicted_relative": float(self.predict_relative([day], inputs)[0]),
        }
