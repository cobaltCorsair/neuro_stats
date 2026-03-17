# coding: utf-8
"""Geometry-preserving parser for tumor Excel files with a-b-c measurements."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (
    extract_date_from_filename,
)
from work_with_prepared_data.radiobioligy_project.data_processing.rat_manager import (
    register_rat_labels,
)


def ellipsoid_volume_from_diameters(a: float, b: float, c: float) -> float:
    """Compute ellipsoid volume from three orthogonal diameters."""
    return float((math.pi * a * b * c) / 6.0)


def equivalent_sphere_diameter(volume: float) -> float:
    """Convert scalar volume to a sphere diameter with the same volume."""
    if not np.isfinite(volume) or volume < 0.0:
        return float("nan")
    return float((6.0 * volume / math.pi) ** (1.0 / 3.0))


@dataclass(frozen=True)
class TumorGeometryDataset:
    """Tumor dataset that preserves the measured ellipsoid axes."""

    path: Path
    experiment_params: Tuple[str, ...]
    time_data: Tuple[str, ...]
    rat_labels: Tuple[str, ...]
    axis_a: np.ndarray
    axis_b: np.ndarray
    axis_c: np.ndarray
    volumes: np.ndarray
    explicit_axes_mask: np.ndarray

    @property
    def rat_count(self) -> int:
        return len(self.rat_labels)

    @property
    def time_count(self) -> int:
        return len(self.time_data)

    def mean_geometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return mean axes and mean volume across rats for each day."""
        mean_a = np.nanmean(self.axis_a, axis=0)
        mean_b = np.nanmean(self.axis_b, axis=0)
        mean_c = np.nanmean(self.axis_c, axis=0)
        mean_volume = np.nanmean(self.volumes, axis=0)
        return mean_a, mean_b, mean_c, mean_volume

    def rat_geometry(self, rat_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return axes and volume trajectory for one rat."""
        return (
            self.axis_a[rat_index],
            self.axis_b[rat_index],
            self.axis_c[rat_index],
            self.volumes[rat_index],
        )


def _normalize_cell_text(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return (
        str(value)
        .strip()
        .replace(",", ".")
        .replace("–", "-")
        .replace("—", "-")
        .replace(" -", "-")
        .replace("- ", "-")
    )


def _format_time_label(value: object) -> str:
    if pd.isna(value):
        return ""
    token = str(value).strip().replace("V", "0").split(" ")[0]
    try:
        numeric = float(token)
    except ValueError:
        return token
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _parse_geometry_cell(value: object) -> Tuple[float, float, float, float, bool]:
    """Parse one tumor cell as diameters a/b/c plus volume."""
    text = _normalize_cell_text(value)
    if not text or text == "NA":
        return (float("nan"), float("nan"), float("nan"), float("nan"), False)

    if "-" in text:
        parts = text.split("-")
        if len(parts) == 3:
            try:
                a, b, c = (float(part) for part in parts)
            except ValueError:
                return (float("nan"), float("nan"), float("nan"), float("nan"), False)
            return (a, b, c, ellipsoid_volume_from_diameters(a, b, c), True)

    try:
        volume = float(text)
    except ValueError:
        return (float("nan"), float("nan"), float("nan"), float("nan"), False)

    diameter = equivalent_sphere_diameter(volume)
    return (diameter, diameter, diameter, volume, False)


def process_tumor_geometry_excel(file_path: str | Path) -> TumorGeometryDataset:
    """Read tumor Excel data while preserving ellipsoid axes from a-b-c cells."""
    path = Path(file_path).expanduser().resolve()
    data = pd.read_excel(path, header=None)

    experiment_params = data.iloc[0, :].dropna().astype(str).tolist()
    if experiment_params and "ч" in experiment_params[-1]:
        irradiation_time = experiment_params.pop().strip()
        experiment_params.append(f"Irradiation Time={irradiation_time}")

    time_data = tuple(
        _format_time_label(item)
        for item in data.iloc[1, 1:]
        if not pd.isna(item)
    )
    tumor_data = data.iloc[2:, :].copy()
    rat_labels = tuple(tumor_data.iloc[:, 0].astype(str).tolist())

    axis_a_rows: List[List[float]] = []
    axis_b_rows: List[List[float]] = []
    axis_c_rows: List[List[float]] = []
    volume_rows: List[List[float]] = []
    explicit_rows: List[List[bool]] = []

    for _, row in tumor_data.iterrows():
        axis_a_row: List[float] = []
        axis_b_row: List[float] = []
        axis_c_row: List[float] = []
        volume_row: List[float] = []
        explicit_row: List[bool] = []
        for item in row[1:]:
            a, b, c, volume, explicit = _parse_geometry_cell(item)
            axis_a_row.append(a)
            axis_b_row.append(b)
            axis_c_row.append(c)
            volume_row.append(volume)
            explicit_row.append(explicit)
        axis_a_rows.append(axis_a_row)
        axis_b_rows.append(axis_b_row)
        axis_c_rows.append(axis_c_row)
        volume_rows.append(volume_row)
        explicit_rows.append(explicit_row)

    formatted_date = extract_date_from_filename(str(path))
    if formatted_date:
        experiment_params.append(f"Date={formatted_date}")

    register_rat_labels(list(rat_labels), os.path.basename(path))

    return TumorGeometryDataset(
        path=path,
        experiment_params=tuple(experiment_params),
        time_data=time_data,
        rat_labels=rat_labels,
        axis_a=np.asarray(axis_a_rows, dtype=float),
        axis_b=np.asarray(axis_b_rows, dtype=float),
        axis_c=np.asarray(axis_c_rows, dtype=float),
        volumes=np.asarray(volume_rows, dtype=float),
        explicit_axes_mask=np.asarray(explicit_rows, dtype=bool),
    )
