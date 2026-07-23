from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import openpyxl
import pandas as pd
import pydicom
import scipy
from matplotlib.lines import Line2D
from pydicom.dataset import Dataset


DEFAULT_RESULTS = Path(
    r"D:\Диссертация\Результаты\Задача_2_Томография\2.3_Итоговый_анализ"
)
DEFAULT_MORPHOMETRY_ROOT = Path(
    r"D:\Диссертация\Результаты\Задача_2_Томография"
    r"\2.1_Морфометрия_двух_крыс\01_Исходные_таблицы"
)
DEFAULT_RAT_2024 = Path(r"C:\dev\dissertation\2024-06-21_Rat_M1_Obninsk")
DEFAULT_RATS_2025 = Path(r"C:\dev\dissertation\Rats sarcoma M1 2025")
DEFAULT_M6 = Path(r"C:\dev\dissertation\RAT_CT+RTSTRUCT\m6_e")

MORPHOMETRY_FILES = (
    ("Студентка", 1, Path("Наблюдатель_A") / "1 крыса.xlsx", 5, 6, "cm"),
    ("Студентка", 2, Path("Наблюдатель_A") / "2 крыса.xlsx", 6, 7, "cm"),
    (
        "Автор",
        1,
        Path("Наблюдатель_B") / "Крыса_1_LWH_08-08_03-09.xlsx",
        2,
        3,
        "cm",
    ),
    (
        "Автор",
        2,
        Path("Наблюдатель_B") / "Крыса_2_LWH_08-08_03-09.xlsx",
        2,
        3,
        "cm",
    ),
)

LONGITUDINAL_DATES = (
    "2025-08-08",
    "2025-08-11",
    "2025-08-13",
    "2025-08-15",
    "2025-08-18",
    "2025-08-20",
    "2025-08-22",
    "2025-08-25",
    "2025-08-27",
    "2025-08-29",
    "2025-09-01",
    "2025-09-03",
)


@dataclass(frozen=True)
class DicomRoot:
    key: str
    path: Path
    subject_fallback: str


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_date_cell(value: object, year: int = 2025) -> date | None:
    if isinstance(value, (date, pd.Timestamp)):
        return date(value.year, value.month, value.day)
    match = re.search(r"(\d{1,2})[./](\d{1,2})", str(value or ""))
    if not match:
        return None
    day, month = map(int, match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_lwh(value: object, unit: str = "cm") -> tuple[float, float, float]:
    # Tumor dimensions are non-negative; hyphens in source cells are separators,
    # not numeric signs.
    numbers = re.findall(r"\d+(?:[.,]\d+)?", str(value or ""))
    parsed = [float(number.replace(",", ".")) for number in numbers]
    if len(parsed) != 3:
        raise ValueError(f"Expected an L-W-H triple, received {value!r}")
    if unit.lower() == "mm":
        parsed = [number / 10.0 for number in parsed]
    return tuple(parsed)  # type: ignore[return-value]


def locate_measurement_rows(
    worksheet: openpyxl.worksheet.worksheet.Worksheet,
) -> tuple[int, int]:
    values = list(worksheet.iter_rows(values_only=True))
    for header_index, row in enumerate(values):
        date_count = sum(parse_date_cell(value) is not None for value in row)
        if date_count < 10:
            continue
        for value_index in range(header_index + 1, min(header_index + 5, len(values))):
            triple_count = 0
            for value in values[value_index]:
                try:
                    parse_lwh(value)
                    triple_count += 1
                except ValueError:
                    pass
            if triple_count >= 10:
                return header_index + 1, value_index + 1
    raise ValueError(f"Could not locate date and L-W-H rows in {worksheet.title}")


def read_morphometry_file(
    path: Path,
    observer: str,
    animal: int,
    header_row: int | None = None,
    value_row: int | None = None,
    unit: str = "cm",
) -> list[dict[str, object]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook.worksheets[0]
    if header_row is None or value_row is None:
        header_row, value_row = locate_measurement_rows(worksheet)
    records: list[dict[str, object]] = []
    first_date: date | None = None
    for column in range(1, worksheet.max_column + 1):
        observation_date = parse_date_cell(worksheet.cell(header_row, column).value)
        if observation_date is None:
            continue
        try:
            length, width, height = parse_lwh(
                worksheet.cell(value_row, column).value, unit=unit
            )
        except ValueError:
            continue
        first_date = first_date or observation_date
        records.append(
            {
                "animal": animal,
                "date": observation_date.isoformat(),
                "day_from_first_scan": (observation_date - first_date).days,
                "observer": observer,
                "source_file": path.name,
                "source_cell": f"{worksheet.title}!{worksheet.cell(value_row, column).coordinate}",
                "length_cm": length,
                "width_cm": width,
                "height_cm": height,
                "ellipsoid_volume_cm3": math.pi / 6.0 * length * width * height,
            }
        )
    if len(records) != 12:
        raise ValueError(f"Expected 12 dates in {path}, found {len(records)}")
    return records


def load_morphometry(root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for observer, animal, relative, header_row, value_row, unit in MORPHOMETRY_FILES:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        records.extend(
            read_morphometry_file(
                path,
                observer,
                animal,
                header_row=header_row,
                value_row=value_row,
                unit=unit,
            )
        )
        manifest.append(
            {
                "role": "morphometry_source",
                "observer": observer,
                "animal": animal,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(records).sort_values(
        ["animal", "date", "observer"]
    )
    if len(frame) != 48:
        raise ValueError(f"Expected 48 morphometry records, found {len(frame)}")
    observer_means = frame.groupby("observer")["ellipsoid_volume_cm3"].mean()
    mean_ratio = float(observer_means.max() / observer_means.min())
    if (
        (frame["ellipsoid_volume_cm3"] <= 0).any()
        or frame["ellipsoid_volume_cm3"].max() > 500.0
        or mean_ratio > 5.0
    ):
        raise ValueError(
            "Implausible morphometry scale: check source rows and cm/mm units"
        )
    return frame.reset_index(drop=True), manifest


def concordance_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) != len(y_array) or len(x_array) < 2:
        return float("nan")
    covariance = float(np.mean((x_array - x_array.mean()) * (y_array - y_array.mean())))
    variance_x = float(np.var(x_array, ddof=0))
    variance_y = float(np.var(y_array, ddof=0))
    denominator = variance_x + variance_y + (
        float(np.mean(x_array)) - float(np.mean(y_array))
    ) ** 2
    return 2.0 * covariance / denominator if denominator else float("nan")


def paired_morphometry(raw: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["animal", "date", "day_from_first_scan"]
    value_columns = [
        "length_cm",
        "width_cm",
        "height_cm",
        "ellipsoid_volume_cm3",
    ]
    pivot = raw.pivot(
        index=index_columns,
        columns="observer",
        values=value_columns,
    )
    pivot.columns = [f"{metric}_{observer}" for metric, observer in pivot.columns]
    result = pivot.reset_index()
    for metric in value_columns:
        author = result[f"{metric}_Автор"]
        student = result[f"{metric}_Студентка"]
        result[f"{metric}_difference_author_minus_student"] = author - student
        result[f"{metric}_absolute_difference"] = (author - student).abs()
    author_volume = result["ellipsoid_volume_cm3_Автор"]
    student_volume = result["ellipsoid_volume_cm3_Студентка"]
    result["mean_volume_cm3"] = (author_volume + student_volume) / 2.0
    result["symmetric_volume_difference"] = (
        (author_volume - student_volume) / result["mean_volume_cm3"]
    )
    result["absolute_symmetric_volume_difference"] = result[
        "symmetric_volume_difference"
    ].abs()
    result["log_volume_ratio_author_student"] = np.log(
        author_volume / student_volume
    )
    return result.sort_values(["animal", "date"]).reset_index(drop=True)


def agreement_summary(paired: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, label, unit in (
        ("length_cm", "L", "cm"),
        ("width_cm", "W", "cm"),
        ("height_cm", "H", "cm"),
        ("ellipsoid_volume_cm3", "V", "cm3"),
    ):
        author = paired[f"{metric}_Автор"].to_numpy(float)
        student = paired[f"{metric}_Студентка"].to_numpy(float)
        difference = author - student
        rows.append(
            {
                "scope": "all",
                "metric": label,
                "unit": unit,
                "n_paired_timepoints": len(paired),
                "mean_author": float(np.mean(author)),
                "mean_student": float(np.mean(student)),
                "mean_bias_author_minus_student": float(np.mean(difference)),
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "pearson_r": float(np.corrcoef(author, student)[0, 1]),
                "lin_ccc": concordance_correlation(author, student),
            }
        )
    log_ratio = paired["log_volume_ratio_author_student"].to_numpy(float)
    log_mean = float(np.mean(log_ratio))
    log_sd = float(np.std(log_ratio, ddof=1))
    rows.append(
        {
            "scope": "all",
            "metric": "V_log_ratio",
            "unit": "ratio",
            "n_paired_timepoints": len(paired),
            "mean_author": "",
            "mean_student": "",
            "mean_bias_author_minus_student": "",
            "mean_absolute_difference": float(
                paired["absolute_symmetric_volume_difference"].mean()
            ),
            "pearson_r": "",
            "lin_ccc": "",
            "geometric_mean_ratio_author_student": math.exp(log_mean),
            "lower_descriptive_limit_ratio": math.exp(log_mean - 1.96 * log_sd),
            "upper_descriptive_limit_ratio": math.exp(log_mean + 1.96 * log_sd),
            "median_absolute_symmetric_difference": float(
                paired["absolute_symmetric_volume_difference"].median()
            ),
            "maximum_absolute_symmetric_difference": float(
                paired["absolute_symmetric_volume_difference"].max()
            ),
        }
    )
    for animal, subset in paired.groupby("animal"):
        author = subset["ellipsoid_volume_cm3_Автор"].to_numpy(float)
        student = subset["ellipsoid_volume_cm3_Студентка"].to_numpy(float)
        rows.append(
            {
                "scope": f"animal_{animal}",
                "metric": "V",
                "unit": "cm3",
                "n_paired_timepoints": len(subset),
                "mean_author": float(np.mean(author)),
                "mean_student": float(np.mean(student)),
                "mean_bias_author_minus_student": float(
                    np.mean(author - student)
                ),
                "mean_absolute_difference": float(
                    np.mean(np.abs(author - student))
                ),
                "pearson_r": float(np.corrcoef(author, student)[0, 1]),
                "lin_ccc": concordance_correlation(author, student),
            }
        )
    return rows


def growth_summary(raw: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (animal, observer), subset in raw.groupby(["animal", "observer"]):
        subset = subset.sort_values("day_from_first_scan")
        days = subset["day_from_first_scan"].to_numpy(float)
        volumes = subset["ellipsoid_volume_cm3"].to_numpy(float)
        slope, intercept = np.polyfit(days, np.log(volumes), 1)
        fitted = intercept + slope * days
        residual_sum = float(np.sum((np.log(volumes) - fitted) ** 2))
        total_sum = float(
            np.sum((np.log(volumes) - np.mean(np.log(volumes))) ** 2)
        )
        rows.append(
            {
                "animal": int(animal),
                "observer": observer,
                "n_timepoints": len(subset),
                "first_date": subset.iloc[0]["date"],
                "last_date": subset.iloc[-1]["date"],
                "volume_first_cm3": float(volumes[0]),
                "volume_last_cm3": float(volumes[-1]),
                "fold_change": float(volumes[-1] / volumes[0]),
                "exponential_rate_per_day": float(slope),
                "doubling_time_days": float(math.log(2.0) / slope),
                "r2_log_linear": 1.0 - residual_sum / total_sum,
            }
        )
    return rows


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(item) for item in value)
    return str(value)


def subject_from_path(root: DicomRoot, path: Path, dataset: Dataset) -> str:
    relative = str(path.relative_to(root.path))
    raw = " ".join(
        (
            as_text(getattr(dataset, "PatientID", "")),
            as_text(getattr(dataset, "PatientName", "")),
            relative,
        )
    ).lower()
    if re.search(r"rat[_\s-]*1\b", raw):
        return "rat_1"
    if re.search(r"rat[_\s-]*2\b", raw):
        return "rat_2"
    return root.subject_fallback


def dicom_candidates(root: Path) -> Iterable[Path]:
    excluded_suffixes = {
        ".zip",
        ".nii",
        ".gz",
        ".vti",
        ".xlsx",
        ".csv",
        ".json",
        ".md",
        ".png",
    }
    for path in root.rglob("*"):
        if path.is_file() and path.name.upper() != "DICOMDIR":
            if path.suffix.lower() not in excluded_suffixes:
                yield path


def read_dicom_header(path: Path) -> Dataset | None:
    try:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )
    except Exception:
        return None
    return dataset if getattr(dataset, "SOPClassUID", None) else None


def referenced_series_uids(dataset: Dataset) -> list[str]:
    result: list[str] = []
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []):
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for series in getattr(study, "RTReferencedSeriesSequence", []):
                uid = as_text(getattr(series, "SeriesInstanceUID", ""))
                if uid:
                    result.append(uid)
    return sorted(set(result))


def inventory_dicom(
    roots: Sequence[DicomRoot],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[tuple[Path, Dataset]]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    for root in roots:
        for path in dicom_candidates(root.path):
            dataset = read_dicom_header(path)
            if dataset is None:
                failures.append(
                    {
                        "source_root": root.key,
                        "relative_path": str(path.relative_to(root.path)),
                    }
                )
                continue
            series_uid = as_text(getattr(dataset, "SeriesInstanceUID", ""))
            groups[(root.key, series_uid or f"NO_SERIES::{path.parent}")].append(
                (path, dataset)
            )
    roots_by_key = {root.key: root for root in roots}
    series_rows: list[dict[str, object]] = []
    rtstruct_rows: list[dict[str, object]] = []
    for (root_key, series_uid), items in sorted(groups.items()):
        root = roots_by_key[root_key]
        first_path, dataset = items[0]
        relative = str(first_path.relative_to(root.path))
        series_rows.append(
            {
                "source_root": root_key,
                "subject": subject_from_path(root, first_path, dataset),
                "modality": as_text(getattr(dataset, "Modality", "")),
                "study_date": as_text(getattr(dataset, "StudyDate", "")),
                "series_date": as_text(getattr(dataset, "SeriesDate", "")),
                "acquisition_date": as_text(
                    getattr(dataset, "AcquisitionDate", "")
                ),
                "acquisition_time": as_text(
                    getattr(dataset, "AcquisitionTime", "")
                ),
                "study_description": as_text(
                    getattr(dataset, "StudyDescription", "")
                ),
                "series_description": as_text(
                    getattr(dataset, "SeriesDescription", "")
                ),
                "protocol_name": as_text(getattr(dataset, "ProtocolName", "")),
                "manufacturer": as_text(getattr(dataset, "Manufacturer", "")),
                "model": as_text(
                    getattr(dataset, "ManufacturerModelName", "")
                ),
                "rows": as_text(getattr(dataset, "Rows", "")),
                "columns": as_text(getattr(dataset, "Columns", "")),
                "pixel_spacing_mm": as_text(
                    getattr(dataset, "PixelSpacing", "")
                ),
                "slice_thickness_mm": as_text(
                    getattr(dataset, "SliceThickness", "")
                ),
                "file_count": len(items),
                "frame_count_total": sum(
                    int(getattr(item, "NumberOfFrames", 1) or 1)
                    for _, item in items
                ),
                "series_instance_uid": series_uid,
                "frame_of_reference_uid": as_text(
                    getattr(dataset, "FrameOfReferenceUID", "")
                ),
                "sop_class_uid": as_text(
                    getattr(dataset, "SOPClassUID", "")
                ),
                "relative_folder": str(first_path.parent.relative_to(root.path)),
                "example_file": relative,
            }
        )
        if as_text(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT":
            continue
        contour_by_roi = {
            int(item.ReferencedROINumber): item
            for item in getattr(dataset, "ROIContourSequence", [])
        }
        observations = {
            int(item.ReferencedROINumber): item
            for item in getattr(dataset, "RTROIObservationsSequence", [])
        }
        references = "\\".join(referenced_series_uids(dataset))
        for roi in getattr(dataset, "StructureSetROISequence", []):
            roi_number = int(roi.ROINumber)
            roi_contour = contour_by_roi.get(roi_number)
            contours = list(
                getattr(roi_contour, "ContourSequence", [])
                if roi_contour is not None
                else []
            )
            points = sum(
                int(getattr(contour, "NumberOfContourPoints", 0) or 0)
                for contour in contours
            )
            observation = observations.get(roi_number)
            rtstruct_rows.append(
                {
                    "source_root": root_key,
                    "subject": subject_from_path(root, first_path, dataset),
                    "relative_path": relative,
                    "study_date": as_text(getattr(dataset, "StudyDate", "")),
                    "structure_set_label": as_text(
                        getattr(dataset, "StructureSetLabel", "")
                    ),
                    "series_instance_uid": series_uid,
                    "referenced_series_uids": references,
                    "roi_number": roi_number,
                    "roi_name": as_text(getattr(roi, "ROIName", "")),
                    "interpreted_type": as_text(
                        getattr(observation, "RTROIInterpretedType", "")
                        if observation is not None
                        else ""
                    ),
                    "contour_count": len(contours),
                    "contour_point_count": points,
                    "has_contours": bool(contours and points),
                }
            )
    return series_rows, rtstruct_rows, failures


def inventory_nifti(root: Path) -> list[dict[str, object]]:
    paths = sorted(set(root.rglob("*.nii")) | set(root.rglob("*.nii.gz")))
    basename_counts = Counter(path.name for path in paths)
    rows: list[dict[str, object]] = []
    for path in paths:
        image = nib.load(str(path), mmap=True)
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "shape": "x".join(map(str, image.shape)),
                "voxel_size_mm": "x".join(
                    f"{value:.6g}" for value in image.header.get_zooms()[:3]
                ),
                "dtype": str(image.header.get_data_dtype()),
                "qform_code": int(image.header["qform_code"]),
                "sform_code": int(image.header["sform_code"]),
                "repeated_basename": basename_counts[path.name] > 1,
            }
        )
    return rows


def longitudinal_matrix(
    dicom_series: Sequence[dict[str, object]],
    nifti_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    mri = {
        (str(row["subject"]), str(row["study_date"])): row
        for row in dicom_series
        if row["source_root"] == "rats_2025" and row["modality"] == "MR"
    }
    ct: dict[tuple[str, str, str], dict[str, object]] = {}
    pattern = re.compile(
        r"CT_2025-(\d{2})-(\d{2})_.*rat_(1|2)[ _](TB|UF)\.nii$",
        re.IGNORECASE,
    )
    for row in nifti_rows:
        if Path(str(row["relative_path"])).parent != Path("."):
            continue
        match = pattern.match(str(row["file_name"]))
        if not match:
            continue
        month, day, animal, mode = match.groups()
        ct[(f"rat_{animal}", f"2025{month}{day}", mode.upper())] = row
    rows: list[dict[str, object]] = []
    first = date.fromisoformat(LONGITUDINAL_DATES[0])
    for animal in ("rat_1", "rat_2"):
        for iso_date in LONGITUDINAL_DATES:
            compact = iso_date.replace("-", "")
            current = date.fromisoformat(iso_date)
            mr = mri.get((animal, compact))
            tb = ct.get((animal, compact, "TB"))
            uf = ct.get((animal, compact, "UF"))
            rows.append(
                {
                    "animal": animal,
                    "date": iso_date,
                    "day_from_first_scan": (current - first).days,
                    "morphometry_author": "yes",
                    "morphometry_student": "yes",
                    "mri_dicom": "yes" if mr else "no",
                    "mri_shape": (
                        f"{mr['rows']}x{mr['columns']}x{mr['file_count']}"
                        if mr
                        else ""
                    ),
                    "mri_pixel_spacing_mm": mr["pixel_spacing_mm"] if mr else "",
                    "mri_slice_thickness_mm": (
                        mr["slice_thickness_mm"] if mr else ""
                    ),
                    "ct_tb_nifti": "yes" if tb else "no",
                    "ct_tb_shape": tb["shape"] if tb else "",
                    "ct_tb_voxel_size_mm": tb["voxel_size_mm"] if tb else "",
                    "ct_uf_nifti": "yes" if uf else "no",
                    "ct_uf_shape": uf["shape"] if uf else "",
                    "ct_uf_voxel_size_mm": uf["voxel_size_mm"] if uf else "",
                    "matched_segmentation": "no",
                }
            )
    return rows


def shoelace_area_xy(points: np.ndarray) -> float:
    x_values = points[:, 0]
    y_values = points[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_values, np.roll(y_values, 1))
            - np.dot(y_values, np.roll(x_values, 1))
        )
    )


def rtstruct_geometry(source: str, path: Path) -> list[dict[str, object]]:
    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    names = {
        int(item.ROINumber): str(item.ROIName)
        for item in getattr(dataset, "StructureSetROISequence", [])
    }
    types = {
        int(item.ReferencedROINumber): str(
            getattr(item, "RTROIInterpretedType", "")
        )
        for item in getattr(dataset, "RTROIObservationsSequence", [])
    }
    rows: list[dict[str, object]] = []
    for roi_contour in getattr(dataset, "ROIContourSequence", []):
        number = int(roi_contour.ReferencedROINumber)
        contours = list(getattr(roi_contour, "ContourSequence", []))
        all_points: list[np.ndarray] = []
        areas_by_z: defaultdict[float, float] = defaultdict(float)
        for contour in contours:
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            if not points.size:
                continue
            all_points.append(points)
            areas_by_z[round(float(np.median(points[:, 2])), 5)] += (
                shoelace_area_xy(points)
            )
        if not all_points:
            continue
        combined = np.vstack(all_points)
        extents = combined.max(axis=0) - combined.min(axis=0)
        z_values = np.asarray(sorted(areas_by_z), dtype=float)
        areas = np.asarray([areas_by_z[value] for value in z_values], dtype=float)
        spacing = (
            float(np.median(np.diff(z_values))) if len(z_values) > 1 else float("nan")
        )
        trapezoid = (
            float(np.trapz(areas, z_values)) if len(z_values) > 1 else float("nan")
        )
        slice_sum = float(np.sum(areas) * spacing)
        rows.append(
            {
                "source": source,
                "file": str(path),
                "roi_number": number,
                "roi_name": names.get(number, ""),
                "interpreted_type": types.get(number, ""),
                "contour_count": len(contours),
                "unique_slice_count": len(z_values),
                "median_contour_spacing_mm": spacing,
                "extent_x_mm": float(extents[0]),
                "extent_y_mm": float(extents[1]),
                "extent_z_mm": float(extents[2]),
                "contour_volume_trapezoid_cm3": trapezoid / 1000.0,
                "contour_volume_slice_sum_cm3": slice_sum / 1000.0,
                "bounding_ellipsoid_volume_cm3": float(
                    math.pi / 6.0 * np.prod(extents) / 1000.0
                ),
            }
        )
    return rows


def ct_series_files(root: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in root.glob("*.dcm"):
        dataset = read_dicom_header(path)
        if dataset is None or getattr(dataset, "Modality", "") != "CT":
            continue
        files.append((int(getattr(dataset, "InstanceNumber", 0)), path))
    return sorted(files)


def pixel_series_sha256(files: Sequence[tuple[int, Path]]) -> str:
    digest = hashlib.sha256()
    for _, path in files:
        dataset = pydicom.dcmread(str(path), force=True)
        digest.update(dataset.PixelData)
    return digest.hexdigest()


def verify_m6_linkage(rat_2024: Path, m6_root: Path) -> dict[str, object]:
    original_root = rat_2024 / "CT" / "2024-06-21_WB_DICOM"
    original = ct_series_files(original_root)
    exported = ct_series_files(m6_root)
    original_hash = pixel_series_sha256(original)
    exported_hash = pixel_series_sha256(exported)
    original_header = pydicom.dcmread(
        str(original[0][1]), stop_before_pixels=True, force=True
    )
    exported_header = pydicom.dcmread(
        str(exported[0][1]), stop_before_pixels=True, force=True
    )
    return {
        "dataset": "m6_e",
        "linked_dataset": "rat_2024",
        "link_status": (
            "exact_pixel_match" if original_hash == exported_hash else "not_matched"
        ),
        "inferred_subject": "rat_2024",
        "inferred_ct_acquisition_date": as_text(
            getattr(original_header, "AcquisitionDate", "")
        ),
        "original_slice_count": len(original),
        "export_slice_count": len(exported),
        "pixel_data_sha256_original": original_hash,
        "pixel_data_sha256_export": exported_hash,
        "same_pixel_hash": original_hash == exported_hash,
        "original_series_uid": as_text(original_header.SeriesInstanceUID),
        "export_series_uid": as_text(exported_header.SeriesInstanceUID),
        "original_frame_uid": as_text(original_header.FrameOfReferenceUID),
        "export_frame_uid": as_text(exported_header.FrameOfReferenceUID),
        "interpretation": (
            "m6_e is a re-export of the 2024-06-24 whole-body CT; "
            "it is not one of the two longitudinal 2025 rats"
        ),
    }


def find_rtstruct(root: Path) -> Path:
    candidates = sorted(root.glob("RS.*.dcm"))
    if candidates:
        return candidates[0]
    candidates = sorted(root.rglob("RTSTRUCT.dcm"))
    if not candidates:
        raise FileNotFoundError(f"No RTSTRUCT under {root}")
    return candidates[0]


def render_longitudinal(raw: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    styles = {"Автор": ("#B23A48", "o"), "Студентка": ("#2F6690", "s")}
    for animal, axis in zip((1, 2), axes):
        for observer, (color, marker) in styles.items():
            subset = raw[
                (raw["animal"] == animal) & (raw["observer"] == observer)
            ].sort_values("day_from_first_scan")
            axis.plot(
                subset["day_from_first_scan"],
                subset["ellipsoid_volume_cm3"],
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=4.5,
                label=observer,
            )
        axis.set_title(f"Крыса {animal}")
        axis.set_xlabel("Сутки от 08.08.2025")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Эллипсоидальный объём, см³")
    axes[1].legend(frameon=False)
    fig.suptitle("Продольная томографическая морфометрия саркомы М-1")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_agreement(paired: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    x_values = np.sqrt(
        paired["ellipsoid_volume_cm3_Автор"]
        * paired["ellipsoid_volume_cm3_Студентка"]
    )
    y_values = 100.0 * paired["log_volume_ratio_author_student"]
    mean = float(y_values.mean())
    sd = float(y_values.std(ddof=1))
    fig, axis = plt.subplots(figsize=(7.3, 4.5))
    colors = paired["animal"].map({1: "#2F6690", 2: "#B23A48"})
    axis.scatter(x_values, y_values, c=colors, s=36, alpha=0.85)
    axis.axhline(mean, color="#222222", linewidth=1.4)
    axis.axhline(mean - 1.96 * sd, color="#777777", linestyle="--", linewidth=1.2)
    axis.axhline(
        mean + 1.96 * sd,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
    )
    axis.set_xlabel("Геометрическое среднее объёма двух наблюдателей, см³")
    axis.set_ylabel("100 × ln(Vавтор / Vстудент), %")
    axis.set_title("Согласованность расчётного объёма")
    axis.grid(True, alpha=0.2)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color="#2F6690", label="Крыса 1"),
            Line2D([], [], marker="o", linestyle="", color="#B23A48", label="Крыса 2"),
            Line2D([], [], color="#222222", linewidth=1.4, label="Среднее"),
            Line2D(
                [],
                [],
                color="#777777",
                linestyle="--",
                linewidth=1.2,
                label="Описательные пределы ±1,96 SD",
            ),
        ],
        frameon=False,
        fontsize=8.5,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_availability(matrix: Sequence[dict[str, object]], output: Path) -> None:
    frame = pd.DataFrame(matrix)
    modalities = [
        "morphometry_author",
        "morphometry_student",
        "mri_dicom",
        "ct_tb_nifti",
        "ct_uf_nifti",
        "matched_segmentation",
    ]
    values = (frame[modalities] == "yes").astype(int).to_numpy().T
    labels = [
        "Морфометрия: автор",
        "Морфометрия: студентка",
        "МРТ",
        "КТ TB",
        "КТ UF",
        "Сегментация",
    ]
    columns = [
        f"{row.animal[-1]}:{row.date[5:]}"
        for row in frame[["animal", "date"]].itertuples(index=False)
    ]
    fig, axis = plt.subplots(figsize=(12, 3.6))
    axis.imshow(values, cmap=matplotlib.colors.ListedColormap(["#F1F1F1", "#2F75B5"]), aspect="auto")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=7)
    axis.set_xlabel("Крыса:дата")
    axis.set_title("Полнота продольного массива 2025 года")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "✓" if values[row, column] else "—",
                ha="center",
                va="center",
                color="white" if values[row, column] else "#777777",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_rtstruct_overlay(m6_root: Path, rtstruct_path: Path, output: Path) -> None:
    structure = pydicom.dcmread(str(rtstruct_path), force=True)
    names = {
        int(item.ROINumber): str(item.ROIName)
        for item in structure.StructureSetROISequence
    }
    selected: list[tuple[str, np.ndarray]] = []
    for roi in structure.ROIContourSequence:
        name = names.get(int(roi.ReferencedROINumber), "")
        if name not in {"GTVp", "GTVp1", "PTV_High"}:
            continue
        for contour in getattr(roi, "ContourSequence", []):
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            selected.append((name, points))
    if not selected:
        raise ValueError("No selected tumor contours in m6_e RTSTRUCT")
    target_name, target_points = max(
        selected, key=lambda item: shoelace_area_xy(item[1])
    )
    target_z = float(np.median(target_points[:, 2]))
    slices: list[tuple[float, Path]] = []
    for _, path in ct_series_files(m6_root):
        header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        slices.append((float(header.ImagePositionPatient[2]), path))
    _, ct_path = min(slices, key=lambda item: abs(item[0] - target_z))
    dataset = pydicom.dcmread(str(ct_path), force=True)
    pixels = dataset.pixel_array.astype(float)
    origin = np.asarray(dataset.ImagePositionPatient, dtype=float)
    spacing = np.asarray(dataset.PixelSpacing, dtype=float)
    row_direction = np.asarray(dataset.ImageOrientationPatient[:3], dtype=float)
    column_direction = np.asarray(dataset.ImageOrientationPatient[3:], dtype=float)
    colors = {"GTVp": "#FF3B30", "GTVp1": "#FFD60A", "PTV_High": "#34C759"}
    fig, axis = plt.subplots(figsize=(6.2, 6.2))
    axis.imshow(pixels, cmap="gray", vmin=-250, vmax=700)
    plotted: set[str] = set()
    all_pixel_points: list[np.ndarray] = []
    for name, points in selected:
        if abs(float(np.median(points[:, 2])) - float(dataset.ImagePositionPatient[2])) > 0.11:
            continue
        displacement = points - origin
        columns = displacement @ row_direction / spacing[1]
        rows = displacement @ column_direction / spacing[0]
        coords = np.column_stack((columns, rows))
        all_pixel_points.append(coords)
        axis.plot(
            columns,
            rows,
            color=colors[name],
            linewidth=1.5,
            label=name if name not in plotted else None,
        )
        plotted.add(name)
    if all_pixel_points:
        combined = np.vstack(all_pixel_points)
        margin = 30
        axis.set_xlim(combined[:, 0].min() - margin, combined[:, 0].max() + margin)
        axis.set_ylim(combined[:, 1].max() + margin, combined[:, 1].min() - margin)
    axis.set_title(
        f"КТ крысы 2024 г.: RTSTRUCT на срезе GTVp (z={target_z:.1f} мм)"
    )
    axis.set_xlabel("Столбец изображения")
    axis.set_ylabel("Строка изображения")
    axis.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_report(
    output: Path,
    raw: pd.DataFrame,
    paired: pd.DataFrame,
    agreement: Sequence[dict[str, object]],
    growth: Sequence[dict[str, object]],
    dicom_series: Sequence[dict[str, object]],
    nifti_rows: Sequence[dict[str, object]],
    matrix: Sequence[dict[str, object]],
    geometry: Sequence[dict[str, object]],
    linkage: dict[str, object],
) -> None:
    volume = next(row for row in agreement if row["metric"] == "V_log_ratio")
    volume_basic = next(
        row
        for row in agreement
        if row["scope"] == "all" and row["metric"] == "V"
    )
    tumor_geometry = [
        row
        for row in geometry
        if row["roi_name"] in {"GTVp", "GTVp1", "PTV_High", "CTV_High"}
    ]
    modality_counts = Counter(str(row["modality"]) for row in dicom_series)
    report = f"""# Итоговый анализ задачи 2

Дата воспроизводимого расчёта: 23.07.2026.

## Постановка

Задачу 2 закрывали как пилотный количественный анализ томографических данных
животных-опухоленосителей: инвентаризация КТ/МРТ, продольная морфометрия двух
крыс, техническая оценка согласованности двух наблюдателей и подготовка
контурированной индивидуальной геометрии для физического моделирования.

## Состав данных

- две крысы 2025 года, по 12 совпадающих календарных дат;
- 48 наборов осевых измерений (2 крысы × 12 дат × 2 наблюдателя);
- МРТ: {sum(row['mri_dicom'] == 'yes' for row in matrix)} из 24 сессий;
- КТ TB: {sum(row['ct_tb_nifti'] == 'yes' for row in matrix)} из 24 сессий;
- КТ UF: {sum(row['ct_uf_nifti'] == 'yes' for row in matrix)} из 24 сессий;
- DICOM-серий: {len(dicom_series)} ({dict(modality_counts)});
- NIfTI-файлов: {len(nifti_rows)};
- готовых продольных масок опухоли у крыс 2025 года: 0.

Файлы с кодом `LWH` принадлежат автору диссертации; файлы `1 крыса.xlsx` и
`2 крыса.xlsx` — студентке. Наблюдения объединяли по животному и календарной
дате, поскольку исходные текстовые номера суток расходились на одни сутки.

## Согласованность морфометрии

- парных временных точек: {len(paired)};
- средний объём автора: {paired['ellipsoid_volume_cm3_Автор'].mean():.3f} см³;
- средний объём студентки: {paired['ellipsoid_volume_cm3_Студентка'].mean():.3f} см³;
- среднее смещение автор − студентка: {volume_basic['mean_bias_author_minus_student']:.3f} см³;
- среднее абсолютное симметричное расхождение: {100 * volume['mean_absolute_difference']:.1f}%;
- медианное абсолютное симметричное расхождение: {100 * volume['median_absolute_symmetric_difference']:.1f}%;
- корреляция объёмов: {volume_basic['pearson_r']:.3f};
- коэффициент конкордации Лина: {volume_basic['lin_ccc']:.3f};
- геометрическое среднее отношения автор/студентка: {volume['geometric_mean_ratio_author_student']:.3f};
- описательные 95%-е пределы отношения: {volume['lower_descriptive_limit_ratio']:.3f}–{volume['upper_descriptive_limit_ratio']:.3f}.

Корреляция характеризует сходство формы траекторий, но не доказывает
взаимозаменяемость измерений. Из-за всего двух животных пределы согласия
считаются описательными; 24 повторные даты не трактуются как 24 независимых
животных.

## Продольный рост

Экспоненциальная аппроксимация логарифма объёма использована только для
описания индивидуальных траекторий:

| Крыса | Наблюдатель | Скорость, сут⁻¹ | Удвоение, сут | R² |
|---:|---|---:|---:|---:|
"""
    for row in growth:
        report += (
            f"| {row['animal']} | {row['observer']} | "
            f"{row['exponential_rate_per_day']:.3f} | "
            f"{row['doubling_time_days']:.2f} | "
            f"{row['r2_log_linear']:.3f} |\n"
        )
    report += f"""

## Происхождение m6_e

Комплект `m6_e` не относится к продольным крысам 2025 года. КТ `m6_e` и
whole-body КТ крысы 2024 года имеют по {linkage['original_slice_count']} среза
и идентичный объединённый SHA-256 пиксельных данных:
`{linkage['pixel_data_sha256_original']}`. Исходная КТ получена
24.06.2024; `m6_e` является её переэкспортированной контурированной версией
с новыми DICOM UID и собственной системой координат.

## Контуры

"""
    for row in tumor_geometry:
        report += (
            f"- {row['source']} / {row['roi_name']}: "
            f"{row['extent_x_mm']:.2f} × {row['extent_y_mm']:.2f} × "
            f"{row['extent_z_mm']:.2f} мм; контурный объём "
            f"{row['contour_volume_trapezoid_cm3']:.3f} см³.\n"
        )
    report += """

RTSTRUCT в `m6_e` непосредственно ссылается на КТ того же экспорта и пригоден
для передачи ROI в расчётный контур. Второй RTSTRUCT, лежащий рядом с исходной
КТ 2024 года, ссылается на отсутствующую переэкспортированную серию; его
геометрию можно описывать численно, но прямое наложение на исходную КТ без
восстановления регистрации не считается проверенным.

## Допустимый вывод

Задача 2 решена в пилотном морфометрическом и геометрическом объёме:
сформирован полный реестр доступных исследований, показана воспроизводимость
формы двух индивидуальных кривых роста, количественно описано расхождение
двух наблюдателей и установлено происхождение контурированного набора,
пригодного для физического моделирования.

Материал не позволяет заявлять популяционную радиомику, оценивать Dice/HD95
для продольных серий или доказывать прогностическую ценность томографических
признаков: для этого отсутствуют продольные маски и достаточное число
независимых животных.

По сведениям исследователя, животным томографического массива была перевита
саркома М-1, но лучевое воздействие им не проводили. Поэтому полученные
траектории характеризуют рост опухоли у необлучённых животных. DICOM- и
NIfTI-метаданные сами по себе не содержат сведений об облучении; статус
зафиксирован как атрибут происхождения экспериментального массива.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def run_analysis(
    results: Path = DEFAULT_RESULTS,
    morphometry_root: Path = DEFAULT_MORPHOMETRY_ROOT,
    rat_2024: Path = DEFAULT_RAT_2024,
    rats_2025: Path = DEFAULT_RATS_2025,
    m6_root: Path = DEFAULT_M6,
) -> dict[str, object]:
    results.mkdir(parents=True, exist_ok=True)
    figures = results / "figures"
    figures.mkdir(exist_ok=True)

    raw, manifest = load_morphometry(morphometry_root)
    paired = paired_morphometry(raw)
    agreement = agreement_summary(paired)
    growth = growth_summary(raw)

    roots = (
        DicomRoot("rat_2024", rat_2024, "rat_2024"),
        DicomRoot("rats_2025", rats_2025, "unknown"),
        DicomRoot("m6_e", m6_root, "rat_2024_reexport"),
    )
    dicom_series, rtstruct_rois, failures = inventory_dicom(roots)
    nifti_rows = inventory_nifti(rats_2025)
    matrix = longitudinal_matrix(dicom_series, nifti_rows)

    original_rtstruct = find_rtstruct(
        rat_2024 / "CT" / "2024-06-21_WB_DICOM"
    )
    m6_rtstruct = find_rtstruct(m6_root)
    geometry = rtstruct_geometry("rat_2024_unlinked_rtstruct", original_rtstruct)
    geometry.extend(rtstruct_geometry("rat_2024_m6_e", m6_rtstruct))
    linkage = verify_m6_linkage(rat_2024, m6_root)

    raw.to_csv(results / "morphometry_long.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(
        results / "morphometry_paired.csv", index=False, encoding="utf-8-sig"
    )
    write_csv(results / "morphometry_agreement_summary.csv", agreement)
    write_csv(results / "longitudinal_growth_summary.csv", growth)
    write_csv(results / "dicom_series_registry.csv", dicom_series)
    write_csv(results / "rtstruct_roi_registry.csv", rtstruct_rois)
    write_csv(results / "rtstruct_geometry.csv", geometry)
    write_csv(results / "nifti_registry.csv", nifti_rows)
    write_csv(results / "longitudinal_session_matrix_2025.csv", matrix)
    write_csv(results / "unrecognized_dicom_candidates.csv", failures)
    write_csv(results / "dataset_linkage.csv", [linkage])
    write_csv(results / "source_manifest.csv", manifest)

    render_longitudinal(raw, figures / "fig_2_1_longitudinal_morphometry.png")
    render_agreement(paired, figures / "fig_2_2_volume_agreement.png")
    render_availability(matrix, figures / "fig_2_3_imaging_availability.png")
    render_rtstruct_overlay(
        m6_root, m6_rtstruct, figures / "fig_2_4_rtstruct_overlay_m6_e.png"
    )

    validation = {
        "raw_morphometry_records": len(raw),
        "paired_timepoints": len(paired),
        "animals": int(raw["animal"].nunique()),
        "dates_per_animal": {
            str(animal): int(count)
            for animal, count in raw.groupby("animal")["date"].nunique().items()
        },
        "mri_longitudinal_sessions": sum(
            row["mri_dicom"] == "yes" for row in matrix
        ),
        "ct_tb_longitudinal_sessions": sum(
            row["ct_tb_nifti"] == "yes" for row in matrix
        ),
        "ct_uf_longitudinal_sessions": sum(
            row["ct_uf_nifti"] == "yes" for row in matrix
        ),
        "longitudinal_segmentations": sum(
            row["matched_segmentation"] == "yes" for row in matrix
        ),
        "dicom_series": len(dicom_series),
        "rtstruct_roi_count": len(rtstruct_rois),
        "nifti_file_count": len(nifti_rows),
        "m6_exact_pixel_match_to_2024_ct": bool(linkage["same_pixel_hash"]),
        "all_required_checks_passed": all(
            (
                len(raw) == 48,
                len(paired) == 24,
                raw["animal"].nunique() == 2,
                all(
                    count == 12
                    for count in raw.groupby("animal")["date"].nunique()
                ),
                bool(linkage["same_pixel_hash"]),
            )
        ),
    }
    (results / "validation_summary.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pydicom": pydicom.__version__,
        "nibabel": nib.__version__,
        "openpyxl": openpyxl.__version__,
    }
    (results / "software_versions.json").write_text(
        json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_report(
        results,
        raw,
        paired,
        agreement,
        growth,
        dicom_series,
        nifti_rows,
        matrix,
        geometry,
        linkage,
    )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible analysis for dissertation task 2."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--morphometry-root", type=Path, default=DEFAULT_MORPHOMETRY_ROOT
    )
    parser.add_argument("--rat-2024", type=Path, default=DEFAULT_RAT_2024)
    parser.add_argument("--rats-2025", type=Path, default=DEFAULT_RATS_2025)
    parser.add_argument("--m6-root", type=Path, default=DEFAULT_M6)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            run_analysis(
                results=arguments.results,
                morphometry_root=arguments.morphometry_root,
                rat_2024=arguments.rat_2024,
                rats_2025=arguments.rats_2025,
                m6_root=arguments.m6_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
