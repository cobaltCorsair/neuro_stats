"""Parse and summarise ROKUS-AM OCTAVIUS MCC plane-dose measurements.

The MCC headers retain a generic 100 x 100 mm field value in every file.
The delivered field is therefore inferred from the file name and the
Russian COMMENT field (50, 100, 150, 180 or 220 mm at SSD 750 mm).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


@dataclass
class PlaneMeasurement:
    path: Path
    field_mm: float
    comment: str
    ssd_mm: float
    measurement_date: str
    x_mm: np.ndarray
    y_mm: np.ndarray
    dose_gy: np.ndarray


def parse_mcc(path: Path) -> PlaneMeasurement:
    text = path.read_bytes().decode("cp1251")
    field_match = re.search(r"F(\d{2})x\d{2}", path.name, re.IGNORECASE)
    if field_match is None:
        raise ValueError(f"Cannot infer field size from {path.name}")
    field_mm = 10.0 * float(field_match.group(1))

    current: dict[str, str] = {}
    data_mode = False
    points: list[tuple[float, float, float]] = []
    first_comment = ""
    first_ssd = np.nan
    first_date = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("BEGIN_SCAN "):
            current = {}
            data_mode = False
            continue
        if line == "BEGIN_DATA":
            data_mode = True
            if not first_comment:
                first_comment = current.get("COMMENT", "")
                first_ssd = float(current.get("SSD", "nan"))
                first_date = current.get("MEAS_DATE", "")
            continue
        if line == "END_DATA":
            data_mode = False
            continue
        if data_mode:
            fields = line.split()
            if len(fields) >= 2:
                points.append(
                    (
                        float(fields[0]),
                        float(current["SCAN_OFFAXIS_INPLANE"]),
                        float(fields[1]),
                    )
                )
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()

    array = np.asarray(points, dtype=float)
    if array.shape[0] != 1405:
        raise RuntimeError(
            f"{path.name}: expected 1405 OCTAVIUS points, "
            f"found {array.shape[0]}"
        )
    return PlaneMeasurement(
        path=path,
        field_mm=field_mm,
        comment=first_comment,
        ssd_mm=first_ssd,
        measurement_date=first_date,
        x_mm=array[:, 0],
        y_mm=array[:, 1],
        dose_gy=array[:, 2],
    )


def crossing(distance: np.ndarray, profile: np.ndarray, level: float) -> float:
    peak_index = int(np.argmax(profile))
    indices = np.flatnonzero(profile[peak_index:] <= level) + peak_index
    if indices.size == 0:
        return np.nan
    index = int(indices[0])
    if index == 0:
        return float(distance[index])
    x0, x1 = distance[index - 1 : index + 1]
    y0, y1 = profile[index - 1 : index + 1]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def axis_profile(
    measurement: PlaneMeasurement,
    axis_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.column_stack((measurement.x_mm, measurement.y_mm))
    profile = griddata(
        points,
        measurement.dose_gy,
        (axis_mm, np.zeros_like(axis_mm)),
        method="linear",
    )
    profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
    profile = gaussian_filter1d(profile, sigma=4.0)
    central = float(np.mean(profile[np.abs(axis_mm) <= 5.0]))
    return profile, profile / central, central


def profile_metrics(
    measurement: PlaneMeasurement,
    axis_mm: np.ndarray,
    relative: np.ndarray,
    central_dose: float,
) -> dict[str, object]:
    positive = axis_mm >= 0.0
    distance = axis_mm[positive]
    right = relative[positive]
    left = relative[::-1][positive]
    symmetric = 0.5 * (left + right)
    x80 = crossing(distance, symmetric, 0.8)
    x50 = crossing(distance, symmetric, 0.5)
    x20 = crossing(distance, symmetric, 0.2)
    central_region = distance <= 0.8 * x50
    flat_values = symmetric[central_region]
    flatness = (
        100.0
        * (float(np.max(flat_values)) - float(np.min(flat_values)))
        / (float(np.max(flat_values)) + float(np.min(flat_values)))
    )
    # Restrict symmetry to the central 80% of the measured field.  Including
    # the steep penumbra would turn a sub-millimetre edge displacement into a
    # misleadingly large dose asymmetry.
    symmetry_region = distance <= 0.8 * x50
    symmetry = 100.0 * float(
        np.max(np.abs(left[symmetry_region] - right[symmetry_region]))
    )
    return {
        "file": measurement.path.name,
        "field_nominal_mm": measurement.field_mm,
        "ssd_mm": measurement.ssd_mm,
        "measurement_date": measurement.measurement_date,
        "central_dose_Gy_60s": central_dose,
        "x80_mm": x80,
        "x50_mm": x50,
        "x20_mm": x20,
        "full_width_50_percent_mm": 2.0 * x50,
        "field_width_error_mm": 2.0 * x50 - measurement.field_mm,
        "penumbra_80_20_mm": x20 - x80,
        "central_80pct_flatness_percent": flatness,
        "symmetry_max_difference_percent_of_axis": symmetry,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = [
        parse_mcc(path)
        for path in sorted(input_dir.glob("F*x*_to75_ssd75.mcc"))
    ]
    if len(measurements) != 5:
        raise RuntimeError(
            f"Expected five MCC field files in {input_dir}, "
            f"found {len(measurements)}"
        )

    axis_mm = np.linspace(-130.0, 130.0, 1041)
    profiles: dict[float, np.ndarray] = {}
    metrics: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for measurement in measurements:
        _, relative, central = axis_profile(measurement, axis_mm)
        profiles[measurement.field_mm] = relative
        metrics.append(
            profile_metrics(measurement, axis_mm, relative, central)
        )
        for coordinate, value in zip(axis_mm, relative):
            long_rows.append(
                {
                    "field_nominal_mm": measurement.field_mm,
                    "crossplane_mm": coordinate,
                    "relative_dose": value,
                }
            )

    reference_output = next(
        float(row["central_dose_Gy_60s"])
        for row in metrics
        if float(row["field_nominal_mm"]) == 100.0
    )
    for row in metrics:
        row["output_factor_relative_to_100mm"] = (
            float(row["central_dose_Gy_60s"]) / reference_output
        )

    write_csv(output_dir / "rokus_mcc_field_metrics.csv", metrics)
    write_csv(output_dir / "rokus_mcc_crossplane_profiles.csv", long_rows)
    raw_rows: list[dict[str, object]] = []
    for measurement in measurements:
        for x, y, dose in zip(
            measurement.x_mm,
            measurement.y_mm,
            measurement.dose_gy,
        ):
            raw_rows.append(
                {
                    "source_file": measurement.path.name,
                    "field_nominal_mm": measurement.field_mm,
                    "crossplane_mm": x,
                    "inplane_mm": y,
                    "dose_Gy_60s": dose,
                }
            )
    write_csv(output_dir / "rokus_mcc_raw_points.csv", raw_rows)

    MatplotlibConfigurator().apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(measurements)))

    ax = axes[0, 0]
    for measurement, color in zip(measurements, colors):
        row = next(
            item
            for item in metrics
            if float(item["field_nominal_mm"]) == measurement.field_mm
        )
        ax.plot(
            axis_mm,
            profiles[measurement.field_mm],
            color=color,
            linewidth=1.9,
            label=(
                f"{measurement.field_mm / 10:.0f}×"
                f"{measurement.field_mm / 10:.0f} см; "
                f"W50={float(row['full_width_50_percent_mm']):.1f} мм"
            ),
        )
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0)
    ax.set(
        title="Измеренные поперечные профили при SSD 750 мм",
        xlabel="Поперечная координата, мм",
        ylabel="Доза / центральная доза",
        xlim=(-130.0, 130.0),
        ylim=(-0.02, 1.10),
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)

    ax = axes[0, 1]
    field_values = np.asarray(
        [float(row["field_nominal_mm"]) for row in metrics]
    )
    width_values = np.asarray(
        [float(row["full_width_50_percent_mm"]) for row in metrics]
    )
    ax.plot(
        field_values,
        width_values,
        "o-",
        color="#1b9e77",
        linewidth=2.0,
        label="измеренная ширина по 50%",
    )
    ax.plot(
        [0.0, 230.0],
        [0.0, 230.0],
        "--",
        color="#555555",
        label="номинал = измерение",
    )
    ax.set(
        title="Воспроизведение номинального размера поля",
        xlabel="Номинальный размер, мм",
        ylabel="Полная ширина по 50%, мм",
        xlim=(35.0, 230.0),
        ylim=(35.0, 230.0),
    )
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    penumbra = np.asarray(
        [float(row["penumbra_80_20_mm"]) for row in metrics]
    )
    ax.plot(
        field_values,
        penumbra,
        "o-",
        color="#d95f02",
        linewidth=2.0,
    )
    for x, y in zip(field_values, penumbra):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        title="Измеренная полутень 80–20%",
        xlabel="Номинальный размер поля, мм",
        ylabel="Полутень, мм",
    )
    ax.grid(alpha=0.25)

    measurement_100 = next(
        item for item in measurements if item.field_mm == 100.0
    )
    grid_axis = np.linspace(-130.0, 130.0, 261)
    gx, gy = np.meshgrid(grid_axis, grid_axis)
    dose_grid = griddata(
        np.column_stack(
            (measurement_100.x_mm, measurement_100.y_mm)
        ),
        measurement_100.dose_gy,
        (gx, gy),
        method="linear",
    )
    central_100 = next(
        float(row["central_dose_Gy_60s"])
        for row in metrics
        if float(row["field_nominal_mm"]) == 100.0
    )
    ax = axes[1, 1]
    image = ax.imshow(
        dose_grid / central_100,
        extent=(-130.0, 130.0, -130.0, 130.0),
        origin="lower",
        cmap="turbo",
        vmin=0.0,
        vmax=1.05,
        aspect="equal",
    )
    ax.contour(
        gx,
        gy,
        dose_grid / central_100,
        levels=(0.2, 0.5, 0.8),
        colors=("white", "black", "white"),
        linewidths=(0.8, 1.3, 0.8),
    )
    ax.set(
        title="Измеренная плоскость поля 100×100 мм",
        xlabel="Crossplane, мм",
        ylabel="Inplane, мм",
    )
    fig.colorbar(image, ax=ax, label="Доза / центральная доза")

    fig.suptitle(
        "РОКУС-АМ: опорная дозиметрия OCTAVIUS 1500 XDR, 24.10.2023",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"rokus_mcc_dosimetry.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    summary = {
        "status": "local_machine_dosimetry_reference",
        "source": "PTW OCTAVIUS 1500 XDR MCC export",
        "measurement_date": measurements[0].measurement_date,
        "ssd_mm": measurements[0].ssd_mm,
        "fields_mm": [item.field_mm for item in measurements],
        "points_per_field": 1405,
        "field_size_source": (
            "file name and COMMENT; FIELD_* header is stale at 100 mm"
        ),
        "metrics": metrics,
    }
    (output_dir / "rokus_mcc_dosimetry.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = analyse(args.input_dir, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
