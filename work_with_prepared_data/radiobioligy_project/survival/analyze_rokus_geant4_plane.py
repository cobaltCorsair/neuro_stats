"""Compare a transported ROKUS-AM Geant4 field with the local MCC profile.

The Geant4 detector is a thin water scoring plane at the nominal 750-mm
measurement plane.  The comparison is shape-only: deposited energy is
normalised to the central plateau because the archived MCC files contain a
60-s irradiation, whereas a Geant4 primary has no source-activity scale.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from work_with_prepared_data.radiobioligy_project.survival.dose_reader import (
        read_dose_map,
    )
    from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
        MatplotlibConfigurator,
    )
except ModuleNotFoundError:
    from survival.dose_reader import read_dose_map
    from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
        MatplotlibConfigurator,
    )


def read_measured_profile(
    path: Path, field_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["field_nominal_mm"]) - field_mm) < 1.0e-8:
                rows.append(
                    (
                        float(row["crossplane_mm"]),
                        float(row["relative_dose"]),
                    )
                )
    if not rows:
        raise ValueError(f"No measured profile for {field_mm:g} mm in {path}")
    return (
        np.asarray([row[0] for row in rows], dtype=float),
        np.asarray([row[1] for row in rows], dtype=float),
    )


def crossing(distance: np.ndarray, profile: np.ndarray, level: float) -> float:
    peak_index = int(np.argmax(profile))
    candidates = np.flatnonzero(profile[peak_index:] <= level) + peak_index
    if candidates.size == 0:
        return float("nan")
    index = int(candidates[0])
    if index == 0:
        return float(distance[0])
    x0, x1 = distance[index - 1 : index + 1]
    y0, y1 = profile[index - 1 : index + 1]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def profile_metrics(axis: np.ndarray, profile: np.ndarray) -> dict[str, float]:
    positive = axis >= 0.0
    distance = axis[positive]
    right = profile[positive]
    left = np.interp(-distance, axis, profile)
    symmetric = 0.5 * (left + right)
    x80 = crossing(distance, symmetric, 0.8)
    x50 = crossing(distance, symmetric, 0.5)
    x20 = crossing(distance, symmetric, 0.2)
    return {
        "full_width_50_percent_mm": 2.0 * x50,
        "penumbra_80_20_mm": x20 - x80,
    }


def load_geant4_profile(
    dose_path: Path,
    geometry_path: Path,
    *,
    central_band_half_width_mm: float,
    smoothing_sigma_voxels: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    dose_map = read_dose_map(dose_path, geometry_path)
    nx, ny, nz = dose_map.grid_shape
    dx, _, dz = dose_map.voxel_size_mm
    energy = np.zeros(nx * ny * nz, dtype=float)
    for voxel_id, voxel in dose_map.voxels.items():
        if 0 <= voxel_id < energy.size:
            energy[voxel_id] = voxel.dep_energy_mev
    energy_zyx = energy.reshape((nz, ny, nx), order="C")
    x_mm = (np.arange(nx, dtype=float) + 0.5 - 0.5 * nx) * dx
    z_mm = (np.arange(nz, dtype=float) + 0.5 - 0.5 * nz) * dz
    central_z = np.abs(z_mm) <= central_band_half_width_mm
    if not np.any(central_z):
        central_z[np.argmin(np.abs(z_mm))] = True
    profile = np.sum(energy_zyx[central_z, :, :], axis=(0, 1))
    profile = gaussian_filter1d(profile, sigma=smoothing_sigma_voxels)
    plateau = np.abs(x_mm) <= 10.0
    normalisation = float(np.mean(profile[plateau]))
    if normalisation <= 0.0:
        raise ValueError("The Geant4 central plateau contains no deposited energy.")
    profile /= normalisation
    metadata = {
        "grid": [nx, ny, nz],
        "voxel_mm": [dx, dose_map.voxel_size_mm[1], dz],
        "nonzero_voxels": int(np.count_nonzero(energy)),
        "total_deposited_energy_MeV": float(np.sum(energy)),
        "central_band_half_width_mm": central_band_half_width_mm,
        "smoothing_sigma_voxels": smoothing_sigma_voxels,
        "smoothing_sigma_mm": smoothing_sigma_voxels * dx,
        "scoring_depth_mm": ny * dose_map.voxel_size_mm[1],
    }
    return x_mm, profile, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose-map", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--measured-profiles", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--field-mm", type=float, default=100.0)
    parser.add_argument("--central-band-half-width-mm", type=float, default=10.0)
    parser.add_argument("--smoothing-sigma-voxels", type=float, default=1.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measured_x, measured = read_measured_profile(
        args.measured_profiles, args.field_mm
    )
    calculated_x, calculated, metadata = load_geant4_profile(
        args.dose_map,
        args.geometry,
        central_band_half_width_mm=args.central_band_half_width_mm,
        smoothing_sigma_voxels=args.smoothing_sigma_voxels,
    )
    measured_metrics = profile_metrics(measured_x, measured)
    calculated_metrics = profile_metrics(calculated_x, calculated)
    calculated_on_measured = np.interp(
        measured_x, calculated_x, calculated, left=0.0, right=0.0
    )
    comparison = measured >= 0.02
    rmse = float(
        np.sqrt(
            np.mean((calculated_on_measured[comparison] - measured[comparison]) ** 2)
        )
    )
    result = {
        **metadata,
        "field_nominal_mm": args.field_mm,
        "measured_width50_mm": measured_metrics["full_width_50_percent_mm"],
        "geant4_width50_mm": calculated_metrics["full_width_50_percent_mm"],
        "width50_difference_mm": (
            calculated_metrics["full_width_50_percent_mm"]
            - measured_metrics["full_width_50_percent_mm"]
        ),
        "measured_penumbra80_20_mm": measured_metrics["penumbra_80_20_mm"],
        "geant4_penumbra80_20_mm": calculated_metrics["penumbra_80_20_mm"],
        "penumbra_difference_mm": (
            calculated_metrics["penumbra_80_20_mm"]
            - measured_metrics["penumbra_80_20_mm"]
        ),
        "profile_RMSE_for_measured_dose_ge_2pct": rmse,
        "comparison_scope": "relative field shape; not absolute-dose calibration",
    }
    (args.output_dir / "rokus_geant4_plane_vs_mcc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "rokus_geant4_plane_vs_mcc.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    with (args.output_dir / "rokus_geant4_plane_profile.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["crossplane_mm", "relative_deposited_energy"])
        writer.writerows(zip(calculated_x, calculated))

    MatplotlibConfigurator().apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    scoring_depth_mm = float(result["scoring_depth_mm"])
    scoring_description = (
        "тонкая плоскость регистрации"
        if scoring_depth_mm <= 1.01
        else f"энерговклад за {scoring_depth_mm:g} мм воды"
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    ax.plot(
        measured_x,
        measured,
        color="#1b9e77",
        linewidth=2.2,
        label="измерение OCTAVIUS",
    )
    ax.plot(
        calculated_x,
        calculated,
        color="#d95f02",
        linewidth=2.0,
        label="Geant4: источник, капсула и шторки",
    )
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0)
    ax.set(
        xlim=(-90.0, 90.0),
        ylim=(-0.03, 1.12),
        xlabel="Crossplane, мм",
        ylabel="Относительная доза / энерговклад",
        title=(
            "РОКУС-АМ: поле 10×10 см (плоскость 750 мм)\n"
            f"Geant4: {scoring_description}\n"
            f"ΔW50={result['width50_difference_mm']:+.1f} мм; "
            f"ΔP={result['penumbra_difference_mm']:+.1f} мм"
        ),
    )
    ax.legend(loc="lower center")
    ax.grid(alpha=0.25)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"rokus_geant4_plane_vs_mcc.{extension}",
            dpi=220 if extension == "png" else None,
        )
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
