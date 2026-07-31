"""Analyse Geant4 validation runs for the L->R proton energy window."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import struct

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_geant4_proton_axis_audit import (
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    containing_segment,
)
from plot_geant4_proton_beam_geometry import (
    LOW_PX,
    LOW_PY,
    LOW_PZ,
    geometry_and_contours,
    load_ct_crop,
    rasterise_gtv,
)
from stream_utils import _skip_field, _varint
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


TASK_ROOT = Path(r"C:\dev\dissertation\task4_5")
DEFAULT_OUTPUT = TASK_ROOT / "outputs" / "geant4_livermore_20260724" / (
    "proton_initial_energy_window"
)
DEFAULT_MAPS = {
    15.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_validation_250"
    / "vox_proton_L_to_R_energy_window_validation_250_0",
    40.0: TASK_ROOT
    / "scoring_v2_proton_40MeV_L_to_R_coarse_10k"
    / "vox_proton_40MeV_L_to_R_coarse_10k_0",
    20.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_refinement_1k"
    / "20MeV"
    / "vox_proton_20MeV_L_to_R_1k_0",
    21.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_final_brackets_1k"
    / "21MeV"
    / "vox_proton_21MeV_L_to_R_1k_0",
    46.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_separate_250"
    / "46MeV"
    / "vox_proton_46MeV_L_to_R_250_0",
    51.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_separate_250"
    / "51MeV"
    / "vox_proton_51MeV_L_to_R_250_0",
    52.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_final_brackets_1k"
    / "52MeV"
    / "vox_proton_52MeV_L_to_R_1k_0",
    53.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_final_brackets_1k"
    / "53MeV"
    / "vox_proton_53MeV_L_to_R_1k_0",
    50.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_refinement_1k"
    / "50MeV"
    / "vox_proton_50MeV_L_to_R_1k_0",
    54.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_refinement_1k"
    / "54MeV"
    / "vox_proton_54MeV_L_to_R_1k_0",
    74.0: TASK_ROOT
    / "scoring_v2_proton_L_to_R_energy_window_separate_250"
    / "74MeV"
    / "vox_proton_74MeV_L_to_R_250_0",
}


def energy_profile_y(
    path: Path,
    centre_x: float,
    centre_z: float,
    core_radius_mm: float,
    nx: int = 64,
    ny: int = 64,
    nz: int = 100,
    sx_mm: float = 0.8,
    sy_mm: float = 0.8,
    sz_mm: float = 39.8 / 100.0,
) -> np.ndarray:
    """Sum deposited energy by Geant4 y in a central x-z cylinder."""
    profile = np.zeros(ny, dtype=float)
    with path.open("rb") as handle:
        while True:
            try:
                tag = _varint(handle)
            except EOFError:
                break
            field, wire = tag >> 3, tag & 7
            if field != 1 or wire != 2:
                _skip_field(handle, wire)
                continue
            entry_len = _varint(handle)
            entry_end = handle.tell() + entry_len
            voxel_id = None
            dep_energy = 0.0
            while handle.tell() < entry_end:
                entry_tag = _varint(handle)
                entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_field == 1 and entry_wire == 0:
                    voxel_id = _varint(handle)
                elif entry_field == 2 and entry_wire == 2:
                    value_len = _varint(handle)
                    value_end = handle.tell() + value_len
                    while handle.tell() < value_end:
                        value_tag = _varint(handle)
                        value_field, value_wire = value_tag >> 3, value_tag & 7
                        if value_field == 2 and value_wire == 1:
                            dep_energy = struct.unpack("<d", handle.read(8))[0]
                        else:
                            _skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    _skip_field(handle, entry_wire)
            handle.seek(entry_end)
            if voxel_id is None or dep_energy <= 0.0:
                continue
            gx = voxel_id % nx
            gy = (voxel_id // nx) % ny
            gz = voxel_id // (nx * ny)
            if gz < 0 or gz >= nz:
                continue
            x_mm = (gx + 0.5) * sx_mm
            z_mm = (gz + 0.5) * sz_mm
            if (
                (x_mm - centre_x) ** 2 + (z_mm - centre_z) ** 2
                <= core_radius_mm**2
            ):
                profile[gy] += dep_energy
    return profile


def central_boundaries(
    ct_volume: np.ndarray,
    gtv_mask: np.ndarray,
    centroid: np.ndarray,
) -> dict[str, float | int]:
    iz = int(round(float(centroid[2]) / LOW_PZ - 0.5))
    irow = int(round(float(centroid[1]) / LOW_PY - 0.5))
    hu_line = ct_volume[iz, irow]
    gtv_line = gtv_mask[iz, irow]
    gtv_indices = np.flatnonzero(gtv_line)
    body_lo, body_hi = containing_segment(hu_line > -500.0, int(round(
        float(centroid[0]) / LOW_PX - 0.5
    )))
    gtv_lo = int(gtv_indices.min())
    gtv_hi = int(gtv_indices.max())
    entry_face_mm = (body_hi + 0.5) * LOW_PX
    return {
        "body_entry_column": body_hi,
        "body_exit_column": body_lo,
        "gtv_proximal_column": gtv_hi,
        "gtv_distal_column": gtv_lo,
        "body_entry_coordinate_mm": entry_face_mm,
        "gtv_proximal_depth_mm": (body_hi - gtv_hi) * LOW_PX,
        "gtv_distal_depth_mm": (body_hi - gtv_lo + 1) * LOW_PX,
        "paw_distal_depth_mm": (body_hi - body_lo + 1) * LOW_PX,
    }


def analyse_profiles(
    maps: dict[float, Path],
    centroid: np.ndarray,
    boundaries: dict[str, float | int],
    core_radius_mm: float,
) -> tuple[list[dict[str, float | str]], dict[float, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, float | str]] = []
    plotted: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    ny = 64
    sy_mm = 0.8
    entry_face = float(boundaries["body_entry_coordinate_mm"])
    body_depth = float(boundaries["paw_distal_depth_mm"])
    y_centres = (np.arange(ny, dtype=float) + 0.5) * sy_mm
    depths = entry_face - y_centres
    inside = (depths >= 0.0) & (depths <= body_depth)
    order = np.argsort(depths[inside])
    ordered_depth = depths[inside][order]

    for energy, path in sorted(maps.items()):
        if not path.exists():
            rows.append(
                {
                    "incident_energy_MeV": energy,
                    "status": "missing",
                    "map": str(path),
                }
            )
            continue
        profile = energy_profile_y(
            path,
            centre_x=float(centroid[1]),
            centre_z=float(centroid[2]),
            core_radius_mm=core_radius_mm,
        )
        ordered_profile = profile[inside][order]
        smooth = np.convolve(
            ordered_profile,
            np.ones(3, dtype=float) / 3.0,
            mode="same",
        )
        peak_index = int(np.argmax(smooth))
        peak_depth = float(ordered_depth[peak_index])
        peak_location = "inside_paw"
        if peak_index <= 1:
            peak_location = "at_entry_edge"
        elif peak_index >= len(smooth) - 2:
            peak_location = "at_exit_edge_or_beyond"
        rows.append(
            {
                "incident_energy_MeV": energy,
                "status": "ok",
                "map": str(path),
                "core_radius_mm": core_radius_mm,
                "total_depEnergy_in_paw_core_keV": float(
                    ordered_profile.sum()
                ),
                "smoothed_peak_depth_mm": peak_depth,
                "peak_location": peak_location,
                "gtv_proximal_depth_mm": float(
                    boundaries["gtv_proximal_depth_mm"]
                ),
                "gtv_distal_depth_mm": float(
                    boundaries["gtv_distal_depth_mm"]
                ),
                "paw_distal_depth_mm": body_depth,
            }
        )
        if np.max(smooth) > 0.0:
            plotted[energy] = (ordered_depth, smooth / np.max(smooth))
    return rows, plotted


def plot_profiles(
    profiles: dict[float, tuple[np.ndarray, np.ndarray]],
    boundaries: dict[str, float | int],
    output_stem: Path,
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
        colours = {
            15.0: "#0072b2",
            20.0: "#56b4e9",
            21.0: "#004c6d",
            40.0: "#009e73",
            46.0: "#d55e00",
            50.0: "#e69f00",
            51.0: "#cc79a7",
            52.0: "#a05195",
            53.0: "#665191",
            54.0: "#7b3294",
            74.0: "#4d4d4d",
        }
        for energy, (depth, profile) in profiles.items():
            ax.plot(
                depth,
                profile,
                linewidth=1.8,
                color=colours.get(energy),
                label=f"{energy:g} МэВ",
            )
        proximal = float(boundaries["gtv_proximal_depth_mm"])
        distal = float(boundaries["gtv_distal_depth_mm"])
        paw_distal = float(boundaries["paw_distal_depth_mm"])
        ax.axvspan(proximal, distal, color="#e41a1c", alpha=0.09)
        ax.axvline(proximal, color="#e41a1c", linestyle="--", linewidth=1.2)
        ax.axvline(distal, color="#e41a1c", linestyle="--", linewidth=1.2)
        ax.axvline(paw_distal, color="black", linestyle=":", linewidth=1.2)
        ax.set(
            xlim=(0.0, paw_distal + 0.5),
            ylim=(0.0, 1.08),
            xlabel="Глубина от входной поверхности лапы, мм",
            ylabel=(
                "Энерговклад / собственный максимум\n"
                "для данной начальной энергии"
            ),
            title="Проверка энергетического окна в Geant4",
        )
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_size(14)
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(
                output_stem.with_suffix(suffix),
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def interpolate_boundaries(
    rows: list[dict[str, float | str]],
    boundaries: dict[str, float | int],
) -> list[dict[str, float | str]]:
    """Interpolate incident energy from monotonic stopping-peak positions."""
    by_energy = sorted(
        (
            float(row["incident_energy_MeV"]),
            float(row["smoothed_peak_depth_mm"]),
        )
        for row in rows
        if row.get("status") == "ok"
        and float(row["incident_energy_MeV"]) <= 53.0
    )
    # A peak that ceases to move distally indicates that the Bragg maximum is
    # no longer resolved inside the central paw.  Retain only the strictly
    # increasing stopping-peak envelope for interpolation.
    calibration: list[tuple[float, float]] = []
    last_depth = -np.inf
    for energy, depth in by_energy:
        if depth > last_depth + 1e-9:
            calibration.append((energy, depth))
            last_depth = depth
    energies = np.asarray([item[0] for item in calibration], dtype=float)
    depths = np.asarray([item[1] for item in calibration], dtype=float)
    result: list[dict[str, float | str]] = []
    for label, key in (
        ("central_gtv_proximal", "gtv_proximal_depth_mm"),
        ("central_gtv_distal", "gtv_distal_depth_mm"),
        ("central_paw_distal", "paw_distal_depth_mm"),
    ):
        target_depth = float(boundaries[key])
        if target_depth < depths[0] or target_depth > depths[-1]:
            energy = float("nan")
            status = "outside_calibration"
        else:
            energy = float(np.interp(target_depth, depths, energies))
            status = "interpolated"
        result.append(
            {
                "boundary": label,
                "target_depth_mm": target_depth,
                "incident_energy_MeV": energy,
                "status": status,
                "calibration_min_energy_MeV": float(np.min(energies)),
                "calibration_max_energy_MeV": float(np.max(energies)),
            }
        )
    return result


def plot_calibration(
    rows: list[dict[str, float | str]],
    boundary_rows: list[dict[str, float | str]],
    boundaries: dict[str, float | int],
    output_stem: Path,
) -> None:
    stopping = sorted(
        (
            float(row["incident_energy_MeV"]),
            float(row["smoothed_peak_depth_mm"]),
        )
        for row in rows
        if row.get("status") == "ok"
        and float(row["incident_energy_MeV"]) <= 51.0
    )
    unresolved = sorted(
        (
            float(row["incident_energy_MeV"]),
            float(row["smoothed_peak_depth_mm"]),
        )
        for row in rows
        if row.get("status") == "ok"
        and 51.0 < float(row["incident_energy_MeV"]) <= 54.0
    )
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
        energy = np.asarray([item[0] for item in stopping])
        depth = np.asarray([item[1] for item in stopping])
        ax.plot(
            energy,
            depth,
            color="#0072b2",
            marker="o",
            linewidth=1.8,
            markersize=5,
            label="разрешённый максимум энерговклада",
        )
        if unresolved:
            ax.scatter(
                [item[0] for item in unresolved],
                [item[1] for item in unresolved],
                facecolors="none",
                edgecolors="#7b3294",
                marker="s",
                s=44,
                linewidths=1.3,
                label="максимум упирается в дистальный край",
            )
        proximal = float(boundaries["gtv_proximal_depth_mm"])
        distal = float(boundaries["gtv_distal_depth_mm"])
        paw_distal = float(boundaries["paw_distal_depth_mm"])
        ax.axhspan(
            proximal,
            distal,
            color="#e41a1c",
            alpha=0.10,
            label="GTVp по центральному лучу",
        )
        ax.axhline(paw_distal, color="black", linestyle=":", linewidth=1.2)
        for boundary in boundary_rows[:2]:
            if boundary["status"] != "interpolated":
                continue
            x = float(boundary["incident_energy_MeV"])
            y = float(boundary["target_depth_mm"])
            ax.scatter(
                [x],
                [y],
                color="#c00000",
                edgecolor="white",
                linewidth=0.8,
                s=62,
                zorder=6,
            )
            ax.annotate(
                f"{x:.1f} МэВ",
                xy=(x, y),
                xytext=(7, 8),
                textcoords="offset points",
                fontsize=9,
                color="#8b0000",
            )
        ax.set(
            xlabel="Начальная энергия протонов, МэВ",
            ylabel="Глубина максимума энерговклада, мм",
            title="Калибровка энергетического окна по Geant4",
            xlim=(13.5, 55.5),
            ylim=(0.0, paw_distal + 1.0),
        )
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_size(14)
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=8.8)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(
                output_stem.with_suffix(suffix),
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--core-radius-mm", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    z_values, by_z, crop, contours, centroid = geometry_and_contours(
        args.ct_dir,
        args.rtstruct,
    )
    ct_volume = load_ct_crop(z_values, by_z, crop)
    gtv_mask = rasterise_gtv(contours)
    boundaries = central_boundaries(ct_volume, gtv_mask, centroid)
    rows, profiles = analyse_profiles(
        DEFAULT_MAPS,
        centroid,
        boundaries,
        args.core_radius_mm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "geant4_energy_window_validation.csv", rows)
    boundary_rows = interpolate_boundaries(rows, boundaries)
    write_csv(
        args.output_dir / "geant4_interpolated_energy_boundaries.csv",
        boundary_rows,
    )
    plot_profiles(
        profiles,
        boundaries,
        args.output_dir / "geant4_energy_window_validation",
    )
    plot_calibration(
        rows,
        boundary_rows,
        boundaries,
        args.output_dir / "geant4_energy_window_calibration",
    )
    for row in rows:
        if row["status"] == "ok":
            print(
                f'{float(row["incident_energy_MeV"]):g} MeV: '
                f'peak={float(row["smoothed_peak_depth_mm"]):.2f} mm; '
                f'{row["peak_location"]}'
            )
        else:
            print(f'{float(row["incident_energy_MeV"]):g} MeV: missing')
    for row in boundary_rows:
        if row["status"] == "interpolated":
            print(
                f'{row["boundary"]}: '
                f'{float(row["incident_energy_MeV"]):.2f} MeV'
            )
        else:
            print(f'{row["boundary"]}: outside calibration')


if __name__ == "__main__":
    main()
