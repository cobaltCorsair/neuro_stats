"""Analyse the 100-MeV proton/PMMA screening scan in the rat phantom.

The beam travels along Geant4 ``-y``.  Profiles are therefore reported as
depth from the phantom's ``+y`` face.  This is a coarse-grid, 1000-history
screening analysis: it can localise the central maximum and reject unsuitable
PMMA thicknesses, but it is not used for final GTV DVH or LET estimates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import struct
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from stream_utils import _skip_field, _varint  # noqa: E402
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


DEFAULT_SCAN_ROOT = (
    TASK_DIR / "scoring_v2_proton_100MeV_L_to_R_pmma_scan_coarse_1k"
)
DEFAULT_OUTPUT = (
    TASK_DIR
    / "outputs"
    / "geant4_livermore_20260724"
    / "proton_100MeV_pmma_scan"
)

NX = 64
NY = 64
NZ = 100
PX_MM = 0.8
PY_MM = 0.8
PZ_MM = 0.398
HISTORIES = 1000
CENTRAL_RADIUS_MM = 5.0
# The beam enters from the Geant4 +y face and travels along -y.  These values
# are derived from the rasterised GTVp contour within the same 5-mm central
# cylinder used for the screening profile.
GTV_ENTRY_DEPTH_MM = 17.8
GTV_DISTAL_DEPTH_MM = 34.6
GTV_CENTRE_DEPTH_MM = 26.2
SMOOTHING_BINS = 3


def thickness_from_case(case_name: str) -> float:
    if case_name == "through_0mm":
        return 0.0
    match = re.fullmatch(r"pmma_(\d+(?:p\d+)?)mm", case_name)
    if match is None:
        raise ValueError(f"Unrecognised scan directory: {case_name}")
    return float(match.group(1).replace("p", "."))


def find_voxel_map(case_dir: Path) -> Path:
    matches = sorted(case_dir.glob("vox_proton_100MeV_L_to_R_*_0"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one completed voxel map in {case_dir}, found "
            f"{len(matches)}"
        )
    return matches[0]


def read_native_log(path: Path) -> str:
    """Read PowerShell-redirected native output on Windows."""
    payload = path.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16", errors="replace")
    if b"\x00" in payload[:200]:
        return payload.decode("utf-16-le", errors="replace")
    return payload.decode("utf-8", errors="replace")


def stream_y_profiles(path: Path) -> dict[str, np.ndarray]:
    """Accumulate energy and LET numerator by depth from the +y face."""
    whole_energy = np.zeros(NY, dtype=float)
    central_energy = np.zeros(NY, dtype=float)
    central_let_base = np.zeros(NY, dtype=float)

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

            entry_length = _varint(handle)
            entry_end = handle.tell() + entry_length
            voxel_id: int | None = None
            dep_energy = 0.0
            let_base = 0.0
            while handle.tell() < entry_end:
                entry_tag = _varint(handle)
                entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_field == 1 and entry_wire == 0:
                    voxel_id = _varint(handle)
                elif entry_field == 2 and entry_wire == 2:
                    value_length = _varint(handle)
                    value_end = handle.tell() + value_length
                    while handle.tell() < value_end:
                        value_tag = _varint(handle)
                        value_field, value_wire = value_tag >> 3, value_tag & 7
                        if value_field == 2 and value_wire == 1:
                            dep_energy = struct.unpack(
                                "<d", handle.read(8)
                            )[0]
                        elif value_field == 4 and value_wire == 1:
                            let_base = struct.unpack(
                                "<d", handle.read(8)
                            )[0]
                        else:
                            _skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    _skip_field(handle, entry_wire)
            handle.seek(entry_end)

            if voxel_id is None or dep_energy <= 0.0:
                continue
            ix = voxel_id % NX
            iy = (voxel_id // NX) % NY
            iz = voxel_id // (NX * NY)
            depth_index = NY - 1 - iy
            whole_energy[depth_index] += dep_energy

            x_mm = (ix + 0.5 - NX / 2.0) * PX_MM
            z_mm = (iz + 0.5 - NZ / 2.0) * PZ_MM
            if x_mm * x_mm + z_mm * z_mm <= CENTRAL_RADIUS_MM**2:
                central_energy[depth_index] += dep_energy
                central_let_base[depth_index] += let_base

    return {
        "whole_energy_keV": whole_energy,
        "central_energy_keV": central_energy,
        "central_let_base_keV2_um": central_let_base,
    }


def smooth(values: np.ndarray) -> np.ndarray:
    kernel = np.ones(SMOOTHING_BINS, dtype=float) / SMOOTHING_BINS
    return np.convolve(values, kernel, mode="same")


def analyse_case(case_dir: Path) -> tuple[dict[str, float | str | bool], dict[str, np.ndarray]]:
    case_name = case_dir.name
    thickness = thickness_from_case(case_name)
    voxel_map = find_voxel_map(case_dir)
    profiles = stream_y_profiles(voxel_map)
    central = profiles["central_energy_keV"]
    smoothed = smooth(central)
    depths = (np.arange(NY, dtype=float) + 0.5) * PY_MM

    has_central_signal = bool(np.any(smoothed > 0.0))
    if has_central_signal:
        peak_index = int(np.argmax(smoothed))
        peak_depth = float(depths[peak_index])
    else:
        peak_depth = np.nan
    gtv_depth_mask = (
        (depths >= GTV_ENTRY_DEPTH_MM)
        & (depths <= GTV_DISTAL_DEPTH_MM)
    )
    gtv_energy = float(central[gtv_depth_mask].sum())
    central_total = float(central.sum())
    gtv_let_base = float(
        profiles["central_let_base_keV2_um"][gtv_depth_mask].sum()
    )

    stdout_path = case_dir / "run_stdout.log"
    stdout = read_native_log(stdout_path)
    completed = f"Number of events processed : {HISTORIES}" in stdout
    used_serial = "G4TaskRunManager" not in stdout

    row: dict[str, float | str | bool] = {
        "case": case_name,
        "pmma_thickness_mm": thickness,
        "energy_MeV": 100.0,
        "histories": HISTORIES,
        "run_completed": completed,
        "serial_run_manager": used_serial,
        "central_peak_depth_mm": peak_depth,
        "peak_inside_gtv_depth_interval": bool(
            has_central_signal
            and GTV_ENTRY_DEPTH_MM <= peak_depth <= GTV_DISTAL_DEPTH_MM
        ),
        "distance_from_gtv_centre_mm": abs(
            peak_depth - GTV_CENTRE_DEPTH_MM
        ) if has_central_signal else np.nan,
        "whole_phantom_depenergy_MeV_per_primary": (
            float(profiles["whole_energy_keV"].sum())
            / 1000.0
            / HISTORIES
        ),
        "central_cylinder_depenergy_MeV_per_primary": (
            central_total / 1000.0 / HISTORIES
        ),
        "central_energy_fraction_in_gtv_depth_interval": (
            gtv_energy / central_total if central_total > 0.0 else np.nan
        ),
        "central_gtv_depth_LETd_keV_um": (
            gtv_let_base / gtv_energy if gtv_energy > 0.0 else np.nan
        ),
        "central_signal_present": has_central_signal,
        "voxel_map": str(voxel_map),
    }
    profiles["depth_mm"] = depths
    profiles["central_energy_smoothed_keV"] = smoothed
    return row, profiles


def write_profile(
    output_path: Path,
    profiles: dict[str, np.ndarray],
) -> None:
    columns = [
        "depth_mm",
        "whole_energy_keV",
        "central_energy_keV",
        "central_energy_smoothed_keV",
        "central_let_base_keV2_um",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for values in zip(*(profiles[column] for column in columns)):
            writer.writerow(values)


def plot_scan(
    rows: list[dict[str, float | str | bool]],
    profiles_by_case: dict[str, dict[str, np.ndarray]],
    output_stem: Path,
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 15,
                "axes.labelsize": 14,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 10,
            }
        )
        fig, (ax_profile, ax_position) = plt.subplots(
            2,
            1,
            figsize=(11.5, 10.0),
            constrained_layout=True,
        )
        colours = plt.cm.viridis(
            np.linspace(0.05, 0.92, len(rows))
        )

        for colour, row in zip(colours, rows):
            case = str(row["case"])
            profile = profiles_by_case[case]
            energy = profile["central_energy_smoothed_keV"]
            thickness = float(row["pmma_thickness_mm"])
            if (
                not np.any(energy > 0.0)
                or (
                    thickness > 0.0
                    and not bool(
                        row["adequate_central_signal_for_screening"]
                    )
                )
            ):
                continue
            normalised = energy / energy.max()
            label = (
                "без ПММА (прострел)"
                if thickness == 0.0
                else f"ПММА {thickness:g} мм"
            )
            ax_profile.plot(
                profile["depth_mm"],
                normalised,
                color=colour,
                linewidth=2.0,
                label=label,
            )

        for axis in (ax_profile, ax_position):
            axis.axvspan(
                GTV_ENTRY_DEPTH_MM,
                GTV_DISTAL_DEPTH_MM,
                color="#e31a1c",
                alpha=0.10,
                label="интервал GTVp на центральном луче",
            )
            axis.axvline(
                GTV_CENTRE_DEPTH_MM,
                color="#e31a1c",
                linestyle="--",
                linewidth=1.5,
            )

        ax_profile.set(
            xlim=(0.0, NY * PY_MM),
            ylim=(0.0, 1.08),
            xlabel="Глубина от поверхности опухолевой лапы, мм",
            ylabel="Нормированный энерговклад",
            title=(
                "Протоны 100 МэВ: влияние толщины ПММА "
                "на глубинный профиль"
            ),
        )
        ax_profile.legend(ncol=3, loc="upper right", frameon=True)
        ax_profile.grid(alpha=0.22)

        pmma_rows = [
            row for row in rows
            if float(row["pmma_thickness_mm"]) > 0.0
            and np.isfinite(float(row["central_peak_depth_mm"]))
        ]
        thickness = np.array(
            [float(row["pmma_thickness_mm"]) for row in pmma_rows]
        )
        peak_depth = np.array(
            [float(row["central_peak_depth_mm"]) for row in pmma_rows]
        )
        energy_per_primary = np.array(
            [
                float(row["central_cylinder_depenergy_MeV_per_primary"])
                for row in pmma_rows
            ]
        )
        ax_position.plot(
            peak_depth,
            thickness,
            color="#2166ac",
            marker="o",
            linewidth=2.0,
            label="положение максимума",
        )
        ax_position.set(
            xlim=(0.0, NY * PY_MM),
            xlabel="Глубина максимума от поверхности лапы, мм",
            ylabel="Толщина ПММА, мм",
            title="Скрининговая локализация максимума",
        )
        ax_position.grid(alpha=0.22)

        energy_axis = ax_position.twiny()
        energy_axis.plot(
            energy_per_primary,
            thickness,
            color="#d95f02",
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label="энерговклад в центральном цилиндре",
        )
        energy_axis.set_xlabel(
            "Энерговклад в центральном цилиндре, МэВ/первичный протон"
        )

        handles1, labels1 = ax_position.get_legend_handles_labels()
        handles2, labels2 = energy_axis.get_legend_handles_labels()
        ax_position.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="best",
            frameon=True,
        )
        fig.suptitle(
            "Грубый фантом 0,8 мм; 1000 историй на вариант; "
            "QGSP_INCLXX + Livermore",
            fontsize=15,
        )
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    case_dirs = [
        path
        for path in args.scan_root.iterdir()
        if path.is_dir()
        and (
            path.name == "through_0mm"
            or re.fullmatch(r"pmma_\d+(?:p\d+)?mm", path.name)
        )
    ]
    case_dirs.sort(key=lambda path: thickness_from_case(path.name))
    if not case_dirs:
        raise RuntimeError(f"No completed scan cases in {args.scan_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | bool]] = []
    profiles_by_case: dict[str, dict[str, np.ndarray]] = {}
    for case_dir in case_dirs:
        row, profiles = analyse_case(case_dir)
        if not bool(row["run_completed"]):
            raise RuntimeError(f"Incomplete Geant4 run: {case_dir}")
        if not bool(row["serial_run_manager"]):
            raise RuntimeError(
                f"Non-serial result is excluded from paired scan: {case_dir}"
            )
        rows.append(row)
        profiles_by_case[str(row["case"])] = profiles
        write_profile(
            args.output_dir / f"{row['case']}_depth_profile.csv",
            profiles,
        )

    max_pmma_energy = max(
        float(row["central_cylinder_depenergy_MeV_per_primary"])
        for row in rows
        if float(row["pmma_thickness_mm"]) > 0.0
    )
    for row in rows:
        thickness = float(row["pmma_thickness_mm"])
        adequate_signal = (
            thickness > 0.0
            and float(row["central_cylinder_depenergy_MeV_per_primary"])
            >= 0.10 * max_pmma_energy
        )
        row["adequate_central_signal_for_screening"] = adequate_signal
        row["screening_candidate"] = (
            adequate_signal
            and bool(row["peak_inside_gtv_depth_interval"])
        )

    candidates = [
        row for row in rows if bool(row["screening_candidate"])
    ]
    selected_case = ""
    if candidates:
        selected = min(
            candidates,
            key=lambda row: float(row["distance_from_gtv_centre_mm"]),
        )
        selected_case = str(selected["case"])
    for row in rows:
        row["closest_screening_candidate_to_gtv_centre"] = (
            str(row["case"]) == selected_case
        )

    summary_path = args.output_dir / "pmma_screening_summary.csv"
    with summary_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_scan(
        rows,
        profiles_by_case,
        args.output_dir / "proton_100MeV_pmma_screening",
    )
    print(f"summary={summary_path}")
    print(f"screening_candidate={selected_case or 'none'}")
    for row in rows:
        print(
            f"{row['case']}: peak={row['central_peak_depth_mm']:.2f} mm, "
            f"Ecentral={row['central_cylinder_depenergy_MeV_per_primary']:.4g} "
            "MeV/primary"
        )


if __name__ == "__main__":
    main()
