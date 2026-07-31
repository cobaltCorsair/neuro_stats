"""Analyse an internal Co-60 transport benchmark in homogeneous water.

The calculation uses an equivalent parallel circular field at the phantom
surface.  Therefore the resulting relative depth-dose curve is a transport
and scoring check, not a clinical PDD of a particular cobalt installation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


NX, NY, NZ = 100, 80, 80
DX_MM = DY_MM = DZ_MM = 1.0
CENTRAL_RADIUS_MM = 5.0
PROFILE_DEPTHS_MM = (5.0, 20.0, 40.0, 60.0)


def read_varint(handle) -> int:
    value = 0
    shift = 0
    while True:
        raw = handle.read(1)
        if not raw:
            raise EOFError
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value
        shift += 7
        if shift > 63:
            raise ValueError("Invalid protobuf varint")


def skip_field(handle, wire: int) -> None:
    if wire == 0:
        read_varint(handle)
    elif wire == 1:
        handle.seek(8, 1)
    elif wire == 2:
        handle.seek(read_varint(handle), 1)
    elif wire == 5:
        handle.seek(4, 1)
    else:
        raise ValueError(f"Unsupported protobuf wire type: {wire}")


def find_main_map(run_dir: Path) -> Path:
    matches = [
        path
        for path in run_dir.glob("vox_*_0")
        if "_component_" not in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one main voxel map in {run_dir}, found {len(matches)}"
        )
    return matches[0]


def parse_map(path: Path) -> dict[str, np.ndarray | int]:
    size = NX * NY * NZ
    energy = np.zeros(size, dtype=np.float64)
    energy2 = np.zeros(size, dtype=np.float64)
    let_base = np.zeros(size, dtype=np.float64)
    dose = np.zeros(size, dtype=np.float64)
    entries = 0

    with path.open("rb") as handle:
        while True:
            try:
                tag = read_varint(handle)
            except EOFError:
                break
            field, wire = tag >> 3, tag & 7
            if field != 1 or wire != 2:
                skip_field(handle, wire)
                continue
            entry_length = read_varint(handle)
            entry_end = handle.tell() + entry_length
            voxel_id: int | None = None
            dep_energy = dep_energy2 = dep_let_base = dep_dose = 0.0
            while handle.tell() < entry_end:
                entry_tag = read_varint(handle)
                entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_field == 1 and entry_wire == 0:
                    voxel_id = read_varint(handle)
                elif entry_field == 2 and entry_wire == 2:
                    value_length = read_varint(handle)
                    value_end = handle.tell() + value_length
                    while handle.tell() < value_end:
                        value_tag = read_varint(handle)
                        value_field, value_wire = value_tag >> 3, value_tag & 7
                        if value_wire == 1 and value_field in (2, 3, 4, 9):
                            value = struct.unpack("<d", handle.read(8))[0]
                            if value_field == 2:
                                dep_energy = value
                            elif value_field == 3:
                                dep_energy2 = value
                            elif value_field == 4:
                                dep_let_base = value
                            else:
                                dep_dose = value
                        else:
                            skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    skip_field(handle, entry_wire)
            handle.seek(entry_end)
            if voxel_id is None or not 0 <= voxel_id < size:
                continue
            if dep_energy > 0.0:
                entries += 1
                energy[voxel_id] += dep_energy
                energy2[voxel_id] += dep_energy2
                let_base[voxel_id] += dep_let_base
                dose[voxel_id] += dep_dose

    shape = (NZ, NY, NX)
    return {
        "entries": entries,
        "energy_keV": energy.reshape(shape),
        "energy2_keV2": energy2.reshape(shape),
        "let_base_keV2_um": let_base.reshape(shape),
        "dose_Gy": dose.reshape(shape),
    }


def smooth(values: np.ndarray, bins: int = 3) -> np.ndarray:
    if bins <= 1:
        return values.copy()
    kernel = np.ones(bins, dtype=float) / bins
    return np.convolve(values, kernel, mode="same")


def profile_fwhm(x_mm: np.ndarray, profile: np.ndarray) -> float:
    if not np.any(profile > 0.0):
        return math.nan
    norm = profile / float(np.max(profile))
    selected = x_mm[norm >= 0.5]
    if selected.size < 2:
        return math.nan
    return float(selected[-1] - selected[0] + DX_MM)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(
    run_dir: Path,
    output_dir: Path,
    histories: int,
    field_radius_mm: float,
    skip_plots: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_map(find_main_map(run_dir))
    energy = np.asarray(parsed["energy_keV"])
    energy2 = np.asarray(parsed["energy2_keV2"])
    let_base = np.asarray(parsed["let_base_keV2_um"])
    dose = np.asarray(parsed["dose_Gy"])

    x_mm = (np.arange(NX, dtype=float) + 0.5 - NX / 2.0) * DX_MM
    z_mm = (np.arange(NZ, dtype=float) + 0.5 - NZ / 2.0) * DZ_MM
    depth_mm = (np.arange(NY, dtype=float) + 0.5) * DY_MM
    xx, zz = np.meshgrid(x_mm, z_mm)
    radius2 = xx * xx + zz * zz
    field_mask = radius2 <= field_radius_mm**2
    central_mask = radius2 <= CENTRAL_RADIUS_MM**2

    # Reverse Geant4 y: photons enter at +y, while array index grows from -y.
    energy_depth = energy[:, ::-1, :].transpose(1, 0, 2)
    let_depth = let_base[:, ::-1, :].transpose(1, 0, 2)
    dose_depth = dose[:, ::-1, :].transpose(1, 0, 2)
    energy2_depth = energy2[:, ::-1, :].transpose(1, 0, 2)

    field_dose = np.mean(dose_depth[:, field_mask], axis=1)
    central_dose = np.mean(dose_depth[:, central_mask], axis=1)
    field_energy = np.sum(energy_depth[:, field_mask], axis=1)
    field_let_base = np.sum(let_depth[:, field_mask], axis=1)
    central_energy = np.sum(energy_depth[:, central_mask], axis=1)
    central_let_base = np.sum(let_depth[:, central_mask], axis=1)
    field_let = np.divide(
        field_let_base,
        field_energy,
        out=np.full(NY, np.nan),
        where=field_energy > 0.0,
    )
    central_let = np.divide(
        central_let_base,
        central_energy,
        out=np.full(NY, np.nan),
        where=central_energy > 0.0,
    )
    field_pdd = 100.0 * field_dose / float(np.max(field_dose))
    central_pdd = 100.0 * central_dose / float(np.max(central_dose))
    field_pdd_smooth = 100.0 * smooth(field_dose) / float(
        np.max(smooth(field_dose))
    )
    central_pdd_smooth = 100.0 * smooth(central_dose) / float(
        np.max(smooth(central_dose))
    )

    rel_se = np.full(energy_depth.shape, np.nan, dtype=float)
    positive = energy_depth > 0.0
    if histories > 1:
        variance_numerator = np.maximum(
            histories * energy2_depth - energy_depth * energy_depth,
            0.0,
        )
        rel_se[positive] = (
            np.sqrt(variance_numerator[positive] / (histories - 1.0))
            / energy_depth[positive]
        )
    field_positive = positive[:, field_mask]
    field_rel_se = rel_se[:, field_mask][field_positive]
    field_voxels = int(np.count_nonzero(field_mask) * NY)

    depth_rows: list[dict[str, object]] = []
    for index, depth in enumerate(depth_mm):
        depth_rows.append(
            {
                "depth_mm": depth,
                "field_mean_dose_Gy_per_primary": (
                    field_dose[index] / histories
                ),
                "central_mean_dose_Gy_per_primary": (
                    central_dose[index] / histories
                ),
                "field_relative_depth_dose_percent": field_pdd[index],
                "central_relative_depth_dose_percent": central_pdd[index],
                "field_smoothed_relative_depth_dose_percent": (
                    field_pdd_smooth[index]
                ),
                "central_smoothed_relative_depth_dose_percent": (
                    central_pdd_smooth[index]
                ),
                "field_LETd_w_keV_um": field_let[index],
                "central_LETd_w_keV_um": central_let[index],
            }
        )
    write_csv(output_dir / "gamma_co60_water_depth_profile.csv", depth_rows)

    profile_rows: list[dict[str, object]] = []
    profiles: dict[float, np.ndarray] = {}
    fwhm: dict[float, float] = {}
    z_band = np.abs(z_mm) <= 2.5
    for requested_depth in PROFILE_DEPTHS_MM:
        depth_index = int(np.argmin(np.abs(depth_mm - requested_depth)))
        profile = np.mean(dose_depth[depth_index, z_band, :], axis=0)
        if np.max(profile) > 0.0:
            profile = profile / float(np.max(profile))
        profiles[requested_depth] = profile
        fwhm[requested_depth] = profile_fwhm(x_mm, profile)
        for x_value, relative_dose in zip(x_mm, profile):
            profile_rows.append(
                {
                    "requested_depth_mm": requested_depth,
                    "actual_depth_mm": depth_mm[depth_index],
                    "x_mm": x_value,
                    "relative_dose": relative_dose,
                }
            )
    write_csv(output_dir / "gamma_co60_water_lateral_profiles.csv", profile_rows)

    overall_field_energy = float(np.sum(energy_depth[:, field_mask]))
    overall_field_let_base = float(np.sum(let_depth[:, field_mask]))
    peak_field_index = int(np.argmax(field_pdd_smooth))
    peak_central_index = int(np.argmax(central_pdd_smooth))
    summary: dict[str, object] = {
        "status": "internal_water_transport_benchmark_not_clinical_pdd",
        "histories": histories,
        "grid": f"{NX}x{NY}x{NZ}",
        "voxel_mm": "1x1x1",
        "field_radius_mm": field_radius_mm,
        "central_radius_mm": CENTRAL_RADIUS_MM,
        "nonzero_map_entries": int(parsed["entries"]),
        "field_nonzero_voxel_fraction": (
            float(np.count_nonzero(field_positive)) / field_voxels
        ),
        "field_depEnergy_keV_per_primary": overall_field_energy / histories,
        "field_mean_dose_Gy_per_primary": (
            float(np.mean(dose_depth[:, field_mask])) / histories
        ),
        "field_LETd_w_keV_um": (
            overall_field_let_base / overall_field_energy
        ),
        "central_LETd_w_keV_um": (
            float(np.sum(let_depth[:, central_mask]))
            / float(np.sum(energy_depth[:, central_mask]))
        ),
        "field_smoothed_depth_of_maximum_mm": depth_mm[peak_field_index],
        "central_smoothed_depth_of_maximum_mm": depth_mm[peak_central_index],
        "positive_field_voxel_rel_SE_median": (
            float(np.median(field_rel_se)) if field_rel_se.size else math.nan
        ),
        "positive_field_voxel_fraction_rel_SE_le_0p20": (
            float(np.mean(field_rel_se <= 0.20))
            if field_rel_se.size
            else math.nan
        ),
    }
    for requested_depth in PROFILE_DEPTHS_MM:
        summary[f"lateral_FWHM_at_{requested_depth:g}mm_mm"] = fwhm[
            requested_depth
        ]
    for requested_depth in (5.0, 20.0, 40.0, 60.0, 75.0):
        index = int(np.argmin(np.abs(depth_mm - requested_depth)))
        summary[
            f"central_smoothed_relative_dose_at_{requested_depth:g}mm_percent"
        ] = float(central_pdd_smooth[index])

    write_csv(output_dir / "gamma_co60_water_summary.csv", [summary])
    (output_dir / "gamma_co60_water_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not skip_plots:
        MatplotlibConfigurator().apply_custom_styles()
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
            }
        )
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

        ax = axes[0, 0]
        ax.plot(
            depth_mm,
            central_pdd_smooth,
            color="#d95f02",
            linewidth=2.2,
            label=f"центр, r ≤ {CENTRAL_RADIUS_MM:g} мм",
        )
        ax.plot(
            depth_mm,
            field_pdd_smooth,
            color="#1b9e77",
            linewidth=2.0,
            label=f"поле, r ≤ {field_radius_mm:g} мм",
        )
        ax.set(
            xlabel="Глубина в воде, мм",
            ylabel="Относительная доза, %",
            title="Относительный глубинный профиль",
            xlim=(0.0, 80.0),
            ylim=(0.0, 106.0),
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

        ax = axes[0, 1]
        ax.plot(
            depth_mm,
            central_let,
            color="#7570b3",
            linewidth=2.0,
            label="центр",
        )
        ax.plot(
            depth_mm,
            field_let,
            color="#1b9e77",
            linewidth=1.8,
            label="поле",
        )
        ax.set(
            xlabel="Глубина в воде, мм",
            ylabel=r"$LET_{d,w}$, кэВ/мкм",
            title="Дозо-взвешенная ЛПЭ",
            xlim=(0.0, 80.0),
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

        ax = axes[1, 0]
        colors = ("#e7298a", "#d95f02", "#1b9e77", "#7570b3")
        for depth, color in zip(PROFILE_DEPTHS_MM, colors):
            ax.plot(
                x_mm,
                100.0 * profiles[depth],
                color=color,
                linewidth=1.8,
                label=(
                    f"{depth:g} мм; FWHM {fwhm[depth]:.1f} мм"
                    if np.isfinite(fwhm[depth])
                    else f"{depth:g} мм"
                ),
            )
        ax.axvline(-field_radius_mm, color="#555555", linestyle="--", alpha=0.6)
        ax.axvline(field_radius_mm, color="#555555", linestyle="--", alpha=0.6)
        ax.set(
            xlabel="Поперечная координата x, мм",
            ylabel="Относительная доза, %",
            title="Латеральные профили в центральной полосе",
            xlim=(-30.0, 30.0),
            ylim=(0.0, 110.0),
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncol=2)

        ax = axes[1, 1]
        positive_by_depth = np.mean(positive[:, field_mask], axis=1) * 100.0
        ax.plot(
            depth_mm,
            positive_by_depth,
            color="#377eb8",
            linewidth=2.0,
        )
        ax.set(
            xlabel="Глубина в воде, мм",
            ylabel="Воксели с энерговкладом, %",
            title="Статистическое заполнение поля",
            xlim=(0.0, 80.0),
            ylim=(0.0, 105.0),
        )
        ax.grid(alpha=0.25)
        text = (
            f"N = {histories:,}\n"
            f"медиана отн. SE = {summary['positive_field_voxel_rel_SE_median']:.2f}\n"
            f"SE ≤ 20%: "
            f"{100.0 * summary['positive_field_voxel_fraction_rel_SE_le_0p20']:.1f}%"
        )
        ax.text(
            0.98,
            0.96,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        fig.suptitle(
            "Co-60 в однородной воде: внутренний тест переноса и скоринга\n"
            "параллельное поле на поверхности; не клиническая PDD установки",
            fontsize=14,
        )
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                output_dir / f"gamma_co60_water_benchmark.{suffix}",
                dpi=220 if suffix == "png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--field-radius-mm", type=float, default=14.4)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    summary = analyse(
        args.run_dir,
        args.output_dir,
        args.histories,
        args.field_radius_mm,
        args.skip_plots,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
