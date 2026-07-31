"""Plot the preliminary Geant4 proton peak geometry and LET profile."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


DEFAULT_INPUT = Path(
    r"C:\dev\dissertation\task4_5"
    r"\physics_sensitivity_proton_60MeV_peak_inclxx_livermore_dose_components_1k"
)
DEFAULT_OUTPUT = Path(
    r"C:\dev\dissertation\task4_5\outputs\geant4_livermore_20260724"
    r"\proton_peak_geometry"
)
GTV_MIN_MM = 16.85
GTV_MAX_MM = 33.45


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data: dict[str, np.ndarray] = {}
    for field in rows[0]:
        data[field] = np.array(
            [
                float(row[field]) if row[field].strip() else np.nan
                for row in rows
            ],
            dtype=float,
        )
    return data


def weighted_let(data: dict[str, np.ndarray], mask: np.ndarray) -> float:
    dep = data["depEnergy_keV"][mask]
    let_base = data["letBase_keV2_um"][mask]
    return float(np.nansum(let_base) / np.nansum(dep))


def write_summary(
    path: Path,
    central: dict[str, np.ndarray],
    hydrogen: dict[str, np.ndarray],
    gtv: dict[str, np.ndarray],
    gtv_hydrogen: dict[str, np.ndarray],
) -> None:
    dose = central["smoothed_mean_dose_per_selected_voxel_Gy"]
    peak_index = int(np.nanargmax(dose))
    peak_depth = float(central["depth_mm"][peak_index])
    gtv_dose = gtv["smoothed_mean_dose_per_roi_voxel_Gy"]
    roi_counts = gtv["gtv_roi_voxels"]
    valid_gtv = (
        (gtv["depth_mm"] >= GTV_MIN_MM)
        & (gtv["depth_mm"] <= GTV_MAX_MM)
        & (roi_counts >= 0.1 * np.nanmax(roi_counts))
    )
    gtv_peak_local = int(np.nanargmax(np.where(valid_gtv, gtv_dose, np.nan)))
    peak_band = np.abs(gtv["depth_mm"] - peak_depth) <= 0.5
    peak_band_h = np.abs(gtv_hydrogen["depth_mm"] - peak_depth) <= 0.5
    gtv_total_voxels = 1_217_483
    gtv_positive_voxels = int(np.nansum(gtv["gtv_voxels_with_energy"]))
    gtv_dose_sum = float(np.nansum(gtv["dose_sum_Gy"]))
    gtv_coverage_percent = 100.0 * gtv_positive_voxels / gtv_total_voxels

    summary = [
        ("physics_list", "QGSP_INCLXX + G4EmLivermore"),
        ("primary", "proton"),
        ("energy_MeV", "60"),
        ("histories", "1000"),
        ("central_cylinder_radius_mm", "2"),
        ("smoothing_width_mm", "1.1"),
        ("gtv_proximal_depth_mm", f"{GTV_MIN_MM:.2f}"),
        ("gtv_distal_depth_mm", f"{GTV_MAX_MM:.2f}"),
        ("central_mean_dose_peak_depth_mm", f"{peak_depth:.3f}"),
        (
            "central_mean_dose_peak_Gy_per_1000_primaries",
            f"{dose[peak_index]:.12g}",
        ),
        (
            "central_all_charged_LETd_at_dose_peak_keV_um",
            f"{central['smoothed_LETd_w_keV_um'][peak_index]:.6g}",
        ),
        (
            "central_hydrogen_LETd_at_dose_peak_keV_um",
            f"{hydrogen['smoothed_LETd_w_keV_um'][peak_index]:.6g}",
        ),
        (
            "whole_gtv_mean_dose_peak_depth_mm",
            f"{gtv['depth_mm'][gtv_peak_local]:.3f}",
        ),
        (
            "whole_gtv_nonzero_dose_voxels",
            str(gtv_positive_voxels),
        ),
        (
            "whole_gtv_nonzero_dose_coverage_percent",
            f"{gtv_coverage_percent:.6g}",
        ),
        (
            "whole_gtv_mean_dose_including_zeros_Gy",
            f"{gtv_dose_sum / gtv_total_voxels:.12g}",
        ),
        ("whole_gtv_D50_Gy", "0"),
        ("whole_gtv_D90_Gy", "0"),
        (
            "whole_gtv_all_charged_LETd_keV_um",
            f"{weighted_let(gtv, gtv['depEnergy_keV'] > 0):.6g}",
        ),
        (
            "whole_gtv_hydrogen_LETd_keV_um",
            f"{weighted_let(gtv_hydrogen, gtv_hydrogen['depEnergy_keV'] > 0):.6g}",
        ),
        (
            "gtv_band_pm0p5mm_all_charged_LETd_keV_um",
            f"{weighted_let(gtv, peak_band):.6g}",
        ),
        (
            "gtv_band_pm0p5mm_hydrogen_LETd_keV_um",
            f"{weighted_let(gtv_hydrogen, peak_band_h):.6g}",
        ),
        (
            "status",
            "preliminary; fragment-sensitive LET requires more histories/seeds",
        ),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)


def plot_profiles(
    central: dict[str, np.ndarray],
    hydrogen: dict[str, np.ndarray],
    gtv: dict[str, np.ndarray],
    output_stem: Path,
) -> None:
    depth = central["depth_mm"]
    central_dose = central["smoothed_mean_dose_per_selected_voxel_Gy"]
    central_dose_norm = central_dose / np.nanmax(central_dose)
    peak_index = int(np.nanargmax(central_dose))
    peak_depth = float(depth[peak_index])

    gtv_depth = gtv["depth_mm"]
    gtv_dose = gtv["smoothed_mean_dose_per_roi_voxel_Gy"]
    roi_counts = gtv["gtv_roi_voxels"]
    valid_gtv = (
        (gtv_depth >= GTV_MIN_MM)
        & (gtv_depth <= GTV_MAX_MM)
        & (roi_counts >= 0.1 * np.nanmax(roi_counts))
    )
    gtv_dose_plot = np.where(valid_gtv, gtv_dose, np.nan)
    gtv_dose_norm = gtv_dose_plot / np.nanmax(gtv_dose_plot)

    in_plot = (depth >= 15.5) & (depth <= 36.5)
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig, (ax_dose, ax_let) = plt.subplots(
            2,
            1,
            figsize=(15, 12),
            sharex=True,
            constrained_layout=True,
        )

        for axis in (ax_dose, ax_let):
            axis.axvspan(
                GTV_MIN_MM,
                GTV_MAX_MM,
                color="#d9d9d9",
                alpha=0.45,
                zorder=0,
                label="GTVp" if axis is ax_dose else None,
            )
            axis.axvline(
                peak_depth,
                color="#a61c00",
                linestyle="--",
                linewidth=1.8,
                zorder=2,
                label=(
                    f"максимум дозы: {peak_depth:.2f} мм"
                    if axis is ax_dose
                    else None
                ),
            )
            axis.grid(alpha=0.25, linewidth=0.8)

        ax_dose.plot(
            depth[in_plot],
            central_dose_norm[in_plot],
            color="#c23b22",
            linewidth=2.6,
            label="центральный цилиндр, r = 2 мм",
        )
        ax_dose.plot(
            gtv_depth,
            gtv_dose_norm,
            color="#2f6b9a",
            linewidth=2.3,
            label="средняя доза по поперечному сечению GTVp",
        )
        ax_dose.set_ylabel("Нормированная доза")
        ax_dose.set_ylim(0.0, 1.08)
        ax_dose.set_title(
            "Протоны 60 МэВ: положение дозового максимума относительно GTVp\n"
            "предварительный расчёт, 1000 первичных частиц"
        )
        ax_dose.legend(loc="lower center", ncol=2, frameon=True)

        ax_let.axhspan(
            12.0,
            13.0,
            color="#f2c14e",
            alpha=0.28,
            label="ранее принятый ориентир 12–13 кэВ/мкм",
        )
        ax_let.plot(
            depth[in_plot],
            central["smoothed_LETd_w_keV_um"][in_plot],
            color="#6b6b6b",
            linewidth=1.8,
            label="все заряженные частицы",
        )
        ax_let.plot(
            depth[in_plot],
            hydrogen["smoothed_LETd_w_keV_um"][in_plot],
            color="#1f77b4",
            linewidth=2.6,
            label="водородный компонент",
        )
        ax_let.set_yscale("log")
        ax_let.set_ylim(0.9, 100.0)
        ax_let.set_ylabel(r"$LET_D$, кэВ/мкм")
        ax_let.set_xlabel("Глубина от входной границы фантома, мм")
        ax_let.legend(loc="upper right", frameon=True)

        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(
                output_stem.with_suffix(suffix),
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    central = read_numeric_csv(
        args.input_dir / "central_axis_r2mm_depth_profile.csv"
    )
    hydrogen = read_numeric_csv(
        args.input_dir / "central_axis_r2mm_depth_profile_hydrogen.csv"
    )
    gtv = read_numeric_csv(args.input_dir / "gtv_depth_profile.csv")
    gtv_hydrogen = read_numeric_csv(
        args.input_dir / "gtv_depth_profile_hydrogen.csv"
    )

    output_stem = args.output_dir / "proton_60MeV_peak_geometry"
    plot_profiles(central, hydrogen, gtv, output_stem)
    write_summary(
        args.output_dir / "proton_60MeV_peak_geometry_summary.csv",
        central,
        hydrogen,
        gtv,
        gtv_hydrogen,
    )
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
