"""Analyse monoenergetic NOVAC reference calculations in water and rat GTVp.

The incident electron energies are computational reference scenarios.  They do
not reconstruct the NOVAC head, PMMA applicator or a measured incident energy
spectrum.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_proton_100MeV_reduced_gtv import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    dose_at_volume,
    find_main_map,
    gtv_geometry,
    locally_aggregate_gtv_dose,
    parse_voxel_map,
    smooth,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


ENERGIES_MEV = (4.0, 6.0, 8.0, 10.0)
ENERGY_COLOURS = {
    4.0: "#3b4cc0",
    6.0: "#2a7f9e",
    8.0: "#35a16b",
    10.0: "#d97706",
}
GTV_REPORTING_FACTORS = (4, 4, 4)


def configure_plots() -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def token(energy_mev: float) -> str:
    return f"{energy_mev:g}".replace(".", "p")


def case_dir(root: Path, energy_mev: float) -> Path:
    result = root / f"E{token(energy_mev)}MeV"
    if not result.is_dir():
        raise FileNotFoundError(result)
    return result


def float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray(
        [float(row[name]) if row[name] else np.nan for row in rows],
        dtype=float,
    )


def distal_crossing(
    depth: np.ndarray,
    relative_dose: np.ndarray,
    level: float,
) -> float:
    peak_index = int(np.nanargmax(relative_dose))
    x = depth[peak_index:]
    y = relative_dose[peak_index:]
    below = np.flatnonzero(y <= level)
    if below.size == 0:
        return float("nan")
    index = int(below[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if y1 == y0:
        return x1
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def load_water_case(
    root: Path,
    energy_mev: float,
) -> tuple[dict[str, float | str], dict[str, np.ndarray]]:
    directory = case_dir(root, energy_mev)
    profile_path = directory / "depth_profile_central_r10mm.csv"
    with profile_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    depth = float_column(rows, "depth_mm")
    dose = float_column(
        rows,
        "smoothed_mean_dose_per_selected_voxel_Gy",
    )
    dep_energy = float_column(rows, "depEnergy_keV")
    let_base = float_column(rows, "letBase_keV2_um")
    let_depth = float_column(rows, "smoothed_LETd_w_keV_um")
    dose_max = float(np.nanmax(dose))
    relative_dose = dose / dose_max if dose_max > 0.0 else dose
    peak_index = int(np.nanargmax(relative_dose))
    integrated_let = float(np.nansum(let_base) / np.nansum(dep_energy))
    metadata_path = directory / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    summary: dict[str, float | str] = {
        "energy_MeV": energy_mev,
        "physics": str(metadata["physics"]),
        "histories": float(metadata["histories"]),
        "dmax_mm": float(depth[peak_index]),
        "R90_mm": distal_crossing(depth, relative_dose, 0.90),
        "R80_mm": distal_crossing(depth, relative_dose, 0.80),
        "R50_mm": distal_crossing(depth, relative_dose, 0.50),
        "R10_mm": distal_crossing(depth, relative_dose, 0.10),
        "LETd_w_at_dmax_keV_um": float(let_depth[peak_index]),
        "integrated_LETd_w_keV_um": integrated_let,
    }
    return summary, {
        "depth_mm": depth,
        "relative_dose": relative_dose,
        "LETd_w_keV_um": let_depth,
        "dep_energy_keV": dep_energy,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyse_water(
    root: Path,
    output_dir: Path,
    comparison_root: Path | None,
    energies_mev: tuple[float, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_rows: list[dict[str, object]] = []
    primary_profiles: dict[float, dict[str, np.ndarray]] = {}
    for energy in energies_mev:
        summary, profile = load_water_case(root, energy)
        primary_rows.append(summary)
        primary_profiles[energy] = profile
    write_rows(output_dir / "electron_water_summary.csv", primary_rows)

    comparison_rows: list[dict[str, object]] = []
    comparison_profiles: dict[float, dict[str, np.ndarray]] = {}
    if comparison_root is not None:
        for energy in energies_mev:
            try:
                summary, profile = load_water_case(comparison_root, energy)
            except FileNotFoundError:
                continue
            comparison_rows.append(summary)
            comparison_profiles[energy] = profile
        if comparison_rows:
            write_rows(
                output_dir / "electron_water_physics_comparison.csv",
                comparison_rows,
            )

    configure_plots()
    figure, (axis_dose, axis_let) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.4),
        sharex=True,
        constrained_layout=True,
    )
    for energy in energies_mev:
        profile = primary_profiles[energy]
        colour = ENERGY_COLOURS[energy]
        axis_dose.plot(
            profile["depth_mm"],
            100.0 * profile["relative_dose"],
            color=colour,
            linewidth=2.2,
            label=f"{energy:g} МэВ",
        )
        reliable = profile["relative_dose"] >= 0.05
        axis_let.plot(
            profile["depth_mm"],
            profile["LETd_w_keV_um"],
            color=colour,
            linewidth=2.0,
            alpha=0.28,
            linestyle="--",
        )
        axis_let.plot(
            profile["depth_mm"],
            np.where(
                reliable,
                profile["LETd_w_keV_um"],
                np.nan,
            ),
            color=colour,
            linewidth=2.2,
            label=f"{energy:g} МэВ",
        )
    axis_dose.axhline(90.0, color="#777777", linewidth=0.8, linestyle=":")
    axis_dose.axhline(80.0, color="#777777", linewidth=0.8, linestyle=":")
    axis_dose.set(
        title=(
            "Моноэнергетические электроны NOVAC: водный benchmark\n"
            "Livermore; источник задан на входной плоскости"
        ),
        ylabel="Относительная доза, %",
        xlim=(0.0, 55.0),
        ylim=(0.0, 108.0),
    )
    axis_dose.grid(alpha=0.2)
    axis_dose.legend(frameon=False, ncol=2)
    axis_let.set(
        xlabel="Глубина в воде, мм",
        ylabel=r"$LET_{D,w}$, кэВ/мкм",
        ylim=(0.0, 2.2),
    )
    axis_let.grid(alpha=0.2)
    axis_let.legend(frameon=False, ncol=2)
    axis_let.text(
        0.98,
        0.96,
        (
            "Сплошная линия: доза ≥5% максимума\n"
            "Штриховая: низкодозный диагностический хвост"
        ),
        transform=axis_let.transAxes,
        ha="right",
        va="top",
        color="#555555",
    )
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            output_dir / f"electron_novac_water_depth{suffix}",
            dpi=220,
        )
    plt.close(figure)

    if comparison_profiles:
        compare_figure, (compare_dose, compare_let) = plt.subplots(
            1,
            2,
            figsize=(10.8, 4.5),
            constrained_layout=True,
        )
        for energy in sorted(comparison_profiles):
            primary = primary_profiles[energy]
            secondary = comparison_profiles[energy]
            compare_dose.plot(
                primary["depth_mm"],
                100.0 * primary["relative_dose"],
                color=ENERGY_COLOURS[energy],
                linewidth=2.2,
                label=f"{energy:g} МэВ, Livermore",
            )
            compare_dose.plot(
                secondary["depth_mm"],
                100.0 * secondary["relative_dose"],
                color=ENERGY_COLOURS[energy],
                linewidth=1.8,
                linestyle="--",
                label=f"{energy:g} МэВ, option4",
            )
            reliable = primary["relative_dose"] >= 0.05
            compare_let.plot(
                primary["depth_mm"],
                np.where(
                    reliable,
                    primary["LETd_w_keV_um"],
                    np.nan,
                ),
                color=ENERGY_COLOURS[energy],
                linewidth=2.2,
                label=f"{energy:g} МэВ, Livermore",
            )
            secondary_reliable = secondary["relative_dose"] >= 0.05
            compare_let.plot(
                secondary["depth_mm"],
                np.where(
                    secondary_reliable,
                    secondary["LETd_w_keV_um"],
                    np.nan,
                ),
                color=ENERGY_COLOURS[energy],
                linewidth=1.8,
                linestyle="--",
                label=f"{energy:g} МэВ, option4",
            )
        compare_dose.set(
            title="Глубинная доза",
            xlabel="Глубина в воде, мм",
            ylabel="Относительная доза, %",
            xlim=(0.0, 55.0),
            ylim=(0.0, 108.0),
        )
        compare_let.set(
            title=r"$LET_{D,w}$ в дозовом диапазоне ≥5%",
            xlabel="Глубина в воде, мм",
            ylabel=r"$LET_{D,w}$, кэВ/мкм",
            xlim=(0.0, 55.0),
        )
        for axis in (compare_dose, compare_let):
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
        for suffix in (".png", ".svg", ".pdf"):
            compare_figure.savefig(
                output_dir / f"electron_novac_water_physics{suffix}",
                dpi=220,
            )
        plt.close(compare_figure)


def normalised_dvh(
    dose: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    mean_dose = float(np.mean(dose))
    if mean_dose <= 0.0:
        return np.zeros_like(x_grid)
    normalised = np.sort(dose / mean_dose)
    indices = np.searchsorted(normalised, x_grid, side="left")
    return 100.0 * (normalised.size - indices) / normalised.size


def analyse_rat_case(
    root: Path,
    energy_mev: float,
    gtv_ids: np.ndarray,
    gtv_lookup: np.ndarray,
    gtv_depth_interval: tuple[float, float],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    directory = case_dir(root, energy_mev)
    metadata = json.loads(
        (directory / "run_metadata.json").read_text(encoding="utf-8-sig")
    )
    parsed = parse_voxel_map(
        find_main_map(directory),
        gtv_lookup,
        include_dose=True,
    )
    dose = np.asarray(parsed["gtv_dose_Gy"], dtype=float)
    energy = np.asarray(parsed["gtv_energy_keV"], dtype=float)
    let_base = np.asarray(parsed["gtv_let_base_keV2_um"], dtype=float)
    reporting_dose, reporting_blocks = locally_aggregate_gtv_dose(
        dose,
        gtv_ids,
        GTV_REPORTING_FACTORS,
    )
    mean_dose = float(np.mean(reporting_dose))
    d2 = dose_at_volume(reporting_dose, 2.0)
    d50 = dose_at_volume(reporting_dose, 50.0)
    d80 = dose_at_volume(reporting_dose, 80.0)
    d90 = dose_at_volume(reporting_dose, 90.0)
    d95 = dose_at_volume(reporting_dose, 95.0)
    d98 = dose_at_volume(reporting_dose, 98.0)
    integrated_let = (
        float(np.sum(let_base) / np.sum(energy))
        if np.sum(energy) > 0.0
        else float("nan")
    )
    central_energy = np.asarray(
        parsed["central_energy_keV"],
        dtype=float,
    )
    central_let_base = np.asarray(
        parsed["central_let_base_keV2_um"],
        dtype=float,
    )
    smooth_energy = smooth(central_energy)
    smooth_let_base = smooth(central_let_base)
    central_let = np.divide(
        smooth_let_base,
        smooth_energy,
        out=np.full_like(smooth_energy, np.nan),
        where=smooth_energy > 0.0,
    )
    depth = (np.arange(smooth_energy.size, dtype=float) + 0.5) * 0.4
    in_gtv = (
        (depth >= gtv_depth_interval[0])
        & (depth <= gtv_depth_interval[1])
    )
    normalisation = float(np.mean(smooth_energy[in_gtv]))
    relative_energy = (
        smooth_energy / normalisation
        if normalisation > 0.0
        else smooth_energy
    )
    summary: dict[str, object] = {
        "run_root": str(root),
        "energy_MeV": energy_mev,
        "histories": int(metadata["histories"]),
        "seed1": int(metadata["seed1"]),
        "seed2": int(metadata["seed2"]),
        "physics": str(metadata["physics"]),
        "source_radius_mm": float(metadata["source_radius_mm"]),
        "gtv_reporting_blocks": reporting_blocks,
        "gtv_nonzero_fraction_reporting": float(
            np.mean(reporting_dose > 0.0)
        ),
        "gtv_Dmean_Gy": mean_dose,
        "D2_over_Dmean": d2 / mean_dose,
        "D50_over_Dmean": d50 / mean_dose,
        "D80_over_Dmean": d80 / mean_dose,
        "D90_over_Dmean": d90 / mean_dose,
        "D95_over_Dmean": d95 / mean_dose,
        "D98_over_Dmean": d98 / mean_dose,
        "HI98": (d2 - d98) / d50 if d50 > 0.0 else float("nan"),
        "LETd_w_GTVp_keV_um": integrated_let,
    }
    return summary, {
        "reporting_dose_Gy": reporting_dose,
        "depth_mm": depth,
        "relative_energy": relative_energy,
        "central_LETd_w_keV_um": central_let,
    }


def analyse_rat(
    roots: list[Path],
    output_dir: Path,
    ct_dir: Path,
    rtstruct: Path,
    energies_mev: tuple[float, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, gtv_depth_interval = gtv_geometry(
        ct_dir,
        rtstruct,
    )
    rows: list[dict[str, object]] = []
    profiles: dict[float, list[dict[str, np.ndarray]]] = {
        energy: [] for energy in energies_mev
    }
    for root in roots:
        for energy in energies_mev:
            try:
                summary, profile = analyse_rat_case(
                    root,
                    energy,
                    gtv_ids,
                    gtv_lookup,
                    gtv_depth_interval,
                )
            except FileNotFoundError:
                continue
            rows.append(summary)
            profiles[energy].append(profile)
    if not rows:
        raise FileNotFoundError(
            "No requested electron energy directory was found in any run root."
        )
    write_rows(output_dir / "electron_rat_seed_metrics.csv", rows)

    aggregate_rows: list[dict[str, object]] = []
    metric_names = (
        "gtv_nonzero_fraction_reporting",
        "gtv_Dmean_Gy",
        "D2_over_Dmean",
        "D50_over_Dmean",
        "D80_over_Dmean",
        "D90_over_Dmean",
        "D95_over_Dmean",
        "D98_over_Dmean",
        "HI98",
        "LETd_w_GTVp_keV_um",
    )
    for energy in energies_mev:
        selected = [
            row for row in rows if float(row["energy_MeV"]) == energy
        ]
        if not selected:
            continue
        aggregate: dict[str, object] = {
            "energy_MeV": energy,
            "n_seeds": len(selected),
            "physics": str(selected[0]["physics"]),
            "source_radius_mm": float(selected[0]["source_radius_mm"]),
        }
        for name in metric_names:
            values = np.asarray(
                [float(row[name]) for row in selected],
                dtype=float,
            )
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_min"] = float(np.min(values))
            aggregate[f"{name}_max"] = float(np.max(values))
        aggregate_rows.append(aggregate)
    write_rows(output_dir / "electron_rat_summary.csv", aggregate_rows)

    configure_plots()
    figure, (axis_depth, axis_dvh, axis_let) = plt.subplots(
        3,
        1,
        figsize=(9.3, 12.0),
        constrained_layout=True,
    )
    common_x = np.linspace(0.0, 2.2, 441)
    for energy in energies_mev:
        selected_profiles = profiles[energy]
        if not selected_profiles:
            continue
        depth = selected_profiles[0]["depth_mm"]
        depth_curves = np.asarray(
            [profile["relative_energy"] for profile in selected_profiles]
        )
        let_curves = np.asarray(
            [
                profile["central_LETd_w_keV_um"]
                for profile in selected_profiles
            ]
        )
        dvh_curves = np.asarray(
            [
                normalised_dvh(profile["reporting_dose_Gy"], common_x)
                for profile in selected_profiles
            ]
        )
        colour = ENERGY_COLOURS[energy]
        axis_depth.fill_between(
            depth,
            np.min(depth_curves, axis=0),
            np.max(depth_curves, axis=0),
            color=colour,
            alpha=0.10,
        )
        axis_depth.plot(
            depth,
            np.mean(depth_curves, axis=0),
            color=colour,
            linewidth=2.1,
            label=f"{energy:g} МэВ",
        )
        axis_dvh.fill_between(
            common_x,
            np.min(dvh_curves, axis=0),
            np.max(dvh_curves, axis=0),
            color=colour,
            alpha=0.10,
        )
        axis_dvh.plot(
            common_x,
            np.mean(dvh_curves, axis=0),
            color=colour,
            linewidth=2.1,
            label=f"{energy:g} МэВ",
        )
        mean_depth_curve = np.mean(depth_curves, axis=0)
        reliable = mean_depth_curve >= 0.05 * np.nanmax(mean_depth_curve)
        supported_depth = np.any(np.isfinite(let_curves), axis=0)
        mean_let = np.full(let_curves.shape[1], np.nan, dtype=float)
        mean_let[supported_depth] = np.nanmean(
            let_curves[:, supported_depth],
            axis=0,
        )
        axis_let.plot(
            depth,
            mean_let,
            color=colour,
            linewidth=1.3,
            linestyle="--",
            alpha=0.35,
        )
        axis_let.plot(
            depth,
            np.where(reliable, mean_let, np.nan),
            color=colour,
            linewidth=2.1,
            label=f"{energy:g} МэВ",
        )
    for axis in (axis_depth, axis_let):
        axis.axvspan(
            gtv_depth_interval[0],
            gtv_depth_interval[1],
            color="#e31a1c",
            alpha=0.07,
            label="аксиальный интервал GTVp",
        )
    axis_depth.axhline(1.0, color="#777777", linewidth=0.8)
    axis_depth.set(
        title=(
            "Электроны NOVAC в воксельном фантоме крысы\n"
            "моноэнергетические входные сценарии; среднее и диапазон seed"
        ),
        ylabel="Энерговклад / среднее в GTVp по глубине",
        xlim=(0.0, 50.0),
    )
    axis_depth.grid(alpha=0.2)
    axis_depth.legend(frameon=False, ncol=2)
    axis_dvh.set(
        title="DVH GTVp; отчётная сетка 1,6×1,6×0,8 мм",
        xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
        ylabel="Объём GTVp, получивший ≥ дозы, %",
        xlim=(0.0, 2.2),
        ylim=(0.0, 101.5),
    )
    axis_dvh.grid(alpha=0.2)
    axis_dvh.legend(frameon=False, ncol=2)
    axis_let.set(
        title=r"Центральный профиль $LET_{D,w}$",
        xlabel="Глубина от входной границы +y, мм",
        ylabel=r"$LET_{D,w}$, кэВ/мкм",
        xlim=(0.0, 50.0),
    )
    axis_let.grid(alpha=0.2)
    axis_let.legend(frameon=False, ncol=2)
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            output_dir / f"electron_novac_rat_gtv{suffix}",
            dpi=220,
        )
    plt.close(figure)

    dvh_figure, dvh_axis = plt.subplots(
        1,
        1,
        figsize=(8.4, 5.3),
        constrained_layout=True,
    )
    for energy in energies_mev:
        selected_profiles = profiles[energy]
        if not selected_profiles:
            continue
        dvh_curves = np.asarray(
            [
                normalised_dvh(profile["reporting_dose_Gy"], common_x)
                for profile in selected_profiles
            ]
        )
        colour = ENERGY_COLOURS[energy]
        dvh_axis.fill_between(
            common_x,
            np.min(dvh_curves, axis=0),
            np.max(dvh_curves, axis=0),
            color=colour,
            alpha=0.10,
        )
        dvh_axis.plot(
            common_x,
            np.mean(dvh_curves, axis=0),
            color=colour,
            linewidth=2.4,
            label=f"{energy:g} МэВ",
        )
    for volume_percent in (90.0, 95.0, 98.0):
        dvh_axis.axhline(
            volume_percent,
            color="#777777",
            linewidth=0.8,
            linestyle=":",
        )
    dvh_axis.set(
        title=(
            "Электроны NOVAC: DVH GTVp\n"
            "входные моноэнергетические сценарии; "
            "отчётная сетка 1,6×1,6×0,8 мм"
        ),
        xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
        ylabel="Объём GTVp, получивший ≥ дозы, %",
        xlim=(0.0, 1.8),
        ylim=(0.0, 101.5),
    )
    dvh_axis.grid(alpha=0.2)
    dvh_axis.legend(frameon=False)
    for suffix in (".png", ".svg", ".pdf"):
        dvh_figure.savefig(
            output_dir / f"electron_novac_rat_dvh{suffix}",
            dpi=220,
        )
    plt.close(dvh_figure)

    profile_figure, (profile_axis, let_axis) = plt.subplots(
        2,
        1,
        figsize=(8.7, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    profile_full_let_max = 0.46
    for energy in energies_mev:
        selected_profiles = profiles[energy]
        if not selected_profiles:
            continue
        depth = selected_profiles[0]["depth_mm"]
        depth_curves = np.asarray(
            [profile["relative_energy"] for profile in selected_profiles]
        )
        let_curves = np.asarray(
            [
                profile["central_LETd_w_keV_um"]
                for profile in selected_profiles
            ]
        )
        colour = ENERGY_COLOURS[energy]
        profile_axis.fill_between(
            depth,
            np.min(depth_curves, axis=0),
            np.max(depth_curves, axis=0),
            color=colour,
            alpha=0.10,
        )
        mean_depth_curve = np.mean(depth_curves, axis=0)
        profile_axis.plot(
            depth,
            mean_depth_curve,
            color=colour,
            linewidth=2.3,
            label=f"{energy:g} МэВ",
        )
        supported_depth = np.any(np.isfinite(let_curves), axis=0)
        mean_let = np.full(let_curves.shape[1], np.nan, dtype=float)
        mean_let[supported_depth] = np.nanmean(
            let_curves[:, supported_depth],
            axis=0,
        )
        finite_let = mean_let[np.isfinite(mean_let)]
        if finite_let.size:
            profile_full_let_max = max(
                profile_full_let_max,
                float(np.nanmax(finite_let)),
            )
        reliable = mean_depth_curve >= 0.05 * np.nanmax(mean_depth_curve)
        let_axis.plot(
            depth,
            mean_let,
            color=colour,
            linewidth=1.8,
            linestyle="--",
            alpha=0.35,
        )
        let_axis.plot(
            depth,
            np.where(reliable, mean_let, np.nan),
            color=colour,
            linewidth=2.3,
            label=f"{energy:g} МэВ",
        )
    for axis in (profile_axis, let_axis):
        axis.axvspan(
            gtv_depth_interval[0],
            gtv_depth_interval[1],
            color="#e31a1c",
            alpha=0.07,
            label="аксиальный интервал GTVp",
        )
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=2)
    profile_axis.axhline(1.0, color="#777777", linewidth=0.8)
    profile_axis.set(
        title=(
            "Электроны NOVAC в КТ-фантоме крысы\n"
            "глубинный энерговклад и дозовзвешенная ЛПЭ"
        ),
        ylabel="Энерговклад / среднее в GTVp",
        xlim=(0.0, 50.0),
    )
    let_axis.set(
        xlabel="Глубина от входной границы +y, мм",
        ylabel=r"$LET_{D,w}$, кэВ/мкм",
        ylim=(0.18, 1.05 * profile_full_let_max),
    )
    let_axis.set_title(
        "Сплошная линия: энерговклад ≥5%; "
        "штриховая: низкодозный хвост"
    )
    for suffix in (".png", ".svg", ".pdf"):
        profile_figure.savefig(
            output_dir / f"electron_novac_rat_depth_let{suffix}",
            dpi=220,
        )
    plt.close(profile_figure)


def analyse_rat_scenarios(
    cases: list[tuple[str, Path]],
    energy_mev: float,
    output_dir: Path,
    ct_dir: Path,
    rtstruct: Path,
    comparison_title: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, gtv_depth_interval = gtv_geometry(
        ct_dir,
        rtstruct,
    )
    scenario_data: list[
        tuple[str, dict[str, object], dict[str, np.ndarray]]
    ] = []
    for label, root in cases:
        summary, profile = analyse_rat_case(
            root,
            energy_mev,
            gtv_ids,
            gtv_lookup,
            gtv_depth_interval,
        )
        summary = {"scenario": label, **summary}
        scenario_data.append((label, summary, profile))
    write_rows(
        output_dir / "electron_rat_scenario_comparison.csv",
        [summary for _, summary, _ in scenario_data],
    )

    configure_plots()
    figure, (axis_depth, axis_dvh, axis_let) = plt.subplots(
        3,
        1,
        figsize=(9.0, 11.4),
        constrained_layout=True,
    )
    colours = ("#3366aa", "#d97706", "#35a16b", "#7b4ab5")
    common_x = np.linspace(0.0, 2.2, 441)
    scenario_full_let_max = 0.46
    for index, (label, _, profile) in enumerate(scenario_data):
        colour = colours[index % len(colours)]
        axis_depth.plot(
            profile["depth_mm"],
            profile["relative_energy"],
            color=colour,
            linewidth=2.3,
            label=label,
        )
        axis_dvh.plot(
            common_x,
            normalised_dvh(profile["reporting_dose_Gy"], common_x),
            color=colour,
            linewidth=2.3,
            label=label,
        )
        reliable = (
            profile["relative_energy"]
            >= 0.05 * np.nanmax(profile["relative_energy"])
        )
        full_let = profile["central_LETd_w_keV_um"]
        finite_let = full_let[np.isfinite(full_let)]
        if finite_let.size:
            scenario_full_let_max = max(
                scenario_full_let_max,
                float(np.nanmax(finite_let)),
            )
        axis_let.plot(
            profile["depth_mm"],
            full_let,
            color=colour,
            linewidth=1.8,
            linestyle="--",
            alpha=0.35,
        )
        axis_let.plot(
            profile["depth_mm"],
            np.where(
                reliable,
                full_let,
                np.nan,
            ),
            color=colour,
            linewidth=2.3,
            label=label,
        )
    for axis in (axis_depth, axis_let):
        axis.axvspan(
            gtv_depth_interval[0],
            gtv_depth_interval[1],
            color="#e31a1c",
            alpha=0.07,
            label="аксиальный интервал GTVp",
        )
    axis_depth.axhline(1.0, color="#777777", linewidth=0.8)
    axis_depth.set(
        title=(
            comparison_title
            or (
                f"Электроны NOVAC {energy_mev:g} МэВ: "
                "сравнение расчётных сценариев"
            )
        ),
        ylabel="Энерговклад / среднее в GTVp",
        xlim=(0.0, 50.0),
    )
    axis_dvh.set(
        title="DVH GTVp; отчётная сетка 1,6×1,6×0,8 мм",
        xlabel=r"Доза в элементе / $D_{\mathrm{mean}}$ GTVp",
        ylabel="Объём GTVp, получивший ≥ дозы, %",
        xlim=(0.0, 2.2),
        ylim=(0.0, 101.5),
    )
    axis_let.set(
        title=(
            r"Центральный профиль $LET_{D,w}$; "
            "пунктир — низкодозный хвост <5%"
        ),
        xlabel="Глубина от входной границы +y, мм",
        ylabel=r"$LET_{D,w}$, кэВ/мкм",
        xlim=(0.0, 50.0),
        ylim=(0.18, 1.05 * scenario_full_let_max),
    )
    for axis in (axis_depth, axis_dvh, axis_let):
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=2)
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            output_dir / f"electron_novac_rat_field_sensitivity{suffix}",
            dpi=220,
        )
    plt.close(figure)


def parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, path_text = value.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError(
            "Expected LABEL=PATH for --case."
        )
    return label.strip(), Path(path_text.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    water = subparsers.add_parser("water")
    water.add_argument("--run-root", type=Path, required=True)
    water.add_argument("--comparison-root", type=Path)
    water.add_argument("--output-dir", type=Path, required=True)
    water.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=ENERGIES_MEV,
    )

    rat = subparsers.add_parser("rat")
    rat.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
    )
    rat.add_argument("--output-dir", type=Path, required=True)
    rat.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    rat.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    rat.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=ENERGIES_MEV,
    )

    scenarios = subparsers.add_parser("rat-scenarios")
    scenarios.add_argument(
        "--case",
        type=parse_labelled_path,
        action="append",
        required=True,
    )
    scenarios.add_argument("--energy", type=float, required=True)
    scenarios.add_argument("--output-dir", type=Path, required=True)
    scenarios.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    scenarios.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    scenarios.add_argument("--title")

    args = parser.parse_args()
    if args.command == "water":
        analyse_water(
            args.run_root,
            args.output_dir,
            args.comparison_root,
            tuple(args.energies),
        )
    elif args.command == "rat":
        analyse_rat(
            args.run_root,
            args.output_dir,
            args.ct_dir,
            args.rtstruct,
            tuple(args.energies),
        )
    else:
        analyse_rat_scenarios(
            args.case,
            args.energy,
            args.output_dir,
            args.ct_dir,
            args.rtstruct,
            args.title,
        )


if __name__ == "__main__":
    main()
