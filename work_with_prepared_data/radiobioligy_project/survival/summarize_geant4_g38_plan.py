"""Summarise the exact g38Gy_f1fx proton plan in the rat phantom.

This script intentionally keeps the archived biological dose (38 Gy, RBE
1.1) separate from the Monte Carlo physical dose.  It compares two plausible
spot-width models at the selected 40-mm PMMA geometry and records the pilot
PMMA-thickness scan used to position the high-dose region in GTVp.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


TASK_ROOT = Path(r"C:\dev\dissertation\task4_5")
OUTPUT_ROOT = (
    TASK_ROOT / "outputs" / "geant4_livermore_20260724"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "proton_g38_exact_plan_2026"
PLAN_PROTONS = 237_113_000_000.0
ARCHIVE_BIOLOGICAL_DOSE_GY = 38.0
USER_CONFIRMED_RBE = 1.1
ARCHIVE_PHYSICAL_DOSE_GY = (
    ARCHIVE_BIOLOGICAL_DOSE_GY / USER_CONFIRMED_RBE
)
GTV_DEPTH_MM = (17.8, 34.6)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    spot_width_model: str
    result_dir: Path


MAIN_SCENARIOS = (
    Scenario(
        key="powerlaw",
        label=r"$\sigma(E)=4{,}0$–$5{,}2$ мм",
        spot_width_model="экстраполяция makerP",
        result_dir=OUTPUT_ROOT / "proton_g38_exact_plan_100k",
    ),
    Scenario(
        key="fixed5",
        label=r"$\sigma=5{,}0$ мм",
        spot_width_model="фиксированная ширина 5,0 мм",
        result_dir=OUTPUT_ROOT / "proton_g38_exact_plan_100k",
    ),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_row(scenario: Scenario) -> dict[str, str]:
    rows = read_rows(scenario.result_dir / "gtv_summary.csv")
    token = "powerlaw" if scenario.key == "powerlaw" else "fixed5p0"
    matches = [row for row in rows if token in row["case"]]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {scenario.key} row, found {len(matches)}"
        )
    return matches[0]


def profile_path(scenario: Scenario, case: str) -> Path:
    return scenario.result_dir / f"{case}_central_depth_profile.csv"


def dvh_path(scenario: Scenario, case: str) -> Path:
    return scenario.result_dir / f"{case}_normalised_dvh.csv"


def read_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_rows(path)
    depth = np.asarray(
        [float(row["depth_mm_from_plus_y_surface"]) for row in rows]
    )
    energy = np.asarray(
        [float(row["central_energy_smoothed_keV"]) for row in rows]
    )
    maximum = float(np.max(energy))
    if maximum <= 0.0:
        raise RuntimeError(f"Profile is empty: {path}")
    return depth, energy / maximum


def read_dvh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_rows(path)
    dose_over_mean = np.asarray(
        [float(row["dose_over_gtv_mean"]) for row in rows]
    )
    volume_receiving = np.asarray(
        [float(row["volume_receiving_percent"]) for row in rows]
    )
    return dose_over_mean, volume_receiving


def read_dvh_resolution(path: Path) -> dict[str, np.ndarray]:
    rows = read_rows(path)
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def numerical_summary(
    scenario: Scenario,
    row: dict[str, str],
) -> dict[str, float | int | str]:
    physical_dose = (
        float(row["gtv_mean_dose_Gy_per_primary"]) * PLAN_PROTONS
    )
    biological_dose = physical_dose * USER_CONFIRMED_RBE
    return {
        "scenario": scenario.key,
        "spot_width_model": scenario.spot_width_model,
        "histories": int(row["histories"]),
        "pmma_thickness_mm": 40.0,
        "terminal_air_thickness_mm": 559.8,
        "terminal_path_mm": 600.0,
        "total_plan_protons": PLAN_PROTONS,
        "archive_biological_dose_Gy": ARCHIVE_BIOLOGICAL_DOSE_GY,
        "user_confirmed_RBE": USER_CONFIRMED_RBE,
        "archive_physical_dose_Gy": ARCHIVE_PHYSICAL_DOSE_GY,
        "mc_gtv_mean_physical_dose_Gy": physical_dose,
        "mc_gtv_mean_RBE_weighted_dose_Gy": biological_dose,
        "relative_difference_vs_archive_biological": (
            biological_dose / ARCHIVE_BIOLOGICAL_DOSE_GY - 1.0
        ),
        "central_peak_depth_mm": float(row["central_peak_depth_mm"]),
        "peak_inside_gtv_depth_interval": int(
            row["central_peak_inside_gtv_depth_interval"]
        ),
        "gtv_LETd_w_keV_um": float(row["gtv_LETd_w_keV_um"]),
        "central_gtv_LETd_w_keV_um": float(
            row["central_gtv_depth_LETd_w_keV_um"]
        ),
        "central_peak_band_LETd_w_keV_um": float(
            row["central_peak_band_LETd_w_keV_um"]
        ),
        "gtv_nonzero_fine_voxel_fraction": float(
            row["gtv_nonzero_dose_fraction"]
        ),
        "gtv_D50_over_Dmean_1p6x1p6x0p8": float(
            row["gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8"]
        ),
        "gtv_D90_over_Dmean_1p6x1p6x0p8": float(
            row["gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"]
        ),
        "gtv_D90_over_Dmean_0p8x0p8x0p4": float(
            row["gtv_aggregated_D90_over_Dmean_0p8x0p8x0p4"]
        ),
        "gtv_D90_over_Dmean_1p2x1p2x0p6": float(
            row["gtv_aggregated_D90_over_Dmean_1p2x1p2x0p6"]
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_main(
    rows: list[dict[str, float | int | str]],
    profiles: list[tuple[Scenario, np.ndarray, np.ndarray]],
    output_stem: Path,
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
            }
        )
        colours = ("#542788", "#a6d854")
        fig, (ax_profile, ax_dose) = plt.subplots(
            1,
            2,
            figsize=(12.5, 5.2),
            constrained_layout=True,
        )

        ax_profile.axvspan(
            GTV_DEPTH_MM[0],
            GTV_DEPTH_MM[1],
            color="#e64b35",
            alpha=0.11,
            label="аксиальный интервал GTVp",
        )
        for (scenario, depth, profile), colour, summary in zip(
            profiles,
            colours,
            rows,
        ):
            ax_profile.plot(
                depth,
                profile,
                color=colour,
                linewidth=2.7,
                label=scenario.label,
            )
            ax_profile.axvline(
                float(summary["central_peak_depth_mm"]),
                color=colour,
                linestyle=":",
                linewidth=1.8,
                alpha=0.95,
            )
        ax_profile.set(
            xlim=(0.0, 51.2),
            ylim=(0.0, 1.08),
            xlabel=(
                "Глубина вдоль пучка, мм\n"
                "(0 = входная граница воксельного фантома)"
            ),
            ylabel="Нормированный энерговклад",
            title="Профиль внутри фантома; ПММА находится перед x=0",
        )
        ax_profile.annotate(
            "направление пучка",
            xy=(12.0, 1.035),
            xytext=(2.0, 1.035),
            arrowprops={"arrowstyle": "->", "color": "0.3"},
            ha="left",
            va="center",
            color="0.3",
            fontsize=9,
        )
        ax_profile.grid(alpha=0.22)
        ax_profile.legend(loc="lower center", frameon=True)

        x = np.arange(len(rows))
        width = 0.34
        physical = np.asarray(
            [float(row["mc_gtv_mean_physical_dose_Gy"]) for row in rows]
        )
        biological = np.asarray(
            [
                float(row["mc_gtv_mean_RBE_weighted_dose_Gy"])
                for row in rows
            ]
        )
        bars_physical = ax_dose.bar(
            x - width / 2.0,
            physical,
            width,
            color="#4c78a8",
            label="физическая доза",
        )
        bars_biological = ax_dose.bar(
            x + width / 2.0,
            biological,
            width,
            color="#f58518",
            label=r"доза при $RBE=1{,}1$",
        )
        ax_dose.axhline(
            ARCHIVE_PHYSICAL_DOSE_GY,
            color="#4c78a8",
            linestyle="--",
            linewidth=1.5,
            label="архив: 34,55 Гр физ.",
        )
        ax_dose.axhline(
            ARCHIVE_BIOLOGICAL_DOSE_GY,
            color="#f58518",
            linestyle="--",
            linewidth=1.5,
            label="архив: 38 Гр биол.",
        )
        ax_dose.bar_label(
            bars_physical,
            labels=[f"{value:.2f}" for value in physical],
            padding=3,
            fontsize=10,
        )
        ax_dose.bar_label(
            bars_biological,
            labels=[f"{value:.2f}" for value in biological],
            padding=3,
            fontsize=10,
        )
        ax_dose.set_xticks(
            x,
            [scenario.label for scenario in MAIN_SCENARIOS],
        )
        ax_dose.set(
            ylabel="Средняя доза в GTVp, Гр",
            title="Доза, рассчитанная по весам плана",
            ylim=(0.0, 42.0),
        )
        ax_dose.grid(axis="y", alpha=0.22)
        ax_dose.legend(loc="lower right", frameon=True)

        fig.suptitle(
            (
                "План g38Gy_f1fx: 40 мм ПММА в терминальном "
                "600-мм участке пучка"
            ),
            fontsize=15,
        )
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def plot_dvh(
    rows: list[dict[str, float | int | str]],
    dvhs: list[tuple[Scenario, np.ndarray, np.ndarray]],
    output_stem: Path,
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
            }
        )
        colours = ("#542788", "#a6d854")
        fig, ax = plt.subplots(
            figsize=(7.4, 5.4),
            constrained_layout=True,
        )
        for row_index, (
            (scenario, dose_over_mean, volume_receiving),
            colour,
            summary,
        ) in enumerate(zip(dvhs, colours, rows)):
            ax.plot(
                dose_over_mean,
                volume_receiving,
                color=colour,
                linewidth=2.5,
                label=scenario.label,
            )
            d90 = float(
                summary["gtv_D90_over_Dmean_raw_fine_grid"]
            )
            ax.scatter(
                [d90],
                [90.0],
                s=42,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.annotate(
                rf"$D_{{90}}/D_{{mean}}={d90:.2f}$",
                (d90, 90.0),
                xytext=(9, 9 - 17 * row_index),
                textcoords="offset points",
                color=colour,
                fontsize=9,
            )
        ax.axhline(90.0, color="0.45", linestyle="--", linewidth=1.0)
        ax.axvline(1.0, color="0.45", linestyle=":", linewidth=1.0)
        ax.set(
            xlim=(0.0, 3.0),
            ylim=(0.0, 102.0),
            xlabel=r"Доза в вокселе / $D_{\mathrm{mean}}$ GTVp",
            ylabel="Объём GTVp, получивший ≥ дозы, %",
            title="Нормированная DVH: исходная сетка 0,4×0,4×0,2 мм",
        )
        ax.grid(alpha=0.22)
        ax.legend(frameon=True)
        fig.suptitle(
            "План g38Gy_f1fx, 40 мм ПММА, 100 000 историй",
            fontsize=14,
        )
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def plot_dvh_resolution(
    summary: dict[str, float | int | str],
    dvh: dict[str, np.ndarray],
    output_stem: Path,
) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "legend.fontsize": 9,
            }
        )
        dose_ratio = dvh["dose_over_gtv_mean"]
        mean_biological_dose = float(
            summary["mc_gtv_mean_RBE_weighted_dose_Gy"]
        )
        dose_gy = dose_ratio * mean_biological_dose
        definitions = (
            (
                "volume_receiving_percent",
                "0,4×0,4×0,2 мм",
                float(summary["gtv_D90_over_Dmean_raw_fine_grid"]),
                "#7f7f7f",
                "--",
            ),
            (
                "volume_receiving_percent_0p8x0p8x0p4",
                "0,8×0,8×0,4 мм",
                float(summary["gtv_D90_over_Dmean_0p8x0p8x0p4"]),
                "#80b1d3",
                "-.",
            ),
            (
                "volume_receiving_percent_1p2x1p2x0p6",
                "1,2×1,2×0,6 мм",
                float(summary["gtv_D90_over_Dmean_1p2x1p2x0p6"]),
                "#fdb462",
                ":",
            ),
            (
                "volume_receiving_percent_1p6x1p6x0p8",
                "1,6×1,6×0,8 мм",
                float(summary["gtv_D90_over_Dmean_1p6x1p6x0p8"]),
                "#542788",
                "-",
            ),
        )
        fig, ax = plt.subplots(
            figsize=(8.2, 5.5),
            constrained_layout=True,
        )
        for key, grid_label, d90_ratio, colour, linestyle in definitions:
            ax.plot(
                dose_gy,
                dvh[key],
                color=colour,
                linestyle=linestyle,
                linewidth=2.4,
                label=(
                    f"{grid_label}; "
                    rf"$D_{{90}}={d90_ratio * mean_biological_dose:.1f}$ Гр"
                ),
            )
        ax.axhline(90.0, color="0.45", linestyle="--", linewidth=1.0)
        ax.axvline(
            ARCHIVE_BIOLOGICAL_DOSE_GY,
            color="#e64b35",
            linestyle="--",
            linewidth=1.5,
            label="архивная доза 38 Гр",
        )
        ax.set(
            xlim=(0.0, 70.0),
            ylim=(0.0, 102.0),
            xlabel=r"Биологически взвешенная доза при $RBE=1{,}1$, Гр",
            ylabel="Объём GTVp, получивший ≥ дозы, %",
            title="Зависимость DVH от пространственного агрегирования",
        )
        ax.grid(alpha=0.22)
        ax.legend(frameon=True, loc="upper right")
        fig.suptitle(
            (
                "План g38Gy_f1fx, основная модель ширины пятна, "
                "100 000 историй"
            ),
            fontsize=14,
        )
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def pmma_scan_rows() -> list[dict[str, float | int | str]]:
    definitions = (
        (
            0.0,
            10_000,
            OUTPUT_ROOT / "proton_g38_air600_powerlaw_10k",
        ),
        (
            20.0,
            1_000,
            OUTPUT_ROOT / "proton_g38_pmma20_powerlaw_1k",
        ),
        (
            40.0,
            1_000,
            OUTPUT_ROOT / "proton_g38_pmma40_powerlaw_1k",
        ),
        (
            45.0,
            1_000,
            OUTPUT_ROOT / "proton_g38_pmma45_powerlaw_1k",
        ),
    )
    output = []
    for thickness, histories, directory in definitions:
        row = read_rows(directory / "gtv_summary.csv")[0]
        physical = (
            float(row["gtv_mean_dose_Gy_per_primary"]) * PLAN_PROTONS
        )
        output.append(
            {
                "pmma_thickness_mm": thickness,
                "histories": histories,
                "central_peak_depth_mm": float(
                    row["central_peak_depth_mm"]
                ),
                "peak_inside_gtv_depth_interval": int(
                    GTV_DEPTH_MM[0]
                    <= float(row["central_peak_depth_mm"])
                    <= GTV_DEPTH_MM[1]
                ),
                "mc_gtv_mean_physical_dose_Gy": physical,
                "mc_gtv_mean_RBE_weighted_dose_Gy": (
                    physical * USER_CONFIRMED_RBE
                ),
                "gtv_LETd_w_keV_um": float(row["gtv_LETd_w_keV_um"]),
                "interpretation": (
                    "pilot geometry scan; history counts differ"
                ),
            }
        )
    return output


def plot_pmma_scan(rows: list[dict], output_stem: Path) -> None:
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
            }
        )
        thickness = np.asarray(
            [float(row["pmma_thickness_mm"]) for row in rows]
        )
        peak = np.asarray(
            [float(row["central_peak_depth_mm"]) for row in rows]
        )
        biological = np.asarray(
            [
                float(row["mc_gtv_mean_RBE_weighted_dose_Gy"])
                for row in rows
            ]
        )
        fig, (ax_peak, ax_dose) = plt.subplots(
            1,
            2,
            figsize=(11.5, 4.6),
            constrained_layout=True,
        )
        ax_peak.axhspan(
            GTV_DEPTH_MM[0],
            GTV_DEPTH_MM[1],
            color="#e64b35",
            alpha=0.12,
            label="GTVp",
        )
        ax_peak.plot(
            thickness,
            peak,
            color="#542788",
            marker="o",
            linewidth=2.2,
        )
        for x_value, y_value in zip(thickness, peak):
            ax_peak.annotate(
                f"{y_value:.1f}",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
            )
        ax_peak.set(
            xlim=(-2.0, 47.0),
            ylim=(17.0, 48.0),
            xlabel="Толщина ПММА, мм",
            ylabel="Глубина максимума, мм",
            title="Положение максимума",
        )
        ax_peak.grid(alpha=0.22)
        ax_peak.legend(frameon=True)

        ax_dose.axhline(
            ARCHIVE_BIOLOGICAL_DOSE_GY,
            color="#e64b35",
            linestyle="--",
            label="архив: 38 Гр",
        )
        ax_dose.plot(
            thickness,
            biological,
            color="#1b9e77",
            marker="s",
            linewidth=2.2,
        )
        for x_value, y_value in zip(thickness, biological):
            ax_dose.annotate(
                f"{y_value:.1f}",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
            )
        ax_dose.set(
            xlim=(-2.0, 47.0),
            ylim=(18.0, 39.5),
            xlabel="Толщина ПММА, мм",
            ylabel=r"Средняя доза в GTVp при $RBE=1{,}1$, Гр",
            title="Доза, рассчитанная по весам плана",
        )
        ax_dose.grid(alpha=0.22)
        ax_dose.legend(frameon=True)
        fig.suptitle(
            (
                "Чувствительность плана g38Gy_f1fx к толщине ПММА "
                "(пилотные расчёты)"
            ),
            fontsize=14,
        )
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def main() -> None:
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    profiles = []
    dvhs = []
    resolution_dvhs = []
    for scenario in MAIN_SCENARIOS:
        row = selected_row(scenario)
        summary = numerical_summary(scenario, row)
        summary["gtv_D90_over_Dmean_raw_fine_grid"] = float(
            row["gtv_D90_over_Dmean"]
        )
        summaries.append(summary)
        depth, profile = read_profile(
            profile_path(scenario, row["case"])
        )
        profiles.append((scenario, depth, profile))
        dose_over_mean, volume_receiving = read_dvh(
            dvh_path(scenario, row["case"])
        )
        dvhs.append((scenario, dose_over_mean, volume_receiving))
        resolution_dvhs.append(
            read_dvh_resolution(dvh_path(scenario, row["case"]))
        )

    write_csv(DEFAULT_OUTPUT / "g38_plan_summary.csv", summaries)
    plot_main(
        summaries,
        profiles,
        DEFAULT_OUTPUT / "g38_plan_depth_and_dose",
    )
    plot_dvh(
        summaries,
        dvhs,
        DEFAULT_OUTPUT / "g38_plan_normalised_dvh",
    )
    plot_dvh_resolution(
        summaries[0],
        resolution_dvhs[0],
        DEFAULT_OUTPUT / "g38_plan_dvh_resolution_sensitivity",
    )

    scan_rows = pmma_scan_rows()
    write_csv(DEFAULT_OUTPUT / "g38_pmma_scan_summary.csv", scan_rows)
    plot_pmma_scan(
        scan_rows,
        DEFAULT_OUTPUT / "g38_pmma_scan",
    )

    print(f"output={DEFAULT_OUTPUT}")
    for row in summaries:
        print(
            f"{row['scenario']}: "
            f"peak={row['central_peak_depth_mm']:.2f} mm, "
            f"Dphys={row['mc_gtv_mean_physical_dose_Gy']:.3f} Gy, "
            f"DRBE={row['mc_gtv_mean_RBE_weighted_dose_Gy']:.3f} Gy, "
            f"LET={row['gtv_LETd_w_keV_um']:.3f} keV/um"
        )


if __name__ == "__main__":
    main()
