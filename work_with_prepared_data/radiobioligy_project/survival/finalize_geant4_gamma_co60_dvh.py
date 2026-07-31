"""Finalize the Co-60 GTV DVH by pooling independent entrance-field runs.

The material ROKUS-AM head model is used to validate the transverse field
and the integral LET.  Direct transport from the active cobalt volume is
inefficient for a fine tumour DVH because only a small fraction of source
photons reaches the GTV.  This script therefore pools independent runs of
the already formed incident field at the rat surface.  The resulting DVH is
a conditional high-statistics transport calculation and is not an absolute
dose-per-decay estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    PX_MM,
    PY_MM,
    PZ_MM,
    dose_at_volume,
    find_main_map,
    gtv_geometry,
    locally_aggregate_gtv_dose,
    parse_voxel_map,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


AGGREGATIONS = {
    "fine_0p4x0p4x0p2": None,
    "report_0p8x0p8x0p4": (2, 2, 2),
    "report_1p2x1p2x0p6": (3, 3, 3),
    "report_1p6x1p6x0p8": (4, 4, 4),
}
REPORT_GRID = "report_1p6x1p6x0p8"
DX_LEVELS = (2.0, 10.0, 50.0, 90.0, 95.0, 98.0)


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_stats(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
    cv = 100.0 * sd / mean if mean != 0.0 else math.nan
    return mean, sd, cv


def relative_se(
    energy: np.ndarray,
    energy2: np.ndarray,
    histories: int,
) -> np.ndarray:
    result = np.full(energy.size, np.nan, dtype=float)
    positive = energy > 0.0
    if histories <= 1 or not np.any(positive):
        return result
    variance_numerator = np.maximum(
        histories * energy2 - energy * energy,
        0.0,
    )
    result[positive] = (
        np.sqrt(variance_numerator[positive] / (histories - 1.0))
        / energy[positive]
    )
    return result


def grid_dose(
    dose: np.ndarray,
    gtv_ids: np.ndarray,
    factors: tuple[int, int, int] | None,
) -> tuple[np.ndarray, int]:
    if factors is None:
        return dose, int(dose.size)
    return locally_aggregate_gtv_dose(dose, gtv_ids, factors)


def grid_metrics(
    dose: np.ndarray,
    mean_dose: float,
    grid: str,
    report_voxels: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "grid": grid,
        "report_voxels": report_voxels,
        "nonzero_volume_fraction": float(np.mean(dose > 0.0)),
    }
    for level in DX_LEVELS:
        row[f"D{int(level)}_over_Dmean"] = (
            dose_at_volume(dose, level) / mean_dose
        )
    d2 = float(row["D2_over_Dmean"])
    d50 = float(row["D50_over_Dmean"])
    d98 = float(row["D98_over_Dmean"])
    row["HI98"] = (d2 - d98) / d50 if d50 > 0.0 else math.nan
    return row


def dvh_curve(dose: np.ndarray, mean_dose: float) -> tuple[np.ndarray, np.ndarray]:
    sorted_dose = np.sort(dose)
    count = sorted_dose.size
    receiving = 100.0 * (count - np.arange(count)) / count
    return sorted_dose / mean_dose, receiving


def analyse(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, depth_interval = gtv_geometry(
        args.ct_dir,
        args.rtstruct,
    )
    gtv_count = int(gtv_ids.size)
    pooled_energy = np.zeros(gtv_count, dtype=float)
    pooled_energy2 = np.zeros(gtv_count, dtype=float)
    pooled_let_base = np.zeros(gtv_count, dtype=float)
    pooled_dose = np.zeros(gtv_count, dtype=float)
    total_histories = 0
    per_seed: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    progressive_dvh: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for index, run_tag in enumerate(args.run_tags, start=1):
        run_root = args.runs_root / run_tag
        summary_path = run_root / "gtv_analysis" / "gtv_summary.csv"
        metadata_path = run_root / "through" / "run_metadata.json"
        summary = read_single_csv(summary_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        histories = int(summary["histories"])
        if histories != int(metadata["histories"]):
            raise RuntimeError(
                f"{run_tag}: history mismatch between summary and metadata"
            )
        parsed = parse_voxel_map(
            find_main_map(run_root / "through"),
            gtv_lookup,
            include_dose=True,
        )
        pooled_energy += np.asarray(parsed["gtv_energy_keV"])
        pooled_energy2 += np.asarray(parsed["gtv_energy2_keV2"])
        pooled_let_base += np.asarray(parsed["gtv_let_base_keV2_um"])
        pooled_dose += np.asarray(parsed["gtv_dose_Gy"])
        total_histories += histories
        per_seed.append(
            {
                "run_order": index,
                "run_tag": run_tag,
                "seed1": metadata["seed1"],
                "seed2": metadata["seed2"],
                "histories": histories,
                "mean_dose_Gy_per_primary": float(
                    summary["gtv_mean_dose_Gy_per_primary"]
                ),
                "LETd_w_keV_um": float(summary["gtv_LETd_w_keV_um"]),
                "fine_nonzero_fraction": float(
                    summary["gtv_nonzero_dose_fraction"]
                ),
                "report_D50_over_Dmean": float(
                    summary[
                        "gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8"
                    ]
                ),
                "report_D90_over_Dmean": float(
                    summary[
                        "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"
                    ]
                ),
            }
        )

        mean_dose = float(np.mean(pooled_dose))
        report_dose, report_voxels = grid_dose(
            pooled_dose,
            gtv_ids,
            AGGREGATIONS[REPORT_GRID],
        )
        report = grid_metrics(
            report_dose,
            mean_dose,
            REPORT_GRID,
            report_voxels,
        )
        voxel_rel_se = relative_se(
            pooled_energy,
            pooled_energy2,
            total_histories,
        )
        finite_rel_se = voxel_rel_se[np.isfinite(voxel_rel_se)]
        convergence_rows.append(
            {
                "runs_pooled": index,
                "histories_pooled": total_histories,
                "fine_nonzero_fraction": float(np.mean(pooled_dose > 0.0)),
                "fine_positive_voxel_rel_SE_median": float(
                    np.median(finite_rel_se)
                ),
                "fine_positive_voxel_rel_SE_p90": float(
                    np.percentile(finite_rel_se, 90.0)
                ),
                "mean_dose_Gy_per_primary": mean_dose / total_histories,
                "LETd_w_keV_um": (
                    float(np.sum(pooled_let_base))
                    / float(np.sum(pooled_energy))
                ),
                "report_nonzero_volume_fraction": report[
                    "nonzero_volume_fraction"
                ],
                "report_D2_over_Dmean": report["D2_over_Dmean"],
                "report_D10_over_Dmean": report["D10_over_Dmean"],
                "report_D50_over_Dmean": report["D50_over_Dmean"],
                "report_D90_over_Dmean": report["D90_over_Dmean"],
                "report_D95_over_Dmean": report["D95_over_Dmean"],
                "report_D98_over_Dmean": report["D98_over_Dmean"],
                "report_HI98": report["HI98"],
            }
        )
        if index in args.dvh_checkpoints or index == len(args.run_tags):
            progressive_dvh[index] = dvh_curve(report_dose, mean_dose)

    write_csv(args.output_dir / "gamma_co60_dvh_per_seed.csv", per_seed)
    write_csv(
        args.output_dir / "gamma_co60_dvh_convergence.csv",
        convergence_rows,
    )

    mean_dose = float(np.mean(pooled_dose))
    total_energy = float(np.sum(pooled_energy))
    total_let_base = float(np.sum(pooled_let_base))
    pooled_let = total_let_base / total_energy
    voxel_rel_se = relative_se(
        pooled_energy,
        pooled_energy2,
        total_histories,
    )
    finite_rel_se = voxel_rel_se[np.isfinite(voxel_rel_se)]
    grid_rows: list[dict[str, object]] = []
    final_grid_doses: dict[str, np.ndarray] = {}
    final_grid_voxels: dict[str, int] = {}
    for grid, factors in AGGREGATIONS.items():
        dose, report_voxels = grid_dose(pooled_dose, gtv_ids, factors)
        final_grid_doses[grid] = dose
        final_grid_voxels[grid] = report_voxels
        grid_rows.append(
            {
                **grid_metrics(dose, mean_dose, grid, report_voxels),
                "histories_pooled": total_histories,
            }
        )
    write_csv(
        args.output_dir / "gamma_co60_dvh_resolution_sensitivity.csv",
        grid_rows,
    )

    dvh_rows: list[dict[str, object]] = []
    for grid, dose in final_grid_doses.items():
        x_values, y_values = dvh_curve(dose, mean_dose)
        for relative_dose, receiving in zip(x_values, y_values):
            dvh_rows.append(
                {
                    "grid": grid,
                    "dose_over_GTV_mean": relative_dose,
                    "GTV_volume_receiving_at_least_dose_percent": receiving,
                }
            )
    write_csv(args.output_dir / "gamma_co60_final_dvh.csv", dvh_rows)

    dose_values = np.asarray(
        [float(row["mean_dose_Gy_per_primary"]) for row in per_seed]
    )
    let_values = np.asarray([float(row["LETd_w_keV_um"]) for row in per_seed])
    dose_mean, dose_sd, dose_cv = sample_stats(dose_values)
    let_mean, let_sd, let_cv = sample_stats(let_values)
    final_report = next(row for row in grid_rows if row["grid"] == REPORT_GRID)
    summary: dict[str, object] = {
        "status": (
            "final_conditional_high_statistics_DVH_for_validated_incident_"
            "Co60_field_not_absolute_dose_per_decay"
        ),
        "physical_head_role": (
            "ROKUS-AM source, capsule and blinds validate field geometry and "
            "provide a separate integral LET estimate"
        ),
        "dvh_transport_role": (
            "independent photons in the already formed field at the rat "
            "surface; pooled voxel energy deposition"
        ),
        "number_of_independent_runs": len(per_seed),
        "pooled_histories": total_histories,
        "gtv_voxels": gtv_count,
        "gtv_volume_mm3": gtv_count * PX_MM * PY_MM * PZ_MM,
        "gtv_depth_min_mm_from_plus_y_surface": depth_interval[0],
        "gtv_depth_max_mm_from_plus_y_surface": depth_interval[1],
        "incident_field_radius_mm": args.field_radius_mm,
        "fine_nonzero_dose_fraction": float(np.mean(pooled_dose > 0.0)),
        "fine_positive_voxel_rel_SE_median": float(
            np.median(finite_rel_se)
        ),
        "fine_positive_voxel_rel_SE_p90": float(
            np.percentile(finite_rel_se, 90.0)
        ),
        "fine_positive_voxels_rel_SE_le_20pct_fraction": float(
            np.mean(finite_rel_se <= 0.20)
        ),
        "incident_field_mean_dose_Gy_per_primary": mean_dose / total_histories,
        "incident_field_seed_mean_dose_Gy_per_primary_mean": dose_mean,
        "incident_field_seed_mean_dose_Gy_per_primary_sample_SD": dose_sd,
        "incident_field_seed_mean_dose_Gy_per_primary_CV_percent": dose_cv,
        "incident_field_LETd_w_keV_um": pooled_let,
        "incident_field_seed_LETd_w_keV_um_mean": let_mean,
        "incident_field_seed_LETd_w_keV_um_sample_SD": let_sd,
        "incident_field_seed_LETd_w_keV_um_CV_percent": let_cv,
        "physical_head_1M_LETd_w_keV_um": args.physical_head_let,
        "LET_relative_difference_head_vs_incident_percent": (
            100.0 * (args.physical_head_let / pooled_let - 1.0)
        ),
        "report_grid_mm": "1.6x1.6x0.8",
        "report_voxels": final_grid_voxels[REPORT_GRID],
        "report_nonzero_volume_fraction": final_report[
            "nonzero_volume_fraction"
        ],
        "report_D2_over_Dmean": final_report["D2_over_Dmean"],
        "report_D10_over_Dmean": final_report["D10_over_Dmean"],
        "report_D50_over_Dmean": final_report["D50_over_Dmean"],
        "report_D90_over_Dmean": final_report["D90_over_Dmean"],
        "report_D95_over_Dmean": final_report["D95_over_Dmean"],
        "report_D98_over_Dmean": final_report["D98_over_Dmean"],
        "report_HI98": final_report["HI98"],
        "local_MCC_100mm_central_dose_Gy_per_60s_2023_10_24": (
            args.local_central_dose
        ),
        "normalisation": (
            "DVH is reported as D/Dmean; an experimental Gy map is obtained "
            "by scaling to the prescribed or measured GTV mean dose"
        ),
        "limitation": (
            "The high-statistics DVH does not reconstruct the number of "
            "source decays or a series-specific historical irradiation."
        ),
    }
    write_csv(args.output_dir / "gamma_co60_final_dvh_summary.csv", [summary])
    (args.output_dir / "gamma_co60_final_dvh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    MatplotlibConfigurator().apply_custom_styles()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    ax = axes[0, 0]
    fine_x, fine_y = dvh_curve(
        final_grid_doses["fine_0p4x0p4x0p2"],
        mean_dose,
    )
    report_x, report_y = dvh_curve(
        final_grid_doses[REPORT_GRID],
        mean_dose,
    )
    ax.plot(
        fine_x,
        fine_y,
        color="#9e9e9e",
        linewidth=1.3,
        label="исходные воксели 0,4×0,4×0,2 мм",
    )
    ax.plot(
        report_x,
        report_y,
        color="#d95f02",
        linewidth=2.4,
        label="отчётная сетка 1,6×1,6×0,8 мм",
    )
    metric_label_y = {"D50": 4.0, "D90": 12.0, "D10": 20.0}
    for metric, colour in (
        ("D50", "#1b9e77"),
        ("D90", "#e7298a"),
        ("D10", "#4c78a8"),
    ):
        value = float(final_report[f"{metric}_over_Dmean"])
        ax.axvline(value, color=colour, linestyle="--", linewidth=1.2)
        ax.text(
            value,
            metric_label_y[metric],
            f"{metric}={value:.3f}",
            rotation=90,
            va="bottom",
            ha="right",
            color=colour,
        )
    ax.set(
        title=f"Итоговая DVH GTVp, N={total_histories / 1e6:g} млн",
        xlabel=r"$D/D_{\mathrm{mean}}$",
        ylabel="Объём GTVp, получающий ≥ D, %",
        xlim=(0.0, min(1.8, float(np.percentile(fine_x, 99.5)))),
        ylim=(0.0, 101.0),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    checkpoint_colours = plt.cm.viridis(
        np.linspace(0.15, 0.90, len(progressive_dvh))
    )
    for colour, (run_count, (x_values, y_values)) in zip(
        checkpoint_colours,
        progressive_dvh.items(),
    ):
        histories = sum(
            int(row["histories"]) for row in per_seed[:run_count]
        )
        ax.plot(
            x_values,
            y_values,
            color=colour,
            linewidth=1.8 if run_count < len(per_seed) else 2.6,
            label=f"{histories / 1e6:g} млн",
        )
    ax.set(
        title="Сходимость DVH на отчётной сетке",
        xlabel=r"$D/D_{\mathrm{mean}}$",
        ylabel="Объём GTVp, получающий ≥ D, %",
        xlim=(0.0, 1.5),
        ylim=(0.0, 101.0),
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Объединено")

    ax = axes[1, 0]
    histories_million = np.asarray(
        [float(row["histories_pooled"]) / 1e6 for row in convergence_rows]
    )
    for metric, colour, marker in (
        ("report_D10_over_Dmean", "#4c78a8", "v"),
        ("report_D50_over_Dmean", "#1b9e77", "o"),
        ("report_D90_over_Dmean", "#e7298a", "s"),
        ("report_D95_over_Dmean", "#7570b3", "^"),
        ("report_D98_over_Dmean", "#e6ab02", "D"),
    ):
        ax.plot(
            histories_million,
            [float(row[metric]) for row in convergence_rows],
            color=colour,
            marker=marker,
            linewidth=1.7,
            label=metric.split("_")[1],
        )
    ax.set(
        title="Сходимость объёмных показателей",
        xlabel="Число первичных фотонов, млн",
        ylabel=r"$D_x/D_{\mathrm{mean}}$",
        xlim=(0.8, histories_million[-1] + 0.2),
        ylim=(0.0, 1.25),
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)

    ax = axes[1, 1]
    resolution_labels = [
        "0,4×0,4×0,2",
        "0,8×0,8×0,4",
        "1,2×1,2×0,6",
        "1,6×1,6×0,8",
    ]
    x_positions = np.arange(len(grid_rows))
    ax.plot(
        x_positions,
        [float(row["D10_over_Dmean"]) for row in grid_rows],
        "v-",
        color="#4c78a8",
        label=r"$D_{10}/D_{\mathrm{mean}}$",
    )
    ax.plot(
        x_positions,
        [float(row["D90_over_Dmean"]) for row in grid_rows],
        "o-",
        color="#e7298a",
        label=r"$D_{90}/D_{\mathrm{mean}}$",
    )
    ax.plot(
        x_positions,
        [float(row["D98_over_Dmean"]) for row in grid_rows],
        "s-",
        color="#e6ab02",
        label=r"$D_{98}/D_{\mathrm{mean}}$",
    )
    ax.set(
        title="Чувствительность к размеру отчётного элемента",
        xlabel="Размер элемента, мм",
        ylabel="Относительная доза",
        xticks=x_positions,
        xticklabels=resolution_labels,
        ylim=(0.5, 1.35),
    )
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle(
        "Co-60 / РОКУС-АМ: высокостатистическая фотонная DVH в GTVp",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"gamma_co60_final_dvh.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    figure, axis = plt.subplots(figsize=(9.4, 6.3))
    axis.plot(
        fine_x,
        fine_y,
        color="#9e9e9e",
        linewidth=1.4,
        alpha=0.85,
        label="исходные воксели 0,4×0,4×0,2 мм",
    )
    axis.plot(
        report_x,
        report_y,
        color="#d95f02",
        linewidth=3.0,
        label="отчётная сетка 1,6×1,6×0,8 мм",
    )
    metric_points = (
        ("D98", 98.0, "#e6ab02"),
        ("D95", 95.0, "#7570b3"),
        ("D90", 90.0, "#e7298a"),
        ("D50", 50.0, "#1b9e77"),
        ("D10", 10.0, "#4c78a8"),
    )
    annotation_offsets = {
        "D98": (-8, -18, "right", "top"),
        "D95": (12, 8, "left", "bottom"),
        "D90": (12, -9, "left", "top"),
        "D50": (10, -12, "left", "top"),
        "D10": (-8, 8, "right", "bottom"),
    }
    for label, volume, colour in metric_points:
        value = float(final_report[f"{label}_over_Dmean"])
        axis.plot(value, volume, "o", color=colour, markersize=6, zorder=4)
        dx, dy, horizontal, vertical = annotation_offsets[label]
        axis.annotate(
            f"{label}={value:.3f}",
            xy=(value, volume),
            xytext=(dx, dy),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            ha=horizontal,
            va=vertical,
        )
    axis.set(
        title=(
            "Co-60 / РОКУС-АМ: итоговая нормированная DVH GTVp\n"
            f"10 независимых запусков, N={total_histories / 1e6:g} млн"
        ),
        xlabel=r"$D/D_{\mathrm{mean}}$",
        ylabel="Объём GTVp, получающий ≥ D, %",
        xlim=(0.4, 1.4),
        ylim=(0.0, 101.0),
    )
    axis.set_yticks(np.arange(0.0, 101.0, 10.0))
    axis.grid(which="major", alpha=0.28)
    axis.legend(loc="lower left")
    axis.text(
        0.98,
        0.04,
        (
            rf"$D_2/D_{{mean}}={float(final_report['D2_over_Dmean']):.3f}$"
            "\n"
            rf"$HI_{{98}}={float(final_report['HI98']):.3f}$"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#bdbdbd",
            "alpha": 0.9,
        },
    )
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.output_dir / f"gamma_co60_final_dvh_curve.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tags", nargs="+", required=True)
    parser.add_argument(
        "--dvh-checkpoints",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 10],
    )
    parser.add_argument("--field-radius-mm", type=float, default=14.4)
    parser.add_argument(
        "--physical-head-let",
        type=float,
        default=0.3344631694091014,
    )
    parser.add_argument(
        "--local-central-dose",
        type=float,
        default=0.5976791778331468,
    )
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    args = parser.parse_args()
    summary = analyse(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
