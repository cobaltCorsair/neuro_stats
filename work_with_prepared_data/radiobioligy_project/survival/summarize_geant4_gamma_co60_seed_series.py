"""Pool equal-size Co-60 rat runs and quantify random-seed variability.

This is a Monte Carlo repeatability check.  Three independent runs use the
same source, phantom, physics and production cuts; only random seeds differ.
The pooled voxel map is equivalent to the sum of all simulated histories.
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


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_stats(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
    cv = 100.0 * sd / mean if mean != 0.0 else math.nan
    return mean, sd, cv


def dvh_curve(dose: np.ndarray, mean_dose: float) -> tuple[np.ndarray, np.ndarray]:
    sorted_dose = np.sort(dose)
    count = sorted_dose.size
    volume_percent = 100.0 * (count - np.arange(count)) / count
    return sorted_dose / mean_dose, volume_percent


def analyse(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, gtv_ids, gtv_lookup, depth_interval = gtv_geometry(
        args.ct_dir,
        args.rtstruct,
    )
    gtv_count = gtv_ids.size
    pooled_energy = np.zeros(gtv_count, dtype=float)
    pooled_energy2 = np.zeros(gtv_count, dtype=float)
    pooled_let_base = np.zeros(gtv_count, dtype=float)
    pooled_dose = np.zeros(gtv_count, dtype=float)
    per_seed: list[dict[str, object]] = []

    for run_tag in args.run_tags:
        run_root = args.runs_root / run_tag
        summary = read_single_csv(run_root / "gtv_analysis" / "gtv_summary.csv")
        metadata = json.loads(
            (run_root / "through" / "run_metadata.json").read_text(
                encoding="utf-8-sig"
            )
        )
        histories = int(summary["histories"])
        if histories != args.histories_per_seed:
            raise RuntimeError(
                f"{run_tag}: expected {args.histories_per_seed} histories, "
                f"found {histories}"
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
        per_seed.append(
            {
                "run_tag": run_tag,
                "seed1": metadata["seed1"],
                "seed2": metadata["seed2"],
                "histories": histories,
                "gtv_mean_dose_Gy_per_primary": float(
                    summary["gtv_mean_dose_Gy_per_primary"]
                ),
                "gtv_LETd_w_keV_um": float(summary["gtv_LETd_w_keV_um"]),
                "gtv_nonzero_dose_fraction": float(
                    summary["gtv_nonzero_dose_fraction"]
                ),
                "gtv_fine_D50_over_Dmean": float(
                    summary["gtv_D50_over_Dmean"]
                ),
                "gtv_fine_D90_over_Dmean": float(
                    summary["gtv_D90_over_Dmean"]
                ),
                "gtv_report_D50_over_Dmean_1p6x1p6x0p8": float(
                    summary["gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8"]
                ),
                "gtv_report_D90_over_Dmean_1p6x1p6x0p8": float(
                    summary["gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"]
                ),
            }
        )
    write_csv(args.output_dir / "gamma_co60_1M_per_seed.csv", per_seed)

    total_histories = args.histories_per_seed * len(per_seed)
    if total_histories % 1_000_000 == 0:
        pooled_label = f"{total_histories // 1_000_000}M"
    else:
        pooled_label = str(total_histories)
    mean_dose_per_run = float(np.mean(pooled_dose))
    mean_dose_per_primary = mean_dose_per_run / total_histories
    total_energy = float(np.sum(pooled_energy))
    total_let_base = float(np.sum(pooled_let_base))
    pooled_let = total_let_base / total_energy
    fine_d50 = dose_at_volume(pooled_dose, 50.0)
    fine_d90 = dose_at_volume(pooled_dose, 90.0)
    report_dose, report_voxels = locally_aggregate_gtv_dose(
        pooled_dose,
        gtv_ids,
        (4, 4, 4),
    )
    report_d50 = dose_at_volume(report_dose, 50.0)
    report_d90 = dose_at_volume(report_dose, 90.0)
    pooled_rel_se = np.full(gtv_count, np.nan, dtype=float)
    pooled_positive_energy = pooled_energy > 0.0
    if total_histories > 1:
        variance_numerator = np.maximum(
            total_histories * pooled_energy2 - pooled_energy * pooled_energy,
            0.0,
        )
        pooled_rel_se[pooled_positive_energy] = (
            np.sqrt(
                variance_numerator[pooled_positive_energy]
                / (total_histories - 1.0)
            )
            / pooled_energy[pooled_positive_energy]
        )
    finite_pooled_rel_se = pooled_rel_se[np.isfinite(pooled_rel_se)]

    dose_values = np.asarray(
        [row["gtv_mean_dose_Gy_per_primary"] for row in per_seed],
        dtype=float,
    )
    let_values = np.asarray(
        [row["gtv_LETd_w_keV_um"] for row in per_seed],
        dtype=float,
    )
    report_d50_values = np.asarray(
        [
            row["gtv_report_D50_over_Dmean_1p6x1p6x0p8"]
            for row in per_seed
        ],
        dtype=float,
    )
    report_d90_values = np.asarray(
        [
            row["gtv_report_D90_over_Dmean_1p6x1p6x0p8"]
            for row in per_seed
        ],
        dtype=float,
    )
    dose_mean, dose_sd, dose_cv = sample_stats(dose_values)
    let_mean, let_sd, let_cv = sample_stats(let_values)
    d50_mean, d50_sd, d50_cv = sample_stats(report_d50_values)
    d90_mean, d90_sd, d90_cv = sample_stats(report_d90_values)

    summary: dict[str, object] = {
        "status": "internal_monte_carlo_seed_repeatability_not_external_validation",
        "number_of_independent_seeds": len(per_seed),
        "histories_per_seed": args.histories_per_seed,
        "pooled_histories": total_histories,
        "gtv_voxels": gtv_count,
        "gtv_volume_mm3": gtv_count * PX_MM * PY_MM * PZ_MM,
        "gtv_depth_min_mm_from_plus_y_surface": depth_interval[0],
        "gtv_depth_max_mm_from_plus_y_surface": depth_interval[1],
        "seed_mean_dose_Gy_per_primary_mean": dose_mean,
        "seed_mean_dose_Gy_per_primary_sample_SD": dose_sd,
        "seed_mean_dose_Gy_per_primary_CV_percent": dose_cv,
        "seed_LETd_w_keV_um_mean": let_mean,
        "seed_LETd_w_keV_um_sample_SD": let_sd,
        "seed_LETd_w_keV_um_CV_percent": let_cv,
        "seed_report_D50_over_Dmean_mean": d50_mean,
        "seed_report_D50_over_Dmean_sample_SD": d50_sd,
        "seed_report_D50_over_Dmean_CV_percent": d50_cv,
        "seed_report_D90_over_Dmean_mean": d90_mean,
        "seed_report_D90_over_Dmean_sample_SD": d90_sd,
        "seed_report_D90_over_Dmean_CV_percent": d90_cv,
        "pooled_gtv_mean_dose_Gy_per_primary": mean_dose_per_primary,
        "pooled_gtv_LETd_w_keV_um": pooled_let,
        "pooled_gtv_nonzero_dose_fraction": float(
            np.mean(pooled_dose > 0.0)
        ),
        "pooled_positive_voxel_rel_SE_median": float(
            np.median(finite_pooled_rel_se)
        ),
        "pooled_positive_voxel_rel_SE_p90": float(
            np.percentile(finite_pooled_rel_se, 90.0)
        ),
        "pooled_positive_voxels_rel_SE_le_50pct_fraction": float(
            np.mean(finite_pooled_rel_se <= 0.50)
        ),
        "pooled_positive_voxels_rel_SE_le_20pct_fraction": float(
            np.mean(finite_pooled_rel_se <= 0.20)
        ),
        "pooled_fine_D50_over_Dmean": fine_d50 / mean_dose_per_run,
        "pooled_fine_D90_over_Dmean": fine_d90 / mean_dose_per_run,
        "pooled_report_voxels_1p6x1p6x0p8": report_voxels,
        "pooled_report_D50_over_Dmean_1p6x1p6x0p8": (
            report_d50 / mean_dose_per_run
        ),
        "pooled_report_D90_over_Dmean_1p6x1p6x0p8": (
            report_d90 / mean_dose_per_run
        ),
    }
    write_csv(args.output_dir / "gamma_co60_1M_seed_summary.csv", [summary])
    (args.output_dir / "gamma_co60_1M_seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fine_x, fine_y = dvh_curve(pooled_dose, mean_dose_per_run)
    report_x, report_y = dvh_curve(report_dose, mean_dose_per_run)
    dvh_rows: list[dict[str, object]] = []
    for resolution, x_values, y_values in (
        ("fine_0p4x0p4x0p2", fine_x, fine_y),
        ("report_1p6x1p6x0p8", report_x, report_y),
    ):
        for relative_dose, volume_percent in zip(x_values, y_values):
            dvh_rows.append(
                {
                    "resolution": resolution,
                    "dose_over_gtv_mean": relative_dose,
                    "volume_receiving_at_least_dose_percent": volume_percent,
                }
            )
    write_csv(
        args.output_dir / f"gamma_co60_pooled_{pooled_label}_dvh.csv",
        dvh_rows,
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
    x = np.arange(len(per_seed))
    labels = []
    for index, row in enumerate(per_seed, start=1):
        tag_parts = str(row["run_tag"]).split("_")
        seed_part = next(
            (
                part
                for part in tag_parts
                if part.lower().startswith("seed")
            ),
            "",
        )
        labels.append(
            seed_part[4:] if len(seed_part) > 4 else chr(64 + index)
        )

    ax = axes[0, 0]
    ax.plot(x, dose_values * 1e13, "o", color="#d95f02", markersize=8)
    ax.axhline(dose_mean * 1e13, color="#333333", linestyle="--")
    ax.fill_between(
        [-0.4, len(per_seed) - 0.6],
        (dose_mean - dose_sd) * 1e13,
        (dose_mean + dose_sd) * 1e13,
        color="#d95f02",
        alpha=0.15,
    )
    ax.set(
        title=f"Средняя доза в GTVp; CV = {dose_cv:.2f}%",
        ylabel=r"$D/N$, $10^{-13}$ Гр/первичную частицу",
        xticks=x,
        xticklabels=labels,
    )
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.plot(x, let_values, "o", color="#7570b3", markersize=8)
    ax.axhline(let_mean, color="#333333", linestyle="--")
    ax.fill_between(
        [-0.4, len(per_seed) - 0.6],
        let_mean - let_sd,
        let_mean + let_sd,
        color="#7570b3",
        alpha=0.15,
    )
    ax.set(
        title=f"ЛПЭ в GTVp; CV = {let_cv:.2f}%",
        ylabel=r"$LET_{d,w}$, кэВ/мкм",
        xticks=x,
        xticklabels=labels,
    )
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.plot(
        x,
        report_d50_values,
        "o-",
        color="#1b9e77",
        label=r"$D_{50}/D_{mean}$",
    )
    ax.plot(
        x,
        report_d90_values,
        "o-",
        color="#e7298a",
        label=r"$D_{90}/D_{mean}$",
    )
    ax.set(
        title="Показатели DVH на отчётной сетке",
        ylabel="Относительная доза",
        xticks=x,
        xticklabels=labels,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        fine_x,
        fine_y,
        color="#9e9e9e",
        linewidth=1.4,
        label="исходные воксели 0,4×0,4×0,2 мм",
    )
    ax.plot(
        report_x,
        report_y,
        color="#d95f02",
        linewidth=2.2,
        label="отчётная сетка 1,6×1,6×0,8 мм",
    )
    ax.set(
        title=f"Объединённая DVH, N = {total_histories:,}",
        xlabel=r"$D/D_{mean}$",
        ylabel="Объём GTVp, получающий ≥ D, %",
        xlim=(0.0, min(2.5, float(np.percentile(fine_x, 99.5)))),
        ylim=(0.0, 101.0),
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle(
        "Co-60 в воксельном фантоме крысы: воспроизводимость по случайным seed",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"gamma_co60_1M_seed_repeatability.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tags", nargs="+", required=True)
    parser.add_argument("--histories-per-seed", type=int, default=1_000_000)
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    args = parser.parse_args()
    summary = analyse(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
