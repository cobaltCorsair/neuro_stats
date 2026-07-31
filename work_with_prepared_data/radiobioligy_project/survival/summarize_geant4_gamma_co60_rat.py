"""Summarise Co-60 convergence runs in the reduced rat phantom.

The transport runner stores one numerical GTV summary per run.  This script
collects those summaries without rereading the large voxel maps and produces
an auditable convergence table and a four-panel figure.  Runs with different
numbers of histories are treated as a convergence series, not as equal-size
seed replicates.
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

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


DEFAULT_RUN_ROOT = Path(
    r"C:\dev\dissertation\task4_5\scoring_v2_gamma_co60_rat"
)
DEFAULT_OUTPUT_DIR = (
    Path(r"C:\dev\dissertation\task4_5\outputs")
    / "geant4_livermore_20260724"
    / "gamma_co60_rat"
)

FIELDS = (
    "run_tag",
    "histories",
    "production_cuts_um",
    "physics_code",
    "em_model",
    "gtv_mean_dose_Gy_per_primary",
    "gtv_LETd_w_keV_um",
    "gtv_nonzero_dose_fraction",
    "gtv_positive_voxel_rel_se_median",
    "gtv_positive_voxel_rel_se_p90",
    "gtv_positive_voxels_rel_se_le_50pct_fraction",
    "gtv_positive_voxels_rel_se_le_20pct_fraction",
    "gtv_aggregated_nonzero_volume_fraction_1p6x1p6x0p8",
    "gtv_aggregated_D2_over_Dmean_1p6x1p6x0p8",
    "gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8",
    "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8",
    "gtv_aggregated_D95_over_Dmean_1p6x1p6x0p8",
    "gtv_aggregated_D98_over_Dmean_1p6x1p6x0p8",
)


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def collect_runs(run_root: Path) -> list[dict[str, float | int | str]]:
    runs: list[dict[str, float | int | str]] = []
    for summary_path in sorted(run_root.glob("*/gtv_analysis/gtv_summary.csv")):
        run_dir = summary_path.parents[1]
        metadata_path = run_dir / "through" / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != 1:
            raise RuntimeError(
                f"Expected one row in {summary_path}, found {len(source_rows)}"
            )
        source = source_rows[0]
        row: dict[str, float | int | str] = {
            "run_tag": str(metadata["run_tag"]),
            "histories": int(metadata["histories"]),
            "production_cuts_um": str(metadata["production_cuts_um"]),
            "physics_code": int(metadata["physics_code"]),
            "em_model": str(metadata["em_model"]),
        }
        for field in FIELDS[5:]:
            row[field] = as_float(source, field)
        runs.append(row)
    return sorted(
        runs,
        key=lambda item: (
            str(item["production_cuts_um"]),
            str(item["em_model"]),
            int(item["histories"]),
            str(item["run_tag"]),
        ),
    )


def write_table(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(
    path_stem: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    baseline = [
        row
        for row in rows
        if (
            str(row["production_cuts_um"]).startswith("700")
            and str(row["em_model"]) == "G4EmLivermorePhysics"
        )
    ]
    if not baseline:
        return

    histories = np.asarray([int(row["histories"]) for row in baseline])
    dose = np.asarray(
        [float(row["gtv_mean_dose_Gy_per_primary"]) for row in baseline]
    )
    letd = np.asarray([float(row["gtv_LETd_w_keV_um"]) for row in baseline])
    fine_coverage = 100.0 * np.asarray(
        [float(row["gtv_nonzero_dose_fraction"]) for row in baseline]
    )
    report_coverage = 100.0 * np.asarray(
        [
            float(
                row[
                    "gtv_aggregated_nonzero_volume_fraction_1p6x1p6x0p8"
                ]
            )
            for row in baseline
        ]
    )
    d50 = np.asarray(
        [
            float(
                row["gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8"]
            )
            for row in baseline
        ]
    )
    d90 = np.asarray(
        [
            float(
                row["gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"]
            )
            for row in baseline
        ]
    )

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            }
        )
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12.2, 8.6),
        )
        ax_dose, ax_let, ax_coverage, ax_dvh = axes.ravel()

        reference_dose = dose[-1]
        ax_dose.plot(
            histories,
            dose / reference_dose,
            marker="o",
            linewidth=2.0,
            color="#2166ac",
        )
        ax_dose.axhline(1.0, color="0.45", linestyle="--", linewidth=1.0)
        ax_dose.set(
            xscale="log",
            xlabel="Число первичных фотонов",
            ylabel="Доза на первичный / значение\nмаксимального прогона",
            title="Сходимость средней дозы GTVp",
        )

        ax_let.plot(
            histories,
            letd,
            marker="o",
            linewidth=2.0,
            color="#762a83",
        )
        ax_let.set(
            xscale="log",
            xlabel="Число первичных фотонов",
            ylabel=r"$LET_{D,w}$, кэВ/мкм",
            title="Сходимость дозо-взвешенной ЛПЭ",
        )

        ax_coverage.plot(
            histories,
            fine_coverage,
            marker="o",
            linewidth=1.8,
            label="исходная сетка 0,4×0,4×0,2 мм",
        )
        ax_coverage.plot(
            histories,
            report_coverage,
            marker="s",
            linewidth=2.0,
            label="отчётная сетка 1,6×1,6×0,8 мм",
        )
        ax_coverage.set(
            xscale="log",
            xlabel="Число первичных фотонов",
            ylabel="Элементы с ненулевой дозой, %",
            title="Статистическое покрытие GTVp",
            ylim=(0.0, 103.0),
        )
        ax_coverage.legend(frameon=True)

        ax_dvh.plot(
            histories,
            d50,
            marker="o",
            linewidth=2.0,
            label=r"$D_{50}/D_{\mathrm{mean}}$",
        )
        ax_dvh.plot(
            histories,
            d90,
            marker="s",
            linewidth=2.0,
            label=r"$D_{90}/D_{\mathrm{mean}}$",
        )
        ax_dvh.set(
            xscale="log",
            xlabel="Число первичных фотонов",
            ylabel="Относительная доза",
            title="Сходимость DVH на отчётной сетке",
            ylim=(0.0, max(1.05, float(np.nanmax(d50)) * 1.08)),
        )
        ax_dvh.legend(frameon=True)

        for axis in axes.ravel():
            axis.grid(alpha=0.24)
        fig.suptitle(
            "Сходимость расчёта поля Co-60 в воксельном фантоме крысы",
            fontsize=15,
            y=0.99,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(path_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def write_summary(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    baseline = [
        row
        for row in rows
        if (
            str(row["production_cuts_um"]).startswith("700")
            and str(row["em_model"]) == "G4EmLivermorePhysics"
        )
    ]
    largest = max(baseline, key=lambda row: int(row["histories"]))
    payload = {
        "status": "convergence_series_not_equal_size_seed_replication",
        "number_of_runs": len(rows),
        "baseline_runs": len(baseline),
        "largest_run_tag": largest["run_tag"],
        "largest_histories": largest["histories"],
        "largest_gtv_mean_dose_Gy_per_primary": largest[
            "gtv_mean_dose_Gy_per_primary"
        ],
        "largest_gtv_LETd_w_keV_um": largest["gtv_LETd_w_keV_um"],
        "largest_gtv_D50_over_Dmean_report_grid": largest[
            "gtv_aggregated_D50_over_Dmean_1p6x1p6x0p8"
        ],
        "largest_gtv_D90_over_Dmean_report_grid": largest[
            "gtv_aggregated_D90_over_Dmean_1p6x1p6x0p8"
        ],
        "note": (
            "Different history counts quantify numerical convergence. "
            "ROI uncertainty still requires equal-size independent seeds."
        ),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = collect_runs(args.run_root)
    if not rows:
        raise RuntimeError(f"No completed Co-60 runs found in {args.run_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.output_dir / "gamma_co60_convergence.csv", rows)
    write_summary(args.output_dir / "gamma_co60_convergence.json", rows)
    plot_convergence(
        args.output_dir / "gamma_co60_convergence",
        rows,
    )
    print(f"runs={len(rows)}; output={args.output_dir}")


if __name__ == "__main__":
    main()
