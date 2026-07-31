"""Compare parallel and ideal divergent Co-60 fields in the rat phantom.

The ideal point-source cases project a uniform circle or rectangle onto the
isocentre plane.  They quantify sensitivity to source divergence, an air path
and field shape.  They are not treatment-head reconstructions.
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
    NX,
    NY,
    NZ,
    PX_MM,
    PY_MM,
    PZ_MM,
    gtv_geometry,
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


def geometry_coordinates(
    ct_dir: Path,
    rtstruct: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, ids, _, _ = gtv_geometry(ct_dir, rtstruct)
    ix = ids % NX
    iy = (ids // NX) % NY
    iz = ids // (NX * NY)
    x_mm = (ix + 0.5 - NX / 2.0) * PX_MM
    y_mm = (iy + 0.5 - NY / 2.0) * PY_MM
    z_mm = (iz + 0.5 - NZ / 2.0) * PZ_MM
    return x_mm, y_mm, z_mm


def ideal_field_coverage(
    metadata: dict[str, object],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
) -> float:
    source_y = float(metadata["source_axis_distance_mm"])
    target_y = float(metadata["target_plane_y_mm"])
    scale = (source_y - y_mm) / (source_y - target_y)
    if metadata["field_shape"] == "circle":
        radius = float(metadata["field_radius_mm"]) * scale
        inside = x_mm * x_mm + z_mm * z_mm <= radius * radius
    else:
        half_x = float(metadata["field_half_x_mm"]) * scale
        half_z = float(metadata["field_half_z_mm"]) * scale
        inside = (np.abs(x_mm) <= half_x) & (np.abs(z_mm) <= half_z)
    return float(np.mean(inside))


def analyse(args: argparse.Namespace) -> list[dict[str, object]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_mm, y_mm, z_mm = geometry_coordinates(args.ct_dir, args.rtstruct)
    baseline = read_single_csv(args.baseline_summary)
    per_seed_path = args.baseline_summary.parent / "gamma_co60_1M_per_seed.csv"
    with per_seed_path.open(encoding="utf-8-sig", newline="") as handle:
        baseline_seed_rows = list(csv.DictReader(handle))
    if not baseline_seed_rows:
        raise RuntimeError(f"No baseline seed rows in {per_seed_path}")
    baseline_nonzero = float(
        np.mean(
            [
                float(row["gtv_nonzero_dose_fraction"])
                for row in baseline_seed_rows
            ]
        )
    )
    baseline_dose = float(
        baseline["seed_mean_dose_Gy_per_primary_mean"]
    )
    rows: list[dict[str, object]] = [
        {
            "case": "parallel_surface_mean_of_three_1M",
            "display_label": "параллельное\n3×1 млн",
            "status": "mean of three equal-size entrance-plane runs",
            "histories": int(baseline["histories_per_seed"]),
            "field_shape": "circle",
            "transport_medium": "not transported upstream of entrance",
            "source_y_mm": 25.7,
            "target_y_mm": 25.7,
            "geometric_gtv_coverage_fraction": float(
                np.mean(x_mm * x_mm + z_mm * z_mm <= 14.4**2)
            ),
            "gtv_mean_dose_Gy_per_primary": baseline_dose,
            "dose_relative_to_parallel": 1.0,
            "gtv_LETd_w_keV_um": float(
                baseline["seed_LETd_w_keV_um_mean"]
            ),
            "gtv_nonzero_dose_fraction": baseline_nonzero,
            "report_D50_over_Dmean": float(
                baseline["seed_report_D50_over_Dmean_mean"]
            ),
            "report_D90_over_Dmean": float(
                baseline["seed_report_D90_over_Dmean_mean"]
            ),
        }
    ]

    for run_tag in args.run_tags:
        run_root = args.runs_root / run_tag
        summary = read_single_csv(run_root / "gtv_analysis" / "gtv_summary.csv")
        metadata = json.loads(
            (run_root / "through" / "run_metadata.json").read_text(
                encoding="utf-8-sig"
            )
        )
        shape = str(metadata["field_shape"])
        medium = str(metadata["transport_medium_outside_phantom"])
        shape_label = "круг" if shape == "circle" else "квадрат"
        medium_label = "вакуум" if medium == "vacuum" else "воздух"
        dose = float(summary["gtv_mean_dose_Gy_per_primary"])
        rows.append(
            {
                "case": run_tag,
                "display_label": f"точка, {shape_label}\n{medium_label}",
                "status": metadata["status"],
                "histories": int(summary["histories"]),
                "field_shape": shape,
                "transport_medium": medium,
                "source_y_mm": metadata["source_axis_distance_mm"],
                "target_y_mm": metadata["target_plane_y_mm"],
                "geometric_gtv_coverage_fraction": ideal_field_coverage(
                    metadata,
                    x_mm,
                    y_mm,
                    z_mm,
                ),
                "gtv_mean_dose_Gy_per_primary": dose,
                "dose_relative_to_parallel": dose / baseline_dose,
                "gtv_LETd_w_keV_um": float(
                    summary["gtv_LETd_w_keV_um"]
                ),
                "gtv_nonzero_dose_fraction": float(
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

    write_csv(args.output_dir / "gamma_co60_geometry_sensitivity.csv", rows)
    (args.output_dir / "gamma_co60_geometry_sensitivity.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
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
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.1))
    labels = [str(row["display_label"]) for row in rows]
    positions = np.arange(len(rows))
    colors = ("#6b6b6b", "#7570b3", "#1b9e77", "#d95f02")

    ax = axes[0, 0]
    values = [100.0 * float(row["dose_relative_to_parallel"]) for row in rows]
    ax.bar(positions, values, color=colors[: len(rows)])
    ax.axhline(100.0, color="#222222", linestyle="--", linewidth=1.3)
    ax.set(
        title="Средняя доза на первичный фотон",
        ylabel="Относительно параллельного поля, %",
        xticks=positions,
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    values = [float(row["gtv_LETd_w_keV_um"]) for row in rows]
    ax.plot(positions, values, "o-", color="#7570b3", linewidth=2.0)
    ax.set(
        title=r"Дозо-взвешенная ЛПЭ в GTVp",
        ylabel=r"$LET_{d,w}$, кэВ/мкм",
        xticks=positions,
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    d50 = [float(row["report_D50_over_Dmean"]) for row in rows]
    d90 = [float(row["report_D90_over_Dmean"]) for row in rows]
    ax.plot(
        positions,
        d50,
        "o-",
        color="#1b9e77",
        linewidth=2.0,
        label=r"$D_{50}/D_{mean}$",
    )
    ax.plot(
        positions,
        d90,
        "o-",
        color="#e7298a",
        linewidth=2.0,
        label=r"$D_{90}/D_{mean}$",
    )
    ax.set(
        title="DVH на отчётной сетке 1,6×1,6×0,8 мм",
        ylabel="Относительная доза",
        xticks=positions,
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    projected = np.unique(
        np.column_stack(
            (
                np.round(x_mm, 4),
                np.round(z_mm, 4),
            )
        ),
        axis=0,
    )
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=7,
        color="#bdbdbd",
        alpha=0.75,
        label="проекция GTVp",
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        14.4 * np.cos(theta),
        14.4 * np.sin(theta),
        color="#1b9e77",
        linewidth=2.0,
        label="круг r=14,4 мм",
    )
    square_x = np.array([-14.4, 14.4, 14.4, -14.4, -14.4])
    square_z = np.array([-14.4, -14.4, 14.4, 14.4, -14.4])
    ax.plot(
        square_x,
        square_z,
        color="#d95f02",
        linewidth=2.0,
        label="квадрат 28,8×28,8 мм",
    )
    ax.set(
        title="Идеальные границы поля в изоцентре",
        xlabel="x, мм",
        ylabel="z, мм",
        xlim=(-17.0, 17.0),
        ylim=(-17.0, 17.0),
        aspect="equal",
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="best")

    fig.suptitle(
        "Co-60: чувствительность к расходимости, воздушному пути и форме поля\n"
        "идеальные проектируемые поля; головка и физическая диафрагма не моделируются",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"gamma_co60_geometry_sensitivity.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--run-tags", nargs="+", required=True)
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    args = parser.parse_args()
    rows = analyse(args)
    # ASCII-safe console output avoids failures on Windows cp1251 terminals;
    # the saved JSON remains UTF-8 with readable Russian labels.
    print(json.dumps(rows, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
