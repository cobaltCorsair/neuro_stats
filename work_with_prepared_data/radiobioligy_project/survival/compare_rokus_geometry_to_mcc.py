"""Compare the published ROKUS source/aperture geometry with local MCC data.

This is a primary-ray geometric calculation.  It samples an emitting volume
with diameter and height 20 mm, projects rays through both faces of the
published 244--304 mm collimation-blind interval, and scores them in the
750-mm measurement plane.  Head scatter and detector response are not
included; those remain Geant4 and measurement effects, respectively.
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
from scipy.ndimage import gaussian_filter1d


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


def read_profiles(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    profiles: dict[float, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            field = float(row["field_nominal_mm"])
            profiles.setdefault(field, []).append(
                (
                    float(row["crossplane_mm"]),
                    float(row["relative_dose"]),
                )
            )
    return {
        field: (
            np.asarray([item[0] for item in rows]),
            np.asarray([item[1] for item in rows]),
        )
        for field, rows in profiles.items()
    }


def read_metrics(path: Path) -> dict[float, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {float(row["field_nominal_mm"]): row for row in rows}


def crossing(distance: np.ndarray, profile: np.ndarray, level: float) -> float:
    peak_index = int(np.argmax(profile))
    candidates = np.flatnonzero(profile[peak_index:] <= level) + peak_index
    if candidates.size == 0:
        return np.nan
    index = int(candidates[0])
    x0, x1 = distance[index - 1 : index + 1]
    y0, y1 = profile[index - 1 : index + 1]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def profile_metrics(axis: np.ndarray, profile: np.ndarray) -> dict[str, float]:
    positive = axis >= 0.0
    distance = axis[positive]
    right = profile[positive]
    left = profile[::-1][positive]
    symmetric = 0.5 * (left + right)
    x80 = crossing(distance, symmetric, 0.8)
    x50 = crossing(distance, symmetric, 0.5)
    x20 = crossing(distance, symmetric, 0.2)
    return {
        "x80_mm": x80,
        "x50_mm": x50,
        "x20_mm": x20,
        "full_width_50_percent_mm": 2.0 * x50,
        "penumbra_80_20_mm": x20 - x80,
    }


def simulate_profile(
    field_mm: float,
    *,
    histories: int,
    seed: int,
    aperture_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    work_surface_y = 750.0
    active_front_y = 751.25
    active_height = 20.0
    source_radius = 10.0
    upstream_y = work_surface_y - 244.0
    downstream_y = work_surface_y - 304.0
    target_y = 0.0
    upstream_half = (
        aperture_scale * 0.5 * field_mm * 244.0 / 750.0
    )
    downstream_half = (
        aperture_scale * 0.5 * field_mm * 304.0 / 750.0
    )
    edges = np.linspace(-130.125, 130.125, 1042)
    histogram = np.zeros(edges.size - 1, dtype=np.int64)
    accepted = 0
    rng = np.random.default_rng(seed + int(field_mm))
    remaining = histories
    while remaining:
        count = min(500_000, remaining)
        radius = source_radius * np.sqrt(rng.random(count))
        angle = 2.0 * np.pi * rng.random(count)
        source_x = radius * np.cos(angle)
        source_z = radius * np.sin(angle)
        source_y = active_front_y + active_height * rng.random(count)
        target_x = rng.uniform(-130.0, 130.0, count)
        target_z = rng.uniform(-1.25, 1.25, count)

        def at_plane(plane_y: float) -> tuple[np.ndarray, np.ndarray]:
            fraction = (source_y - plane_y) / (source_y - target_y)
            return (
                source_x + (target_x - source_x) * fraction,
                source_z + (target_z - source_z) * fraction,
            )

        up_x, up_z = at_plane(upstream_y)
        down_x, down_z = at_plane(downstream_y)
        passed = (
            (np.abs(up_x) <= upstream_half)
            & (np.abs(up_z) <= upstream_half)
            & (np.abs(down_x) <= downstream_half)
            & (np.abs(down_z) <= downstream_half)
        )
        histogram += np.histogram(target_x[passed], bins=edges)[0]
        accepted += int(np.sum(passed))
        remaining -= count

    axis = 0.5 * (edges[:-1] + edges[1:])
    profile = gaussian_filter1d(histogram.astype(float), sigma=4.0)
    profile /= float(np.mean(profile[np.abs(axis) <= 5.0]))
    return axis, profile, accepted / histories


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(args: argparse.Namespace) -> list[dict[str, object]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measured_profiles = read_profiles(args.measured_profiles)
    measured_metrics = read_metrics(args.measured_metrics)
    comparison_rows: list[dict[str, object]] = []
    predicted_profiles: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    profile_rows: list[dict[str, object]] = []

    for field in sorted(measured_profiles):
        axis, prediction, accepted_fraction = simulate_profile(
            field,
            histories=args.histories,
            seed=args.seed,
            aperture_scale=args.aperture_scale,
        )
        predicted_profiles[field] = (axis, prediction)
        predicted = profile_metrics(axis, prediction)
        measured_axis, measured = measured_profiles[field]
        predicted_on_measured = np.interp(measured_axis, axis, prediction)
        comparison_mask = measured >= 0.02
        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        predicted_on_measured[comparison_mask]
                        - measured[comparison_mask]
                    )
                    ** 2
                )
            )
        )
        measured_row = measured_metrics[field]
        comparison_rows.append(
            {
                "field_nominal_mm": field,
                "geometric_histories": args.histories,
                "accepted_primary_fraction": accepted_fraction,
                "measured_width50_mm": float(
                    measured_row["full_width_50_percent_mm"]
                ),
                "predicted_width50_mm": predicted[
                    "full_width_50_percent_mm"
                ],
                "width50_difference_mm": (
                    predicted["full_width_50_percent_mm"]
                    - float(measured_row["full_width_50_percent_mm"])
                ),
                "measured_penumbra80_20_mm": float(
                    measured_row["penumbra_80_20_mm"]
                ),
                "predicted_penumbra80_20_mm": predicted[
                    "penumbra_80_20_mm"
                ],
                "penumbra_difference_mm": (
                    predicted["penumbra_80_20_mm"]
                    - float(measured_row["penumbra_80_20_mm"])
                ),
                "profile_RMSE_for_measured_dose_ge_2pct": rmse,
            }
        )
        for coordinate, value in zip(axis, prediction):
            profile_rows.append(
                {
                    "field_nominal_mm": field,
                    "crossplane_mm": coordinate,
                    "predicted_primary_fluence_relative": value,
                }
            )

    write_csv(
        args.output_dir / "rokus_geometry_vs_mcc_metrics.csv",
        comparison_rows,
    )
    write_csv(
        args.output_dir / "rokus_geometric_primary_profiles.csv",
        profile_rows,
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
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), sharex=True)
    for ax, field in zip(axes.flat, sorted(measured_profiles)):
        measured_axis, measured = measured_profiles[field]
        predicted_axis, predicted = predicted_profiles[field]
        row = next(
            item
            for item in comparison_rows
            if float(item["field_nominal_mm"]) == field
        )
        ax.plot(
            measured_axis,
            measured,
            color="#1b9e77",
            linewidth=2.0,
            label="OCTAVIUS",
        )
        ax.plot(
            predicted_axis,
            predicted,
            color="#d95f02",
            linewidth=1.8,
            label="геометрия источника и шторок",
        )
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=0.9)
        ax.set(
            title=(
                f"{field / 10:.0f}×{field / 10:.0f} см; "
                f"ΔW50={float(row['width50_difference_mm']):+.1f} мм; "
                f"ΔP={float(row['penumbra_difference_mm']):+.1f} мм"
            ),
            xlim=(-130.0, 130.0),
            ylim=(-0.03, 1.10),
        )
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0, 0].set_ylabel("Относительная доза / первичный флюенс")
    axes[1, 0].set_ylabel("Относительная доза / первичный флюенс")
    for ax in axes[1, :]:
        ax.set_xlabel("Crossplane, мм")
    axes[1, 2].axis("off")
    fig.suptitle(
        "РОКУС-АМ: проверка опубликованной геометрии по локальной дозиметрии",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir / f"rokus_geometry_vs_mcc.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    result = {
        "status": "geometric_primary_validation_against_local_dosimetry",
        "source_geometry": {
            "active_diameter_mm": 20.0,
            "active_height_mm": 20.0,
            "active_front_from_work_surface_mm": 1.25,
        },
        "collimation_blinds_distance_from_work_surface_mm": [
            244.0,
            304.0,
        ],
        "field_plane_distance_from_work_surface_mm": 750.0,
        "aperture_scale_relative_to_central_ray_construction": (
            args.aperture_scale
        ),
        "limitations": [
            "primary-ray geometry only",
            "no head scatter or jaw transmission",
            "no OCTAVIUS detector-response model",
        ],
        "comparisons": comparison_rows,
    }
    (args.output_dir / "rokus_geometry_vs_mcc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return comparison_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured-profiles", type=Path, required=True)
    parser.add_argument("--measured-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--aperture-scale", type=float, default=1.0)
    args = parser.parse_args()
    rows = analyse(args)
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
