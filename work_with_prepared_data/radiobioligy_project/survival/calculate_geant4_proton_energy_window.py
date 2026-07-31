"""Estimate the incident proton-energy window for the rat GTV.

The beam is assumed to enter from patient left and travel to patient right
(DICOM columns high -> low; Geant4 -y).  For every ray crossing GTVp, the
script integrates the CT-derived density from the ipsilateral body surface to
three boundaries:

* the proximal GTV boundary;
* the distal GTV boundary;
* the distal boundary of the contiguous paw/body segment.

The water-equivalent paths are converted to screening energies by inverting
R[cm] = 0.0022 E^1.77.  These energies initialise a Geant4 scan; they are not
a replacement for the Geant4 range calibration.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_geant4_proton_axis_audit import (
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    hu_to_density,
    screening_energy_mev,
)
from plot_geant4_proton_beam_geometry import (
    BEAM_RADIUS_MM,
    LOW_PX,
    LOW_PY,
    LOW_PZ,
    geometry_and_contours,
    load_ct_crop,
    rasterise_gtv,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


DEFAULT_OUTPUT = Path(
    r"C:\dev\dissertation\task4_5\outputs\geant4_livermore_20260724"
) / "proton_initial_energy_window"


def containing_segment(mask: np.ndarray, indices: np.ndarray) -> tuple[int, int]:
    """Return the tissue segment containing the centre of the supplied ROI."""
    centre = int(round(float(np.mean(indices))))
    if not mask[centre]:
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise RuntimeError("No tissue voxels on a GTV-crossing ray")
        centre = int(candidates[np.argmin(np.abs(candidates - centre))])
    lo = centre
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = centre
    while hi + 1 < len(mask) and mask[hi + 1]:
        hi += 1
    return lo, hi


def ray_metrics(
    ct_volume: np.ndarray,
    gtv_mask: np.ndarray,
    centroid: np.ndarray,
    beam_radius_mm: float,
) -> list[dict[str, float | int]]:
    """Measure L->R paths for every beam ray that intersects GTVp."""
    records: list[dict[str, float | int]] = []
    for iz, irow in np.argwhere(np.any(gtv_mask, axis=2)):
        row_mm = (float(irow) + 0.5) * LOW_PY
        z_mm = (float(iz) + 0.5) * LOW_PZ
        radial_mm = float(
            np.hypot(row_mm - float(centroid[1]), z_mm - float(centroid[2]))
        )
        if radial_mm > beam_radius_mm:
            continue

        gtv_indices = np.flatnonzero(gtv_mask[iz, irow])
        if not len(gtv_indices):
            continue
        gtv_lo = int(gtv_indices.min())
        gtv_hi = int(gtv_indices.max())
        hu_line = ct_volume[iz, irow]
        tissue = hu_line > -500.0
        try:
            body_lo, body_hi = containing_segment(tissue, gtv_indices)
        except RuntimeError:
            continue
        if gtv_lo < body_lo or gtv_hi > body_hi:
            continue

        density = hu_to_density(hu_line)
        # Beam direction is high -> low column.  The proximal GTV boundary is
        # the outer face of gtv_hi, so the GTV voxel itself is not included.
        wet_gtv_proximal_mm = float(
            density[gtv_hi + 1 : body_hi + 1].sum() * LOW_PX
        )
        # The distal GTV boundary is the far face of gtv_lo.
        wet_gtv_distal_mm = float(
            density[gtv_lo : body_hi + 1].sum() * LOW_PX
        )
        wet_paw_distal_mm = float(
            density[body_lo : body_hi + 1].sum() * LOW_PX
        )
        records.append(
            {
                "z_index": int(iz),
                "row_index": int(irow),
                "radial_distance_mm": radial_mm,
                "body_entry_column": body_hi,
                "body_exit_column": body_lo,
                "gtv_proximal_column": gtv_hi,
                "gtv_distal_column": gtv_lo,
                "geometric_to_gtv_proximal_mm": (
                    body_hi - gtv_hi
                )
                * LOW_PX,
                "geometric_to_gtv_distal_mm": (
                    body_hi - gtv_lo + 1
                )
                * LOW_PX,
                "geometric_to_paw_distal_mm": (
                    body_hi - body_lo + 1
                )
                * LOW_PX,
                "wet_to_gtv_proximal_mm": wet_gtv_proximal_mm,
                "wet_to_gtv_distal_mm": wet_gtv_distal_mm,
                "wet_to_paw_distal_mm": wet_paw_distal_mm,
                "energy_to_gtv_proximal_MeV": screening_energy_mev(
                    wet_gtv_proximal_mm
                ),
                "energy_to_gtv_distal_MeV": screening_energy_mev(
                    wet_gtv_distal_mm
                ),
                "energy_to_paw_distal_MeV": screening_energy_mev(
                    wet_paw_distal_mm
                ),
            }
        )
    if not records:
        raise RuntimeError("No GTV-crossing rays were measured")
    return records


def central_record(
    records: list[dict[str, float | int]],
    centroid: np.ndarray,
) -> dict[str, float | int]:
    centre_row = float(centroid[1]) / LOW_PY - 0.5
    centre_z = float(centroid[2]) / LOW_PZ - 0.5
    return min(
        records,
        key=lambda row: (
            (float(row["row_index"]) - centre_row) ** 2
            + (float(row["z_index"]) - centre_z) ** 2
        ),
    )


def summary_rows(
    records: list[dict[str, float | int]],
    central: dict[str, float | int],
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    quantities = (
        ("gtv_proximal", "wet_to_gtv_proximal_mm", "energy_to_gtv_proximal_MeV"),
        ("gtv_distal", "wet_to_gtv_distal_mm", "energy_to_gtv_distal_MeV"),
        ("paw_distal", "wet_to_paw_distal_mm", "energy_to_paw_distal_MeV"),
    )
    for boundary, wet_key, energy_key in quantities:
        wet = np.asarray([float(row[wet_key]) for row in records])
        energy = np.asarray([float(row[energy_key]) for row in records])
        rows.append(
            {
                "boundary": boundary,
                "n_gtv_rays": len(records),
                "central_WET_mm": float(central[wet_key]),
                "central_energy_MeV": float(central[energy_key]),
                "min_WET_mm": float(np.min(wet)),
                "p05_WET_mm": float(np.percentile(wet, 5)),
                "median_WET_mm": float(np.median(wet)),
                "p95_WET_mm": float(np.percentile(wet, 95)),
                "max_WET_mm": float(np.max(wet)),
                "min_energy_MeV": float(np.min(energy)),
                "p05_energy_MeV": float(np.percentile(energy, 5)),
                "median_energy_MeV": float(np.median(energy)),
                "p95_energy_MeV": float(np.percentile(energy, 95)),
                "max_energy_MeV": float(np.max(energy)),
            }
        )
    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, float | str | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_energy_window(
    records: list[dict[str, float | int]],
    central: dict[str, float | int],
    output_stem: Path,
) -> None:
    labels = ["Ближняя граница\nGTVp", "Дальняя граница\nGTVp", "Выход из лапы"]
    keys = [
        "energy_to_gtv_proximal_MeV",
        "energy_to_gtv_distal_MeV",
        "energy_to_paw_distal_MeV",
    ]
    values = [
        np.asarray([float(row[key]) for row in records], dtype=float)
        for key in keys
    ]
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
        parts = ax.violinplot(
            values,
            positions=np.arange(1, 4),
            showmeans=False,
            showmedians=True,
            showextrema=True,
            widths=0.72,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#ed7d31")
            body.set_edgecolor("#8b4513")
            body.set_alpha(0.42)
        for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[part_name].set_color("#6b3410")
            parts[part_name].set_linewidth(1.2)
        ax.scatter(
            np.arange(1, 4),
            [float(central[key]) for key in keys],
            color="#c00000",
            edgecolor="white",
            linewidth=0.7,
            s=48,
            zorder=5,
            label="центральный луч",
        )
        ax.set(
            xticks=np.arange(1, 4),
            xticklabels=labels,
            ylabel="Оценка начальной энергии протонов, МэВ",
            title=(
                "Энергетическое окно для пучка L→R\n"
                "по КТ-оценке водоэквивалентного пути"
            ),
        )
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_size(14)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--beam-radius-mm", type=float, default=BEAM_RADIUS_MM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    z_values, by_z, crop, contours, centroid = geometry_and_contours(
        args.ct_dir,
        args.rtstruct,
    )
    ct_volume = load_ct_crop(z_values, by_z, crop)
    gtv_mask = rasterise_gtv(contours)
    records = ray_metrics(
        ct_volume,
        gtv_mask,
        centroid,
        args.beam_radius_mm,
    )
    central = central_record(records, centroid)
    summary = summary_rows(records, central)

    write_csv(args.output_dir / "proton_energy_window_by_ray.csv", records)
    write_csv(args.output_dir / "proton_energy_window_summary.csv", summary)
    plot_energy_window(
        records,
        central,
        args.output_dir / "proton_initial_energy_window",
    )

    print(f"GTV-crossing rays: {len(records)}")
    for row in summary:
        print(
            f'{row["boundary"]}: central={row["central_energy_MeV"]:.2f} MeV; '
            f'min={row["min_energy_MeV"]:.2f}; '
            f'p05={row["p05_energy_MeV"]:.2f}; '
            f'p95={row["p95_energy_MeV"]:.2f}; '
            f'max={row["max_energy_MeV"]:.2f} MeV'
        )
    print(f"Written to {args.output_dir}")


if __name__ == "__main__":
    main()
