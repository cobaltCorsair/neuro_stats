"""Plot CT anatomy, GTVp, electron field and actual deposited energy."""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path
import struct
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from stream_utils import _skip_field, _varint  # noqa: E402
from work_with_prepared_data.radiobioligy_project.survival.analyze_geant4_proton_100MeV_reduced_gtv import (  # noqa: E402
    gtv_geometry,
    parse_voxel_map,
)
from work_with_prepared_data.radiobioligy_project.survival.plot_geant4_proton_axis_audit import (  # noqa: E402
    clean_body_surface,
    draw_box,
    geometry_and_contours,
    load_ct_crop,
    rasterise_gtv,
)
from work_with_prepared_data.radiobioligy_project.survival.plot_geant4_proton_beam_geometry import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    LOW_NX,
    LOW_NY,
    LOW_NZ,
    LOW_PX,
    LOW_PY,
    LOW_PZ,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


NX, NY, NZ = 128, 128, 199
PX_MM, PY_MM, PZ_MM = 0.4, 0.4, 0.2
BEAM_RADIUS_MM = 14.4
BEAM_CENTRE_ROW_MM = 25.6
BEAM_CENTRE_Z_MM = NZ * PZ_MM / 2.0
DEFAULT_VOXEL_MAP = (
    TASK_DIR
    / "scoring_v2_electron_novac_rat"
    / "livermore_10000_seed50131_80309"
    / "E10MeV"
    / "vox_electron_E10MeV_livermore_10000_0"
)
DEFAULT_OUTPUT = (
    TASK_DIR
    / "scoring_v2_electron_novac_rat"
    / "analysis_anatomy"
    / "electron_novac_10MeV_rat_anatomy"
)


def stream_energy_geometry(
    path: Path,
    centroid: np.ndarray,
    slab_mm: float = 2.0,
    max_points: int = 6500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return axial/longitudinal projections and highest-energy voxels."""
    axial = np.zeros((NX, NY), dtype=np.float64)
    longitudinal = np.zeros((NZ, NY), dtype=np.float64)
    point_heap: list[tuple[float, int, int, int]] = []
    centre_x = int(round(float(centroid[1]) / PX_MM - 0.5))
    centre_z = int(round(float(centroid[2]) / PZ_MM - 0.5))
    x_half = max(0, int(round(slab_mm / PX_MM)))
    z_half = max(0, int(round(slab_mm / PZ_MM)))

    with path.open("rb") as handle:
        while True:
            try:
                tag = _varint(handle)
            except EOFError:
                break
            field, wire = tag >> 3, tag & 7
            if field != 1 or wire != 2:
                _skip_field(handle, wire)
                continue
            entry_len = _varint(handle)
            entry_end = handle.tell() + entry_len
            voxel_id = None
            dep_energy = 0.0
            while handle.tell() < entry_end:
                entry_tag = _varint(handle)
                entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_field == 1 and entry_wire == 0:
                    voxel_id = _varint(handle)
                elif entry_field == 2 and entry_wire == 2:
                    value_len = _varint(handle)
                    value_end = handle.tell() + value_len
                    while handle.tell() < value_end:
                        value_tag = _varint(handle)
                        value_field, value_wire = value_tag >> 3, value_tag & 7
                        if value_field == 2 and value_wire == 1:
                            dep_energy = struct.unpack("<d", handle.read(8))[0]
                        else:
                            _skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    _skip_field(handle, entry_wire)
            handle.seek(entry_end)
            if voxel_id is None or dep_energy <= 0.0:
                continue
            gx = voxel_id % NX
            gy = (voxel_id // NX) % NY
            gz = voxel_id // (NX * NY)
            if not 0 <= gz < NZ:
                continue
            if abs(gz - centre_z) <= z_half:
                axial[gx, gy] += dep_energy
            if abs(gx - centre_x) <= x_half:
                longitudinal[gz, gy] += dep_energy
            item = (dep_energy, gx, gy, gz)
            if len(point_heap) < max_points:
                heapq.heappush(point_heap, item)
            elif dep_energy > point_heap[0][0]:
                heapq.heapreplace(point_heap, item)

    points = np.asarray(
        [
            (
                (gy + 0.5) * PY_MM,
                (gx + 0.5) * PX_MM,
                (gz + 0.5) * PZ_MM,
                energy,
            )
            for energy, gx, gy, gz in point_heap
        ],
        dtype=float,
    )
    return axial, longitudinal, points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--voxel-map", type=Path, default=DEFAULT_VOXEL_MAP)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--histories", type=int, default=10000)
    args = parser.parse_args()

    if not args.voxel_map.exists():
        raise FileNotFoundError(args.voxel_map)
    (
        z_values,
        by_z,
        crop_parameters,
        contours,
        centroid,
    ) = geometry_and_contours(args.ct_dir, args.rtstruct)
    ct_volume = load_ct_crop(z_values, by_z, crop_parameters)
    gtv_mask = rasterise_gtv(contours)
    body_surface = clean_body_surface(ct_volume)
    _, gtv_ids, gtv_lookup, gtv_depth_interval = gtv_geometry(
        args.ct_dir,
        args.rtstruct,
    )
    parsed = parse_voxel_map(
        args.voxel_map,
        gtv_lookup,
        include_dose=True,
    )
    axial_energy, longitudinal_energy, energy_points = (
        stream_energy_geometry(args.voxel_map, centroid)
    )

    centre_z = int(
        np.clip(round(centroid[2] / LOW_PZ - 0.5), 0, LOW_NZ - 1)
    )
    centre_row = int(
        np.clip(round(centroid[1] / LOW_PY - 0.5), 0, LOW_NY - 1)
    )
    axial_ct = ct_volume[centre_z]
    axial_gtv = gtv_mask[centre_z]
    longitudinal_ct = ct_volume[:, centre_row, :]
    longitudinal_gtv = np.any(gtv_mask, axis=1)
    axial_extent = (0.0, 51.2, 0.0, 51.2)
    longitudinal_extent = (0.0, 51.2, 0.0, 39.8)

    positive_energy = np.concatenate(
        (
            axial_energy[axial_energy > 0.0],
            longitudinal_energy[longitudinal_energy > 0.0],
        )
    )
    norm = LogNorm(
        vmin=float(np.percentile(positive_energy, 25.0)),
        vmax=float(np.percentile(positive_energy, 99.7)),
    )

    depth = (np.arange(NY, dtype=float) + 0.5) * PY_MM
    central_energy = np.asarray(parsed["central_energy_keV"], dtype=float)
    central_let_base = np.asarray(
        parsed["central_let_base_keV2_um"],
        dtype=float,
    )
    kernel = np.ones(5, dtype=float) / 5.0
    energy_profile = np.convolve(central_energy, kernel, mode="same")
    let_profile = np.divide(
        np.convolve(central_let_base, kernel, mode="same"),
        energy_profile,
        out=np.full_like(energy_profile, np.nan),
        where=energy_profile > 0.0,
    )
    energy_profile /= float(args.histories)

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 9,
            }
        )
        figure = plt.figure(figsize=(16.5, 12.0), constrained_layout=True)
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=(1.0, 1.12),
            width_ratios=(1.04, 0.96),
        )
        ax_axial = figure.add_subplot(grid[0, 0])
        ax_long = figure.add_subplot(grid[0, 1])
        ax_3d = figure.add_subplot(grid[1, 0], projection="3d")
        ax_profile = figure.add_subplot(grid[1, 1])

        heatmap = None
        panels = (
            (
                ax_axial,
                axial_ct,
                axial_gtv,
                axial_energy,
                axial_extent,
                "Поперечный срез через GTVp",
                "DICOM / patient y, мм",
            ),
            (
                ax_long,
                longitudinal_ct,
                longitudinal_gtv,
                longitudinal_energy,
                longitudinal_extent,
                "Продольный срез вдоль опухолевой лапы",
                "DICOM / patient z, мм",
            ),
        )
        for axis, ct, mask, energy, extent, title, ylabel in panels:
            axis.imshow(
                ct,
                cmap="gray",
                vmin=-600,
                vmax=1200,
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            heatmap = axis.imshow(
                np.ma.masked_less_equal(energy, 0.0),
                cmap="inferno",
                norm=norm,
                alpha=0.66,
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            axis.contour(
                mask.astype(float),
                levels=[0.5],
                colors=["#e31a1c"],
                linewidths=2.2,
                origin="lower",
                extent=extent,
            )
            for boundary in (
                BEAM_CENTRE_ROW_MM - BEAM_RADIUS_MM,
                BEAM_CENTRE_ROW_MM + BEAM_RADIUS_MM,
            ):
                axis.axhline(
                    boundary,
                    color="#f2b134",
                    lw=1.25,
                    ls="--",
                )
            axis.annotate(
                "",
                xy=(1.0, BEAM_CENTRE_ROW_MM),
                xytext=(50.5, BEAM_CENTRE_ROW_MM),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 2.7,
                    "color": "#f2b134",
                },
            )
            axis.set(
                title=title,
                xlabel="DICOM / patient x, мм",
                ylabel=ylabel,
                xlim=(0.0, 51.2),
                ylim=(extent[2], extent[3]),
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.12)

        surface = np.argwhere(body_surface)
        if len(surface) > 7500:
            rng = np.random.default_rng(20260729)
            surface = surface[
                rng.choice(len(surface), 7500, replace=False)
            ]
        ax_3d.scatter(
            (surface[:, 2] + 0.5) * LOW_PX * 2,
            (surface[:, 1] + 0.5) * LOW_PY * 2,
            (surface[:, 0] + 0.5) * LOW_PZ * 2,
            s=1.0,
            c="#888888",
            alpha=0.035,
            depthshade=False,
        )
        for contour in contours[:: max(1, len(contours) // 24)]:
            closed = np.vstack((contour, contour[0]))
            ax_3d.plot(
                closed[:, 0],
                closed[:, 1],
                closed[:, 2],
                color="#e31a1c",
                lw=1.7,
                alpha=0.95,
            )
        point_order = np.argsort(energy_points[:, 3])
        plotted = energy_points[point_order]
        ax_3d.scatter(
            plotted[:, 0],
            plotted[:, 1],
            plotted[:, 2],
            c=plotted[:, 3],
            cmap="inferno",
            norm=norm,
            s=12.0,
            alpha=0.76,
            edgecolors="none",
            depthshade=False,
        )
        theta = np.linspace(0.0, 2.0 * np.pi, 60)
        beam_x = np.linspace(0.0, 51.2, 24)
        xx, tt = np.meshgrid(beam_x, theta)
        yy = BEAM_CENTRE_ROW_MM + BEAM_RADIUS_MM * np.cos(tt)
        zz = BEAM_CENTRE_Z_MM + BEAM_RADIUS_MM * np.sin(tt)
        ax_3d.plot_wireframe(
            xx,
            yy,
            zz,
            rstride=8,
            cstride=5,
            color="#f2b134",
            linewidth=0.65,
            alpha=0.18,
        )
        ax_3d.quiver(
            51.2,
            BEAM_CENTRE_ROW_MM,
            BEAM_CENTRE_Z_MM,
            -51.2,
            0.0,
            0.0,
            color="#f2b134",
            linewidth=2.8,
            arrow_length_ratio=0.07,
        )
        draw_box(ax_3d, (0, 51.2), (0, 51.2), (0, 39.8))
        ax_3d.set(
            xlim=(0.0, 51.2),
            ylim=(0.0, 51.2),
            zlim=(0.0, 39.8),
            xlabel="patient x, мм",
            ylabel="patient y, мм",
            zlabel="patient z, мм",
            title="Трёхмерная расчётная геометрия и энерговклад",
        )
        ax_3d.set_box_aspect((51.2, 51.2, 39.8), zoom=1.15)
        ax_3d.view_init(elev=24, azim=-62)
        ax_3d.legend(
            handles=[
                Patch(
                    facecolor="#888888",
                    alpha=0.20,
                    label="поверхность тела",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#e31a1c",
                    lw=2.2,
                    label="GTVp",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#f2b134",
                    lw=2.2,
                    label="поле r = 14,4 мм; Geant4 −y",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#f06f4d",
                    markeredgecolor="none",
                    markersize=6,
                    label="воксели фактического энерговклада",
                ),
            ],
            loc="upper left",
        )

        energy_line = ax_profile.plot(
            depth,
            energy_profile,
            color="#d95f02",
            lw=2.3,
            label="энерговклад",
        )
        ax_profile.axvspan(
            gtv_depth_interval[0],
            gtv_depth_interval[1],
            color="#e31a1c",
            alpha=0.11,
            label="аксиальный интервал GTVp",
        )
        ax_profile.set(
            title="Центральные профили по глубине от входной границы +y",
            xlabel="Глубина, мм",
            ylabel="Энерговклад, кэВ/первичную частицу",
            xlim=(0.0, 51.2),
            ylim=(0.0, None),
        )
        ax_profile.grid(alpha=0.18)
        let_axis = ax_profile.twinx()
        reliable = energy_profile >= 0.05 * np.nanmax(energy_profile)
        let_axis.plot(
            depth,
            let_profile,
            color="#6a1b9a",
            lw=1.2,
            ls="--",
            alpha=0.32,
        )
        let_line = let_axis.plot(
            depth,
            np.where(reliable, let_profile, np.nan),
            color="#6a1b9a",
            lw=2.0,
            label=r"$LET_{D,w}$ (≥5% энерговклада)",
        )
        let_axis.set_ylabel(r"$LET_{D,w}$, кэВ/мкм", color="#6a1b9a")
        let_axis.tick_params(axis="y", colors="#6a1b9a")
        left_lines, left_labels = ax_profile.get_legend_handles_labels()
        ax_profile.legend(
            left_lines + let_line,
            left_labels + [r"$LET_{D,w}$ (≥5% энерговклада)"],
            frameon=False,
            loc="upper right",
        )

        colorbar = figure.colorbar(
            heatmap,
            ax=[ax_axial, ax_long],
            orientation="horizontal",
            fraction=0.044,
            pad=0.07,
            aspect=50,
        )
        colorbar.set_label(
            f"Энерговклад {args.histories:,} электронов 10 МэВ, кэВ "
            "(логарифмическая шкала)".replace(",", " ")
        )
        figure.suptitle(
            "Электроны NOVAC 10 МэВ в воксельном фантоме крысы\n"
            "G4EmLivermorePhysics; цвет показывает рассчитанный энерговклад",
            fontsize=17,
        )

        args.output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg", ".pdf"):
            figure.savefig(
                args.output_stem.with_suffix(suffix),
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(figure)
    finally:
        configurator.restore_original_styles()

    print(f"Written {args.output_stem}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
