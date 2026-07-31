"""Visualise the article-based carbon-ion stand, rat, GTVp and energy deposit."""

from __future__ import annotations

import argparse
import csv
import heapq
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


NEURO_STATS_ROOT = Path(__file__).resolve().parents[3]
if str(NEURO_STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURO_STATS_ROOT))

TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from stream_utils import _skip_field, _varint  # noqa: E402
from work_with_prepared_data.radiobioligy_project.survival.plot_geant4_proton_beam_geometry import (  # noqa: E402
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    geometry_and_contours,
    load_ct_crop,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (  # noqa: E402
    MatplotlibConfigurator,
)


NX = 128
NY = 128
NZ = 199
PX_MM = 0.4
PY_MM = 0.4
PZ_MM = 0.2
PHANTOM_X_MM = NX * PX_MM
PHANTOM_Y_MM = NY * PY_MM
PHANTOM_Z_MM = NZ * PZ_MM
BEAM_RADIUS_MM = 15.0
BEAM_X_MM = 0.5 * PHANTOM_X_MM
BEAM_Z_MM = 0.5 * PHANTOM_Z_MM

CAISSON_CLEARANCE_MM = 0.2
CAISSON_FRONT_MM = 5.0
CAISSON_OTHER_MM = 15.0
TANK_WINDOW_MM = 20.0
DEFAULT_WATER_GAP_MM = 200.0

DEFAULT_RUN = (
    TASK_DIR
    / "scoring_v2_carbon_c12_rat_through_physics"
    / "opt4_1000_434MeVu_articleStand_gap200mm"
)
DEFAULT_OUTPUT = (
    TASK_DIR
    / "scoring_v2_carbon_c12_rat_through_physics"
    / "carbon_c12_article_stand_visualisation"
)


def find_voxel_map(run_root: Path) -> Path:
    matches = [
        path
        for path in (run_root / "through").glob("vox_*_0")
        if "_component_" not in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one main voxel map in {run_root / 'through'}, "
            f"found {len(matches)}"
        )
    return matches[0]


def coarse_ct(
    z_values: list[float],
    by_z: dict[float, Path],
    crop_parameters: np.ndarray,
) -> np.ndarray:
    fine = load_ct_crop(z_values, by_z, crop_parameters)
    if fine.shape != (NZ, 256, 256):
        raise RuntimeError(f"Unexpected CT crop shape: {fine.shape}")
    return fine.reshape(NZ, NX, 2, NY, 2).mean(axis=(2, 4))


def rasterise_gtv(contours: list[np.ndarray]) -> np.ndarray:
    """Return a GTV mask indexed as [z, Geant4 x, Geant4 y]."""
    mask = np.zeros((NZ, NX, NY), dtype=bool)
    for contour in contours:
        iz = int(round(float(np.mean(contour[:, 2])) / PZ_MM - 0.5))
        if iz < 0 or iz >= NZ:
            continue
        image = Image.new("1", (NY, NX), 0)
        polygon = [
            (
                float(point[0] / PY_MM - 0.5),
                float(point[1] / PX_MM - 0.5),
            )
            for point in contour
        ]
        ImageDraw.Draw(image).polygon(polygon, outline=1, fill=1)
        mask[iz] |= np.asarray(image, dtype=bool)
    return mask


def stream_energy(
    voxel_map: Path,
    centre_x_mm: float,
    centre_z_mm: float,
    *,
    slab_mm: float = 2.0,
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read energy deposits into axial/longitudinal projections and 3-D points."""
    axial = np.zeros((NX, NY), dtype=np.float64)
    longitudinal = np.zeros((NZ, NY), dtype=np.float64)
    depth = np.zeros(NY, dtype=np.float64)
    point_heap: list[tuple[float, int, int, int]] = []
    centre_ix = int(round(centre_x_mm / PX_MM - 0.5))
    centre_iz = int(round(centre_z_mm / PZ_MM - 0.5))
    x_half = max(1, int(round(slab_mm / PX_MM)))
    z_half = max(1, int(round(slab_mm / PZ_MM)))

    with voxel_map.open("rb") as handle:
        while True:
            try:
                tag = _varint(handle)
            except EOFError:
                break
            field, wire = tag >> 3, tag & 7
            if field != 1 or wire != 2:
                _skip_field(handle, wire)
                continue

            entry_length = _varint(handle)
            entry_end = handle.tell() + entry_length
            voxel_id: int | None = None
            dep_energy = 0.0
            while handle.tell() < entry_end:
                entry_tag = _varint(handle)
                entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_field == 1 and entry_wire == 0:
                    voxel_id = _varint(handle)
                elif entry_field == 2 and entry_wire == 2:
                    value_length = _varint(handle)
                    value_end = handle.tell() + value_length
                    while handle.tell() < value_end:
                        value_tag = _varint(handle)
                        value_field, value_wire = value_tag >> 3, value_tag & 7
                        if value_field == 2 and value_wire == 1:
                            dep_energy = struct.unpack(
                                "<d", handle.read(8)
                            )[0]
                        else:
                            _skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    _skip_field(handle, entry_wire)
            handle.seek(entry_end)

            if voxel_id is None or dep_energy <= 0.0:
                continue
            ix = voxel_id % NX
            iy = (voxel_id // NX) % NY
            iz = voxel_id // (NX * NY)
            depth[iy] += dep_energy
            if abs(iz - centre_iz) <= z_half:
                axial[ix, iy] += dep_energy
            if abs(ix - centre_ix) <= x_half:
                longitudinal[iz, iy] += dep_energy
            item = (dep_energy, ix, iy, iz)
            if len(point_heap) < max_points:
                heapq.heappush(point_heap, item)
            elif dep_energy > point_heap[0][0]:
                heapq.heapreplace(point_heap, item)

    points = np.asarray(
        [
            (
                (ix + 0.5) * PX_MM,
                (iy + 0.5) * PY_MM,
                (iz + 0.5) * PZ_MM,
                energy,
            )
            for energy, ix, iy, iz in point_heap
        ],
        dtype=float,
    )
    return axial, longitudinal, depth, points


def clean_body_surface(volume: np.ndarray) -> np.ndarray:
    body = volume > -500.0
    body = ndimage.binary_closing(body, iterations=2)
    labels, count = ndimage.label(body)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        body = labels == int(np.argmax(sizes))
    return body & ~ndimage.binary_erosion(body, iterations=1)


def draw_box(
    axis,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> None:
    for x in x_limits:
        for y in y_limits:
            axis.plot(
                [x, x], [y, y], z_limits,
                color="#666666", lw=0.7, alpha=0.45
            )
    for x in x_limits:
        for z in z_limits:
            axis.plot(
                [x, x], y_limits, [z, z],
                color="#666666", lw=0.7, alpha=0.45
            )
    for y in y_limits:
        for z in z_limits:
            axis.plot(
                x_limits, [y, y], [z, z],
                color="#666666", lw=0.7, alpha=0.45
            )


def stand_depths(water_gap_mm: float) -> dict[str, tuple[float, float]]:
    """Return source-based depths along the -y beam direction."""
    source_margin = 20.0
    window = (source_margin, source_margin + TANK_WINDOW_MM)
    water = (window[1], window[1] + water_gap_mm)
    caisson_front = (water[1], water[1] + CAISSON_FRONT_MM)
    cavity_start = caisson_front[1] + CAISSON_CLEARANCE_MM
    rat = (cavity_start, cavity_start + PHANTOM_Y_MM)
    caisson_back = (
        rat[1] + CAISSON_CLEARANCE_MM,
        rat[1] + CAISSON_CLEARANCE_MM + CAISSON_OTHER_MM,
    )
    return {
        "window": window,
        "water": water,
        "caisson_front": caisson_front,
        "rat": rat,
        "caisson_back": caisson_back,
    }


def make_plot(
    ct: np.ndarray,
    gtv: np.ndarray,
    contours: list[np.ndarray],
    centroid_xyz: np.ndarray,
    axial_energy: np.ndarray,
    longitudinal_energy: np.ndarray,
    depth_energy: np.ndarray,
    points: np.ndarray,
    output_stem: Path,
    water_gap_mm: float,
    histories: int,
    letd_keV_um: float | None,
    letd_sd_keV_um: float | None,
    letd_seed_count: int,
) -> None:
    centre_x_index = int(round(centroid_xyz[0] / PX_MM - 0.5))
    centre_z_index = int(round(centroid_xyz[2] / PZ_MM - 0.5))
    centre_x_index = int(np.clip(centre_x_index, 0, NX - 1))
    centre_z_index = int(np.clip(centre_z_index, 0, NZ - 1))

    positive = np.concatenate(
        (
            axial_energy[axial_energy > 0.0],
            longitudinal_energy[longitudinal_energy > 0.0],
        )
    )
    if positive.size == 0:
        raise RuntimeError("No positive energy deposits were found")
    norm = LogNorm(
        vmin=max(float(np.percentile(positive, 15.0)), 1e-12),
        vmax=float(np.percentile(positive, 99.7)),
    )
    gtv_axial = gtv[centre_z_index]
    x_slice = slice(
        max(0, centre_x_index - 2),
        min(NX, centre_x_index + 3),
    )
    gtv_longitudinal = np.any(gtv[:, x_slice, :], axis=1)
    surface = clean_body_surface(ct)

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        figure = plt.figure(figsize=(17.5, 13.2), constrained_layout=True)
        grid = figure.add_gridspec(
            3,
            2,
            height_ratios=(1.0, 0.055, 1.15),
            width_ratios=(1.0, 1.0),
        )
        axis_axial = figure.add_subplot(grid[0, 0])
        axis_longitudinal = figure.add_subplot(grid[0, 1])
        colorbar_axis = figure.add_subplot(grid[1, :])
        axis_3d = figure.add_subplot(grid[2, 0], projection="3d")
        axis_stand = figure.add_subplot(grid[2, 1])

        axial_extent = (0.0, PHANTOM_Y_MM, 0.0, PHANTOM_X_MM)
        longitudinal_extent = (
            0.0,
            PHANTOM_Y_MM,
            0.0,
            PHANTOM_Z_MM,
        )
        axis_axial.imshow(
            ct[centre_z_index],
            cmap="gray",
            vmin=-600,
            vmax=1200,
            origin="lower",
            extent=axial_extent,
            aspect="equal",
        )
        heatmap = axis_axial.imshow(
            np.ma.masked_less_equal(axial_energy, 0.0),
            cmap="inferno",
            norm=norm,
            alpha=0.64,
            origin="lower",
            extent=axial_extent,
            aspect="equal",
        )
        axis_axial.contour(
            gtv_axial.astype(float),
            levels=[0.5],
            colors=["#e31a1c"],
            linewidths=2.4,
            origin="lower",
            extent=axial_extent,
        )

        axis_longitudinal.imshow(
            ct[:, centre_x_index, :],
            cmap="gray",
            vmin=-600,
            vmax=1200,
            origin="lower",
            extent=longitudinal_extent,
            aspect="equal",
        )
        axis_longitudinal.imshow(
            np.ma.masked_less_equal(longitudinal_energy, 0.0),
            cmap="inferno",
            norm=norm,
            alpha=0.64,
            origin="lower",
            extent=longitudinal_extent,
            aspect="equal",
        )
        axis_longitudinal.contour(
            gtv_longitudinal.astype(float),
            levels=[0.5],
            colors=["#e31a1c"],
            linewidths=2.4,
            origin="lower",
            extent=longitudinal_extent,
        )

        for axis, transverse_centre, transverse_name in (
            (axis_axial, BEAM_X_MM, "x"),
            (axis_longitudinal, BEAM_Z_MM, "z"),
        ):
            axis.axhline(
                transverse_centre - BEAM_RADIUS_MM,
                color="#f2b134",
                lw=1.4,
                ls="--",
            )
            axis.axhline(
                transverse_centre + BEAM_RADIUS_MM,
                color="#f2b134",
                lw=1.4,
                ls="--",
            )
            axis.annotate(
                "",
                xy=(1.5, transverse_centre),
                xytext=(49.7, transverse_centre),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 2.8,
                    "color": "#f2b134",
                },
            )
            axis.text(
                49.4,
                transverse_centre + 1.2,
                "источник, +y",
                ha="right",
                va="bottom",
                fontsize=10,
                color="#5b4a20",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
            )
            axis.text(
                3.0,
                transverse_centre + 1.2,
                "направление −y",
                ha="left",
                va="bottom",
                fontsize=10,
                color="#5b4a20",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
            )
            axis.set(
                xlim=(0.0, PHANTOM_Y_MM),
                xlabel="Geant4 y, мм",
                ylabel=f"Geant4 {transverse_name}, мм",
            )
            axis.grid(alpha=0.13)

        axis_axial.set_title(
            f"Поперечный срез через GTVp, z = {centroid_xyz[2]:.1f} мм"
        )
        axis_longitudinal.set_title(
            f"Продольный срез вдоль пучка, x = {centroid_xyz[0]:.1f} мм"
        )

        colorbar = figure.colorbar(
            heatmap,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label(
            f"Энерговклад {histories} первичных ионов, кэВ "
            "(логарифмическая шкала)"
        )

        surface_indices = np.argwhere(surface)
        if len(surface_indices) > 6500:
            generator = np.random.default_rng(20260728)
            surface_indices = surface_indices[
                generator.choice(len(surface_indices), 6500, replace=False)
            ]
        axis_3d.scatter(
            (surface_indices[:, 1] + 0.5) * PX_MM,
            (surface_indices[:, 2] + 0.5) * PY_MM,
            (surface_indices[:, 0] + 0.5) * PZ_MM,
            s=1.2,
            c="#858585",
            alpha=0.055,
            depthshade=False,
        )
        for contour in contours[:: max(1, len(contours) // 24)]:
            closed = np.vstack((contour, contour[0]))
            axis_3d.plot(
                closed[:, 1],
                closed[:, 0],
                closed[:, 2],
                color="#e31a1c",
                lw=1.5,
                alpha=0.9,
            )
        order = np.argsort(points[:, 3])
        selected = points[order]
        axis_3d.scatter(
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            c=selected[:, 3],
            cmap="inferno",
            norm=norm,
            s=8.0,
            alpha=0.58,
            depthshade=False,
        )
        theta = np.linspace(0.0, 2.0 * np.pi, 72)
        beam_y = np.linspace(61.0, -7.0, 24)
        yy, tt = np.meshgrid(beam_y, theta)
        xx = BEAM_X_MM + BEAM_RADIUS_MM * np.cos(tt)
        zz = BEAM_Z_MM + BEAM_RADIUS_MM * np.sin(tt)
        axis_3d.plot_wireframe(
            xx,
            yy,
            zz,
            rstride=8,
            cstride=5,
            color="#f2b134",
            linewidth=0.65,
            alpha=0.20,
        )
        axis_3d.quiver(
            BEAM_X_MM,
            61.0,
            BEAM_Z_MM,
            0.0,
            -68.0,
            0.0,
            color="#f2b134",
            linewidth=2.6,
            arrow_length_ratio=0.065,
        )
        draw_box(
            axis_3d,
            (0.0, PHANTOM_X_MM),
            (0.0, PHANTOM_Y_MM),
            (0.0, PHANTOM_Z_MM),
        )
        axis_3d.set(
            xlim=(0.0, PHANTOM_X_MM),
            ylim=(-7.0, 61.0),
            zlim=(0.0, PHANTOM_Z_MM),
            xlabel="x, мм",
            ylabel="y, мм",
            zlabel="z, мм",
            title="Трёхмерный путь пучка через крысу",
        )
        axis_3d.set_box_aspect((PHANTOM_X_MM, 68.0, PHANTOM_Z_MM))
        axis_3d.view_init(elev=23, azim=-55)
        axis_3d.legend(
            handles=[
                Patch(
                    facecolor="#858585",
                    alpha=0.25,
                    label="поверхность тела",
                ),
                Line2D(
                    [0], [0],
                    color="#e31a1c",
                    lw=2.4,
                    label="GTVp",
                ),
                Line2D(
                    [0], [0],
                    color="#f2b134",
                    lw=2.4,
                    label="поле, r = 15 мм",
                ),
                Line2D(
                    [0], [0],
                    marker="o",
                    ls="",
                    color="#ef6c00",
                    label="энерговклад",
                ),
            ],
            loc="upper left",
            fontsize=10,
        )

        layers = stand_depths(water_gap_mm)
        colours = {
            "window": "#72a8b8",
            "water": "#7db7e8",
            "caisson_front": "#f0a34a",
            "rat": "#b9b9b9",
            "caisson_back": "#72a8b8",
        }
        labels = {
            "window": "окно\n20 мм",
            "water": f"вода\n{water_gap_mm:g} мм",
            "caisson_front": "ПММА\n5 мм",
            "rat": "КТ-фантом\nкрысы",
            "caisson_back": "поликарбонат\n15 мм",
        }
        for name, interval in layers.items():
            axis_stand.axvspan(
                interval[0],
                interval[1],
                color=colours[name],
                alpha=0.30 if name != "water" else 0.22,
                lw=0,
            )
            axis_stand.text(
                0.5 * (interval[0] + interval[1]),
                0.13 if name != "water" else 0.20,
                labels[name],
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90 if interval[1] - interval[0] < 20 else 0,
            )

        rat_start = layers["rat"][0]
        gtv_depths = np.flatnonzero(np.any(gtv, axis=(0, 1)))
        if gtv_depths.size:
            # Geant4 +y is the entrance side; convert iy to beam depth.
            gtv_min = rat_start + (
                NY - int(np.max(gtv_depths)) - 0.5
            ) * PY_MM
            gtv_max = rat_start + (
                NY - int(np.min(gtv_depths)) - 0.5
            ) * PY_MM
            axis_stand.axvspan(
                gtv_min,
                gtv_max,
                color="#e31a1c",
                alpha=0.16,
                label="проекция GTVp",
            )
            axis_stand.text(
                0.5 * (gtv_min + gtv_max),
                0.92,
                "GTVp",
                ha="center",
                va="top",
                color="#8b0000",
                fontsize=10,
            )

        beam_depth = rat_start + (
            NY - np.arange(NY, dtype=float) - 0.5
        ) * PY_MM
        order_depth = np.argsort(beam_depth)
        profile = depth_energy[order_depth]
        if np.max(profile) > 0.0:
            profile = profile / np.max(profile)
        profile = np.convolve(profile, np.ones(3) / 3.0, mode="same")
        axis_stand.plot(
            beam_depth[order_depth],
            profile,
            color="#5b207f",
            lw=2.2,
            label="энерговклад в фантоме",
        )
        axis_stand.annotate(
            "",
            xy=(layers["caisson_back"][1], 1.10),
            xytext=(0.0, 1.10),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 2.6,
                "color": "#f2b134",
            },
        )
        axis_stand.text(
            0.0,
            1.13,
            "источник",
            ha="left",
            va="bottom",
            fontsize=10,
        )
        axis_stand.set(
            xlim=(0.0, layers["caisson_back"][1] + 4.0),
            ylim=(0.0, 1.22),
            xlabel="Путь от источника вдоль направления −y, мм",
            ylabel="Энерговклад, отн. ед.",
            title="Послойная схема стенда и профиль в крысе",
        )
        axis_stand.grid(axis="y", alpha=0.18)
        axis_stand.legend(loc="center left", fontsize=9)

        if letd_keV_um is None:
            let_text = ""
        elif letd_sd_keV_um is None or letd_seed_count < 2:
            let_text = f"; LET$_D$(GTVp) = {letd_keV_um:.2f} кэВ/мкм"
        else:
            let_text = (
                f"; LET$_D$(GTVp) = {letd_keV_um:.2f}"
                f"±{letd_sd_keV_um:.2f} кэВ/мкм "
                f"({letd_seed_count} seed)"
            )
        figure.suptitle(
            r"$^{12}$C, 434 МэВ/нуклон: углеродный прострел "
            "в геометрии статьи\n"
            f"QGSP_INCLXX + G4EmStandardPhysics_option4; "
            f"{histories} первичных ионов{let_text}",
            fontsize=16,
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg", ".pdf"):
            figure.savefig(
                output_stem.with_suffix(suffix),
                dpi=280 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(figure)
    finally:
        configurator.restore_original_styles()


def read_letd_series(
    run_root: Path,
) -> tuple[float | None, float | None, int]:
    prefix = run_root.name.split("_seed", maxsplit=1)[0]
    values = []
    for candidate in sorted(run_root.parent.glob(prefix + "*")):
        summary_path = candidate / "gtv_analysis" / "gtv_summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        value = row.get("gtv_LETd_w_keV_um")
        if value not in (None, ""):
            values.append(float(value))
    if not values:
        return None, None, 0
    array = np.asarray(values, dtype=float)
    standard_deviation = (
        None if len(array) < 2 else float(np.std(array, ddof=1))
    )
    return float(np.mean(array)), standard_deviation, len(array)


def write_summary(
    path: Path,
    centroid_xyz: np.ndarray,
    voxel_map: Path,
    water_gap_mm: float,
    histories: int,
    letd_keV_um: float | None,
    letd_sd_keV_um: float | None,
    letd_seed_count: int,
) -> None:
    rows = [
        ("particle", "C-12"),
        ("energy_per_nucleon_MeV", "434"),
        ("beam_direction", "Geant4 +y to -y"),
        ("beam_radius_mm", f"{BEAM_RADIUS_MM:.3f}"),
        ("water_gap_mm", f"{water_gap_mm:.3f}"),
        ("tank_beam_window_polycarbonate_mm", "20.000"),
        ("caisson_front_pmma_mm", "5.000"),
        ("caisson_other_polycarbonate_mm", "15.000"),
        ("histories", str(histories)),
        ("gtvp_centroid_x_mm", f"{centroid_xyz[0]:.3f}"),
        ("gtvp_centroid_y_mm", f"{centroid_xyz[1]:.3f}"),
        ("gtvp_centroid_z_mm", f"{centroid_xyz[2]:.3f}"),
        (
            "gtvp_LETd_keV_um",
            "" if letd_keV_um is None else f"{letd_keV_um:.9f}",
        ),
        (
            "gtvp_LETd_seed_sd_keV_um",
            ""
            if letd_sd_keV_um is None
            else f"{letd_sd_keV_um:.9f}",
        ),
        ("gtvp_LETd_seed_count", str(letd_seed_count)),
        ("voxel_map", str(voxel_map)),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--water-gap-mm",
        type=float,
        default=DEFAULT_WATER_GAP_MM,
    )
    parser.add_argument("--histories", type=int, default=1000)
    args = parser.parse_args()

    voxel_map = find_voxel_map(args.run_root)
    z_values, by_z, crop, contours, centroid_yxz = geometry_and_contours(
        args.ct_dir,
        args.rtstruct,
    )
    ct = coarse_ct(z_values, by_z, crop)
    gtv = rasterise_gtv(contours)
    centroid_xyz = np.asarray(
        [centroid_yxz[1], centroid_yxz[0], centroid_yxz[2]],
        dtype=float,
    )
    axial, longitudinal, depth, points = stream_energy(
        voxel_map,
        centroid_xyz[0],
        centroid_xyz[2],
    )
    letd, letd_sd, letd_seed_count = read_letd_series(args.run_root)
    make_plot(
        ct,
        gtv,
        contours,
        centroid_xyz,
        axial,
        longitudinal,
        depth,
        points,
        args.output_stem,
        args.water_gap_mm,
        args.histories,
        letd,
        letd_sd,
        letd_seed_count,
    )
    write_summary(
        args.output_stem.with_name(
            args.output_stem.name + "_summary.csv"
        ),
        centroid_xyz,
        voxel_map,
        args.water_gap_mm,
        args.histories,
        letd,
        letd_sd,
        letd_seed_count,
    )
    print(f"GTVp centroid (Geant4 xyz): {centroid_xyz}")
    print(f"Voxel map: {voxel_map}")
    print(f"Written {args.output_stem}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
