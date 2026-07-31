"""Visualise the Geant4 rat phantom, GTVp and the 60 MeV proton beam."""

from __future__ import annotations

import argparse
import csv
import heapq
import os
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
import pydicom
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)

TASK_DIR = Path(r"C:\dev\dissertation\task4_5")
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from stream_utils import _skip_field, _varint


DEFAULT_CT_DIR = Path(
    r"C:\dev\ng in github\data\NEW_RAT_CT+RTSTRUCT_FROM_ECLIPSE\m6_e"
)
DEFAULT_RS = DEFAULT_CT_DIR / (
    "RS.1.2.246.352.221.5252265548251877138.6002514067141073041.dcm"
)
DEFAULT_VOXEL_MAP = TASK_DIR / (
    "physics_sensitivity_proton_60MeV_peak_inclxx_livermore_"
    "dose_components_1k"
) / "vox_proton_60MeV_peak_inclxx_livermore_dose_components_1k_0"
DEFAULT_OUTPUT = TASK_DIR / "outputs" / "geant4_livermore_20260724" / (
    "proton_peak_beam_geometry"
)

NX, NY, NZ = 512, 512, 199
PX, PY, PZ = 0.1, 0.1, 0.2
CT_XY_MM = 0.2
LOW_NX, LOW_NY, LOW_NZ = 256, 256, 199
LOW_PX, LOW_PY, LOW_PZ = 0.2, 0.2, 0.2
BEAM_RADIUS_MM = 14.4
BEAM_CENTRE_Y_MM = NY * PY / 2.0
BEAM_CENTRE_Z_MM = NZ * PZ / 2.0


def load_ct_geometry(
    ct_dir: Path,
) -> tuple[list[float], dict[float, Path], pydicom.dataset.FileDataset]:
    by_z: dict[float, Path] = {}
    first = None
    for path in ct_dir.glob("CT*.dcm"):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        if not hasattr(dataset, "ImagePositionPatient"):
            continue
        z = float(dataset.ImagePositionPatient[2])
        by_z[z] = path
        if first is None or z < float(first.ImagePositionPatient[2]):
            first = dataset
    if first is None:
        raise RuntimeError(f"No CT slices found in {ct_dir}")
    return sorted(by_z), by_z, first


def get_roi_contours(
    rs_path: Path,
    roi_name: str,
) -> list[np.ndarray]:
    dataset = pydicom.dcmread(rs_path)
    roi_number = None
    for roi in dataset.StructureSetROISequence:
        if str(roi.ROIName) == roi_name:
            roi_number = int(roi.ROINumber)
            break
    if roi_number is None:
        raise RuntimeError(f"ROI {roi_name!r} not found")
    for roi_contour in dataset.ROIContourSequence:
        if int(roi_contour.ReferencedROINumber) != roi_number:
            continue
        return [
            np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            for contour in roi_contour.ContourSequence
        ]
    raise RuntimeError(f"No contour sequence for ROI {roi_name!r}")


def patient_to_ct_pixel(
    points: np.ndarray,
    origin: np.ndarray,
    row_cos: np.ndarray,
    col_cos: np.ndarray,
    spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = points - origin[None, :]
    columns = delta @ row_cos / spacing_mm
    rows = delta @ col_cos / spacing_mm
    return columns, rows


def geometry_and_contours(
    ct_dir: Path,
    rs_path: Path,
) -> tuple[
    list[float],
    dict[float, Path],
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
]:
    z_values, by_z, first = load_ct_geometry(ct_dir)
    contours_patient = get_roi_contours(rs_path, "GTVp")
    all_points = np.vstack(contours_patient)
    centroid_patient = np.mean(all_points, axis=0)
    origin = np.asarray(first.ImagePositionPatient, dtype=float)
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    spacing = float(first.PixelSpacing[0])
    centroid_col, centroid_row = patient_to_ct_pixel(
        centroid_patient[None, :],
        origin,
        row_cos,
        col_cos,
        spacing,
    )
    scale = CT_XY_MM / PX
    ct_col_start = float(centroid_col[0]) - (NX / 2.0) / scale
    ct_row_start = float(centroid_row[0]) - (NY / 2.0) / scale
    z_origin_patient = float(centroid_patient[2]) - (NZ / 2.0) * PZ

    contours_phantom: list[np.ndarray] = []
    for contour in contours_patient:
        columns, rows = patient_to_ct_pixel(
            contour,
            origin,
            row_cos,
            col_cos,
            spacing,
        )
        phantom = np.column_stack(
            (
                (columns - ct_col_start) * CT_XY_MM,
                (rows - ct_row_start) * CT_XY_MM,
                contour[:, 2] - z_origin_patient,
            )
        )
        contours_phantom.append(phantom)
    area_centroids = []
    for contour in contours_phantom:
        x = contour[:, 0]
        y = contour[:, 1]
        next_index = np.roll(np.arange(len(contour)), -1)
        cross = x * y[next_index] - x[next_index] * y
        signed_area = 0.5 * np.sum(cross)
        if abs(signed_area) < 1e-12:
            continue
        centroid_x = np.sum((x + x[next_index]) * cross) / (
            6.0 * signed_area
        )
        centroid_y = np.sum((y + y[next_index]) * cross) / (
            6.0 * signed_area
        )
        area_centroids.append(
            (
                abs(signed_area),
                centroid_x,
                centroid_y,
                float(np.mean(contour[:, 2])),
            )
        )
    area_centroids_array = np.asarray(area_centroids, dtype=float)
    centroid_phantom = np.sum(
        area_centroids_array[:, :1] * area_centroids_array[:, 1:],
        axis=0,
    ) / np.sum(area_centroids_array[:, 0])

    crop_parameters = np.asarray(
        [
            ct_col_start,
            ct_row_start,
            z_origin_patient,
            float(origin[0]),
            float(origin[1]),
        ],
        dtype=float,
    )
    return (
        z_values,
        by_z,
        crop_parameters,
        contours_phantom,
        centroid_phantom,
    )


def load_ct_crop(
    z_values: list[float],
    by_z: dict[float, Path],
    crop_parameters: np.ndarray,
) -> np.ndarray:
    ct_col_start, ct_row_start, z_origin_patient, _, _ = crop_parameters
    volume = np.full((LOW_NZ, LOW_NY, LOW_NX), -1000.0, dtype=np.float32)
    source_columns = np.rint(
        ct_col_start + np.arange(LOW_NX, dtype=float)
    ).astype(int)
    source_rows = np.rint(
        ct_row_start + np.arange(LOW_NY, dtype=float)
    ).astype(int)
    valid_columns = (source_columns >= 0) & (source_columns < 320)
    valid_rows = (source_rows >= 0) & (source_rows < 320)
    valid_col_values = source_columns[valid_columns]
    valid_row_values = source_rows[valid_rows]
    z_array = np.asarray(z_values)

    for iz in range(LOW_NZ):
        patient_z = z_origin_patient + (iz + 0.5) * LOW_PZ
        source_z = float(z_array[np.argmin(np.abs(z_array - patient_z))])
        dataset = pydicom.dcmread(by_z[source_z])
        pixels = (
            dataset.pixel_array.astype(np.float32)
            * float(getattr(dataset, "RescaleSlope", 1.0))
            + float(getattr(dataset, "RescaleIntercept", 0.0))
        )
        target = volume[iz]
        target[np.ix_(valid_rows, valid_columns)] = pixels[
            np.ix_(valid_row_values, valid_col_values)
        ]
    return volume


def rasterise_gtv(contours: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros((LOW_NZ, LOW_NY, LOW_NX), dtype=bool)
    for contour in contours:
        iz = int(round(float(np.mean(contour[:, 2])) / LOW_PZ - 0.5))
        if iz < 0 or iz >= LOW_NZ:
            continue
        xy = [
            (
                float(point[0] / LOW_PX - 0.5),
                float(point[1] / LOW_PY - 0.5),
            )
            for point in contour
        ]
        image = Image.new("1", (LOW_NX, LOW_NY), 0)
        ImageDraw.Draw(image).polygon(xy, outline=1, fill=1)
        mask[iz] |= np.asarray(image, dtype=bool)
    return mask


def stream_energy_projections(
    voxel_map: Path,
    centre_y: float,
    centre_z: float,
    slab_mm: float = 2.0,
    max_points: int = 3500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.zeros((NY, NX), dtype=np.float64)
    xz = np.zeros((NZ, NX), dtype=np.float64)
    point_heap: list[tuple[float, int, int, int]] = []
    y_half = int(round(slab_mm / PY))
    z_half = int(round(slab_mm / PZ))
    centre_iy = int(round(centre_y / PY - 0.5))
    centre_iz = int(round(centre_z / PZ - 0.5))

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
                        value_field = value_tag >> 3
                        value_wire = value_tag & 7
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
            if voxel_id is None or dep_energy <= 0:
                continue
            ix = voxel_id % NX
            iy = (voxel_id // NX) % NY
            iz = voxel_id // (NX * NY)
            if abs(iz - centre_iz) <= z_half:
                xy[iy, ix] += dep_energy
            if abs(iy - centre_iy) <= y_half:
                xz[iz, ix] += dep_energy
            item = (dep_energy, ix, iy, iz)
            if len(point_heap) < max_points:
                heapq.heappush(point_heap, item)
            elif dep_energy > point_heap[0][0]:
                heapq.heapreplace(point_heap, item)

    points = np.asarray(
        [
            (
                (ix + 0.5) * PX,
                (iy + 0.5) * PY,
                (iz + 0.5) * PZ,
                energy,
            )
            for energy, ix, iy, iz in point_heap
        ],
        dtype=float,
    )
    return xy, xz, points


def clean_body_surface(volume: np.ndarray) -> np.ndarray:
    coarse = volume[::2, ::2, ::2] > -500.0
    coarse = ndimage.binary_closing(coarse, iterations=2)
    labels, count = ndimage.label(coarse)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        coarse = labels == int(np.argmax(sizes))
    return coarse & ~ndimage.binary_erosion(coarse, iterations=1)


def draw_box(ax, xlim, ylim, zlim) -> None:
    for x in xlim:
        for y in ylim:
            ax.plot([x, x], [y, y], zlim, color="#666666", lw=0.7, alpha=0.5)
    for x in xlim:
        for z in zlim:
            ax.plot([x, x], ylim, [z, z], color="#666666", lw=0.7, alpha=0.5)
    for y in ylim:
        for z in zlim:
            ax.plot(xlim, [y, y], [z, z], color="#666666", lw=0.7, alpha=0.5)


def plot_geometry(
    ct_volume: np.ndarray,
    gtv_mask: np.ndarray,
    contours: list[np.ndarray],
    centroid: np.ndarray,
    xy_energy: np.ndarray,
    xz_energy: np.ndarray,
    top_points: np.ndarray,
    output_stem: Path,
) -> None:
    centre_low_y = int(round(centroid[1] / LOW_PY - 0.5))
    centre_low_z = int(round(centroid[2] / LOW_PZ - 0.5))
    xy_energy_low = xy_energy.reshape(NY // 2, 2, NX // 2, 2).sum((1, 3))
    xz_energy_low = xz_energy.reshape(NZ, NX // 2, 2).sum(2)
    positive = np.concatenate(
        (
            xy_energy_low[xy_energy_low > 0],
            xz_energy_low[xz_energy_low > 0],
        )
    )
    norm = LogNorm(
        vmin=float(np.percentile(positive, 30)),
        vmax=float(np.percentile(positive, 99.7)),
    )
    gtv_xy = gtv_mask[centre_low_z]
    gtv_xz = np.any(gtv_mask, axis=1)
    body_surface = clean_body_surface(ct_volume)

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        fig = plt.figure(figsize=(17, 12.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.18))
        ax_xy = fig.add_subplot(grid[0, 0])
        ax_xz = fig.add_subplot(grid[0, 1])
        ax_3d = fig.add_subplot(grid[1, :], projection="3d")

        image_extent_xy = (0, LOW_NX * LOW_PX, 0, LOW_NY * LOW_PY)
        image_extent_xz = (0, LOW_NX * LOW_PX, 0, LOW_NZ * LOW_PZ)
        ax_xy.imshow(
            ct_volume[centre_low_z],
            cmap="gray",
            vmin=-600,
            vmax=1200,
            origin="lower",
            extent=image_extent_xy,
        )
        energy_xy_masked = np.ma.masked_less_equal(xy_energy_low, 0)
        heat_xy = ax_xy.imshow(
            energy_xy_masked,
            cmap="inferno",
            norm=norm,
            alpha=0.62,
            origin="lower",
            extent=image_extent_xy,
        )
        ax_xy.contour(
            gtv_xy.astype(float),
            levels=[0.5],
            colors=["#e31a1c"],
            linewidths=2.5,
            origin="lower",
            extent=image_extent_xy,
        )

        ax_xz.imshow(
            ct_volume[:, centre_low_y, :],
            cmap="gray",
            vmin=-600,
            vmax=1200,
            origin="lower",
            extent=image_extent_xz,
            aspect="auto",
        )
        energy_xz_masked = np.ma.masked_less_equal(xz_energy_low, 0)
        ax_xz.imshow(
            energy_xz_masked,
            cmap="inferno",
            norm=norm,
            alpha=0.62,
            origin="lower",
            extent=image_extent_xz,
            aspect="auto",
        )
        ax_xz.contour(
            gtv_xz.astype(float),
            levels=[0.5],
            colors=["#e31a1c"],
            linewidths=2.5,
            origin="lower",
            extent=image_extent_xz,
        )

        for axis, centre, transverse_label in (
            (ax_xy, BEAM_CENTRE_Y_MM, "y"),
            (ax_xz, BEAM_CENTRE_Z_MM, "z"),
        ):
            axis.axhline(
                centre - BEAM_RADIUS_MM,
                color="#f2b134",
                lw=1.5,
                ls="--",
            )
            axis.axhline(
                centre + BEAM_RADIUS_MM,
                color="#f2b134",
                lw=1.5,
                ls="--",
            )
            axis.annotate(
                "",
                xy=(49.5, centre),
                xytext=(-4.5, centre),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 2.8,
                    "color": "#f2b134",
                },
                annotation_clip=False,
            )
            axis.text(
                -5.7,
                centre + 1.5,
                "источник",
                ha="left",
                va="bottom",
                fontsize=10,
                color="#5b4a20",
                clip_on=False,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 1.5,
                },
            )
            axis.text(
                44.0,
                centre + 1.5,
                "+x",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#5b4a20",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 1.5,
                },
            )
            axis.set_xlim(-6.5, 52.0)
            axis.grid(alpha=0.15)
            axis.set_xlabel("x — глубина в фантоме, мм")
            axis.set_ylabel(f"{transverse_label}, мм")

        ax_xy.set_title(
            f"Поперечный срез через GTVp, z = {centroid[2]:.1f} мм"
        )
        ax_xz.set_title(
            f"Продольный срез через GTVp, y = {centroid[1]:.1f} мм"
        )

        surface_indices = np.argwhere(body_surface)
        if len(surface_indices) > 7000:
            rng = np.random.default_rng(20260724)
            surface_indices = surface_indices[
                rng.choice(len(surface_indices), 7000, replace=False)
            ]
        ax_3d.scatter(
            (surface_indices[:, 2] + 0.5) * LOW_PX * 2,
            (surface_indices[:, 1] + 0.5) * LOW_PY * 2,
            (surface_indices[:, 0] + 0.5) * LOW_PZ * 2,
            s=1.2,
            c="#8c8c8c",
            alpha=0.08,
            depthshade=False,
        )
        for contour in contours[:: max(1, len(contours) // 22)]:
            closed = np.vstack((contour, contour[0]))
            ax_3d.plot(
                closed[:, 0],
                closed[:, 1],
                closed[:, 2],
                color="#e31a1c",
                lw=1.6,
                alpha=0.9,
            )
        energy_order = np.argsort(top_points[:, 3])
        selected = top_points[energy_order]
        ax_3d.scatter(
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            c=selected[:, 3],
            cmap="inferno",
            norm=norm,
            s=4,
            alpha=0.3,
            depthshade=False,
        )

        theta = np.linspace(0, 2 * np.pi, 64)
        beam_x = np.linspace(-5.0, 51.2, 24)
        xx, tt = np.meshgrid(beam_x, theta)
        yy = BEAM_CENTRE_Y_MM + BEAM_RADIUS_MM * np.cos(tt)
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
            -5.0,
            BEAM_CENTRE_Y_MM,
            BEAM_CENTRE_Z_MM,
            57.0,
            0,
            0,
            color="#f2b134",
            linewidth=2.6,
            arrow_length_ratio=0.06,
        )
        ax_3d.scatter(
            [centroid[0]],
            [centroid[1]],
            [centroid[2]],
            color="#e31a1c",
            s=45,
            marker="o",
            depthshade=False,
        )
        draw_box(
            ax_3d,
            (0, NX * PX),
            (0, NY * PY),
            (0, NZ * PZ),
        )
        ax_3d.set(
            xlim=(-6, 52),
            ylim=(0, 51.2),
            zlim=(0, 39.8),
            xlabel="x, мм",
            ylabel="y, мм",
            zlabel="z, мм",
            title="Трёхмерная расчётная геометрия",
        )
        ax_3d.set_box_aspect((58, 51.2, 39.8), zoom=1.32)
        ax_3d.view_init(elev=24, azim=-62)
        ax_3d.legend(
            handles=[
                Patch(
                    facecolor="#8c8c8c",
                    alpha=0.25,
                    label="поверхность тела",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#e31a1c",
                    lw=2.5,
                    label="GTVp",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#f2b134",
                    lw=2.5,
                    label="границы пучка, r = 14,4 мм",
                ),
            ],
            loc="upper left",
            fontsize=12,
        )

        colorbar = fig.colorbar(
            heat_xy,
            ax=[ax_xy, ax_xz],
            orientation="horizontal",
            fraction=0.055,
            pad=0.11,
            aspect=55,
        )
        colorbar.set_label(
            "Энерговклад seed A, кэВ (логарифмическая шкала)",
            fontsize=14,
        )
        colorbar.ax.tick_params(labelsize=11)
        fig.suptitle(
            "Протоны 60 МэВ: направление пучка в воксельном фантоме крысы\n"
            "QGSP_INCLXX + G4EmLivermore; наложение энерговклада одного "
            "пилотного запуска",
            fontsize=17,
        )
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(
                output_stem.with_suffix(suffix),
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)
    finally:
        configurator.restore_original_styles()


def write_summary(
    path: Path,
    centroid: np.ndarray,
    output_stem: Path,
) -> None:
    rows = [
        ("beam_particle", "proton"),
        ("beam_energy_MeV", "60"),
        ("beam_direction", "+x"),
        ("beam_radius_mm", f"{BEAM_RADIUS_MM:.3f}"),
        ("beam_centre_y_mm", f"{BEAM_CENTRE_Y_MM:.3f}"),
        ("beam_centre_z_mm", f"{BEAM_CENTRE_Z_MM:.3f}"),
        ("gtvp_centroid_x_mm", f"{centroid[0]:.3f}"),
        ("gtvp_centroid_y_mm", f"{centroid[1]:.3f}"),
        ("gtvp_centroid_z_mm", f"{centroid[2]:.3f}"),
        ("energy_overlay", "seed A, 1000 primary protons"),
        ("source_position", "schematic; actual GPS x=-800 mm"),
        ("output_stem", str(output_stem)),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--voxel-map", type=Path, default=DEFAULT_VOXEL_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        z_values,
        by_z,
        crop_parameters,
        contours,
        centroid,
    ) = geometry_and_contours(args.ct_dir, args.rtstruct)
    ct_volume = load_ct_crop(z_values, by_z, crop_parameters)
    gtv_mask = rasterise_gtv(contours)
    xy_energy, xz_energy, top_points = stream_energy_projections(
        args.voxel_map,
        BEAM_CENTRE_Y_MM,
        BEAM_CENTRE_Z_MM,
    )
    output_stem = args.output_dir / "proton_60MeV_rat_phantom_beam_geometry"
    plot_geometry(
        ct_volume,
        gtv_mask,
        contours,
        centroid,
        xy_energy,
        xz_energy,
        top_points,
        output_stem,
    )
    write_summary(
        args.output_dir / "proton_60MeV_rat_phantom_beam_geometry_summary.csv",
        centroid,
        output_stem,
    )
    print(f"GTVp centroid: {centroid}")
    print(f"Written {output_stem}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
