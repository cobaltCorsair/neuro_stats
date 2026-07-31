"""Audit the proton-beam axis against the DICOM geometry of the rat.

The historical phantom converter writes ``numpy[row, column, slice]`` as
``Geant4[x, y, z]``.  Therefore a Geant4 +x beam is directed along increasing
DICOM rows, not along increasing DICOM columns.  This script makes that
mapping explicit, compares it with the previously drawn assumption, and
estimates the radiological path to the GTV centroid from all four in-plane
sides.

The density integral is a screening approximation based on the same HU to
density calibration used by the local DICOM-to-phantom converter.  It is not a
substitute for a Geant4 range calibration or for an experimentally documented
beam entrance side.
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
import struct

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pydicom

from plot_geant4_proton_beam_geometry import (
    BEAM_CENTRE_Y_MM,
    BEAM_RADIUS_MM,
    DEFAULT_CT_DIR,
    DEFAULT_RS,
    DEFAULT_VOXEL_MAP,
    LOW_NX,
    LOW_NY,
    LOW_NZ,
    LOW_PX,
    LOW_PY,
    LOW_PZ,
    NX,
    NY,
    _skip_field,
    _varint,
    clean_body_surface,
    draw_box,
    geometry_and_contours,
    load_ct_crop,
    rasterise_gtv,
    stream_energy_projections,
)
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import (
    MatplotlibConfigurator,
)


DEFAULT_OUTPUT = Path(
    r"C:\dev\dissertation\task4_5\outputs\geant4_livermore_20260724"
) / "proton_beam_axis_audit"
DEFAULT_REDUCED_LATERAL_MAP = Path(
    r"C:\dev\dissertation\task4_5"
) / "scoring_v2_proton_40MeV_L_to_R_coarse_10k" / (
    "vox_proton_40MeV_L_to_R_coarse_10k_0"
)


def hu_to_density(hu: np.ndarray) -> np.ndarray:
    """Return the local converter's HU-to-density approximation in g/cm3."""
    hu = np.asarray(hu, dtype=float)
    density = np.empty_like(hu)
    mask = hu < -98
    density[mask] = (
        1.21e-3
        + (0.93 - 1.21e-3) / (-98.0 + 1024.0) * (hu[mask] + 1024.0)
    )
    mask = (hu >= -98) & (hu < 14)
    density[mask] = 1.018 + 0.893e-3 * hu[mask]
    mask = (hu >= 14) & (hu < 23)
    density[mask] = 1.03
    mask = (hu >= 23) & (hu < 100)
    density[mask] = 1.003 + 1.169e-3 * hu[mask]
    mask = (hu >= 100) & (hu < 2000)
    density[mask] = 1.017 + 0.592e-3 * hu[mask]
    mask = (hu >= 2000) & (hu < 3060)
    density[mask] = (
        2.201
        + (2.550 - 2.020) / (3060.0 - 2000.0) * (hu[mask] - 2000.0)
    )
    density[hu >= 3060] = 4.507
    return np.maximum(density, 1.21e-3)


def containing_segment(mask: np.ndarray, centre_index: int) -> tuple[int, int]:
    """Return the contiguous True segment that contains centre_index."""
    if not mask[centre_index]:
        true_indices = np.flatnonzero(mask)
        if not len(true_indices):
            raise RuntimeError("No body voxels on the selected central ray")
        centre_index = int(true_indices[np.argmin(abs(true_indices - centre_index))])
    lo = centre_index
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = centre_index
    while hi + 1 < len(mask) and mask[hi + 1]:
        hi += 1
    return lo, hi


def screening_energy_mev(water_range_mm: float) -> float:
    """Invert R[cm] = 0.0022 E^1.77 for a screening-only energy estimate."""
    if water_range_mm <= 0:
        return 0.0
    return ((water_range_mm / 10.0) / 0.0022) ** (1.0 / 1.77)


def direction_metrics(
    ct_slice: np.ndarray,
    gtv_slice: np.ndarray,
    centre_row: int,
    centre_col: int,
) -> list[dict[str, float | str]]:
    """Measure geometric and density-integral paths from four image sides."""
    cases = [
        (
            "A→P; текущий Geant4 +x",
            "row",
            "low",
            ct_slice[:, centre_col],
            gtv_slice[:, centre_col],
            centre_row,
            LOW_PY,
        ),
        (
            "P→A; Geant4 −x",
            "row",
            "high",
            ct_slice[:, centre_col],
            gtv_slice[:, centre_col],
            centre_row,
            LOW_PY,
        ),
        (
            "R→L; DICOM +x",
            "column",
            "low",
            ct_slice[centre_row, :],
            gtv_slice[centre_row, :],
            centre_col,
            LOW_PX,
        ),
        (
            "L→R; DICOM −x",
            "column",
            "high",
            ct_slice[centre_row, :],
            gtv_slice[centre_row, :],
            centre_col,
            LOW_PX,
        ),
    ]
    rows: list[dict[str, float | str]] = []
    for label, axis, side, hu_line, gtv_line, centre, spacing in cases:
        body = hu_line > -500.0
        body_lo, body_hi = containing_segment(body, centre)
        gtv_indices = np.flatnonzero(gtv_line)
        if not len(gtv_indices):
            raise RuntimeError(f"The selected {axis} ray does not cross GTVp")
        gtv_lo = int(gtv_indices.min())
        gtv_hi = int(gtv_indices.max())
        density = hu_to_density(hu_line)
        if side == "low":
            geometric_depth = (centre - body_lo) * spacing
            gtv_entry_depth = (gtv_lo - body_lo) * spacing
            gtv_distal_depth = (gtv_hi - body_lo) * spacing
            wet_to_centroid = float(
                density[body_lo : centre + 1].sum() * spacing
            )
            wet_to_gtv_distal = float(
                density[body_lo : gtv_hi + 1].sum() * spacing
            )
            entry_index = body_lo
        else:
            geometric_depth = (body_hi - centre) * spacing
            gtv_entry_depth = (body_hi - gtv_hi) * spacing
            gtv_distal_depth = (body_hi - gtv_lo) * spacing
            wet_to_centroid = float(
                density[centre : body_hi + 1].sum() * spacing
            )
            wet_to_gtv_distal = float(
                density[gtv_lo : body_hi + 1].sum() * spacing
            )
            entry_index = body_hi
        rows.append(
            {
                "direction": label,
                "array_axis": axis,
                "entry_side": side,
                "entry_index": entry_index,
                "body_lo_index": body_lo,
                "body_hi_index": body_hi,
                "gtv_lo_index": gtv_lo,
                "gtv_hi_index": gtv_hi,
                "geometric_depth_to_centroid_mm": geometric_depth,
                "geometric_depth_to_gtv_entry_mm": gtv_entry_depth,
                "geometric_depth_to_gtv_distal_mm": gtv_distal_depth,
                "density_integral_to_centroid_mm": wet_to_centroid,
                "density_integral_to_gtv_distal_mm": wet_to_gtv_distal,
                "screening_energy_to_centroid_MeV": screening_energy_mev(
                    wet_to_centroid
                ),
                "screening_energy_to_gtv_distal_MeV": screening_energy_mev(
                    wet_to_gtv_distal
                ),
            }
        )
    return rows


def central_ct_metadata(ct_dir: Path) -> dict[str, str]:
    first = next(ct_dir.glob("CT*.dcm"))
    dataset = pydicom.dcmread(first, stop_before_pixels=True)
    return {
        "PatientPosition": str(getattr(dataset, "PatientPosition", "")),
        "ImageOrientationPatient": "\\".join(
            str(value)
            for value in getattr(dataset, "ImageOrientationPatient", [])
        ),
    }


def plot_audit(
    ct_slice: np.ndarray,
    gtv_slice: np.ndarray,
    centroid: np.ndarray,
    energy_geant_xy: np.ndarray,
    metrics: list[dict[str, float | str]],
    output_stem: Path,
) -> None:
    positive = energy_geant_xy[energy_geant_xy > 0]
    norm = LogNorm(
        vmin=float(np.percentile(positive, 30)),
        vmax=float(np.percentile(positive, 99.7)),
    )
    extent = (0, LOW_NX * LOW_PX, 0, LOW_NY * LOW_PY)
    centre_row_mm = centroid[1]
    centre_col_mm = centroid[0]

    # Geant4 x is DICOM row in the converter.  The scorer returns
    # energy_geant_xy[geant_y, geant_x] = [DICOM column, DICOM row].
    energy_corrected = energy_geant_xy.T
    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        # The dissertation-wide plotting preset deliberately uses large fonts
        # for single-panel figures.  This four-panel audit needs a compact
        # local override to keep every label inside the canvas.
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(16.5, 11.5),
            constrained_layout=True,
        )
        ax_correct, ax_lateral, ax_profile, ax_table = axes.ravel()

        for axis, energy, title in (
            (
                ax_correct,
                energy_corrected,
                "Фактически рассчитанный старый прогон: A→P (Geant4 +x)",
            ),
            (
                ax_lateral,
                None,
                "Предполагаемый вход через наружную сторону опухоли: L→R (Geant4 −y)",
            ),
        ):
            axis.imshow(
                ct_slice,
                cmap="gray",
                vmin=-600,
                vmax=1200,
                origin="lower",
                extent=extent,
            )
            if energy is not None:
                axis.imshow(
                    np.ma.masked_less_equal(energy, 0),
                    cmap="inferno",
                    norm=norm,
                    alpha=0.60,
                    origin="lower",
                    extent=extent,
                )
            axis.contour(
                gtv_slice.astype(float),
                levels=[0.5],
                colors=["#e31a1c"],
                linewidths=2.4,
                origin="lower",
                extent=extent,
            )
            axis.set(
                xlim=(0, LOW_NX * LOW_PX),
                ylim=(0, LOW_NY * LOW_PY),
                xlabel="столбцы DICOM / patient x, мм",
                ylabel="строки DICOM / patient y, мм",
                title=title,
            )
            axis.title.set_fontsize(12)
            axis.grid(alpha=0.14)

        for x in (
            centre_col_mm - BEAM_RADIUS_MM,
            centre_col_mm + BEAM_RADIUS_MM,
        ):
            ax_correct.axvline(x, color="#f2b134", lw=1.5, ls="--")
        ax_correct.annotate(
            "",
            xy=(centre_col_mm, 49.5),
            xytext=(centre_col_mm, -2.0),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 3.0,
                "color": "#f2b134",
            },
            annotation_clip=False,
        )
        ax_correct.text(
            centre_col_mm + 1.0,
            2.0,
            "GPS x = −800 мм\nA → P",
            color="#5b4a20",
            fontsize=10,
            ha="left",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )

        for y in (
            centre_row_mm - BEAM_RADIUS_MM,
            centre_row_mm + BEAM_RADIUS_MM,
        ):
            ax_lateral.axhline(y, color="#d95f02", lw=1.4, ls="--")
        ax_lateral.annotate(
            "",
            xy=(1.0, centre_row_mm),
            xytext=(51.5, centre_row_mm),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 2.7,
                "color": "#d95f02",
            },
            annotation_clip=False,
        )
        ax_lateral.text(
            49.0,
            centre_row_mm + 1.0,
            "источник: Geant4 y = +800 мм\n"
            "направление: (0, −1, 0)\n"
            "энерговклад ещё не рассчитан",
            color="#8c2d04",
            fontsize=10,
            ha="right",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )

        centre_row = int(round(centroid[1] / LOW_PY - 0.5))
        centre_col = int(round(centroid[0] / LOW_PX - 0.5))
        hu_line = ct_slice[:, centre_col]
        gtv_line = gtv_slice[:, centre_col]
        current = metrics[0]
        body_lo = int(current["body_lo_index"])
        body_hi = int(current["body_hi_index"])
        gtv_lo = int(current["gtv_lo_index"])
        gtv_hi = int(current["gtv_hi_index"])
        depth = (np.arange(len(hu_line)) - body_lo) * LOW_PY
        density = hu_to_density(hu_line)
        cumulative = np.cumsum(density[body_lo : body_hi + 1]) * LOW_PY
        body_depth = depth[body_lo : body_hi + 1]

        ax_profile.plot(depth, hu_line, color="#3b4cc0", lw=1.5)
        ax_profile.axhline(-500, color="#777777", lw=1.0, ls=":")
        ax_profile.axvspan(
            depth[gtv_lo],
            depth[gtv_hi],
            color="#e31a1c",
            alpha=0.13,
            label="GTVp на центральном луче",
        )
        ax_profile.axvline(
            depth[centre_row],
            color="#e31a1c",
            lw=1.6,
            ls="--",
            label="центроид GTVp",
        )
        ax_profile.set(
            xlim=(0, depth[body_hi]),
            xlabel="геометрическая глубина от входной поверхности A→P, мм",
            ylabel="HU",
            title="Центральный луч: тело и GTVp",
        )
        ax_profile.grid(alpha=0.18)
        ax_profile.legend(loc="upper left", fontsize=9)
        wet_axis = ax_profile.twinx()
        wet_axis.plot(
            body_depth,
            cumulative,
            color="#1b9e77",
            lw=2.0,
            label="интеграл плотности",
        )
        wet_axis.set_ylabel(
            "интеграл плотности, мм воды",
            color="#1b9e77",
            fontsize=11,
        )
        wet_axis.tick_params(axis="y", colors="#1b9e77")

        ax_table.axis("off")
        columns = [
            "Направление",
            "до центра,\nмм воды",
            "до дистальной\nграницы GTVp,\nмм воды",
            "E к центру,\nМэВ*",
            "E к дистальной\nгранице,\nМэВ*",
        ]
        cell_text = []
        for row in metrics:
            cell_text.append(
                [
                    str(row["direction"]).split(";")[0],
                    f'{float(row["density_integral_to_centroid_mm"]):.1f}',
                    f'{float(row["density_integral_to_gtv_distal_mm"]):.1f}',
                    f'{float(row["screening_energy_to_centroid_MeV"]):.1f}',
                    f'{float(row["screening_energy_to_gtv_distal_MeV"]):.1f}',
                ]
            )
        table = ax_table.table(
            cellText=cell_text,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            bbox=(0.0, 0.28, 1.0, 0.66),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.2)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e8edf4")
                cell.set_text_props(weight="bold")
            elif row == 1:
                cell.set_facecolor("#fff4d6")
            elif row == 4:
                cell.set_facecolor("#fde4d0")
        ax_table.text(
            0.0,
            0.18,
            "* Энергия — только скрининговая оценка по водному пробегу; "
            "окончательно её выбирают по отдельной Geant4-серии энергий.",
            transform=ax_table.transAxes,
            fontsize=9.5,
            va="top",
            wrap=True,
        )
        ax_table.text(
            0.0,
            0.08,
            "60 МэВ соответствует примерно 31 мм пробега в воде: "
            "для найденных путей это не недостаточная, а, вероятно, "
            "избыточная энергия моноэнергетического пучка.",
            transform=ax_table.transAxes,
            fontsize=10.2,
            weight="bold",
            color="#8c2d04",
            va="top",
            wrap=True,
        )

        fig.suptitle(
            "Аудит направления и глубины протонного пучка в фантоме крысы\n"
            "GTVp, DICOM HFS: старый расчёт A→P и проверяемый вход L→R",
            fontsize=15,
        )
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


def plot_lateral_entry_geometry(
    ct_volume: np.ndarray,
    gtv_mask: np.ndarray,
    centroid: np.ndarray,
    output_stem: Path,
) -> None:
    """Plot axial and leg-longitudinal views for the proposed L-to-R entry."""
    centre_z = int(round(centroid[2] / LOW_PZ - 0.5))
    centre_row = int(round(centroid[1] / LOW_PY - 0.5))
    centre_z = int(np.clip(centre_z, 0, LOW_NZ - 1))
    centre_row = int(np.clip(centre_row, 0, LOW_NY - 1))

    axial_ct = ct_volume[centre_z]
    axial_gtv = gtv_mask[centre_z]
    longitudinal_ct = ct_volume[:, centre_row, :]
    # The projected contour preserves the complete cranio-caudal GTV extent
    # when the hand-drawn contours do not intersect every central-row voxel.
    longitudinal_gtv = np.any(gtv_mask, axis=1)

    axial_extent = (0, LOW_NX * LOW_PX, 0, LOW_NY * LOW_PY)
    longitudinal_extent = (0, LOW_NX * LOW_PX, 0, LOW_NZ * LOW_PZ)
    centre_col_mm = float(centroid[0])
    centre_row_mm = float(centroid[1])
    centre_z_mm = float(centroid[2])

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            }
        )
        fig, (ax_axial, ax_longitudinal) = plt.subplots(
            1,
            2,
            figsize=(15.5, 7.0),
            constrained_layout=True,
        )

        panels = (
            (
                ax_axial,
                axial_ct,
                axial_gtv,
                axial_extent,
                centre_row_mm,
                "Поперечный срез через GTVp",
                "строки DICOM / patient y, мм",
            ),
            (
                ax_longitudinal,
                longitudinal_ct,
                longitudinal_gtv,
                longitudinal_extent,
                centre_z_mm,
                "Продольный срез вдоль лапы через GTVp",
                "ось срезов DICOM / patient z, мм",
            ),
        )
        for axis, ct_image, gtv_image, extent, arrow_y, title, ylabel in panels:
            axis.imshow(
                ct_image,
                cmap="gray",
                vmin=-600,
                vmax=1200,
                origin="lower",
                extent=extent,
            )
            axis.contour(
                gtv_image.astype(float),
                levels=[0.5],
                colors=["#e31a1c"],
                linewidths=2.4,
                origin="lower",
                extent=extent,
            )
            for boundary in (
                arrow_y - BEAM_RADIUS_MM,
                arrow_y + BEAM_RADIUS_MM,
            ):
                axis.axhline(
                    boundary,
                    color="#d95f02",
                    lw=1.4,
                    ls="--",
                )
            axis.annotate(
                "",
                xy=(1.0, arrow_y),
                xytext=(51.5, arrow_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 3.0,
                    "color": "#d95f02",
                },
                annotation_clip=False,
            )
            axis.text(
                49.2,
                arrow_y + 0.9,
                "источник справа\nL→R; Geant4 −y",
                color="#8c2d04",
                fontsize=10,
                ha="right",
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.80,
                },
            )
            axis.set(
                xlim=(0, LOW_NX * LOW_PX),
                ylim=(extent[2], extent[3]),
                xlabel="столбцы DICOM / patient x, мм",
                ylabel=ylabel,
                title=title,
            )
            axis.grid(alpha=0.14)

        fig.suptitle(
            "Проверяемая геометрия протонного пучка L→R\n"
            "энерговклад нового прогона ещё не рассчитан",
            fontsize=15,
        )
        fig.text(
            0.5,
            0.01,
            "Красный контур — GTVp; пунктир — границы поля радиусом 14,4 мм.",
            ha="center",
            fontsize=10,
        )
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


def stream_reduced_lateral_energy(
    voxel_map: Path,
    centroid: np.ndarray,
    nx: int = 64,
    ny: int = 64,
    nz: int = 100,
    sx: float = 0.8,
    sy: float = 0.8,
    sz: float = 39.8 / 100.0,
    slab_mm: float = 2.0,
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a coarse L-to-R energy map without loading protobuf objects."""
    axial = np.zeros((nx, ny), dtype=np.float64)
    longitudinal = np.zeros((nz, ny), dtype=np.float64)
    point_heap: list[tuple[float, int, int, int]] = []
    centre_x = int(round(float(centroid[1]) / sx - 0.5))
    centre_z = int(round(float(centroid[2]) / sz - 0.5))
    x_half = int(round(slab_mm / sx))
    z_half = int(round(slab_mm / sz))

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
                        value_field, value_wire = (
                            value_tag >> 3,
                            value_tag & 7,
                        )
                        if value_field == 2 and value_wire == 1:
                            dep_energy = struct.unpack(
                                "<d",
                                handle.read(8),
                            )[0]
                        else:
                            _skip_field(handle, value_wire)
                    handle.seek(value_end)
                else:
                    _skip_field(handle, entry_wire)
            handle.seek(entry_end)
            if voxel_id is None or dep_energy <= 0.0:
                continue

            # Reduced phantom mapping:
            # Geant4 x = DICOM row, Geant4 y = DICOM column.
            gx = voxel_id % nx
            gy = (voxel_id // nx) % ny
            gz = voxel_id // (nx * ny)
            if gz < 0 or gz >= nz:
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
                (gy + 0.5) * sy,
                (gx + 0.5) * sx,
                (gz + 0.5) * sz,
                energy,
            )
            for energy, gx, gy, gz in point_heap
        ],
        dtype=float,
    )
    return axial, longitudinal, points


def plot_lateral_entry_composite(
    ct_volume: np.ndarray,
    gtv_mask: np.ndarray,
    contours: list[np.ndarray],
    centroid: np.ndarray,
    metrics: list[dict[str, float | str]],
    output_stem: Path,
    lateral_energy: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """Recreate the preferred two-slice/3D layout and add an L-to-R profile."""
    centre_z = int(
        np.clip(
            round(centroid[2] / LOW_PZ - 0.5),
            0,
            LOW_NZ - 1,
        )
    )
    centre_row = int(
        np.clip(
            round(centroid[1] / LOW_PY - 0.5),
            0,
            LOW_NY - 1,
        )
    )
    centre_col = int(
        np.clip(
            round(centroid[0] / LOW_PX - 0.5),
            0,
            LOW_NX - 1,
        )
    )

    axial_ct = ct_volume[centre_z]
    axial_gtv = gtv_mask[centre_z]
    longitudinal_ct = ct_volume[:, centre_row, :]
    longitudinal_gtv = np.any(gtv_mask, axis=1)
    body_surface = clean_body_surface(ct_volume)
    axial_extent = (0, LOW_NX * LOW_PX, 0, LOW_NY * LOW_PY)
    longitudinal_extent = (0, LOW_NX * LOW_PX, 0, LOW_NZ * LOW_PZ)

    # Central-ray profile from the high-column side: image right -> image left,
    # patient L->R and Geant4 -y.
    lateral = metrics[3]
    body_lo = int(lateral["body_lo_index"])
    body_hi = int(lateral["body_hi_index"])
    gtv_lo = int(lateral["gtv_lo_index"])
    gtv_hi = int(lateral["gtv_hi_index"])
    profile_indices = np.arange(body_hi, body_lo - 1, -1)
    profile_depth = (body_hi - profile_indices) * LOW_PX
    hu_profile = axial_ct[centre_row, profile_indices]
    density_profile = hu_to_density(hu_profile)
    wet_profile = np.cumsum(density_profile) * LOW_PX
    gtv_entry_depth = (body_hi - gtv_hi) * LOW_PX
    gtv_distal_depth = (body_hi - gtv_lo) * LOW_PX
    centroid_depth = (body_hi - centre_col) * LOW_PX
    energy_norm = None
    energy_depth_profile = None
    energy_profile_per_primary = None
    fallback_density_overlay = False
    if lateral_energy is not None:
        axial_energy, longitudinal_energy, energy_points = lateral_energy
        positive_energy = np.concatenate(
            (
                axial_energy[axial_energy > 0],
                longitudinal_energy[longitudinal_energy > 0],
            )
        )
        if positive_energy.size:
            energy_norm = LogNorm(
                vmin=float(np.percentile(positive_energy, 30)),
                vmax=float(np.percentile(positive_energy, 99.7)),
            )
        energy_col_spacing = (
            LOW_NX * LOW_PX / axial_energy.shape[1]
        )
        energy_col_mm = (
            np.arange(axial_energy.shape[1]) + 0.5
        ) * energy_col_spacing
        entry_surface_mm = (body_hi + 0.5) * LOW_PX
        energy_depth = entry_surface_mm - energy_col_mm
        energy_valid = (
            (energy_depth >= 0.0)
            & (energy_depth <= float(profile_depth[-1]))
        )
        energy_order = np.argsort(energy_depth[energy_valid])
        energy_depth_profile = energy_depth[energy_valid][energy_order]
        # The map contains the sum over 10 000 primaries.  Integrate the
        # central axial slab over the transverse beam coordinate and report
        # deposited energy per primary particle.
        energy_profile_per_primary = (
            axial_energy.sum(axis=0)[energy_valid][energy_order] / 10000.0
        )
    else:
        fallback_density_overlay = True
        axial_density = hu_to_density(axial_ct)
        axial_row_mm = (np.arange(LOW_NY) + 0.5) * LOW_PY
        axial_field = (
            np.abs(axial_row_mm[:, None] - float(centroid[1]))
            <= BEAM_RADIUS_MM
        )
        axial_energy = np.where(
            axial_field & (axial_ct > -500.0),
            axial_density,
            0.0,
        )
        longitudinal_density = hu_to_density(longitudinal_ct)
        longitudinal_z_mm = (np.arange(LOW_NZ) + 0.5) * LOW_PZ
        longitudinal_field = (
            np.abs(
                longitudinal_z_mm[:, None] - float(centroid[2])
            )
            <= BEAM_RADIUS_MM
        )
        longitudinal_energy = np.where(
            longitudinal_field & (longitudinal_ct > -500.0),
            longitudinal_density,
            0.0,
        )
        energy_norm = Normalize(vmin=0.2, vmax=1.8)
        energy_points = np.empty((0, 4), dtype=float)

    configurator = MatplotlibConfigurator()
    configurator.apply_custom_styles()
    try:
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )
        fig = plt.figure(figsize=(17.0, 12.5), constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=(1.0, 1.15),
            width_ratios=(1.05, 0.95),
        )
        ax_axial = fig.add_subplot(grid[0, 0])
        ax_longitudinal = fig.add_subplot(grid[0, 1])
        ax_3d = fig.add_subplot(grid[1, 0], projection="3d")
        ax_profile = fig.add_subplot(grid[1, 1])

        panels = (
            (
                ax_axial,
                axial_ct,
                axial_gtv,
                axial_energy,
                axial_extent,
                float(centroid[1]),
                (
                    "Поперечный срез через GTVp, "
                    f"z = {centroid[2]:.1f} мм"
                ),
                "строки DICOM / patient y, мм",
            ),
            (
                ax_longitudinal,
                longitudinal_ct,
                longitudinal_gtv,
                longitudinal_energy,
                longitudinal_extent,
                float(centroid[2]),
                (
                    "Продольный срез вдоль лапы через GTVp, "
                    f"y = {centroid[1]:.1f} мм"
                ),
                "ось срезов DICOM / patient z, мм",
            ),
        )
        energy_heatmap = None
        for (
            axis,
            ct_image,
            gtv_image,
            energy_image,
            extent,
            beam_y,
            title,
            ylabel,
        ) in panels:
            axis.imshow(
                ct_image,
                cmap="gray",
                vmin=-600,
                vmax=1200,
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            if energy_norm is not None and energy_image is not None:
                energy_heatmap = axis.imshow(
                    np.ma.masked_less_equal(energy_image, 0),
                    cmap="inferno",
                    norm=energy_norm,
                    alpha=0.62,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                )
            axis.contour(
                gtv_image.astype(float),
                levels=[0.5],
                colors=["#e31a1c"],
                linewidths=2.4,
                origin="lower",
                extent=extent,
            )
            for boundary in (
                beam_y - BEAM_RADIUS_MM,
                beam_y + BEAM_RADIUS_MM,
            ):
                axis.axhline(
                    boundary,
                    color="#f2b134",
                    lw=1.4,
                    ls="--",
                )
            axis.annotate(
                "",
                xy=(0.8, beam_y),
                xytext=(51.0, beam_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 2.8,
                    "color": "#f2b134",
                },
                annotation_clip=False,
            )
            axis.text(
                49.2,
                beam_y + 1.1,
                "источник\nL→R; Geant4 −y",
                ha="right",
                va="bottom",
                color="#5b4a20",
                fontsize=9.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.76,
                    "pad": 1.5,
                },
            )
            axis.set(
                xlim=(0, LOW_NX * LOW_PX),
                ylim=(extent[2], extent[3]),
                xlabel="столбцы DICOM / patient x, мм",
                ylabel=ylabel,
                title=title,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.14)

        surface_indices = np.argwhere(body_surface)
        if len(surface_indices) > 7000:
            rng = np.random.default_rng(20260727)
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
                lw=1.7,
                alpha=0.92,
            )
        if energy_norm is not None and energy_points.size:
            point_order = np.argsort(energy_points[:, 3])
            plotted_points = energy_points[point_order]
            ax_3d.scatter(
                plotted_points[:, 0],
                plotted_points[:, 1],
                plotted_points[:, 2],
                c=plotted_points[:, 3],
                cmap="inferno",
                norm=energy_norm,
                s=12.0,
                alpha=0.68,
                edgecolors="none",
                depthshade=False,
            )
        elif fallback_density_overlay:
            coarse_ct = ct_volume[::2, ::2, ::2]
            coarse_z = (
                np.arange(coarse_ct.shape[0]) + 0.5
            ) * LOW_PZ * 2
            coarse_row = (
                np.arange(coarse_ct.shape[1]) + 0.5
            ) * LOW_PY * 2
            cylinder_cross_section = (
                (
                    coarse_row[None, :, None] - float(centroid[1])
                )
                ** 2
                + (
                    coarse_z[:, None, None] - float(centroid[2])
                )
                ** 2
                <= BEAM_RADIUS_MM**2
            )
            planned_indices = np.argwhere(
                (coarse_ct > -500.0) & cylinder_cross_section
            )
            if len(planned_indices) > 4500:
                rng = np.random.default_rng(20260728)
                planned_indices = planned_indices[
                    rng.choice(len(planned_indices), 4500, replace=False)
                ]
            ax_3d.scatter(
                (planned_indices[:, 2] + 0.5) * LOW_PX * 2,
                (planned_indices[:, 1] + 0.5) * LOW_PY * 2,
                (planned_indices[:, 0] + 0.5) * LOW_PZ * 2,
                color="#f28e2b",
                s=7.0,
                alpha=0.36,
                depthshade=False,
            )

        theta = np.linspace(0, 2 * np.pi, 64)
        beam_x = np.linspace(0.0, LOW_NX * LOW_PX, 24)
        xx, tt = np.meshgrid(beam_x, theta)
        yy = float(centroid[1]) + BEAM_RADIUS_MM * np.cos(tt)
        zz = float(centroid[2]) + BEAM_RADIUS_MM * np.sin(tt)
        ax_3d.plot_wireframe(
            xx,
            yy,
            zz,
            rstride=8,
            cstride=5,
            color="#f2b134",
            linewidth=0.65,
            alpha=0.20,
        )
        ax_3d.quiver(
            LOW_NX * LOW_PX,
            float(centroid[1]),
            float(centroid[2]),
            -(LOW_NX * LOW_PX),
            0,
            0,
            color="#f2b134",
            linewidth=2.8,
            arrow_length_ratio=0.07,
        )
        ax_3d.scatter(
            [centroid[0]],
            [centroid[1]],
            [centroid[2]],
            color="#e31a1c",
            s=42,
            depthshade=False,
        )
        draw_box(
            ax_3d,
            (0, LOW_NX * LOW_PX),
            (0, LOW_NY * LOW_PY),
            (0, LOW_NZ * LOW_PZ),
        )
        ax_3d.set(
            xlim=(0, LOW_NX * LOW_PX),
            ylim=(0, LOW_NY * LOW_PY),
            zlim=(0, LOW_NZ * LOW_PZ),
            xlabel="DICOM x / Geant4 y, мм",
            ylabel="DICOM y / Geant4 x, мм",
            zlabel="z, мм",
            title="Трёхмерная расчётная геометрия",
        )
        ax_3d.set_box_aspect(
            (
                LOW_NX * LOW_PX,
                LOW_NY * LOW_PY,
                LOW_NZ * LOW_PZ,
            ),
            zoom=1.18,
        )
        ax_3d.view_init(elev=24, azim=-62)
        legend_handles = [
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
                    label="поле, r = 14,4 мм; L→R",
                ),
            ]
        if fallback_density_overlay:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#f28e2b",
                    markeredgecolor="none",
                    markersize=6,
                    label="ткань внутри поля (не энерговклад)",
                )
            )
        elif energy_points.size:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#f06f4d",
                    markeredgecolor="none",
                    markersize=6,
                    label="воксели энерговклада (цвет = кэВ)",
                )
            )
        ax_3d.legend(
            handles=legend_handles,
            loc="upper left",
        )

        if (
            energy_depth_profile is not None
            and energy_profile_per_primary is not None
        ):
            ax_profile.plot(
                energy_depth_profile,
                energy_profile_per_primary,
                color="#d95f02",
                lw=2.2,
                marker="o",
                markersize=3.2,
                label="энерговклад, кэВ/протон",
            )
        else:
            ax_profile.plot(
                profile_depth,
                hu_profile,
                color="#3b4cc0",
                lw=1.5,
                label="HU",
            )
            ax_profile.axhline(-500, color="#777777", lw=1.0, ls=":")
        ax_profile.axvspan(
            gtv_entry_depth,
            gtv_distal_depth,
            color="#e31a1c",
            alpha=0.13,
            label="GTVp",
        )
        ax_profile.axvline(
            centroid_depth,
            color="#e31a1c",
            lw=1.5,
            ls="--",
            label="центр GTVp",
        )
        if energy_profile_per_primary is not None:
            ax_profile.set(
                xlim=(0, float(profile_depth[-1])),
                ylim=(0, None),
                xlabel="глубина от поверхности опухолевой лапы L→R, мм",
                ylabel="энерговклад в центральном слое, кэВ/протон",
                title="Энерговклад по глубине и радиологическая глубина",
            )
        else:
            ax_profile.set(
                xlim=(0, float(profile_depth[-1])),
                xlabel="глубина от поверхности опухолевой лапы L→R, мм",
                ylabel="HU",
                title="Центральный луч: HU и радиологическая глубина",
            )
        ax_profile.grid(alpha=0.18)
        wet_axis = ax_profile.twinx()
        wet_axis.plot(
            profile_depth,
            wet_profile,
            color="#1b9e77",
            lw=2.1,
            label="радиологическая глубина",
        )
        wet_axis.set_ylabel(
            "интеграл плотности, мм воды",
            color="#1b9e77",
        )
        wet_axis.tick_params(axis="y", colors="#1b9e77")
        lines_left, labels_left = ax_profile.get_legend_handles_labels()
        lines_right, labels_right = wet_axis.get_legend_handles_labels()
        ax_profile.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            loc="upper left",
        )
        ax_profile.text(
            0.98,
            0.04,
            (
                "скрининг: "
                f"{float(lateral['screening_energy_to_centroid_MeV']):.1f} "
                "МэВ к центру; "
                f"{float(lateral['screening_energy_to_gtv_distal_MeV']):.1f} "
                "МэВ к дистальной границе"
            ),
            transform=ax_profile.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            bbox={
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.86,
            },
        )

        if energy_heatmap is not None:
            colorbar = fig.colorbar(
                energy_heatmap,
                ax=[ax_axial, ax_longitudinal],
                orientation="horizontal",
                fraction=0.045,
                pad=0.07,
                aspect=50,
            )
            if fallback_density_overlay:
                colorbar.set_label(
                    "Плотность ткани по HU внутри геометрического поля, "
                    "г/см³ (не энерговклад)",
                    fontsize=10,
                )
                title_second_line = (
                    "геометрический контроль L→R; цвет — плотность ткани "
                    "в поле, не энерговклад"
                )
            else:
                colorbar.set_label(
                    "Энерговклад 10 000 протонов 40 МэВ, кэВ "
                    "(сетка 0,8×0,8×0,398 мм; логарифмическая шкала)",
                    fontsize=10,
                )
                title_second_line = (
                    "QGSP_INCLXX + G4EmLivermore; "
                    "рассчитанный энерговклад 10 000 протонов 40 МэВ"
                )
        else:
            title_second_line = (
                "QGSP_INCLXX + G4EmLivermore; "
                "новый энерговклад ещё не рассчитан"
            )
        fig.suptitle(
            "Протонный пучок в воксельном фантоме крысы: "
            "проверяемое направление L→R\n"
            + title_second_line,
            fontsize=16,
        )
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


def write_summary(
    path: Path,
    metrics: list[dict[str, float | str]],
    centroid: np.ndarray,
    metadata: dict[str, str],
) -> None:
    fieldnames = list(metrics[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    metadata_path = path.with_name("proton_axis_audit_metadata.csv")
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerow(["gtvp_centroid_column_mm", f"{centroid[0]:.4f}"])
        writer.writerow(["gtvp_centroid_row_mm", f"{centroid[1]:.4f}"])
        writer.writerow(["gtvp_centroid_z_mm", f"{centroid[2]:.4f}"])
        writer.writerow(["body_threshold_HU", "-500"])
        writer.writerow(
            [
                "phantom_axis_mapping",
                "Geant4 x=DICOM row; Geant4 y=DICOM column",
            ]
        )
        for key, value in metadata.items():
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, default=DEFAULT_CT_DIR)
    parser.add_argument("--rtstruct", type=Path, default=DEFAULT_RS)
    parser.add_argument("--voxel-map", type=Path, default=DEFAULT_VOXEL_MAP)
    parser.add_argument(
        "--reduced-lateral-map",
        type=Path,
        default=DEFAULT_REDUCED_LATERAL_MAP,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    (
        z_values,
        by_z,
        crop_parameters,
        contours,
        centroid,
    ) = geometry_and_contours(args.ct_dir, args.rtstruct)
    ct_volume = load_ct_crop(z_values, by_z, crop_parameters)
    gtv_mask = rasterise_gtv(contours)
    centre_z = int(round(centroid[2] / LOW_PZ - 0.5))
    centre_row = int(round(centroid[1] / LOW_PY - 0.5))
    centre_col = int(round(centroid[0] / LOW_PX - 0.5))
    ct_slice = ct_volume[centre_z]
    gtv_slice = gtv_mask[centre_z]

    energy_xy, _, _ = stream_energy_projections(
        args.voxel_map,
        BEAM_CENTRE_Y_MM,
        centroid[2],
    )
    energy_low = energy_xy.reshape(
        NY // 2,
        2,
        NX // 2,
        2,
    ).sum((1, 3))
    metrics = direction_metrics(
        ct_slice,
        gtv_slice,
        centre_row,
        centre_col,
    )
    output_stem = args.output_dir / "proton_beam_current_vs_lateral_entry"
    plot_audit(
        ct_slice,
        gtv_slice,
        centroid,
        energy_low,
        metrics,
        output_stem,
    )
    longitudinal_stem = (
        args.output_dir / "proton_L_to_R_axial_and_leg_longitudinal_geometry"
    )
    plot_lateral_entry_geometry(
        ct_volume,
        gtv_mask,
        centroid,
        longitudinal_stem,
    )
    composite_stem = (
        args.output_dir / "proton_L_to_R_geometry_3d_and_depth_profile"
    )
    if not args.reduced_lateral_map.exists():
        raise FileNotFoundError(
            "A calculated L-to-R energy map is required; "
            f"not found: {args.reduced_lateral_map}"
        )
    lateral_energy = stream_reduced_lateral_energy(
        args.reduced_lateral_map,
        centroid,
    )
    plot_lateral_entry_composite(
        ct_volume,
        gtv_mask,
        contours,
        centroid,
        metrics,
        composite_stem,
        lateral_energy,
    )
    write_summary(
        args.output_dir / "proton_beam_direction_metrics.csv",
        metrics,
        centroid,
        central_ct_metadata(args.ct_dir),
    )
    print(f"Written {output_stem}.png/.svg/.pdf")
    print(f"Written {longitudinal_stem}.png/.svg/.pdf")
    print(f"Written {composite_stem}.png/.svg/.pdf")
    for row in metrics:
        console_direction = (
            str(row["direction"]).replace("→", "->").replace("−", "-")
        )
        print(
            f"{console_direction}: "
            f'WET(center)={row["density_integral_to_centroid_mm"]:.2f} mm, '
            f'E_screen={row["screening_energy_to_centroid_MeV"]:.1f} MeV'
        )


if __name__ == "__main__":
    main()
