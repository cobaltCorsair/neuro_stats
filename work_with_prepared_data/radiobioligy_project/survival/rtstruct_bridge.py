# coding: utf-8
"""Rasterize RT Structure Set contours onto an aligned RT Dose grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import pydicom
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments without pydicom
    pydicom = None

_DEFAULT_TUMOR_ALIASES = ("tumor", "target", "gtv", "ctv", "ptv")


@dataclass(frozen=True)
class RTDoseGridGeometry:
    """Dose-grid geometry expressed in patient coordinates."""

    grid_shape: Tuple[int, int, int]
    origin_mm: Tuple[float, float, float]
    axis_x_direction: Tuple[float, float, float]
    axis_y_direction: Tuple[float, float, float]
    frame_offsets_mm: Tuple[float, ...]
    voxel_size_mm: Tuple[float, float, float]


@dataclass(frozen=True)
class StructureAssignments:
    """Resolved structure names and per-voxel membership for one contour source."""

    structure_ids: Dict[int, str]
    voxel_structure_ids: Dict[int, Tuple[int, ...]]


def build_structure_assignments_from_rtstruct(
    rtstruct_path: Path,
    geometry: RTDoseGridGeometry,
    *,
    structure_name: Optional[str] = None,
    base_structure_ids: Optional[Mapping[int, str]] = None,
    base_voxel_structure_ids: Optional[Mapping[int, Sequence[int]]] = None,
) -> StructureAssignments:
    """Convert RTSTRUCT contours into voxel structure assignments on the aligned dose grid."""
    if pydicom is None:
        raise ModuleNotFoundError(
            "RT Structure Set support requires `pydicom`. Install it in the project environment first."
        )

    dataset = pydicom.dcmread(str(rtstruct_path), stop_before_pixels=False)
    modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
    if modality and modality != "RTSTRUCT":
        raise ValueError(f"{rtstruct_path} is not an RT Structure Set dataset; Modality={modality!r}.")

    roi_names = {
        int(item.ROINumber): str(getattr(item, "ROIName", "") or f"structure_{int(item.ROINumber)}")
        for item in getattr(dataset, "StructureSetROISequence", [])
    }
    selected_roi_ids = _select_roi_numbers(structure_name, roi_names)
    if not selected_roi_ids:
        available_names = ", ".join(sorted(roi_names.values())) or "<none>"
        raise ValueError(
            f"{rtstruct_path} does not contain a structure matching {structure_name!r}. "
            f"Available ROI names: {available_names}."
        )

    structure_ids = {
        int(structure_id): str(name)
        for structure_id, name in (base_structure_ids or {}).items()
    }
    voxel_structure_ids = {
        int(voxel_id): tuple(int(structure_id) for structure_id in structure_ids_for_voxel)
        for voxel_id, structure_ids_for_voxel in (base_voxel_structure_ids or {}).items()
    }

    for roi_id in selected_roi_ids:
        structure_ids.setdefault(int(roi_id), roi_names.get(int(roi_id), f"structure_{int(roi_id)}"))

    roi_contours = {
        int(getattr(item, "ReferencedROINumber", -1)): item
        for item in getattr(dataset, "ROIContourSequence", [])
    }

    for roi_id in selected_roi_ids:
        contour_item = roi_contours.get(int(roi_id))
        if contour_item is None:
            continue
        slice_masks = _rasterize_roi_contours(contour_item, geometry)
        for slice_index, slice_mask in slice_masks.items():
            row_indices, column_indices = np.nonzero(slice_mask)
            for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
                voxel_id = (
                    int(column_index)
                    + int(geometry.grid_shape[0]) * int(row_index)
                    + int(geometry.grid_shape[0]) * int(geometry.grid_shape[1]) * int(slice_index)
                )
                existing_ids = set(voxel_structure_ids.get(voxel_id, ()))
                existing_ids.add(int(roi_id))
                voxel_structure_ids[voxel_id] = tuple(sorted(existing_ids))

    return StructureAssignments(
        structure_ids=structure_ids,
        voxel_structure_ids=voxel_structure_ids,
    )


def _select_roi_numbers(structure_name: Optional[str], roi_names: Mapping[int, str]) -> list[int]:
    normalized_requested = _normalize_structure_name(structure_name or "")
    if not normalized_requested:
        return sorted(int(roi_id) for roi_id in roi_names)

    selected: set[int] = set()
    if normalized_requested.isdigit():
        requested_id = int(normalized_requested)
        if requested_id in roi_names:
            selected.add(requested_id)

    for roi_id, roi_name in roi_names.items():
        normalized_roi_name = _normalize_structure_name(roi_name)
        if normalized_requested in normalized_roi_name:
            selected.add(int(roi_id))
            continue
        if normalized_requested in _DEFAULT_TUMOR_ALIASES and any(
            alias in normalized_roi_name for alias in _DEFAULT_TUMOR_ALIASES
        ):
            selected.add(int(roi_id))
    return sorted(selected)


def _rasterize_roi_contours(contour_item, geometry: RTDoseGridGeometry) -> Dict[int, np.ndarray]:
    slice_masks: Dict[int, np.ndarray] = {}
    for contour in getattr(contour_item, "ContourSequence", []):
        if str(getattr(contour, "ContourGeometricType", "") or "").strip().upper() != "CLOSED_PLANAR":
            continue
        raw_data = getattr(contour, "ContourData", None)
        if raw_data is None:
            continue
        points = np.asarray(list(raw_data), dtype=float)
        if points.size < 9:
            continue
        polygon_xyz = points.reshape(-1, 3)
        slice_index = _resolve_slice_index(polygon_xyz, geometry)
        polygon_mask = _rasterize_single_polygon(polygon_xyz, geometry, slice_index)
        if not np.any(polygon_mask):
            continue
        if slice_index not in slice_masks:
            slice_masks[slice_index] = polygon_mask
        else:
            slice_masks[slice_index] ^= polygon_mask
    return slice_masks


def _resolve_slice_index(polygon_xyz: np.ndarray, geometry: RTDoseGridGeometry) -> int:
    origin = np.asarray(geometry.origin_mm, dtype=float)
    axis_x = _normalized_vector(np.asarray(geometry.axis_x_direction, dtype=float))
    axis_y = _normalized_vector(np.asarray(geometry.axis_y_direction, dtype=float))
    axis_z = _normalized_vector(np.cross(axis_x, axis_y))
    frame_offsets = np.asarray(geometry.frame_offsets_mm, dtype=float)

    offsets = np.matmul(polygon_xyz - origin, axis_z)
    mean_offset = float(np.mean(offsets))
    slice_index = int(np.argmin(np.abs(frame_offsets - mean_offset)))
    slice_spacing = float(geometry.voxel_size_mm[2])
    tolerance = max(slice_spacing * 0.55, 1e-3)
    if np.max(np.abs(offsets - frame_offsets[slice_index])) > tolerance:
        raise ValueError(
            "RTSTRUCT contour plane does not align with the RT Dose z-grid within tolerance."
        )
    return slice_index


def _rasterize_single_polygon(
    polygon_xyz: np.ndarray,
    geometry: RTDoseGridGeometry,
    slice_index: int,
) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    origin = np.asarray(geometry.origin_mm, dtype=float)
    axis_x = _normalized_vector(np.asarray(geometry.axis_x_direction, dtype=float))
    axis_y = _normalized_vector(np.asarray(geometry.axis_y_direction, dtype=float))
    axis_z = _normalized_vector(np.cross(axis_x, axis_y))
    slice_origin = origin + axis_z * float(geometry.frame_offsets_mm[slice_index])

    polygon_relative = polygon_xyz - slice_origin
    polygon_x = np.matmul(polygon_relative, axis_x) / float(geometry.voxel_size_mm[0])
    polygon_y = np.matmul(polygon_relative, axis_y) / float(geometry.voxel_size_mm[1])
    polygon_xy = np.column_stack([polygon_x, polygon_y])

    x_min = max(int(np.floor(float(np.min(polygon_x)))) - 1, 0)
    x_max = min(int(np.ceil(float(np.max(polygon_x)))) + 1, int(geometry.grid_shape[0]) - 1)
    y_min = max(int(np.floor(float(np.min(polygon_y)))) - 1, 0)
    y_max = min(int(np.ceil(float(np.max(polygon_y)))) + 1, int(geometry.grid_shape[1]) - 1)
    if x_max < x_min or y_max < y_min:
        return np.zeros((int(geometry.grid_shape[1]), int(geometry.grid_shape[0])), dtype=bool)

    x_indices = np.arange(x_min, x_max + 1, dtype=float)
    y_indices = np.arange(y_min, y_max + 1, dtype=float)
    grid_x, grid_y = np.meshgrid(x_indices, y_indices)
    sample_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    closed_polygon_xy = polygon_xy
    if not np.allclose(closed_polygon_xy[0], closed_polygon_xy[-1]):
        closed_polygon_xy = np.vstack([closed_polygon_xy, closed_polygon_xy[0]])
    local_mask = MplPath(closed_polygon_xy).contains_points(sample_points, radius=1e-6)
    local_mask = local_mask.reshape(len(y_indices), len(x_indices))

    slice_mask = np.zeros((int(geometry.grid_shape[1]), int(geometry.grid_shape[0])), dtype=bool)
    slice_mask[y_min : y_max + 1, x_min : x_max + 1] = local_mask
    return slice_mask


def _normalized_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Dose grid orientation contains a zero-length direction vector.")
    return vector / norm


def _normalize_structure_name(value: str) -> str:
    return "".join(character.lower() for character in str(value).strip() if character.isalnum())
