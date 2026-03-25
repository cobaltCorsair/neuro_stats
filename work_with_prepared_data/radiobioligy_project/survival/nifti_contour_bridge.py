# coding: utf-8
"""Bridge a 3D Slicer NIfTI mask onto GEANT4 voxel structure assignments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments without nibabel
    nib = None


@dataclass(frozen=True)
class StructureAssignments:
    """Resolved structure names and per-voxel membership for one contour source."""

    structure_ids: Dict[int, str]
    voxel_structure_ids: Dict[int, Tuple[int, ...]]


def is_nifti_contour_path(path: Path) -> bool:
    """Return True when the path points to a NIfTI contour mask."""
    path_text = str(Path(path)).strip().lower()
    return path_text.endswith(".nii") or path_text.endswith(".nii.gz")


def build_structure_assignments_from_nifti(
    mask_path: Path,
    geometry_message,
    *,
    structure_name: Optional[str] = None,
    base_structure_ids: Optional[Mapping[int, str]] = None,
    base_voxel_structure_ids: Optional[Mapping[int, Sequence[int]]] = None,
) -> StructureAssignments:
    """Convert a binary NIfTI label map into GEANT4-style voxel structure assignments.

    The current bridge intentionally supports one binary structure per file.
    The mask must already be resampled to the GEANT4 voxel grid.
    """
    if nib is None:
        raise ModuleNotFoundError(
            "NIfTI contour support requires `nibabel`. Install it in the project environment first."
        )

    expected_shape = (
        int(geometry_message.xLen),
        int(geometry_message.yLen),
        int(geometry_message.zLen),
    )
    geometry_voxel_ids = {int(voxel_id) for voxel_id in geometry_message.voxData.keys()}
    return _build_structure_assignments_from_mask(
        mask_path=mask_path,
        expected_shape=expected_shape,
        geometry_voxel_ids=geometry_voxel_ids,
        structure_name=structure_name,
        base_structure_ids=base_structure_ids,
        base_voxel_structure_ids=base_voxel_structure_ids,
    )


def build_structure_assignments_from_nifti_grid(
    mask_path: Path,
    grid_shape: Sequence[int],
    *,
    structure_name: Optional[str] = None,
    base_structure_ids: Optional[Mapping[int, str]] = None,
    base_voxel_structure_ids: Optional[Mapping[int, Sequence[int]]] = None,
) -> StructureAssignments:
    """Convert a binary NIfTI label map into voxel assignments for an arbitrary aligned grid."""
    if nib is None:
        raise ModuleNotFoundError(
            "NIfTI contour support requires `nibabel`. Install it in the project environment first."
        )
    expected_shape = tuple(int(item) for item in grid_shape)
    if len(expected_shape) != 3:
        raise ValueError(f"grid_shape must contain exactly three dimensions; got {grid_shape}.")
    return _build_structure_assignments_from_mask(
        mask_path=mask_path,
        expected_shape=expected_shape,
        geometry_voxel_ids=None,
        structure_name=structure_name,
        base_structure_ids=base_structure_ids,
        base_voxel_structure_ids=base_voxel_structure_ids,
    )


def _build_structure_assignments_from_mask(
    *,
    mask_path: Path,
    expected_shape: tuple[int, int, int],
    geometry_voxel_ids: Optional[set[int]],
    structure_name: Optional[str],
    base_structure_ids: Optional[Mapping[int, str]],
    base_voxel_structure_ids: Optional[Mapping[int, Sequence[int]]],
) -> StructureAssignments:
    image = nib.load(str(mask_path))
    data = np.asarray(image.get_fdata())
    if data.ndim == 2 and expected_shape[2] == 1:
        data = data[:, :, np.newaxis]
    elif data.ndim > 3:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(
            f"{mask_path} must be a 3D NIfTI mask after normalizing singleton axes; got shape {tuple(data.shape)}."
        )
    if tuple(int(item) for item in data.shape) != expected_shape:
        raise ValueError(
            f"{mask_path} shape {tuple(int(item) for item in data.shape)} does not match "
            f"the GEANT4 grid {expected_shape}. Export or resample the Slicer mask onto the GEANT4 geometry first."
        )

    positive_mask = np.asarray(data > 0, dtype=bool)
    if not np.any(positive_mask):
        raise ValueError(f"{mask_path} does not contain any positive contour voxels.")

    positive_values = np.unique(data[positive_mask])
    if positive_values.size > 1:
        raise ValueError(
            f"{mask_path} contains multiple positive label values {positive_values.tolist()}. "
            "Export one binary NIfTI mask per target structure from 3D Slicer."
        )

    linear_ids = np.flatnonzero(positive_mask.ravel(order="F")).astype(int)
    resolved_offset = (
        _resolve_voxel_id_offset(linear_ids, geometry_voxel_ids)
        if geometry_voxel_ids is not None
        else 0
    )

    merged_structure_ids = {
        int(structure_id): str(name)
        for structure_id, name in (base_structure_ids or {}).items()
    }
    merged_voxel_structure_ids = {
        int(voxel_id): tuple(int(structure_id) for structure_id in structure_ids)
        for voxel_id, structure_ids in (base_voxel_structure_ids or {}).items()
    }

    new_structure_id = _choose_structure_id(
        structure_ids=merged_structure_ids,
        voxel_structure_ids=merged_voxel_structure_ids,
    )
    merged_structure_ids[new_structure_id] = _resolve_structure_name(
        structure_name=structure_name,
        mask_path=mask_path,
    )

    for linear_id in linear_ids:
        voxel_id = int(linear_id + resolved_offset)
        existing_ids = set(merged_voxel_structure_ids.get(voxel_id, ()))
        existing_ids.add(new_structure_id)
        merged_voxel_structure_ids[voxel_id] = tuple(sorted(existing_ids))

    return StructureAssignments(
        structure_ids=merged_structure_ids,
        voxel_structure_ids=merged_voxel_structure_ids,
    )


def _resolve_voxel_id_offset(linear_ids: np.ndarray, geometry_voxel_ids: set[int]) -> int:
    if linear_ids.size == 0:
        return 0
    if all(int(linear_id) in geometry_voxel_ids for linear_id in linear_ids):
        return 0
    if all(int(linear_id) + 1 in geometry_voxel_ids for linear_id in linear_ids):
        return 1
    raise ValueError(
        "Could not align the NIfTI contour mask with geometry voxel ids. "
        "Expected GEANT4 voxel ids to match either zero-based linear indices "
        "(x + Nx*y + Nx*Ny*z) or those indices plus one."
    )


def _choose_structure_id(
    *,
    structure_ids: Mapping[int, str],
    voxel_structure_ids: Mapping[int, Sequence[int]],
) -> int:
    used_ids = {int(structure_id) for structure_id in structure_ids}
    used_ids.update(
        int(structure_id)
        for structure_tuple in voxel_structure_ids.values()
        for structure_id in structure_tuple
    )
    return max(used_ids, default=0) + 1


def _resolve_structure_name(structure_name: Optional[str], mask_path: Path) -> str:
    normalized = str(structure_name or "").strip()
    if normalized:
        return normalized
    name = Path(mask_path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return Path(mask_path).stem
