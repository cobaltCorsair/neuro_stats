# coding: utf-8
"""Bridge GEANT4/NPLibrary voxel-dose protobuf payloads into NumPy-friendly objects."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

try:
    from work_with_prepared_data.radiobioligy_project.survival.nifti_contour_bridge import (
        build_structure_assignments_from_nifti,
        build_structure_assignments_from_nifti_grid,
        is_nifti_contour_path,
    )
    from work_with_prepared_data.radiobioligy_project.survival.rtstruct_bridge import (
        RTDoseGridGeometry,
        build_structure_assignments_from_rtstruct,
    )
except ModuleNotFoundError:
    from survival.nifti_contour_bridge import (
        build_structure_assignments_from_nifti,
        build_structure_assignments_from_nifti_grid,
        is_nifti_contour_path,
    )
    from survival.rtstruct_bridge import RTDoseGridGeometry, build_structure_assignments_from_rtstruct

try:
    import pydicom
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments without pydicom
    pydicom = None

_PROTO_IMPORT_BASES = (
    "work_with_prepared_data.radiobioligy_project.survival.proto",
    "survival.proto",
)
_DEFAULT_TUMOR_ALIASES = ("tumor", "target", "gtv", "ctv", "ptv")
_RT_DOSE_SUFFIXES = (".dcm", ".dicom")


@dataclass(frozen=True)
class VoxelDose:
    """One entry from a GEANT4 voxel dose map."""

    voxel_id: int
    dose_gy: float
    let_kev_um: float
    dep_energy_mev: float
    n_events: int
    rel_error: float
    scaled_dose: float
    eqd_gy: float
    mev2gy: float


@dataclass(frozen=True)
class DoseMap:
    """Deserialized dose map with geometry and structure metadata."""

    voxels: Dict[int, VoxelDose]
    grid_shape: Tuple[int, int, int]
    voxel_size_mm: Tuple[float, float, float]
    structure_ids: Dict[int, str]
    voxel_structure_ids: Dict[int, Tuple[int, ...]]

    def tumor_voxel_ids(self, structure_name: str = "tumor") -> List[int]:
        """Return voxel ids that belong to the requested structure."""
        candidate_ids = _resolve_structure_ids(
            structure_name=structure_name,
            structure_ids=self.structure_ids,
            voxel_structure_ids=self.voxel_structure_ids,
        )
        if not candidate_ids:
            return []
        return sorted(
            voxel_id
            for voxel_id, structure_ids in self.voxel_structure_ids.items()
            if any(structure_id in candidate_ids for structure_id in structure_ids)
        )

    def dose_volume_histogram(self, voxel_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return a cumulative DVH as (dose_thresholds_gy, volume_fraction_ge_threshold)."""
        doses = self._select_values(voxel_ids, "dose_gy")
        if doses.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        thresholds, counts = np.unique(np.sort(doses), return_counts=True)
        cumulative_fraction = np.cumsum(counts[::-1], dtype=float)[::-1] / float(doses.size)
        return thresholds.astype(float), cumulative_fraction.astype(float)

    def mean_dose(self, voxel_ids: List[int]) -> float:
        """Return the arithmetic mean dose over the selected voxels."""
        doses = self._select_values(voxel_ids, "dose_gy")
        if doses.size == 0:
            return 0.0
        return float(np.mean(doses))

    def mean_let(self, voxel_ids: List[int]) -> float:
        """Return the arithmetic mean LET over the selected voxels."""
        lets = self._select_values(voxel_ids, "let_kev_um")
        if lets.size == 0:
            return 0.0
        return float(np.mean(lets))

    def _select_values(self, voxel_ids: Iterable[int], attribute: str) -> np.ndarray:
        values = [
            float(getattr(self.voxels[voxel_id], attribute))
            for voxel_id in voxel_ids
            if voxel_id in self.voxels
        ]
        return np.asarray(values, dtype=float)


def read_dose_map(
    dose_path: Path,
    geometry_path: Optional[Path] = None,
    contour_path: Optional[Path] = None,
    contour_structure_name: Optional[str] = None,
) -> DoseMap:
    """Deserialize either a GEANT4 dose protobuf or RT Dose DICOM into a DoseMap."""
    dose_path = Path(dose_path)
    if is_rt_dose_path(dose_path):
        return _read_rt_dose_map(
            dose_path,
            contour_path=contour_path,
            contour_structure_name=contour_structure_name,
        )
    if geometry_path is None:
        raise ValueError("geometry_path is required for GEANT4 protobuf dose maps.")
    geometry_message = _read_input_voxel_map(geometry_path)
    structure_ids, voxel_structure_ids = _resolve_structure_context(
        geometry_message=geometry_message,
        contour_path=contour_path,
        contour_structure_name=contour_structure_name,
    )
    wise_message = _read_full_voxel_map(dose_path)
    if not wise_message.totDose:
        raise ValueError(
            f"{dose_path} does not contain `totDose`; use read_full_dose_map() for component maps."
        )
    return _build_dose_map(
        entries=wise_message.totDose,
        geometry_message=geometry_message,
        structure_ids=structure_ids,
        voxel_structure_ids=voxel_structure_ids,
    )


def read_full_dose_map(
    path: Path,
    geometry_path: Optional[Path],
    contour_path: Optional[Path] = None,
    contour_structure_name: Optional[str] = None,
) -> Dict[str, DoseMap]:
    """Deserialize a fullVoxelMap payload into per-component DoseMap objects."""
    path = Path(path)
    if is_rt_dose_path(path):
        raise ValueError("RT Dose inputs do not support mixed-field component maps.")
    if geometry_path is None:
        raise ValueError("geometry_path is required for GEANT4 mixed-field protobuf inputs.")
    geometry_message = _read_input_voxel_map(geometry_path)
    structure_ids, voxel_structure_ids = _resolve_structure_context(
        geometry_message=geometry_message,
        contour_path=contour_path,
        contour_structure_name=contour_structure_name,
    )
    wise_message = _read_full_voxel_map(path)
    component_names = ("totDose", "protonDose", "midDose", "mainDose", "stuffDose")
    return {
        component_name: _build_dose_map(
            entries=getattr(wise_message, component_name),
            geometry_message=geometry_message,
            structure_ids=structure_ids,
            voxel_structure_ids=voxel_structure_ids,
        )
        for component_name in component_names
    }


def is_rt_dose_path(path: Path) -> bool:
    """Return True when the path looks like a DICOM RT Dose file."""
    normalized = str(Path(path)).strip().lower()
    return normalized.endswith(_RT_DOSE_SUFFIXES)


def is_rtstruct_path(path: Path) -> bool:
    """Return True when the path points to a DICOM RT Structure Set dataset."""
    if pydicom is None or not is_rt_dose_path(path):
        return False
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["Modality"])
    except Exception:
        return False
    modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
    return modality == "RTSTRUCT"


def _load_proto_modules():
    errors: list[ModuleNotFoundError] = []
    for base in _PROTO_IMPORT_BASES:
        try:
            input_module = importlib.import_module(f"{base}.NPInputVoxelData_pb2")
            wise_module = importlib.import_module(f"{base}.NPWiseVoxelData_pb2")
            return input_module, wise_module
        except ModuleNotFoundError as exc:
            errors.append(exc)
    raise ModuleNotFoundError(
        "GEANT4 protobuf bridge is unavailable. Generate bindings in "
        "`survival/proto` and install the `protobuf` runtime first."
    ) from errors[-1]


def _read_input_voxel_map(path: Path):
    input_module, _ = _load_proto_modules()
    message = input_module.InputVoxelMap()
    message.ParseFromString(Path(path).read_bytes())
    if message.xLen <= 0 or message.yLen <= 0 or message.zLen <= 0:
        raise ValueError(f"{path} does not contain a valid InputVoxelMap grid.")
    return message


def _read_contour_meta(path: Path):
    input_module, _ = _load_proto_modules()
    message = input_module.ContourMeta()
    message.ParseFromString(Path(path).read_bytes())
    return message


def _read_full_voxel_map(path: Path):
    _, wise_module = _load_proto_modules()
    message = wise_module.fullVoxelMap()
    message.ParseFromString(Path(path).read_bytes())
    return message


def _read_rt_dose_map(
    path: Path,
    *,
    contour_path: Optional[Path],
    contour_structure_name: Optional[str],
) -> DoseMap:
    dataset = _read_rt_dose_dataset(path)
    dose_grid_zyx = _extract_rt_dose_grid_gy(dataset, path)
    geometry = _extract_rt_dose_geometry(dataset, dose_grid_zyx=dose_grid_zyx, path=path)
    frame_count, row_count, column_count = dose_grid_zyx.shape
    structure_ids, voxel_structure_ids = _resolve_rt_dose_structure_context(
        geometry=geometry,
        dose_grid_zyx=dose_grid_zyx,
        contour_path=contour_path,
        contour_structure_name=contour_structure_name,
    )

    voxels: Dict[int, VoxelDose] = {}
    for z_index in range(frame_count):
        for y_index in range(row_count):
            for x_index in range(column_count):
                voxel_id = x_index + column_count * y_index + column_count * row_count * z_index
                dose_gy = float(dose_grid_zyx[z_index, y_index, x_index])
                voxels[voxel_id] = VoxelDose(
                    voxel_id=voxel_id,
                    dose_gy=dose_gy,
                    let_kev_um=0.0,
                    dep_energy_mev=0.0,
                    n_events=0,
                    rel_error=0.0,
                    scaled_dose=dose_gy,
                    eqd_gy=dose_gy,
                    mev2gy=0.0,
                )

    return DoseMap(
        voxels=voxels,
        grid_shape=geometry.grid_shape,
        voxel_size_mm=geometry.voxel_size_mm,
        structure_ids=structure_ids,
        voxel_structure_ids=voxel_structure_ids,
    )


def _read_rt_dose_dataset(path: Path):
    if pydicom is None:
        raise ModuleNotFoundError(
            "RT Dose support requires `pydicom`. Install it in the project environment first."
        )
    dataset = pydicom.dcmread(str(path))
    modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
    if modality and modality != "RTDOSE":
        raise ValueError(f"{path} is not an RT Dose dataset; Modality={modality!r}.")
    return dataset


def _extract_rt_dose_grid_gy(dataset, path: Path) -> np.ndarray:
    dose_units = str(getattr(dataset, "DoseUnits", "") or "").strip().upper()
    if dose_units and dose_units != "GY":
        raise ValueError(f"{path} uses DoseUnits={dose_units!r}; only absolute Gy RT Dose is supported.")

    scaling = float(getattr(dataset, "DoseGridScaling", 1.0) or 1.0)
    dose_grid = np.asarray(dataset.pixel_array, dtype=float)
    if dose_grid.ndim == 2:
        dose_grid = dose_grid[np.newaxis, :, :]
    if dose_grid.ndim != 3:
        raise ValueError(f"{path} must contain a 3D RT Dose grid; got array shape {tuple(dose_grid.shape)}.")

    rows = int(getattr(dataset, "Rows", dose_grid.shape[-2]) or dose_grid.shape[-2])
    columns = int(getattr(dataset, "Columns", dose_grid.shape[-1]) or dose_grid.shape[-1])
    if dose_grid.shape[1] != rows or dose_grid.shape[2] != columns:
        raise ValueError(
            f"{path} dose grid shape {tuple(dose_grid.shape)} does not match Rows/Columns ({rows}, {columns})."
        )
    return dose_grid.astype(float) * scaling


def _extract_rt_dose_geometry(dataset, *, dose_grid_zyx: np.ndarray, path: Path) -> RTDoseGridGeometry:
    frame_count, row_count, column_count = dose_grid_zyx.shape
    voxel_size_mm = _extract_rt_dose_voxel_size_mm(dataset, frame_count, path)
    frame_offsets_mm = _extract_rt_dose_frame_offsets_mm(dataset, frame_count=frame_count, path=path)
    image_position = getattr(dataset, "ImagePositionPatient", None)
    if image_position is None or len(image_position) < 3:
        raise ValueError(f"{path} does not define ImagePositionPatient for the RT Dose grid.")
    image_orientation = getattr(dataset, "ImageOrientationPatient", None)
    if image_orientation is None or len(image_orientation) < 6:
        raise ValueError(f"{path} does not define ImageOrientationPatient for the RT Dose grid.")
    axis_x_direction = tuple(float(value) for value in image_orientation[:3])
    axis_y_direction = tuple(float(value) for value in image_orientation[3:6])
    return RTDoseGridGeometry(
        grid_shape=(int(column_count), int(row_count), int(frame_count)),
        origin_mm=tuple(float(value) for value in image_position[:3]),
        axis_x_direction=axis_x_direction,
        axis_y_direction=axis_y_direction,
        frame_offsets_mm=frame_offsets_mm,
        voxel_size_mm=voxel_size_mm,
    )


def _extract_rt_dose_voxel_size_mm(dataset, frame_count: int, path: Path) -> Tuple[float, float, float]:
    pixel_spacing = getattr(dataset, "PixelSpacing", None)
    if pixel_spacing is None or len(pixel_spacing) < 2:
        raise ValueError(f"{path} does not define PixelSpacing for the RT Dose grid.")
    row_spacing_mm = float(pixel_spacing[0])
    column_spacing_mm = float(pixel_spacing[1])
    slice_spacing_mm = _resolve_rt_dose_slice_spacing_mm(dataset, frame_count=frame_count, path=path)
    return (column_spacing_mm, row_spacing_mm, slice_spacing_mm)


def _extract_rt_dose_frame_offsets_mm(dataset, *, frame_count: int, path: Path) -> Tuple[float, ...]:
    offsets = getattr(dataset, "GridFrameOffsetVector", None)
    if offsets is not None:
        if isinstance(offsets, (str, bytes)):
            offset_values = np.asarray([float(offsets)], dtype=float)
        else:
            raw_values = np.atleast_1d(offsets)
            offset_values = np.asarray([float(value) for value in raw_values], dtype=float)
        if offset_values.size == frame_count:
            return tuple(float(value) for value in offset_values.tolist())
        if frame_count == 1 and offset_values.size == 1:
            return (float(offset_values[0]),)

    for attribute_name in ("SliceThickness", "SpacingBetweenSlices"):
        attribute_value = getattr(dataset, attribute_name, None)
        if attribute_value is None:
            continue
        spacing_mm = float(attribute_value)
        if spacing_mm > 0.0:
            return tuple(float(index * spacing_mm) for index in range(frame_count))

    if frame_count == 1:
        return (0.0,)
    raise ValueError(
        f"{path} does not provide enough z-offset information for the RT Dose frames."
    )


def _resolve_rt_dose_slice_spacing_mm(dataset, *, frame_count: int, path: Path) -> float:
    offset_values = np.asarray(_extract_rt_dose_frame_offsets_mm(dataset, frame_count=frame_count, path=path), dtype=float)
    if offset_values.size >= 2:
        diffs = np.abs(np.diff(offset_values))
        if not np.allclose(diffs, diffs[0], atol=1e-6):
            raise ValueError(f"{path} uses a non-uniform GridFrameOffsetVector, which is not supported yet.")
        return float(diffs[0])
    for attribute_name in ("SliceThickness", "SpacingBetweenSlices"):
        attribute_value = getattr(dataset, attribute_name, None)
        if attribute_value is None:
            continue
        spacing_mm = float(attribute_value)
        if spacing_mm > 0.0:
            return spacing_mm
    if frame_count <= 1:
        return 1.0
    raise ValueError(
        f"{path} does not provide enough z-spacing information for a multi-frame RT Dose grid."
    )


def _resolve_rt_dose_structure_context(
    *,
    geometry: RTDoseGridGeometry,
    dose_grid_zyx: np.ndarray,
    contour_path: Optional[Path],
    contour_structure_name: Optional[str],
) -> tuple[Dict[int, str], Dict[int, Tuple[int, ...]]]:
    if contour_path is not None:
        contour_path = Path(contour_path)
        if not is_nifti_contour_path(contour_path):
            if is_rtstruct_path(contour_path):
                assignments = build_structure_assignments_from_rtstruct(
                    contour_path,
                    geometry,
                    structure_name=contour_structure_name,
                    base_structure_ids={},
                    base_voxel_structure_ids={},
                )
                return assignments.structure_ids, assignments.voxel_structure_ids
            raise ValueError(
                "RT Dose contour support currently accepts aligned NIfTI masks or RTSTRUCT DICOM. "
                "ContourMeta protobuf requires GEANT4 InputVoxelMap structure membership."
            )
        assignments = build_structure_assignments_from_nifti_grid(
            contour_path,
            geometry.grid_shape,
            structure_name=contour_structure_name,
            base_structure_ids={},
            base_voxel_structure_ids={},
        )
        return assignments.structure_ids, assignments.voxel_structure_ids

    resolved_name = str(contour_structure_name or "tumor").strip() or "tumor"
    structure_ids = {1: resolved_name}
    positive_voxel_ids = np.flatnonzero(dose_grid_zyx.ravel(order="C") > 0.0).astype(int)
    if positive_voxel_ids.size == 0:
        positive_voxel_ids = np.arange(int(dose_grid_zyx.size), dtype=int)
    voxel_structure_ids = {int(voxel_id): (1,) for voxel_id in positive_voxel_ids}
    return structure_ids, voxel_structure_ids


def _build_dose_map(
    entries: Mapping[int, object],
    geometry_message,
    structure_ids: Mapping[int, str],
    voxel_structure_ids: Mapping[int, Tuple[int, ...]],
) -> DoseMap:
    voxels = {
        int(voxel_id): _convert_voxel_entry(int(voxel_id), entry)
        for voxel_id, entry in entries.items()
    }
    return DoseMap(
        voxels=voxels,
        grid_shape=(
            int(geometry_message.xLen),
            int(geometry_message.yLen),
            int(geometry_message.zLen),
        ),
        voxel_size_mm=(
            float(geometry_message.xSize),
            float(geometry_message.ySize),
            float(geometry_message.zSize),
        ),
        structure_ids=structure_ids,
        voxel_structure_ids=voxel_structure_ids,
    )


def _resolve_structure_context(
    *,
    geometry_message,
    contour_path: Optional[Path],
    contour_structure_name: Optional[str],
) -> tuple[Dict[int, str], Dict[int, Tuple[int, ...]]]:
    voxel_structure_ids = _extract_voxel_structure_ids(geometry_message)
    structure_ids = _build_structure_name_map(
        geometry_message=geometry_message,
        contour_message=None,
        voxel_structure_ids=voxel_structure_ids,
    )
    if contour_path is None:
        return structure_ids, voxel_structure_ids

    contour_path = Path(contour_path)
    if is_nifti_contour_path(contour_path):
        assignments = build_structure_assignments_from_nifti(
            contour_path,
            geometry_message,
            structure_name=contour_structure_name,
            base_structure_ids=structure_ids,
            base_voxel_structure_ids=voxel_structure_ids,
        )
        return assignments.structure_ids, assignments.voxel_structure_ids

    contour_message = _read_contour_meta(contour_path)
    structure_ids = _build_structure_name_map(
        geometry_message=geometry_message,
        contour_message=contour_message,
        voxel_structure_ids=voxel_structure_ids,
    )
    return structure_ids, voxel_structure_ids


def _extract_voxel_structure_ids(geometry_message) -> Dict[int, Tuple[int, ...]]:
    voxel_structure_ids: Dict[int, Tuple[int, ...]] = {}
    for voxel_id, entry in geometry_message.voxData.items():
        structures = sorted(
            int(structure_id)
            for structure_id, marker in entry.voxelStructureId.items()
            if int(marker) != 0
        )
        if not structures:
            marker = int(round(float(entry.voxelStructureMarker)))
            if marker != 0:
                structures = [marker]
        voxel_structure_ids[int(voxel_id)] = tuple(structures)
    return voxel_structure_ids


def _build_structure_name_map(
    geometry_message,
    contour_message,
    voxel_structure_ids: Mapping[int, Tuple[int, ...]],
) -> Dict[int, str]:
    structure_ids: Dict[int, str] = {}
    if contour_message is not None:
        structure_ids.update(
            {int(structure_id): str(name) for structure_id, name in contour_message.voxelStructureNames.items()}
        )
    for structure_tuple in voxel_structure_ids.values():
        for structure_id in structure_tuple:
            structure_ids.setdefault(int(structure_id), f"structure_{int(structure_id)}")
    return structure_ids


def _convert_voxel_entry(voxel_id: int, entry) -> VoxelDose:
    return VoxelDose(
        voxel_id=int(getattr(entry, "vId", voxel_id) or voxel_id),
        dose_gy=float(getattr(entry, "dose", 0.0)),
        let_kev_um=float(getattr(entry, "letd", 0.0)),
        dep_energy_mev=float(getattr(entry, "depEnergy", 0.0)),
        n_events=int(getattr(entry, "nEvents", 0)),
        rel_error=_calculate_relative_error(
            dep_energy=float(getattr(entry, "depEnergy", 0.0)),
            dep_energy_sq=float(getattr(entry, "depEnergy2", 0.0)),
            n_events=int(getattr(entry, "nEvents", 0)),
        ),
        scaled_dose=float(getattr(entry, "scaledDose", 0.0)),
        eqd_gy=float(getattr(entry, "doseGyEQD", 0.0)),
        mev2gy=float(getattr(entry, "mev2gy", 0.0)),
    )


def _calculate_relative_error(dep_energy: float, dep_energy_sq: float, n_events: int) -> float:
    if n_events <= 1 or dep_energy <= 0.0:
        return 0.0
    mean_energy = dep_energy / float(n_events)
    if mean_energy <= 0.0:
        return 0.0
    variance = max(dep_energy_sq / float(n_events) - mean_energy * mean_energy, 0.0)
    return float(math.sqrt(variance) / mean_energy)


def _resolve_structure_ids(
    structure_name: str,
    structure_ids: Mapping[int, str],
    voxel_structure_ids: Mapping[int, Tuple[int, ...]],
) -> set[int]:
    requested = _normalize_structure_name(structure_name)
    if not requested:
        return set()

    matches: set[int] = set()
    if requested.isdigit():
        matches.add(int(requested))

    for structure_id, raw_name in structure_ids.items():
        normalized_name = _normalize_structure_name(raw_name)
        if requested in normalized_name:
            matches.add(int(structure_id))
            continue
        if requested in _DEFAULT_TUMOR_ALIASES and any(
            alias in normalized_name for alias in _DEFAULT_TUMOR_ALIASES
        ):
            matches.add(int(structure_id))

    if matches:
        return matches

    if requested in _DEFAULT_TUMOR_ALIASES:
        unique_structure_ids = {
            int(structure_id)
            for structure_tuple in voxel_structure_ids.values()
            for structure_id in structure_tuple
        }
        if len(unique_structure_ids) == 1:
            return unique_structure_ids

    return set()


def _normalize_structure_name(value: str) -> str:
    return "".join(character.lower() for character in str(value).strip() if character.isalnum())
