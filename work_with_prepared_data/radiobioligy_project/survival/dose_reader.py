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
        is_nifti_contour_path,
    )
except ModuleNotFoundError:
    from survival.nifti_contour_bridge import (
        build_structure_assignments_from_nifti,
        is_nifti_contour_path,
    )

_PROTO_IMPORT_BASES = (
    "work_with_prepared_data.radiobioligy_project.survival.proto",
    "survival.proto",
)
_DEFAULT_TUMOR_ALIASES = ("tumor", "target", "gtv", "ctv", "ptv")


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
    geometry_path: Path,
    contour_path: Optional[Path] = None,
    contour_structure_name: Optional[str] = None,
) -> DoseMap:
    """Deserialize a totDoseVoxelMap or fullVoxelMap payload into a DoseMap."""
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
    geometry_path: Path,
    contour_path: Optional[Path] = None,
    contour_structure_name: Optional[str] = None,
) -> Dict[str, DoseMap]:
    """Deserialize a fullVoxelMap payload into per-component DoseMap objects."""
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
