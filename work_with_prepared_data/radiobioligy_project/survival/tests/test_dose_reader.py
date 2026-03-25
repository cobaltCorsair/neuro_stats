import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments without nibabel
    nib = None

try:
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, generate_uid
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments without pydicom
    Dataset = None
    FileDataset = None
    FileMetaDataset = None
    ExplicitVRLittleEndian = None
    RTDoseStorage = None
    generate_uid = None

DOSE_READER_IMPORT_ERROR = None
PROTO_IMPORT_ERROR = None

try:
    from survival.dose_reader import read_dose_map, read_full_dose_map
except ModuleNotFoundError as exc:
    DOSE_READER_IMPORT_ERROR = exc
    try:
        from work_with_prepared_data.radiobioligy_project.survival.dose_reader import (
            read_dose_map,
            read_full_dose_map,
        )
        DOSE_READER_IMPORT_ERROR = None
    except ModuleNotFoundError as fallback_exc:
        DOSE_READER_IMPORT_ERROR = fallback_exc

if DOSE_READER_IMPORT_ERROR is None:
    try:
        from survival.proto import NPInputVoxelData_pb2, NPWiseVoxelData_pb2
    except ModuleNotFoundError as exc:
        PROTO_IMPORT_ERROR = exc
        try:
            from work_with_prepared_data.radiobioligy_project.survival.proto import (
                NPInputVoxelData_pb2,
                NPWiseVoxelData_pb2,
            )
            PROTO_IMPORT_ERROR = None
        except ModuleNotFoundError as fallback_exc:
            PROTO_IMPORT_ERROR = fallback_exc
else:
    NPInputVoxelData_pb2 = None
    NPWiseVoxelData_pb2 = None


@unittest.skipIf(
    DOSE_READER_IMPORT_ERROR is not None or PROTO_IMPORT_ERROR is not None,
    f"protobuf runtime unavailable: {PROTO_IMPORT_ERROR or DOSE_READER_IMPORT_ERROR}",
)
class DoseReaderTests(unittest.TestCase):


    def test_read_dose_map_deserializes_geometry_structures_and_voxel_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            geometry_path = tmp_path / "geometry.ivz"
            contour_path = tmp_path / "contour.pb"
            dose_path = tmp_path / "dose.pb"

            geometry = NPInputVoxelData_pb2.InputVoxelMap(
                xLen=10,
                yLen=10,
                zLen=10,
                xSize=1.0,
                ySize=1.0,
                zSize=1.0,
            )
            geometry.voxData[101].vId = 101
            geometry.voxData[101].voxelStructureId[7] = 1
            geometry.voxData[102].vId = 102
            geometry.voxData[102].voxelStructureId[7] = 1
            geometry.voxData[203].vId = 203
            geometry.voxData[203].voxelStructureId[9] = 1

            contour = NPInputVoxelData_pb2.ContourMeta()
            contour.voxelStructureNames[7] = "Tumor"
            contour.voxelStructureNames[9] = "NormalTissue"

            dose_map_message = NPWiseVoxelData_pb2.totDoseVoxelMap()
            tumor_a = dose_map_message.totDose[101]
            tumor_a.vId = 101
            tumor_a.depEnergy = 12.0
            tumor_a.depEnergy2 = 44.0
            tumor_a.nEvents = 4
            tumor_a.letd = 1.0
            tumor_a.mev2gy = 0.5
            tumor_a.dose = 2.0
            tumor_a.scaledDose = 2.5
            tumor_a.doseGyEQD = 2.2

            tumor_b = dose_map_message.totDose[102]
            tumor_b.vId = 102
            tumor_b.depEnergy = 20.0
            tumor_b.depEnergy2 = 108.0
            tumor_b.nEvents = 4
            tumor_b.letd = 2.0
            tumor_b.mev2gy = 0.5
            tumor_b.dose = 4.0
            tumor_b.scaledDose = 4.5
            tumor_b.doseGyEQD = 4.1

            healthy = dose_map_message.totDose[203]
            healthy.vId = 203
            healthy.depEnergy = 5.0
            healthy.depEnergy2 = 25.0
            healthy.nEvents = 1
            healthy.letd = 0.5
            healthy.mev2gy = 0.5
            healthy.dose = 1.0
            healthy.scaledDose = 1.0
            healthy.doseGyEQD = 1.0

            geometry_path.write_bytes(geometry.SerializeToString())
            contour_path.write_bytes(contour.SerializeToString())
            dose_path.write_bytes(dose_map_message.SerializeToString())

            dose_map = read_dose_map(dose_path, geometry_path, contour_path)
            tumor_voxels = dose_map.tumor_voxel_ids()
            dvh_doses, dvh_volume = dose_map.dose_volume_histogram(tumor_voxels)

            self.assertEqual(dose_map.grid_shape, (10, 10, 10))
            self.assertEqual(dose_map.voxel_size_mm, (1.0, 1.0, 1.0))
            self.assertEqual(dose_map.structure_ids[7], "Tumor")
            self.assertEqual(tumor_voxels, [101, 102])
            self.assertAlmostEqual(dose_map.mean_dose(tumor_voxels), 3.0, places=8)
            self.assertAlmostEqual(dose_map.mean_let(tumor_voxels), 1.5, places=8)
            self.assertTrue(np.allclose(dvh_doses, [2.0, 4.0]))
            self.assertTrue(np.allclose(dvh_volume, [1.0, 0.5]))
            self.assertAlmostEqual(
                dose_map.voxels[101].rel_error,
                math.sqrt(44.0 / 4.0 - (12.0 / 4.0) ** 2) / (12.0 / 4.0),
                places=8,
            )

    def test_read_full_dose_map_returns_all_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            geometry_path = tmp_path / "geometry.ivz"
            dose_path = tmp_path / "dose.pb"

            geometry = NPInputVoxelData_pb2.InputVoxelMap(
                xLen=2,
                yLen=2,
                zLen=1,
                xSize=0.5,
                ySize=0.5,
                zSize=1.0,
            )
            geometry.voxData[1].vId = 1
            geometry.voxData[1].voxelStructureId[5] = 1
            geometry_path.write_bytes(geometry.SerializeToString())

            dose_map_message = NPWiseVoxelData_pb2.fullVoxelMap()
            dose_map_message.totDose[1].vId = 1
            dose_map_message.totDose[1].dose = 6.0
            dose_map_message.protonDose[1].vId = 1
            dose_map_message.protonDose[1].dose = 4.0
            dose_map_message.mainDose[1].vId = 1
            dose_map_message.mainDose[1].dose = 2.0
            dose_path.write_bytes(dose_map_message.SerializeToString())

            component_maps = read_full_dose_map(dose_path, geometry_path)

            self.assertEqual(
                set(component_maps),
                {"totDose", "protonDose", "midDose", "mainDose", "stuffDose"},
            )
            self.assertAlmostEqual(component_maps["totDose"].voxels[1].dose_gy, 6.0, places=8)
            self.assertAlmostEqual(component_maps["protonDose"].voxels[1].dose_gy, 4.0, places=8)
            self.assertAlmostEqual(component_maps["mainDose"].voxels[1].dose_gy, 2.0, places=8)
            self.assertEqual(component_maps["midDose"].voxels, {})
            self.assertEqual(component_maps["stuffDose"].grid_shape, (2, 2, 1))

    @unittest.skipIf(nib is None, "nibabel runtime unavailable")
    def test_read_dose_map_accepts_binary_nifti_contour_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            geometry_path = tmp_path / "geometry.ivz"
            dose_path = tmp_path / "dose.pb"
            mask_path = tmp_path / "tumor_mask.nii.gz"

            geometry = NPInputVoxelData_pb2.InputVoxelMap(
                xLen=2,
                yLen=2,
                zLen=1,
                xSize=1.0,
                ySize=1.0,
                zSize=1.0,
            )
            geometry.voxData[1].vId = 1
            geometry.voxData[2].vId = 2
            geometry.voxData[3].vId = 3
            geometry.voxData[4].vId = 4

            dose_map_message = NPWiseVoxelData_pb2.totDoseVoxelMap()
            dose_map_message.totDose[1].vId = 1
            dose_map_message.totDose[1].dose = 2.0
            dose_map_message.totDose[2].vId = 2
            dose_map_message.totDose[2].dose = 4.0
            dose_map_message.totDose[3].vId = 3
            dose_map_message.totDose[3].dose = 1.0

            mask = np.zeros((2, 2, 1), dtype=np.uint8)
            mask[0, 0, 0] = 1
            mask[1, 0, 0] = 1

            geometry_path.write_bytes(geometry.SerializeToString())
            dose_path.write_bytes(dose_map_message.SerializeToString())
            nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

            dose_map = read_dose_map(
                dose_path,
                geometry_path,
                contour_path=mask_path,
                contour_structure_name="tumor",
            )

            tumor_voxels = dose_map.tumor_voxel_ids("tumor")
            self.assertEqual(tumor_voxels, [1, 2])
            self.assertAlmostEqual(dose_map.mean_dose(tumor_voxels), 3.0, places=8)


@unittest.skipIf(DOSE_READER_IMPORT_ERROR is not None, f"dose reader unavailable: {DOSE_READER_IMPORT_ERROR}")
class RTDoseReaderTests(unittest.TestCase):
    @unittest.skipIf(FileDataset is None, "pydicom runtime unavailable")
    def test_read_dose_map_accepts_rt_dose_without_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dose_path = _write_rt_dose_file(
                tmp_path / "dose.dcm",
                np.asarray([[[1.0, 2.0], [0.0, 4.0]]], dtype=float),
                row_spacing_mm=1.25,
                column_spacing_mm=1.5,
                slice_spacing_mm=2.0,
            )

            dose_map = read_dose_map(
                dose_path,
                None,
                contour_path=None,
                contour_structure_name="tumor",
            )

            tumor_voxels = dose_map.tumor_voxel_ids("tumor")
            self.assertEqual(dose_map.grid_shape, (2, 2, 1))
            self.assertEqual(dose_map.voxel_size_mm, (1.5, 1.25, 2.0))
            self.assertEqual(tumor_voxels, [0, 1, 3])
            self.assertAlmostEqual(dose_map.mean_dose(tumor_voxels), 7.0 / 3.0, places=8)
            self.assertAlmostEqual(dose_map.voxels[2].dose_gy, 0.0, places=8)
            self.assertAlmostEqual(dose_map.mean_let(tumor_voxels), 0.0, places=8)

    @unittest.skipIf(FileDataset is None or nib is None, "pydicom/nibabel runtime unavailable")
    def test_read_dose_map_accepts_rt_dose_with_aligned_nifti_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dose_path = _write_rt_dose_file(
                tmp_path / "dose.dcm",
                np.asarray([[[1.0, 2.0], [0.0, 4.0]]], dtype=float),
            )
            mask_path = tmp_path / "tumor_mask.nii.gz"
            mask = np.zeros((2, 2, 1), dtype=np.uint8)
            mask[0, 0, 0] = 1
            mask[1, 0, 0] = 1
            nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

            dose_map = read_dose_map(
                dose_path,
                None,
                contour_path=mask_path,
                contour_structure_name="tumor",
            )

            tumor_voxels = dose_map.tumor_voxel_ids("tumor")
            self.assertEqual(tumor_voxels, [0, 1])
            self.assertAlmostEqual(dose_map.mean_dose(tumor_voxels), 1.5, places=8)

    @unittest.skipIf(FileDataset is None, "pydicom runtime unavailable")
    def test_read_dose_map_accepts_rtstruct_contours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dose_path = _write_rt_dose_file(
                tmp_path / "dose.dcm",
                np.asarray([[[1.0, 2.0], [0.0, 4.0]]], dtype=float),
            )
            rtstruct_path = _write_rtstruct_file(
                tmp_path / "struct.dcm",
                roi_name="PTV_High",
                polygon_points_mm=[
                    (-0.5, -0.5, 0.0),
                    (1.5, -0.5, 0.0),
                    (1.5, 0.5, 0.0),
                    (-0.5, 0.5, 0.0),
                ],
            )

            dose_map = read_dose_map(
                dose_path,
                None,
                contour_path=rtstruct_path,
                contour_structure_name="PTV_High",
            )

            tumor_voxels = dose_map.tumor_voxel_ids("PTV_High")
            self.assertEqual(tumor_voxels, [0, 1])
            self.assertAlmostEqual(dose_map.mean_dose(tumor_voxels), 1.5, places=8)
            self.assertIn("PTV_High", dose_map.structure_ids.values())


def _write_rt_dose_file(
    path: Path,
    dose_grid_zyx: np.ndarray,
    *,
    row_spacing_mm: float = 1.0,
    column_spacing_mm: float = 1.0,
    slice_spacing_mm: float = 1.0,
) -> Path:
    if FileDataset is None or FileMetaDataset is None or ExplicitVRLittleEndian is None:
        raise RuntimeError("pydicom runtime unavailable")

    dose_grid = np.asarray(dose_grid_zyx, dtype=float)
    if dose_grid.ndim == 2:
        dose_grid = dose_grid[np.newaxis, :, :]
    scaling = 0.01
    stored_grid = np.round(dose_grid / scaling).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTDoseStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "RTDOSE"
    dataset.DoseUnits = "GY"
    dataset.DoseType = "PHYSICAL"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = int(stored_grid.shape[1])
    dataset.Columns = int(stored_grid.shape[2])
    dataset.NumberOfFrames = str(int(stored_grid.shape[0]))
    dataset.PixelSpacing = [float(row_spacing_mm), float(column_spacing_mm)]
    dataset.GridFrameOffsetVector = [float(index * slice_spacing_mm) for index in range(stored_grid.shape[0])]
    dataset.SliceThickness = float(slice_spacing_mm)
    dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.FrameIncrementPointer = [0x3004000C]
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.DoseGridScaling = float(scaling)
    dataset.PixelData = stored_grid.tobytes()
    dataset.save_as(str(path), enforce_file_format=True)
    return path


def _write_rtstruct_file(
    path: Path,
    *,
    roi_name: str,
    polygon_points_mm: list[tuple[float, float, float]],
) -> Path:
    if FileDataset is None or FileMetaDataset is None or Dataset is None or ExplicitVRLittleEndian is None:
        raise RuntimeError("pydicom runtime unavailable")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "RTSTRUCT"
    dataset.StructureSetLabel = "TEST"
    dataset.StructureSetDate = "20260325"
    dataset.StructureSetTime = "120000"
    dataset.FrameOfReferenceUID = generate_uid()

    structure_roi = Dataset()
    structure_roi.ROINumber = 1
    structure_roi.ReferencedFrameOfReferenceUID = dataset.FrameOfReferenceUID
    structure_roi.ROIName = roi_name
    structure_roi.ROIGenerationAlgorithm = "MANUAL"
    dataset.StructureSetROISequence = [structure_roi]

    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = len(polygon_points_mm)
    contour.ContourData = [float(value) for point in polygon_points_mm for value in point]

    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = [contour]
    dataset.ROIContourSequence = [roi_contour]

    observation = Dataset()
    observation.ObservationNumber = 1
    observation.ReferencedROINumber = 1
    observation.RTROIInterpretedType = "PTV"
    observation.ROIObservationLabel = roi_name
    dataset.RTROIObservationsSequence = [observation]

    dataset.save_as(str(path), enforce_file_format=True)
    return path


if __name__ == "__main__":
    unittest.main()
