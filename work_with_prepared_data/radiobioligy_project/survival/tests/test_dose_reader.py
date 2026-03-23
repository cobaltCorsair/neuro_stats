import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments without nibabel
    nib = None

PROTO_IMPORT_ERROR = None

try:
    from survival.dose_reader import read_dose_map, read_full_dose_map
    from survival.proto import NPInputVoxelData_pb2, NPWiseVoxelData_pb2
except ModuleNotFoundError as exc:
    PROTO_IMPORT_ERROR = exc
    try:
        from work_with_prepared_data.radiobioligy_project.survival.dose_reader import (
            read_dose_map,
            read_full_dose_map,
        )
        from work_with_prepared_data.radiobioligy_project.survival.proto import (
            NPInputVoxelData_pb2,
            NPWiseVoxelData_pb2,
        )
        PROTO_IMPORT_ERROR = None
    except ModuleNotFoundError as fallback_exc:
        PROTO_IMPORT_ERROR = fallback_exc


@unittest.skipIf(PROTO_IMPORT_ERROR is not None, f"protobuf runtime unavailable: {PROTO_IMPORT_ERROR}")
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


if __name__ == "__main__":
    unittest.main()
