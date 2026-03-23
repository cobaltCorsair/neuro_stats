import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments without nibabel
    nib = None

try:
    import survival.pipeline_geant4_to_prediction as pipeline_module
    from survival.pipeline_geant4_to_prediction import cli_main, parse_cli, run_prediction_pipeline
    from survival.proto import NPInputVoxelData_pb2, NPWiseVoxelData_pb2
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival import (
        pipeline_geant4_to_prediction as pipeline_module,
    )
    from work_with_prepared_data.radiobioligy_project.survival.pipeline_geant4_to_prediction import (
        cli_main,
        parse_cli,
        run_prediction_pipeline,
    )
    from work_with_prepared_data.radiobioligy_project.survival.proto import (
        NPInputVoxelData_pb2,
        NPWiseVoxelData_pb2,
    )


class PredictionPipelineTests(unittest.TestCase):
    def test_parse_cli_accepts_manual_and_mixed_field_options(self) -> None:
        args = parse_cli(
            [
                "dose.pb",
                "geometry.ivz",
                "--contour-path",
                "contour.pb",
                "--alpha",
                "0.1",
                "--beta",
                "0.02",
                "--schedule-days",
                "0,1,2",
                "--mixed-field",
                "--component-family-map",
                "protonDose=p,mainDose=y",
                "--bed-eqd2-fractions",
                "1,5,10",
            ]
        )

        self.assertEqual(args.dose_pb_path, "dose.pb")
        self.assertEqual(args.geometry_ivz_path, "geometry.ivz")
        self.assertEqual(args.contour_path, "contour.pb")
        self.assertAlmostEqual(args.alpha, 0.1, places=8)
        self.assertAlmostEqual(args.beta, 0.02, places=8)
        self.assertEqual(args.schedule_days, "0,1,2")
        self.assertTrue(args.mixed_field)
        self.assertEqual(args.component_family_map, "protonDose=p,mainDose=y")
        self.assertEqual(args.bed_eqd2_fractions, "1,5,10")

    def test_pipeline_writes_expected_outputs_from_manual_alpha_beta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            geometry_path, contour_path, dose_path = _write_simple_geant_inputs(tmp_path)
            output_dir = tmp_path / "output"

            result = run_prediction_pipeline(
                dose_pb_path=dose_path,
                geometry_ivz_path=geometry_path,
                contour_path=contour_path,
                manual_alpha_beta=(0.1, 0.02),
                structure_name="tumor",
                schedule_days=[0.0],
                growth_duration_days=2.0,
                growth_time_step_days=1.0,
                output_dir=output_dir,
            )

            self.assertAlmostEqual(result["volumetric_sf"].mean_dose_gy, 3.0, places=8)
            self.assertEqual(len(result["dvh"][0]), 2)
            expected_files = {
                output_dir / "dvh_tumor.csv",
                output_dir / "sf_per_voxel.csv",
                output_dir / "aggregated_params.json",
                output_dir / "growth_curve.csv",
                output_dir / "bed_eqd2_table.csv",
                output_dir / "summary.json",
            }
            self.assertTrue(expected_files.issubset(set(result["output_files"])))
            for path in expected_files:
                self.assertTrue(path.exists(), msg=str(path))

            aggregated = json.loads((output_dir / "aggregated_params.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(aggregated["effective_alpha"], 0.1, places=8)
            self.assertAlmostEqual(aggregated["effective_beta"], 0.02, places=8)
            self.assertEqual(summary["structure_name"], "tumor")
            self.assertEqual(summary["schedule_days"], [0.0])

    def test_pipeline_can_resolve_let_params_from_fit_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            geometry_path, contour_path, dose_path = _write_simple_geant_inputs(tmp_path)
            summary_csv = tmp_path / "fit_results.csv"
            with summary_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "sf_mode",
                        "response_mode",
                        "model_kind",
                        "family",
                        "status",
                        "train_count",
                        "alpha",
                        "beta",
                        "alpha_0",
                        "lambda_alpha",
                    ]
                )
                writer.writerow(["absolute", "scalar", "classic_lq", "y", "ok", 3, 0.12, 0.03, "", ""])

            result = run_prediction_pipeline(
                dose_pb_path=dose_path,
                geometry_ivz_path=geometry_path,
                contour_path=contour_path,
                fit_results_csv=summary_csv,
                structure_name="tumor",
                schedule_days=[0.0],
                growth_duration_days=1.0,
                growth_time_step_days=1.0,
                output_dir=tmp_path / "output_from_csv",
            )

            self.assertAlmostEqual(result["volumetric_sf"].effective_alpha, 0.12, places=8)
            self.assertAlmostEqual(result["volumetric_sf"].effective_beta, 0.03, places=8)

    @unittest.skipIf(nib is None, "nibabel runtime unavailable")
    def test_pipeline_accepts_nifti_contour_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            geometry_path, _, dose_path = _write_simple_geant_inputs(tmp_path)
            mask_path = tmp_path / "tumor_mask.nii.gz"
            output_dir = tmp_path / "output_from_nifti"

            mask = np.zeros((2, 2, 1), dtype=np.uint8)
            mask[0, 0, 0] = 1
            mask[1, 0, 0] = 1
            nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

            result = run_prediction_pipeline(
                dose_pb_path=dose_path,
                geometry_ivz_path=geometry_path,
                contour_path=mask_path,
                manual_alpha_beta=(0.1, 0.02),
                structure_name="tumor",
                schedule_days=[0.0],
                growth_duration_days=1.0,
                growth_time_step_days=1.0,
                output_dir=output_dir,
            )

            self.assertAlmostEqual(result["volumetric_sf"].mean_dose_gy, 3.0, places=8)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["structure_name"], "tumor")

    def test_cli_main_invokes_pipeline_and_prints_summary(self) -> None:
        fake_result = {
            "volumetric_sf": SimpleNamespace(
                mean_dose_gy=3.0,
                mean_sf=0.55,
                effective_alpha=0.1,
                effective_beta=0.02,
            ),
            "output_files": [Path("C:/tmp/summary.json")],
        }

        with patch.object(pipeline_module, "run_prediction_pipeline", return_value=fake_result) as mocked_run:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                result = cli_main(
                    [
                        "dose.pb",
                        "geometry.ivz",
                        "--contour-path",
                        "contour.pb",
                        "--alpha",
                        "0.1",
                        "--beta",
                        "0.02",
                        "--schedule-days",
                        "0,1,2",
                        "--component-family-map",
                        "protonDose=p,mainDose=y",
                        "--bed-eqd2-fractions",
                        "1,5",
                        "--output-dir",
                        "out_dir",
                    ]
                )

        mocked_run.assert_called_once()
        kwargs = mocked_run.call_args.kwargs
        self.assertEqual(kwargs["dose_pb_path"], Path("dose.pb"))
        self.assertEqual(kwargs["geometry_ivz_path"], Path("geometry.ivz"))
        self.assertEqual(kwargs["contour_path"], Path("contour.pb"))
        self.assertEqual(kwargs["manual_alpha_beta"], (0.1, 0.02))
        self.assertEqual(kwargs["schedule_days"], [0.0, 1.0, 2.0])
        self.assertEqual(kwargs["component_family_map"], {"protonDose": "p", "mainDose": "y"})
        self.assertEqual(kwargs["bed_eqd2_fractions"], [1, 5])
        self.assertEqual(kwargs["output_dir"], Path("out_dir"))
        self.assertIs(result, fake_result)
        stdout = buffer.getvalue()
        self.assertIn("mean_dose_gy=3.000000", stdout)
        self.assertIn("summary_json=C:\\tmp\\summary.json", stdout.replace("/", "\\"))


def _write_simple_geant_inputs(base_dir: Path) -> tuple[Path, Path, Path]:
    geometry_path = base_dir / "geometry.ivz"
    contour_path = base_dir / "contour.pb"
    dose_path = base_dir / "dose.pb"

    geometry = NPInputVoxelData_pb2.InputVoxelMap(
        xLen=2,
        yLen=2,
        zLen=1,
        xSize=1.0,
        ySize=1.0,
        zSize=1.0,
    )
    geometry.voxData[1].vId = 1
    geometry.voxData[1].voxelStructureId[7] = 1
    geometry.voxData[2].vId = 2
    geometry.voxData[2].voxelStructureId[7] = 1

    contour = NPInputVoxelData_pb2.ContourMeta()
    contour.voxelStructureNames[7] = "Tumor"

    dose_map_message = NPWiseVoxelData_pb2.fullVoxelMap()
    first = dose_map_message.totDose[1]
    first.vId = 1
    first.dose = 2.0
    first.letd = 5.0
    first.scaledDose = 2.0
    first.doseGyEQD = 2.0
    first.nEvents = 1

    second = dose_map_message.totDose[2]
    second.vId = 2
    second.dose = 4.0
    second.letd = 5.0
    second.scaledDose = 4.0
    second.doseGyEQD = 4.0
    second.nEvents = 1

    geometry_path.write_bytes(geometry.SerializeToString())
    contour_path.write_bytes(contour.SerializeToString())
    dose_path.write_bytes(dose_map_message.SerializeToString())
    return geometry_path, contour_path, dose_path


if __name__ == "__main__":
    unittest.main()
