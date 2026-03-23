import math
import unittest

import numpy as np

try:
    from survival.dose_reader import DoseMap, VoxelDose
    from survival.let_parametrization import LETDependentParams
    from survival.tumor_growth_predictor import (
        GeometryReference,
        GrowthModelParameters,
        TreatmentFraction,
        simulate_growth,
    )
    from survival.voxel_sf_calculator import compute_mixed_voxel_sf, compute_voxel_sf
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.dose_reader import (
        DoseMap,
        VoxelDose,
    )
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
    )
    from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
        GeometryReference,
        GrowthModelParameters,
        TreatmentFraction,
        simulate_growth,
    )
    from work_with_prepared_data.radiobioligy_project.survival.voxel_sf_calculator import (
        compute_mixed_voxel_sf,
        compute_voxel_sf,
    )


def _build_dose_map(tumor_doses: list[float], tumor_lets: list[float]) -> DoseMap:
    voxels = {}
    voxel_structure_ids = {}
    for index, (dose_value, let_value) in enumerate(zip(tumor_doses, tumor_lets), start=1):
        voxels[index] = VoxelDose(
            voxel_id=index,
            dose_gy=float(dose_value),
            let_kev_um=float(let_value),
            dep_energy_mev=0.0,
            n_events=1,
            rel_error=0.0,
            scaled_dose=float(dose_value),
            eqd_gy=float(dose_value),
            mev2gy=1.0,
        )
        voxel_structure_ids[index] = (7,)
    voxels[999] = VoxelDose(
        voxel_id=999,
        dose_gy=0.5,
        let_kev_um=0.2,
        dep_energy_mev=0.0,
        n_events=1,
        rel_error=0.0,
        scaled_dose=0.5,
        eqd_gy=0.5,
        mev2gy=1.0,
    )
    voxel_structure_ids[999] = (9,)
    return DoseMap(
        voxels=voxels,
        grid_shape=(4, 4, 4),
        voxel_size_mm=(1.0, 1.0, 1.0),
        structure_ids={7: "tumor", 9: "normal"},
        voxel_structure_ids=voxel_structure_ids,
    )


class VoxelSFCalculatorTests(unittest.TestCase):
    def test_uniform_voxel_dose_matches_analytic_lq(self) -> None:
        dose_map = _build_dose_map([2.0, 2.0, 2.0], [5.0, 5.0, 5.0])
        let_params = LETDependentParams(
            alpha_0=0.1,
            lambda_alpha=0.0,
            beta_0=0.02,
            lambda_beta=0.0,
        )

        result = compute_voxel_sf(dose_map, let_params)

        expected_sf = math.exp(-(0.1 * 2.0 + 0.02 * 4.0))
        self.assertEqual(len(result.voxel_results), 3)
        self.assertAlmostEqual(result.mean_sf, expected_sf, places=8)
        self.assertAlmostEqual(result.volume_weighted_sf, expected_sf, places=8)
        self.assertAlmostEqual(result.mean_dose_gy, 2.0, places=8)
        self.assertAlmostEqual(result.mean_let_kev_um, 5.0, places=8)
        self.assertAlmostEqual(result.d90, 2.0, places=8)
        self.assertAlmostEqual(result.d50, 2.0, places=8)
        self.assertAlmostEqual(result.v20, 0.0, places=8)
        self.assertAlmostEqual(result.effective_alpha, 0.1, places=8)
        self.assertAlmostEqual(result.effective_beta, 0.02, places=8)
        self.assertAlmostEqual(result.equivalent_uniform_dose, 2.0, places=8)

    def test_nonuniform_dose_obeys_jensen_gap_and_dvh_metrics(self) -> None:
        dose_map = _build_dose_map([1.0, 5.0], [3.0, 3.0])
        let_params = LETDependentParams(
            alpha_0=0.1,
            lambda_alpha=0.0,
            beta_0=0.02,
            lambda_beta=0.0,
        )

        result = compute_voxel_sf(dose_map, let_params)

        sf_at_mean_dose = math.exp(-(0.1 * 3.0 + 0.02 * 9.0))
        self.assertNotAlmostEqual(result.mean_sf, sf_at_mean_dose, places=4)
        self.assertAlmostEqual(result.d90, 1.0, places=8)
        self.assertAlmostEqual(result.d50, 5.0, places=8)
        self.assertAlmostEqual(result.volume_weighted_sf, result.mean_sf, places=8)

    def test_repair_model_penalizes_short_inter_fraction_gap(self) -> None:
        dose_map = _build_dose_map([4.0], [5.0])
        let_params = LETDependentParams(
            alpha_0=0.0,
            lambda_alpha=0.0,
            beta_0=0.04,
            lambda_beta=0.0,
        )

        short_gap = compute_voxel_sf(
            dose_map,
            let_params,
            model_kind="repair_lq",
            repair_half_time_hours=1.0,
            n_fractions=2,
            schedule_days=(0.0, 1.0 / 24.0),
        )
        long_gap = compute_voxel_sf(
            dose_map,
            let_params,
            model_kind="repair_lq",
            repair_half_time_hours=1.0,
            n_fractions=2,
            schedule_days=(0.0, 1.0),
        )

        self.assertLess(short_gap.mean_sf, long_gap.mean_sf)

    def test_simulate_growth_uses_volumetric_sf_override(self) -> None:
        dose_map = _build_dose_map([2.0, 2.0], [5.0, 5.0])
        let_params = LETDependentParams(
            alpha_0=0.1,
            lambda_alpha=0.0,
            beta_0=0.02,
            lambda_beta=0.0,
        )
        volumetric_sf = compute_voxel_sf(dose_map, let_params)
        reference = GeometryReference(axis_a=2.0, axis_b=4.0, axis_c=6.0, volume=12.0)
        neutral_parameters = GrowthModelParameters(
            alpha=0.0,
            beta=0.0,
            growth_rate=0.0,
            carrying_capacity=50.0,
            clearance_rate=0.0,
        )
        direct_parameters = GrowthModelParameters(
            alpha=volumetric_sf.effective_alpha,
            beta=volumetric_sf.effective_beta,
            growth_rate=0.0,
            carrying_capacity=50.0,
            clearance_rate=0.0,
        )
        schedule = [TreatmentFraction(day=0.0, dose=2.0)]

        overridden = simulate_growth(
            sample_times=[0.0, 1.0],
            parameters=neutral_parameters,
            reference=reference,
            schedule=schedule,
            volumetric_sf=volumetric_sf,
        )
        direct = simulate_growth(
            sample_times=[0.0, 1.0],
            parameters=direct_parameters,
            reference=reference,
            schedule=schedule,
        )

        self.assertTrue(np.allclose(overridden.live_volume, direct.live_volume))
        self.assertTrue(np.allclose(overridden.dead_volume, direct.dead_volume))
        self.assertTrue(np.allclose(overridden.total_volume, direct.total_volume))

    def test_compute_mixed_voxel_sf_aggregates_component_dose_maps(self) -> None:
        proton_map = _build_dose_map([2.0], [8.0])
        neutron_map = _build_dose_map([1.0], [30.0])
        let_params = LETDependentParams(
            alpha_0=0.08,
            lambda_alpha=0.002,
            beta_0=0.002,
            lambda_beta=0.0,
        )

        result = compute_mixed_voxel_sf(
            {"protonDose": proton_map, "mainDose": neutron_map},
            {"protonDose": "p", "mainDose": "n"},
            let_params,
            method="tdra",
        )

        self.assertEqual(len(result.voxel_results), 1)
        self.assertAlmostEqual(result.mean_dose_gy, 3.0, places=8)
        self.assertGreater(result.effective_alpha, 0.08)
        self.assertGreater(result.mean_let_kev_um, 8.0)
        self.assertLess(result.mean_sf, 1.0)
        self.assertGreater(result.mean_sf, 0.0)


if __name__ == "__main__":
    unittest.main()
