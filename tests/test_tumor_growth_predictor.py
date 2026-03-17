import unittest

import numpy as np

from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
    GeometryReference,
    GrowthModelParameters,
    TreatmentFraction,
    build_schedule_from_intervals,
    fit_geometry_scaling,
    fit_gompertz_to_control,
    gompertz_volume,
    parse_irradiation_intervals_days,
    simulate_growth,
    surviving_fraction,
)


class TumorGrowthPredictorTests(unittest.TestCase):
    def test_surviving_fraction_matches_lq_formula(self) -> None:
        sf = surviving_fraction(0.1, 0.02, 2.0)
        self.assertAlmostEqual(sf, np.exp(-(0.1 * 2.0 + 0.02 * 4.0)), places=8)

    def test_surviving_fraction_uses_unrepaired_dose_memory(self) -> None:
        sf_without_memory = surviving_fraction(0.1, 0.02, 2.0)
        sf_with_memory = surviving_fraction(0.1, 0.02, 2.0, prior_unrepaired_dose=1.5)

        self.assertLess(sf_with_memory, sf_without_memory)

    def test_single_fraction_moves_volume_from_live_to_dead(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=4.0, axis_c=6.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.2,
            beta=0.0,
            growth_rate=0.0,
            carrying_capacity=50.0,
            clearance_rate=0.0,
        )
        result = simulate_growth(
            sample_times=[0.0, 1.0, 2.0],
            parameters=parameters,
            reference=reference,
            schedule=[TreatmentFraction(day=0.0, dose=2.0)],
        )

        expected_sf = surviving_fraction(0.2, 0.0, 2.0)
        self.assertAlmostEqual(result.live_volume[0], reference.volume * expected_sf, places=6)
        self.assertAlmostEqual(result.dead_volume[0], reference.volume * (1.0 - expected_sf), places=6)
        self.assertAlmostEqual(result.total_volume[0], reference.volume, places=6)

    def test_axes_follow_cube_root_of_volume(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=4.0, axis_c=6.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.0,
            beta=0.0,
            growth_rate=0.0,
            carrying_capacity=50.0,
            clearance_rate=0.0,
        )
        result = simulate_growth(
            sample_times=[0.0, 1.0],
            parameters=parameters,
            reference=reference,
            schedule=[],
        )

        self.assertTrue(np.allclose(result.axis_a, [2.0, 2.0]))
        self.assertTrue(np.allclose(result.axis_b, [4.0, 4.0]))
        self.assertTrue(np.allclose(result.axis_c, [6.0, 6.0]))

    def test_fit_geometry_scaling_recovers_power_law_axes(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=3.0, axis_c=4.0, volume=10.0)
        volumes = np.array([10.0, 20.0, 40.0, 80.0])
        axis_a = 1.5 * np.power(volumes, 0.20)
        axis_b = 0.8 * np.power(volumes, 0.35)
        axis_c = 2.1 * np.power(volumes, 0.15)

        model = fit_geometry_scaling(volumes, axis_a, axis_b, axis_c, reference)

        self.assertEqual(model.mode, "fitted")
        self.assertAlmostEqual(model.coeff_a, 1.5, places=3)
        self.assertAlmostEqual(model.power_a, 0.20, places=3)
        self.assertAlmostEqual(model.coeff_b, 0.8, places=3)
        self.assertAlmostEqual(model.power_b, 0.35, places=3)
        self.assertAlmostEqual(model.coeff_c, 2.1, places=3)
        self.assertAlmostEqual(model.power_c, 0.15, places=3)

    def test_fit_gompertz_to_control_recovers_synthetic_parameters(self) -> None:
        initial_volume = 10.0
        growth_rate = 0.18
        carrying_capacity = 150.0
        times = np.array([0.0, 2.0, 5.0, 8.0, 11.0])
        volumes = gompertz_volume(times, initial_volume, growth_rate, carrying_capacity)
        fit = fit_gompertz_to_control(times, volumes)

        self.assertAlmostEqual(fit.initial_volume, initial_volume, places=6)
        self.assertAlmostEqual(fit.growth_rate, growth_rate, places=3)
        self.assertAlmostEqual(fit.carrying_capacity, carrying_capacity, places=1)

    def test_parse_irradiation_intervals_days_converts_hours(self) -> None:
        intervals = parse_irradiation_intervals_days(
            ("y = 4 Гр", "y = 4 Гр", "y = 32 Гр", "Irradiation Time=t = 1 ч")
        )

        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0], 1.0 / 24.0, places=8)

    def test_build_schedule_from_intervals_uses_subday_spacing(self) -> None:
        schedule = build_schedule_from_intervals(
            [4.0, 4.0, 32.0],
            [1.0 / 24.0],
        )

        self.assertEqual([event.dose for event in schedule], [4.0, 4.0, 32.0])
        self.assertAlmostEqual(schedule[0].day, 0.0, places=8)
        self.assertAlmostEqual(schedule[1].day, 1.0 / 24.0, places=8)
        self.assertAlmostEqual(schedule[2].day, 2.0 / 24.0, places=8)

    def test_repair_aware_simulation_penalizes_short_intervals(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=4.0, axis_c=6.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.0,
            beta=0.04,
            growth_rate=0.0,
            carrying_capacity=50.0,
            clearance_rate=0.0,
            repair_half_time_hours=1.0,
        )

        short_gap = simulate_growth(
            sample_times=[0.0, 1.0],
            parameters=parameters,
            reference=reference,
            schedule=[TreatmentFraction(day=0.0, dose=2.0), TreatmentFraction(day=1.0 / 24.0, dose=2.0)],
        )
        long_gap = simulate_growth(
            sample_times=[0.0, 1.0],
            parameters=parameters,
            reference=reference,
            schedule=[TreatmentFraction(day=0.0, dose=2.0), TreatmentFraction(day=1.0, dose=2.0)],
        )

        self.assertLess(short_gap.live_volume[-1], long_gap.live_volume[-1])
        self.assertGreater(short_gap.dead_volume[-1], long_gap.dead_volume[-1])

    def test_zero_repair_half_time_keeps_fraction_kill_independent(self) -> None:
        reference = GeometryReference(axis_a=2.0, axis_b=4.0, axis_c=6.0, volume=12.0)
        parameters = GrowthModelParameters(
            alpha=0.0,
            beta=0.04,
            growth_rate=0.0,
            carrying_capacity=50.0,
            clearance_rate=0.0,
            repair_half_time_hours=0.0,
        )

        short_gap = simulate_growth(
            sample_times=[0.0, 1.0],
            parameters=parameters,
            reference=reference,
            schedule=[TreatmentFraction(day=0.0, dose=2.0), TreatmentFraction(day=1.0 / 24.0, dose=2.0)],
        )
        long_gap = simulate_growth(
            sample_times=[0.0, 1.0],
            parameters=parameters,
            reference=reference,
            schedule=[TreatmentFraction(day=0.0, dose=2.0), TreatmentFraction(day=1.0, dose=2.0)],
        )

        self.assertAlmostEqual(short_gap.live_volume[-1], long_gap.live_volume[-1], places=8)
        self.assertAlmostEqual(short_gap.dead_volume[-1], long_gap.dead_volume[-1], places=8)


if __name__ == "__main__":
    unittest.main()
