"""Focused checks for the regularised hierarchical LKB implementation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hierarchical_lkb import HierarchicalLKBModel, prepare_hierarchical_lkb_data


def synthetic_clustered_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    layouts = {
        "2020-01-01": (10.0, [0, 0, 0, 0], 18.0, [1, 1, 1, 1]),
        "2020-02-01": (14.0, [0, 0, 1, 0], 22.0, [1, 1, 1, 1]),
        "2020-03-01": (18.0, [0, 1, 1, 1], 26.0, [1, 1, 1, 1]),
        "2020-04-01": (22.0, [1, 1, 1, 1], 30.0, [1, 1, 1, 1]),
    }
    for block, (dose_a, outcome_a, dose_b, outcome_b) in layouts.items():
        for outcome in outcome_a:
            rows.append(
                {
                    "family": "a",
                    "calendar_series": block,
                    "physical_total_dose": dose_a,
                    "complication": outcome,
                }
            )
        for outcome in outcome_b:
            rows.append(
                {
                    "family": "b",
                    "calendar_series": block,
                    "physical_total_dose": dose_b,
                    "complication": outcome,
                }
            )
    return pd.DataFrame(rows)


class HierarchicalLKBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_clustered_data()
        self.model = HierarchicalLKBModel(quadrature_points=11)

    def test_regularisation_keeps_separated_family_finite(self) -> None:
        result = self.model.fit(self.frame)
        self.assertTrue(result.converged)
        self.assertGreater(result.common_m, 0.0)
        self.assertTrue(np.isfinite(result.common_m))
        self.assertTrue(np.isfinite(result.calendar_sd))
        self.assertTrue(np.all(np.isfinite(result.family_td50)))
        self.assertTrue(np.all(np.asarray(result.family_td50) > 0.0))

    def test_population_probability_is_monotone_and_bounded(self) -> None:
        result = self.model.fit(self.frame)
        probability = self.model.predict_population_probability(
            result,
            "a",
            np.linspace(5.0, 35.0, 101),
        )
        self.assertTrue(np.all(np.diff(probability) >= 0.0))
        self.assertTrue(np.all((probability >= 0.0) & (probability <= 1.0)))

    def test_analytic_gradient_matches_central_difference(self) -> None:
        data = prepare_hierarchical_lkb_data(self.frame)
        priors = self.model.default_priors(self.frame)
        point = self.model.fit(self.frame, priors=priors)
        parameters = np.asarray(point.parameter_vector)
        _, analytic = self.model._negative_log_posterior_and_gradient(
            data,
            parameters,
            priors,
        )
        numeric = np.empty_like(parameters)
        step = 1e-5
        for index in range(len(parameters)):
            upper = parameters.copy()
            lower = parameters.copy()
            upper[index] += step
            lower[index] -= step
            numeric[index] = (
                self.model._negative_log_posterior(data, upper, priors)
                - self.model._negative_log_posterior(data, lower, priors)
            ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
