import json
import unittest
from pathlib import Path

import numpy as np

from growth_prediction.model import CanonicalGrowthModel, SeriesInputs, feature_names


ROOT = Path(__file__).resolve().parents[1] / "growth_prediction"


class CanonicalGrowthPredictionTests(unittest.TestCase):
    def test_artifact_has_complete_97_feature_specification(self):
        artifact = json.loads((ROOT / "canonical_model_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(len(artifact["feature_names"]), 97)
        self.assertEqual(artifact["feature_names"], feature_names(artifact["categories"]))
        self.assertEqual(len(artifact["scaler_scale"]), 97)
        self.assertEqual(len(artifact["scaled_coefficients"]), 97)

    def test_serialized_verification_cases_are_reproduced(self):
        model = CanonicalGrowthModel.load(ROOT / "canonical_model_v1.json")
        for case in model.artifact["verification_cases"]:
            inputs = SeriesInputs(**case["inputs"])
            actual = model.predict_log_relative([case["day"]], inputs)[0]
            self.assertTrue(
                np.isclose(actual, case["predicted_log_relative"], atol=1e-12, rtol=0)
            )

    def test_zero_day_anchor_is_exact(self):
        model = CanonicalGrowthModel.load(ROOT / "canonical_model_v1.json")
        case = model.artifact["verification_cases"][-1]
        inputs = SeriesInputs(**case["inputs"])
        self.assertEqual(model.predict_log_relative([0.0], inputs)[0], 0.0)
        self.assertEqual(model.predict_relative([0.0], inputs)[0], 1.0)


if __name__ == "__main__":
    unittest.main()
