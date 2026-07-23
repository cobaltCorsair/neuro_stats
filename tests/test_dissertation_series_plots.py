import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from work_with_prepared_data.radiobioligy_project.utils.dissertation_series_plots import (
    plot_calendar_block_forest,
    plot_longitudinal_series_response,
    plot_series_dose_response,
)


class DissertationSeriesPlotTests(unittest.TestCase):
    def test_plot_exports_raster_and_vector_formats(self) -> None:
        series_rows = [
            {
                "family": "y",
                "regimen_class": "single_fraction",
                "dose_group_physical_gy": 32.0,
                "auc": 82.0,
                "control_kind": "pooled_historical_fallback",
            },
            {
                "family": "y",
                "regimen_class": "single_fraction",
                "dose_group_physical_gy": 32.0,
                "auc": 88.0,
                "control_kind": "same_date",
            },
        ]
        dose_rows = [
            {
                "family": "y",
                "regimen_class": "single_fraction",
                "dose_group_physical_gy": 32.0,
                "n_series": 2,
                "auc_mean": 85.0,
                "auc_hierarchical_p2_5": 80.0,
                "auc_hierarchical_p97_5": 90.0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            created = plot_series_dose_response(
                series_rows,
                dose_rows,
                endpoint="auc",
                output_path=Path(directory) / "dose_response.png",
                family_labels={"y": "γ"},
                families=["y"],
                y_label="Ингибирование AUC, %",
                title="Тест",
                formats=("png", "svg", "pdf"),
            )
            self.assertEqual({".png", ".svg", ".pdf"}, {path.suffix for path in created})
            for path in created:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 100)

    def test_calendar_block_forest_exports_all_formats(self) -> None:
        rows = [
            {
                "endpoint": "auc",
                "family1": "y",
                "family2": "y",
                "dose1_physical_gy": 32.0,
                "dose2_physical_gy": 36.0,
                "observed_mean_difference": -4.0,
                "block_bootstrap_p2_5": -8.0,
                "block_bootstrap_p97_5": -1.0,
                "block_ci_excludes_zero": 1,
            },
            {
                "endpoint": "auc",
                "family1": "e",
                "family2": "y",
                "dose1_physical_gy": 32.0,
                "dose2_physical_gy": 32.0,
                "observed_mean_difference": 2.0,
                "block_bootstrap_p2_5": -2.0,
                "block_bootstrap_p97_5": 6.0,
                "block_ci_excludes_zero": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            created = plot_calendar_block_forest(
                rows,
                endpoint="auc",
                output_path=Path(directory) / "forest.png",
                family_labels={"y": "γ", "e": "электроны"},
                title="Тест",
            )
            self.assertEqual({".png", ".svg", ".pdf"}, {path.suffix for path in created})
            for path in created:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 100)

    def test_longitudinal_response_exports_all_formats(self) -> None:
        series_rows = []
        summary_rows = []
        availability_rows = []
        for dose, scale in ((32.0, 1.0), (36.0, 0.8)):
            for series_index in range(2):
                for day in (0.0, 7.0, 14.0, 21.0):
                    series_rows.append(
                        {
                            "family": "y",
                            "regimen_class": "single_fraction",
                            "dose_group_physical_gy": dose,
                            "series_key": f"s{series_index}|{dose:g}",
                            "day": day,
                            "volume_response": scale * (1.0 - 0.02 * day),
                            "control_kind": "same_date" if series_index == 0 else "pooled_historical_fallback",
                        }
                    )
            for day in (0.0, 7.0, 14.0, 21.0):
                mean = scale * (1.0 - 0.02 * day)
                summary_rows.append(
                    {
                        "family": "y",
                        "regimen_class": "single_fraction",
                        "dose_group_physical_gy": dose,
                        "day": day,
                        "mean_volume_response": mean,
                        "bootstrap_p2_5": mean - 0.05,
                        "bootstrap_p97_5": mean + 0.05,
                        "n_series": 2,
                    }
                )
                availability_rows.append(
                    {
                        "family": "y",
                        "regimen_class": "single_fraction",
                        "dose_group_physical_gy": dose,
                        "day": day,
                        "n_available_animals": 12 - int(day >= 14.0),
                        "n_registered_deaths_cumulative": int(day >= 14.0),
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            created = plot_longitudinal_series_response(
                series_rows,
                summary_rows,
                availability_rows,
                family="y",
                doses=(32.0, 36.0),
                output_path=Path(directory) / "longitudinal.png",
                family_label="γ",
            )
            self.assertEqual({".png", ".svg", ".pdf"}, {path.suffix for path in created})
            for path in created:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
