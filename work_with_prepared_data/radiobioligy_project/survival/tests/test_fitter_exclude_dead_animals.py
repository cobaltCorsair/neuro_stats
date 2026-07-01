import math
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd

# Ячейки формата "a-b-c" парсятся process_tumor_data_excel как объём эллипсоида
# (pi*a*b*c)/6, а не как сырое число -- используем ту же формулу для ожидаемых значений.
_SURVIVOR_LAST_VOLUME = math.pi * 1.3 * 1.3 * 1.3 / 6

try:
    from survival.fit_alpha_beta_using_processor import Fitter
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
        Fitter,
    )


def _write_xlsx(tmp_path: Path, name: str, header_row, time_labels, data_rows) -> Path:
    """Собирает .xlsx в формате проекта: строка 0 = заголовок (дозы + t=), строка 1 =
    подписи времени, со строки 2 = метка крысы + измерения. Тот же формат, что
    tests/test_survival_events.py, продублирован здесь по тому же принципу, что и остальные
    fixture-хелперы в этом проекте — общего conftest.py между survival/tests и tests нет."""
    width = max(len(header_row), len(time_labels), max((len(r) for r in data_rows), default=0))

    def pad(row):
        return list(row) + [None] * (width - len(row))

    rows = [pad(header_row), pad(time_labels)] + [pad(r) for r in data_rows]
    df = pd.DataFrame(rows)
    file_path = tmp_path / name
    df.to_excel(file_path, header=False, index=False, engine="openpyxl")
    return file_path


class FitterExcludeDeadAnimalsTests(unittest.TestCase):
    """Fitter.collect() не использует ExtractOutliers/exclude_rats (основной GUI-конвейер) —
    у него собственная загрузка файлов, поэтому исключение умерших животных подключено
    отдельно через exclude_dead_animals, проверяем именно этот путь."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

        control_header = ["без облучения"]
        # без "control" в НАЗВАНИИ файла is_control_file его не примет
        self.control_path = _write_xlsx(
            self.tmp_path, "my_control.xlsx", control_header,
            ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03"],
            [["c1", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2"]],
        )

        experiment_header = ["p = 20 Гр"]
        experiment_labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03", "8 сут. - 1.04"]
        experiment_rows = [
            ["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗", None],  # умерла на 6 сут.
            ["rat2", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2", "1.3-1.3-1.3"],  # дожила до 8 сут.
        ]
        self.experiment_path = _write_xlsx(
            self.tmp_path, "experiment.xlsx", experiment_header, experiment_labels, experiment_rows
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def _build_fitter(self, exclude_dead_animals: bool) -> Fitter:
        return Fitter(
            sf_mode="absolute",
            min_sf=0.0,
            alpha_fixed=None,
            verbose=False,
            exclude_dead_animals=exclude_dead_animals,
        )

    def test_default_keeps_dead_animal_row(self):
        fitter = self._build_fitter(exclude_dead_animals=False)
        fitter.collect([self.control_path, self.experiment_path])

        self.assertEqual(len(fitter.raw_experiments), 1)
        self.assertEqual(fitter.raw_experiments[0].volumes.shape[0], 2)
        self.assertEqual(fitter.controls[self.control_path].shape[0], 1)

    def test_enabled_drops_dead_animal_row_from_experiment_and_control(self):
        fitter = self._build_fitter(exclude_dead_animals=True)
        fitter.collect([self.control_path, self.experiment_path])

        self.assertEqual(len(fitter.raw_experiments), 1)
        self.assertEqual(fitter.raw_experiments[0].volumes.shape[0], 1)
        # выжившая крыса (rat2) должна остаться, а не умершая (rat1)
        self.assertAlmostEqual(
            float(fitter.raw_experiments[0].volumes[0, -1]), _SURVIVOR_LAST_VOLUME, places=6
        )
        # в контрольной группе никто не умирал -> строка не должна пропасть
        self.assertEqual(fitter.controls[self.control_path].shape[0], 1)

    def test_enabled_does_not_drop_control_animal_that_died(self):
        # У контрольной группы своя крыса с маркером смерти -- проверяем, что фильтрация
        # применяется и к self.controls, а не только к self.raw_experiments.
        control_header = ["без облучения"]
        control_labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03", "8 сут. - 1.04"]
        control_rows = [
            ["c1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗", None],
            ["c2", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2", "1.3-1.3-1.3"],
        ]
        control_path = _write_xlsx(self.tmp_path, "second_control.xlsx", control_header, control_labels, control_rows)

        fitter = self._build_fitter(exclude_dead_animals=True)
        fitter.collect([control_path, self.experiment_path])

        self.assertEqual(fitter.controls[control_path].shape[0], 1)
        self.assertAlmostEqual(float(fitter.controls[control_path][0, -1]), _SURVIVOR_LAST_VOLUME, places=6)


if __name__ == "__main__":
    unittest.main()
