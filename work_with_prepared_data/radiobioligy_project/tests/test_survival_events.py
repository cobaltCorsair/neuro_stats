import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from data_processing.excel_data_processor import (
    extract_survival_events,
    find_rats_that_died_before_experiment_end,
    parse_irradiation_schedule,
    process_skin_data_excel,
)


def _write_xlsx(tmp_path: Path, name: str, header_row, time_labels, data_rows) -> str:
    """Собирает .xlsx в формате проекта: строка 0 = заголовок (дозы + t=), строка 1 =
    подписи времени, со строки 2 = метка крысы + измерения."""
    width = max(len(header_row), len(time_labels), max((len(r) for r in data_rows), default=0))

    def pad(row):
        return list(row) + [None] * (width - len(row))

    rows = [pad(header_row), pad(time_labels)] + [pad(r) for r in data_rows]
    df = pd.DataFrame(rows)
    file_path = tmp_path / name
    df.to_excel(file_path, header=False, index=False, engine="openpyxl")
    return str(file_path)


class TestExtractSurvivalEvents(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_confirmed_death_with_exact_date_overrides_column_day(self):
        # без t= (контроль): rebased day == номинальный день столбца
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03", "8 сут. - 1.04"]
        rows = [
            ["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗ 29.03", None],
        ]
        path = _write_xlsx(self.tmp_path, "death_exact_date.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e.label, "rat1")
        self.assertTrue(e.event_observed)
        # маркер на 1 день раньше номинальной даты столбца (30.03) -> день 5, не 6
        self.assertEqual(e.day, 5.0)
        self.assertEqual(e.reason, "⊗ 29.03")

    def test_death_symbol_without_date_uses_column_day(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗"]]
        path = _write_xlsx(self.tmp_path, "death_no_date.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        self.assertTrue(events[0].event_observed)
        self.assertEqual(events[0].day, 6.0)

    def test_censoring_marker_is_not_death(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "выгрызла"]]
        path = _write_xlsx(self.tmp_path, "censored.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        self.assertFalse(events[0].event_observed)
        self.assertEqual(events[0].day, 6.0)
        self.assertEqual(events[0].reason, "выгрызла")

    def test_later_death_marker_takes_precedence_over_earlier_censoring(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03", "8 сут. - 1.04"]
        rows = [["rat1", "1.0-1.0-1.0", "выгрызла", None, "⊗ 31.03"]]
        path = _write_xlsx(self.tmp_path, "censor_then_death.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        e = events[0]
        self.assertTrue(e.event_observed)
        self.assertEqual(e.reason, "⊗ 31.03")
        # маркер на 1 день раньше столбца "8 сут." (01.04) -> день 7
        self.assertEqual(e.day, 7.0)

    def test_no_marker_means_censored_at_last_valid_measurement(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2"]]
        path = _write_xlsx(self.tmp_path, "alive.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        self.assertFalse(events[0].event_observed)
        self.assertEqual(events[0].reason, "")
        self.assertEqual(events[0].day, 6.0)

    def test_v_promezhut_rebase_shifts_death_day_for_multi_day_gap(self):
        # p,p с t=5 сут: V промежут. привязывается к дню 5, "2 сут." после него = день 7
        header = ["p = 25.1 Гр", "p = 25.1 Гр", "t = 5 сут"]
        labels = ["Метка", "V исх. - 20.05.26", "V промежут. - 25.05.2026", "2 сут. - 27.05"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗"]]
        path = _write_xlsx(self.tmp_path, "multiday_gap_death.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        self.assertTrue(events[0].event_observed)
        self.assertEqual(events[0].day, 7.0)

    def test_day_unit_sutki_parses_into_schedule_hours(self):
        self.assertEqual(parse_irradiation_schedule("t = 5 сут", ["p", "p"]), [120.0])

    def test_multiday_gap_rebase_without_explicit_v_promezhut_row(self):
        # тот же p,p t=5 сут случай, но без строки 'V промежут.' (как в файлах кожных
        # реакций той же серии измерений) - переразметка должна сработать по календарной
        # дате подписи '2 сут. - 27.05', которая совпадает с днём 7 (5 + 2), а не днём 2.
        header = ["p = 25.1 Гр", "p = 25.1 Гр", "t = 5 сут"]
        labels = ["Метка", "V исх. - 20.05.26", "2 сут. - 27.05", "4 сут. - 29.05"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗"]]
        path = _write_xlsx(self.tmp_path, "multiday_gap_no_v_row.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        self.assertTrue(events[0].event_observed)
        self.assertEqual(events[0].day, 9.0)

    def test_small_labeling_offset_does_not_trigger_false_rebase(self):
        # обычный режим без многосуточных перерывов (фракции в пределах ~суток друг от
        # друга): устойчивое расхождение подписи 'N сут.' с календарной датой на 1 сутки
        # не должно переключать базовую точку - номинальная цифра используется как есть.
        header = ["n = 2.36 Гр", "p = 18.8 Гр", "n = 2.36 Гр", "p = 18.8 Гр", "t1 = 2 ч/24 ч/1 ч 45 мин"]
        labels = ["Метка", "V исх. - 23.03.26", "3 сут. - 27.03", "6 сут. - 1.04"]
        rows = [["rat1", "1.1-1.1-1.1", "1.2-1.2-1.2", "⊗"]]
        path = _write_xlsx(self.tmp_path, "small_offset_no_rebase.xlsx", header, labels, rows)
        events = extract_survival_events(path)
        self.assertTrue(events[0].event_observed)
        self.assertEqual(events[0].day, 6.0)


class TestFindRatsThatDiedBeforeExperimentEnd(unittest.TestCase):
    """
    Общий детектор "умер в середине эксперимента" для автоматического исключения животных
    (ExtractOutliers.exclude_dead_rats, LQ fitter) — переиспользует extract_survival_events,
    не парсит файл заново."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_death_strictly_before_last_day_is_included(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03", "8 сут. - 1.04"]
        rows = [
            ["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗", None],  # умерла на 6 сут.
            ["rat2", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2", "1.3-1.3-1.3"],  # дожила до 8 сут.
        ]
        path = _write_xlsx(self.tmp_path, "mid_death.xlsx", header, labels, rows)
        self.assertEqual(find_rats_that_died_before_experiment_end(path), ["rat1"])

    def test_death_exactly_on_last_day_is_not_included(self):
        # Единственное животное, маркер смерти ровно в последнем столбце -> его день
        # смерти совпадает с last_day, а не строго меньше него.
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗"]]
        path = _write_xlsx(self.tmp_path, "death_on_last_day.xlsx", header, labels, rows)
        self.assertEqual(find_rats_that_died_before_experiment_end(path), [])

    def test_censoring_marker_is_not_treated_as_death(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03", "8 сут. - 1.04"]
        rows = [
            ["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "выгрызла", None],  # цензурирована на 6 сут.
            ["rat2", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2", "1.3-1.3-1.3"],  # дожила до 8 сут.
        ]
        path = _write_xlsx(self.tmp_path, "censored_not_death.xlsx", header, labels, rows)
        self.assertEqual(find_rats_that_died_before_experiment_end(path), [])

    def test_no_marker_survives_to_end_is_not_included(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03"]
        rows = [["rat1", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2"]]
        path = _write_xlsx(self.tmp_path, "no_marker.xlsx", header, labels, rows)
        self.assertEqual(find_rats_that_died_before_experiment_end(path), [])

    def test_mixed_group_returns_only_early_deaths_in_file_order(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26", "3 сут. - 27.03", "6 сут. - 30.03", "8 сут. - 1.04"]
        rows = [
            ["rat1", "1.0-1.0-1.0", "⊗", None, None],               # умерла на 3 сут.
            ["rat2", "1.0-1.0-1.0", "1.1-1.1-1.1", "⊗", None],       # умерла на 6 сут.
            ["rat3", "1.0-1.0-1.0", "1.1-1.1-1.1", "1.2-1.2-1.2", "1.3-1.3-1.3"],  # дожила до 8 сут.
        ]
        path = _write_xlsx(self.tmp_path, "mixed_group.xlsx", header, labels, rows)
        self.assertEqual(find_rats_that_died_before_experiment_end(path), ["rat1", "rat2"])

    def test_empty_file_returns_empty_list(self):
        header = ["без облучения"]
        labels = ["Метка", "V исх. - 24.03.26"]
        path = _write_xlsx(self.tmp_path, "empty.xlsx", header, labels, [])
        self.assertEqual(find_rats_that_died_before_experiment_end(path), [])


class TestProcessSkinDataExcelWithDeathMarker(unittest.TestCase):
    """
    process_skin_data_excel раньше возвращал сырые ячейки без приведения к float.
    Стоило в матрице появиться маркеру события ('⊗', 'death', ...), numpy приводил
    ВЕСЬ массив к строковому dtype, и np.nanmean (используется во всех графиках кожных
    реакций) падал с UFuncTypeError при попытке сложения строк.
    """

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_death_marker_becomes_nan_not_string_and_array_stays_numeric(self):
        header = ["p = 25.1 Гр", "p = 25.1 Гр", "t = 5 сут"]
        labels = ["Метка", "V исх. - 20.05.26", "2 сут. - 27.05", "4 сут. - 29.05"]
        rows = [
            ["rat1", 0, 245, "⊗"],
            ["rat2", 0, 295, 390],
        ]
        path = _write_xlsx(self.tmp_path, "skin_death_marker.xlsx", header, labels, rows)

        _, _, _, skin_reactions = process_skin_data_excel(path)

        self.assertTrue(all(isinstance(v, float) for row in skin_reactions for v in row))
        self.assertTrue(np.isnan(skin_reactions[0][2]))

        arr = np.array(skin_reactions)
        self.assertEqual(arr.dtype, np.float64)
        mean = np.nanmean(arr, axis=0)  # не должно бросать UFuncTypeError
        self.assertAlmostEqual(mean[1], 270.0)


if __name__ == "__main__":
    unittest.main()
