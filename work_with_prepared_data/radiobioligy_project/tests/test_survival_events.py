import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data_processing.excel_data_processor import (
    extract_survival_events,
    parse_irradiation_schedule,
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


if __name__ == "__main__":
    unittest.main()
