# coding: utf-8
"""Small CSV export helpers shared by the PyQt survival GUIs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


def normalize_csv_path(path: str | Path) -> Path:
    """Resolve a user-provided export path and force a .csv suffix."""
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".csv":
        resolved = resolved.with_suffix(".csv")
    return resolved


def related_csv_path(base_path: str | Path, suffix: str) -> Path:
    """Build a sibling CSV path like base_stem_suffix.csv."""
    normalized = normalize_csv_path(base_path)
    safe_suffix = suffix.strip().replace(" ", "_")
    if not safe_suffix:
        return normalized
    return normalized.with_name(f"{normalized.stem}_{safe_suffix}{normalized.suffix}")


def write_csv_rows(
    path: str | Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> Path:
    """Write one header + row matrix to a UTF-8 CSV file."""
    output_path = normalize_csv_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))
        writer.writerows(rows)
    return output_path
