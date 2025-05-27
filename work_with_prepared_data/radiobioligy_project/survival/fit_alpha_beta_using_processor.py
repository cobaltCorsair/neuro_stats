from __future__ import annotations
import argparse
import re
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import nnls

# --- готовые модули ----------------------------------
from work_with_prepared_data.radiobioligy_project.data_processing.data_processing import TumorDataProcessor
from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import process_tumor_data_excel
# ------------------------------------------------------------------

# Регулярные выражения для поиска чисел и суффикса дозы
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
GR_SUFFIX = re.compile(r"гр|gy", re.IGNORECASE)


class TumorExperiment:
    """Контейнер для одного опыта: фракции, SF, доза и вывод отчёта."""

    def __init__(self, path: Path, fractions: List[float], sf: float):
        self.path = path
        self.fractions = fractions
        self.sf = sf

    @property
    def dose_sum(self) -> float:
        """Σ d_i — суммарная доза (Gy)."""
        return sum(self.fractions)

    @property
    def dose2_sum(self) -> float:
        """Σ d_i^2 — сумма квадратов доз (Gy^2)."""
        return sum(d * d for d in self.fractions)

    def report(self) -> str:
        """Строка отчёта: имя файла, фракции, D, D^2 и SF."""
        frac_str = "+".join(f"{d:g}" for d in self.fractions)
        return (f"{self.path.name:30s} ({frac_str})  D={self.dose_sum:5.1f}  "
                f"D²={self.dose2_sum:6.0f}  SF={self.sf:.4f}")


# ------------------- HELPERS -------------------

def parse_fractions(experiment_params: List[str]) -> List[float]:
    """
    Извлекает все числа, за которыми следует 'Гр' или 'Gy'.
    Возвращает список фракций [d1, d2, ...].
    """
    fractions: List[float] = []
    for token in experiment_params:
        tok_l = token.lower()
        if GR_SUFFIX.search(tok_l):
            for num in NUMBER.findall(tok_l):
                fractions.append(float(num.replace(',', '.')))
    return fractions


def compute_sf_relative(mean_rel: np.ndarray, mode: str) -> float:
    """
    Вычисляет S(F) по относительному объёму:
      - 'relative': min(mean_rel[1:])
      - 'index:N': mean_rel[N]
    """
    if mode.startswith("index:"):
        idx = int(mode.split(":")[1])
        if idx >= len(mean_rel):
            raise IndexError(f"index {idx} out of range 0..{len(mean_rel)-1}")
        return float(mean_rel[idx])
    return float(np.nanmin(mean_rel[1:]))  # минимум после baseline


def compute_sf_absolute(mean_abs: np.ndarray, mode: str) -> float:
    """
    Вычисляет S(F) по абсолютному объёму:
      - 'absolute': min(mean_abs[1:]) / mean_abs[0]
      - 'absindex:N': mean_abs[N]/mean_abs[0]
    """
    if mode.startswith("absindex:"):
        idx = int(mode.split(":")[1])
        if idx >= len(mean_abs):
            raise IndexError(f"absindex {idx} out of range 0..{len(mean_abs)-1}")
        return float(mean_abs[idx] / mean_abs[0])
    return float(np.nanmin(mean_abs[1:]) / mean_abs[0])


# ---------------------- FITTER ----------------------
class Fitter:
    """Собирает эксперименты, фильтрует и подбирает α, β."""

    def __init__(self, sf_mode: str, min_sf: float,
                 alpha_fixed: Optional[float], verbose: bool):
        self.sf_mode = sf_mode
        self.min_sf = min_sf
        self.alpha_fixed = alpha_fixed
        self.verbose = verbose
        self.experiments: List[TumorExperiment] = []

    def load_file(self, path: Path) -> None:
        """Загружает один файл и добавляет в список, если SF < min_sf."""
        params, time_data, rat_labels, tumor_volumes = \
            process_tumor_data_excel(str(path))
        # 1) парсим фракции из params
        fractions = parse_fractions(params)
        if not fractions:
            if self.verbose:
                print(f"⚠️  {path.name}: no doses recognised in first row")
            return

        # 2) считаем средние объёмы через TumorDataProcessor
        tdp = TumorDataProcessor(np.array(tumor_volumes, dtype=float))
        mean_rel = tdp.get_mean_relative_tumor_volumes()
        mean_abs = tdp.get_mean_tumor_volumes()

        # 3) вычисляем surviving fraction
        if self.sf_mode.startswith("abs"):
            sf = compute_sf_absolute(mean_abs, self.sf_mode)
        else:
            sf = compute_sf_relative(mean_rel, self.sf_mode)

        # 4) фильтрация: пропустить, если SF ≥ min_sf
        if sf >= self.min_sf:
            if self.verbose:
                print(f"ℹ️  {path.name}: SF={sf:.2f} ≥ min_sf={self.min_sf} → skip")
            return

        # 5) добавляем в список
        self.experiments.append(TumorExperiment(path, fractions, sf))
        if self.verbose:
            print(f"✓ {path.name}: fractions={fractions}, SF={sf:.4f}")

    def collect(self, files: List[Path]) -> None:
        """Собирает и дедуплицирует эксперименты по (D, D²)."""
        for p in files:
            self.load_file(p)
        # удаляем дубликаты схем
        uniq: Dict[Tuple[float, float], TumorExperiment] = {}
        for exp in self.experiments:
            key = (exp.dose_sum, exp.dose2_sum)
            uniq.setdefault(key, exp)
        self.experiments = list(uniq.values())

    def fit(self) -> Tuple[float, float]:
        """
        Подбирает α и β:
          - если α не фиксирован: NNLS по X=[D, D²], y=-ln(SF)
          - если α фиксирован: одномерный NNLS для β
        """
        D = np.array([e.dose_sum for e in self.experiments])
        D2 = np.array([e.dose2_sum for e in self.experiments])
        y = np.array([-math.log(e.sf) for e in self.experiments])

        if self.alpha_fixed is None:
            X = np.column_stack([D, D2])
            alpha, beta = nnls(X, y)[0]
        else:
            # y' = y - alpha_fixed * D
            y_shift = y - self.alpha_fixed * D
            beta = max(0.0, np.dot(D2, y_shift) / np.dot(D2, D2))
            alpha = self.alpha_fixed
        return alpha, beta

    def report(self, alpha: float, beta: float) -> None:
        """Печатает отобранные эксперименты и итоговый результат."""
        print("\n# Selected experiments: ")
        for exp in self.experiments:
            print(exp.report())
        print("\n===== FIT RESULT =====")
        print(f"alpha (Gy^-1): {alpha:.4f}")
        print(f"beta  (Gy^-2): {beta:.5f}")
        if beta > 0:
            print(f"alpha/beta   : {alpha/beta:.2f} Gy")
        print("=======================")


# ---------------------- CLI ----------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit α, β using existing excel_data_processor"
    )
    p.add_argument(
        "--files", nargs="*",
        help="Explicit list of .xlsx files. If omitted – all in cwd"
    )
    p.add_argument(
        "--sf", default="relative",
        help="SF mode: relative|absolute|index:N|absindex:N (default: relative)"
    )
    p.add_argument(
        "--alpha", type=float,
        help="Fix α and fit only β (e.g. --alpha 0.3)"
    )
    p.add_argument(
        "--min-sf", type=float, default=1.0,
        help="Skip experiments with SF ≥ min-sf (default: 1.0)"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Verbose output for debug"
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    files = ([Path(f).resolve() for f in args.files]  \
             if args.files else sorted(Path.cwd().glob("*.xlsx")))

    fitter = Fitter(args.sf, args.min_sf, args.alpha, args.verbose)
    fitter.collect(files)

    if len(fitter.experiments) < 2:
        print("⚠️  Need at least two valid experiments", file=sys.stderr)
        sys.exit(1)

    alpha, beta = fitter.fit()
    fitter.report(alpha, beta)


if __name__ == "__main__":
    main()
