# coding: utf-8
"""
fit_alpha_beta_using_processor.py
---------------------------------

Автономный скрипт: подбирает α и β, используя абсолютные объёмы.

Основные шаги
~~~~~~~~~~~~~
1. Читает все .xlsx (или только перечисленные `--files ...`).
2. Через `process_tumor_data_excel` извлекает:
   • experiment_params – первая строка (текст с дозами)
   • time_data, rat_labels, tumor_volumes – сырые данные объёмов.
3. С помощью `TumorDataProcessor` вычисляет **абсолютные** средние объёмы.
4. Вычисляет surviving fraction (SF) по абсолютным объёмам:
   • `absolute`   – `min(mean_abs[1:]) / mean_abs[0]`
   • `absindex:N` – `mean_abs[N] / mean_abs[0]` (N ≥ 1)
5. Фильтрует эксперименты: `SF < --min-sf` (по умолчанию 1.0).
6. Убирает дубликаты по (ΣD, ΣD²).
7. Подбирает α и β:
   • Свободные α, β – curve_fit по модели `SF = exp(-α·D - β·D²)`.
   • Фиксированный α – одномерная регрессия для β.

Запуск
~~~~~~
    python fit_alpha_beta_using_processor.py                       # абсолютный минимум
    python fit_alpha_beta_using_processor.py --alpha 0.3           # фиксируем α
    python fit_alpha_beta_using_processor.py --sf absindex:2 --verbose
    python fit_alpha_beta_using_processor.py --files a.xlsx b.xlsx

Или задайте параметры прямо в файле (см. блок ### INLINE CONFIG ###)
и запустите «Run» из PyCharm.
"""
from __future__ import annotations
import argparse, re, math, sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import curve_fit

# --- готовые модули ----------------------------------
from work_with_prepared_data.radiobioligy_project.data_processing.data_processing \
    import TumorDataProcessor
from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor \
    import process_tumor_data_excel
# -----------------------------------------------------

# ---------------------- INLINE CONFIG -----------------
# Поставьте USE_INLINE_PARAMS = True и отредактируйте переменные
# ниже, чтобы запускать скрипт без CLI-аргументов.
USE_INLINE_PARAMS = False
INLINE_FILES: Optional[List[str]] = None        # пример: ["11.01.2024_y_40.xlsx"]
INLINE_SF_MODE = "absolute"                     # absolute | absindex:N
INLINE_ALPHA: Optional[float] = None            # например 0.3
INLINE_MIN_SF = 1.0
INLINE_VERBOSE = True
# ------------------------------------------------------

# Регулярные выражения для поиска чисел и суффикса дозы
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
GR_SUFFIX = re.compile(r"гр|gy", re.IGNORECASE)


def parse_fractions(experiment_params: List[str]) -> List[float]:
    """Извлекает все числа, за которыми следует 'Гр' или 'Gy'."""
    fracs: List[float] = []
    for token in experiment_params:
        t = token.lower()
        if GR_SUFFIX.search(t):
            for num in NUMBER.findall(t):
                fracs.append(float(num.replace(",", ".")))
    return fracs


def compute_sf(mean_abs: np.ndarray, mode: str) -> float:
    """Вычисляет SF по абсолютным объёмам."""
    if mode.startswith("absindex:"):
        idx = int(mode.split(":")[1])
        if idx >= len(mean_abs):
            raise IndexError(f"absindex {idx} out of 0..{len(mean_abs)-1}")
        return float(mean_abs[idx] / mean_abs[0])
    # mode == 'absolute'
    return float(np.nanmin(mean_abs[1:]) / mean_abs[0])


class TumorExperiment:
    """Контейнер для одного Excel-файла с дозами и SF."""

    def __init__(self, path: Path, fractions: List[float], sf: float):
        self.path = path
        self.fractions = fractions
        self.sf = sf

    @property
    def dose_sum(self) -> float:
        """Суммарная доза (ΣD)"""
        return sum(self.fractions)

    @property
    def dose2_sum(self) -> float:
        """Сумма квадратов доз (ΣD²)"""
        return sum(d*d for d in self.fractions)

    def report(self) -> str:
        """Форматированный отчёт по одному эксперименту."""
        frac = "+".join(f"{d:g}" for d in self.fractions)
        return (f"{self.path.name:30s} ({frac})  "
                f"D={self.dose_sum:5.1f}  D²={self.dose2_sum:6.0f}  "
                f"SF={self.sf:.4f}")


class Fitter:
    """Основной класс: собирает эксперименты и подбирает параметры α и β."""

    def __init__(self, sf_mode: str, min_sf: float,
                 alpha_fixed: Optional[float], verbose: bool):
        self.sf_mode = sf_mode
        self.min_sf = min_sf
        self.alpha_fixed = alpha_fixed
        self.verbose = verbose
        self.experiments: List[TumorExperiment] = []

    def load_file(self, path: Path):
        """Загружает файл, извлекает фракции и SF, фильтрует."""
        params, _, _, volumes = process_tumor_data_excel(str(path))
        fracs = parse_fractions(params)
        if not fracs:
            if self.verbose:
                print(f"⚠️  {path.name}: дозы не распознаны")
            return
        tdp = TumorDataProcessor(np.array(volumes, dtype=float))
        mean_abs = tdp.get_mean_tumor_volumes()
        sf = compute_sf(mean_abs, self.sf_mode)
        if sf >= self.min_sf:
            if self.verbose:
                print(f"ℹ️  {path.name}: SF={sf:.2f} ≥ {self.min_sf} → skip")
            return
        self.experiments.append(TumorExperiment(path, fracs, sf))
        if self.verbose:
            print(f"✓ {path.name}: fractions={fracs}, SF={sf:.4f}")

    def collect(self, files: List[Path]):
        """Собирает эксперименты из файлов и удаляет дубликаты."""
        for p in files:
            self.load_file(p)
        uniq: Dict[Tuple[float, float], TumorExperiment] = {}
        for e in self.experiments:
            uniq.setdefault((e.dose_sum, e.dose2_sum), e)
        self.experiments = list(uniq.values())

    def fit(self) -> Tuple[float, float]:
        """
        Подбирает параметры α и β:
          • Если α задан — решается одномерная регрессия на β
          • Если α свободен — используется curve_fit для подбора обоих
        """
        D = np.array([e.dose_sum for e in self.experiments])
        y = np.array([e.sf for e in self.experiments])

        if self.alpha_fixed is not None:
            y_log = -np.log(y)
            y_shift = y_log - self.alpha_fixed * D
            D2 = np.array([e.dose2_sum for e in self.experiments])
            beta = np.dot(D2, y_shift) / np.dot(D2, D2)
            alpha = self.alpha_fixed
        else:
            def lq_model(dose, alpha, beta):
                return np.exp(-alpha * dose - beta * dose ** 2)
            p0 = [0.1, 0.01]
            bounds = (0, np.inf)
            popt, _ = curve_fit(lq_model, D, y, p0=p0, bounds=bounds)
            alpha, beta = popt

        return alpha, beta

    def report(self, alpha: float, beta: float):
        """Выводит список экспериментов и результат подбора."""
        print("\n# Отобранные эксперименты:")
        for e in self.experiments:
            print(e.report())
        print("\n===== FIT RESULT =====")
        print(f"alpha (Gy^-1): {alpha:.4f}")
        print(f"beta  (Gy^-2): {beta:.5f}")
        if beta > 0:
            print(f"alpha/beta   : {alpha/beta:.2f} Gy")
        print("=======================")


# ---------- CLI ----------
def parse_cli() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    p = argparse.ArgumentParser(
        description="Fit α, β по абсолютным объёмам (excel_data_processor)"
    )
    p.add_argument("--files", nargs="*",
                   help="Список .xlsx; если опущен — берутся все в cwd")
    p.add_argument("--sf", default="absolute",
                   help="SF: absolute | absindex:N (default absolute)")
    p.add_argument("--alpha", type=float,
                   help="Fix α и подбирать только β (пример --alpha 0.3)")
    p.add_argument("--min-sf", type=float, default=1.0,
                   help="Отбросить эксперименты с SF ≥ MIN_SF (default 1.0)")
    p.add_argument("--verbose", action="store_true",
                   help="Подробный вывод для отладки")
    return p.parse_args()


# ---------- MAIN ----------
def run_fit(files: Optional[List[str]],
            sf: str,
            alpha: Optional[float],
            min_sf: float,
            verbose: bool):
    """Запуск подбора: чтение файлов, подбор параметров, вывод."""
    paths = ([Path(f).resolve() for f in files]
             if files else sorted(Path.cwd().glob("*.xlsx")))
    fitter = Fitter(sf, min_sf, alpha, verbose)
    fitter.collect(paths)
    if len(fitter.experiments) < 2:
        print("⚠️  Need at least two valid experiments", file=sys.stderr)
        return
    a, b = fitter.fit()
    fitter.report(a, b)


def main():
    """Основная точка входа."""
    if USE_INLINE_PARAMS:
        run_fit(INLINE_FILES, INLINE_SF_MODE,
                INLINE_ALPHA, INLINE_MIN_SF, INLINE_VERBOSE)
    else:
        args = parse_cli()
        run_fit(args.files, args.sf, args.alpha, args.min_sf, args.verbose)


if __name__ == "__main__":
    main()
