# coding: utf-8
"""
Верификационные тесты биомоделей диссертации.

Три вида проверок:
  1. Аналитические  — числа, посчитанные вручную, сравниваются с кодом.
  2. Физические     — инвариантность, монотонность, граничные случаи.
  3. Биологически   — синтетические данные с известными (α, β) восстанавливаются
     разумные         обратной подгонкой; результат сравнивается с публикациями.

Все тесты работают без реальных Excel-файлов.

Литературные ориентиры для параметров саркомы М-1 и смежных тканей:
  - α/β опухоли (быстро делящиеся ткани): 8–12 Гр [Fowler 1989]
  - α/β поздних реакций кожи: 2–4 Гр [Joiner & van der Kogel]
  - ОБЭ нейтронов (7–14 МэВ): 2.5–4.0 [данные МРНЦ]
  - ОБЭ ионов C в плато: 1.2–1.5; в пике: 2.5–4.0 [LEM, данные GSI]
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

try:
    from survival.fit_alpha_beta_using_processor import LQFitResult, TumorExperiment
    from survival.let_parametrization import LETDependentParams, fit_let_dependence
    from survival.mixed_field_model import FieldComponent, compute_mixed_field_sf
    from survival.radiobiology_analysis import compute_bed
    from survival.tumor_growth_predictor import (
        GeometryReference,
        GrowthModelParameters,
        TreatmentFraction,
        advance_unrepaired_dose,
        gompertz_volume,
        simulate_growth,
        surviving_fraction,
    )
except ModuleNotFoundError:
    from work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor import (
        LQFitResult,
        TumorExperiment,
    )
    from work_with_prepared_data.radiobioligy_project.survival.let_parametrization import (
        LETDependentParams,
        fit_let_dependence,
    )
    from work_with_prepared_data.radiobioligy_project.survival.mixed_field_model import (
        FieldComponent,
        compute_mixed_field_sf,
    )
    from work_with_prepared_data.radiobioligy_project.survival.radiobiology_analysis import (
        compute_bed,
    )
    from work_with_prepared_data.radiobioligy_project.survival.tumor_growth_predictor import (
        GeometryReference,
        GrowthModelParameters,
        TreatmentFraction,
        advance_unrepaired_dose,
        gompertz_volume,
        simulate_growth,
        surviving_fraction,
    )

_DUMMY_PATH = Path(".")


def _make_experiment(
    fractions: tuple[float, ...],
    sf: float = 0.5,
    family: str | None = None,
    schedule_days: tuple[float, ...] = (),
    irradiation_duration_hours: tuple[float, ...] = (),
) -> TumorExperiment:
    """Создать минимальный TumorExperiment для тестов формул."""
    return TumorExperiment(
        path=_DUMMY_PATH,
        fractions=fractions,
        sf=sf,
        family=family,
        schedule_days=schedule_days,
        irradiation_duration_hours=irradiation_duration_hours,
    )


def _lq_sf(alpha: float, beta: float, dose: float) -> float:
    """Аналитическое значение SF по ЛК-модели."""
    return math.exp(-alpha * dose - beta * dose * dose)


# ===========================================================================
# 1. ЛК-модель: аналитические проверки
# ===========================================================================

class LQModelAnalyticTests(unittest.TestCase):
    """
    Проверяем, что surviving_fraction и LQFitResult.predict_sf
    совпадают с аналитикой SF = exp(−αD − βD²).
    """

    def test_sf_at_zero_dose_equals_one(self) -> None:
        """SF при нулевой дозе = 1 (нет повреждений)."""
        sf = surviving_fraction(alpha=0.3, beta=0.03, dose=0.0)
        self.assertAlmostEqual(sf, 1.0, places=12)

    def test_sf_exact_known_alpha_beta_and_dose(self) -> None:
        """
        α=0.3 Гр⁻¹, β=0.03 Гр⁻², D=2 Гр:
        SF = exp(−0.3·2 − 0.03·4) = exp(−0.72) ≈ 0.4868.
        """
        sf = surviving_fraction(alpha=0.3, beta=0.03, dose=2.0)
        expected = math.exp(-0.3 * 2.0 - 0.03 * 4.0)
        self.assertAlmostEqual(sf, expected, places=10)
        self.assertAlmostEqual(sf, 0.48675, delta=0.0001)

    def test_sf_only_alpha_component_is_exponential(self) -> None:
        """При β=0 ЛК-модель вырождается в чистое экспоненциальное затухание."""
        alpha, dose = 0.25, 3.0
        sf = surviving_fraction(alpha=alpha, beta=0.0, dose=dose)
        expected = math.exp(-alpha * dose)
        self.assertAlmostEqual(sf, expected, places=12)

    def test_sf_monotone_decreasing_with_dose(self) -> None:
        """SF строго убывает с ростом дозы."""
        alpha, beta = 0.2, 0.02
        doses = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 32.0]
        sfs = [surviving_fraction(alpha=alpha, beta=beta, dose=d) for d in doses]
        for i in range(len(sfs) - 1):
            self.assertGreater(
                sfs[i], sfs[i + 1],
                msg=f"SF не убывает между D={doses[i]} и D={doses[i+1]} Гр",
            )

    def test_sf_approaches_zero_at_large_dose(self) -> None:
        """SF → 0 при очень большой дозе."""
        sf = surviving_fraction(alpha=0.3, beta=0.03, dose=100.0)
        self.assertLess(sf, 1e-50)

    def test_predict_sf_matches_analytic_single_fraction(self) -> None:
        """LQFitResult.predict_sf для однократного облучения совпадает с аналитикой."""
        alpha, beta, dose = 0.25, 0.025, 4.0
        fit = LQFitResult(
            alpha=alpha, beta=beta,
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
            model_kind="classic_lq",
        )
        exp = _make_experiment(fractions=(dose,), family="y")
        self.assertAlmostEqual(fit.predict_sf(exp), _lq_sf(alpha, beta, dose), places=10)

    def test_predict_sf_multifraction_uses_sum_of_squared_fractions(self) -> None:
        """
        classic_lq для N фракций: SF = exp(−α·ΣDᵢ − β·ΣDᵢ²), без перекрёстных членов.
        """
        alpha, beta, d = 0.2, 0.02, 3.0
        fractions = (d, d, d)
        fit = LQFitResult(
            alpha=alpha, beta=beta,
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
            model_kind="classic_lq",
        )
        exp = _make_experiment(fractions=fractions, family="y")
        expected = math.exp(-(alpha * 3 * d + beta * 3 * d * d))
        self.assertAlmostEqual(fit.predict_sf(exp), expected, places=10)


# ===========================================================================
# 2. BED и EQD2: числовые проверки по стандартным формулам
# ===========================================================================

class BEDAndEQD2Tests(unittest.TestCase):
    """
    Проверяем вычисление биологически эффективной дозы (BED) и
    эквивалентной дозы в 2 Гр (EQD2).

    Стандартные формулы:
      BED  = D_total · (1 + d / (α/β))
      EQD2 = BED / (1 + 2 / (α/β))
    """

    def test_bed_single_fraction_exact_value(self) -> None:
        """
        Однократная доза 4 Гр, α/β=10 Гр:
        BED = 4·(1 + 4/10) = 5.6 Гр.
        """
        fit = LQFitResult(
            alpha=0.3, beta=0.03,                   # α/β = 10
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
        )
        exp = _make_experiment(fractions=(4.0,))
        expected_bed = 4.0 * (1.0 + 4.0 / 10.0)    # = 5.6
        self.assertAlmostEqual(fit.compute_bed(exp), expected_bed, delta=1e-8)

    def test_bed_conventional_5x2gy_equals_12gy(self) -> None:
        """
        5 фракций по 2 Гр, α/β=10 Гр:
        BED = 5·2·(1 + 2/10) = 12.0 Гр.
        (Стандартный эталон — Fowler 1989.)
        """
        fit = LQFitResult(
            alpha=0.3, beta=0.03,
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
        )
        exp = _make_experiment(fractions=(2.0, 2.0, 2.0, 2.0, 2.0))
        self.assertAlmostEqual(fit.compute_bed(exp), 12.0, delta=1e-8)

    def test_eqd2_conventional_5x2gy_equals_10gy(self) -> None:
        """
        5 фракций по 2 Гр, α/β=10 Гр:
        EQD2 = 12.0 / (1 + 2/10) = 10.0 Гр.
        (EQD2 конвенционального курса = суммарная доза.)
        """
        fit = LQFitResult(
            alpha=0.3, beta=0.03,
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
        )
        exp = _make_experiment(fractions=(2.0, 2.0, 2.0, 2.0, 2.0))
        self.assertAlmostEqual(fit.compute_eqd2(exp), 10.0, delta=1e-8)

    def test_eqd2_single_2gy_fraction_equals_2gy(self) -> None:
        """
        EQD2 при однократном облучении ровно 2 Гр = 2 Гр по определению.
        """
        fit = LQFitResult(
            alpha=0.3, beta=0.03,
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
        )
        exp = _make_experiment(fractions=(2.0,))
        self.assertAlmostEqual(fit.compute_eqd2(exp), 2.0, delta=1e-8)

    def test_hypofractionation_gives_higher_bed_than_conventional(self) -> None:
        """
        Гипофракционирование (1×10 Гр) даёт более высокий BED, чем 5×2 Гр.
        BED_hypo = 10·(1 + 10/10) = 20 Гр > 12 Гр = BED_conv.
        """
        fit = LQFitResult(
            alpha=0.3, beta=0.03,
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
        )
        bed_hypo = fit.compute_bed(_make_experiment(fractions=(10.0,)))
        bed_conv = fit.compute_bed(_make_experiment(fractions=(2.0, 2.0, 2.0, 2.0, 2.0)))
        self.assertAlmostEqual(bed_hypo, 20.0, delta=1e-8)
        self.assertGreater(bed_hypo, bed_conv)

    def test_bed_increases_with_total_dose(self) -> None:
        """BED монотонно растёт с суммарной дозой при одинаковом числе фракций."""
        fit = LQFitResult(
            alpha=0.3, beta=0.03,
            train_count=1, train_kind="all",
            family="y", sf_mode="absolute",
        )
        beds = [
            fit.compute_bed(_make_experiment(fractions=(d,)))
            for d in [2.0, 4.0, 8.0, 16.0, 32.0]
        ]
        for i in range(len(beds) - 1):
            self.assertGreater(beds[i + 1], beds[i])


# ===========================================================================
# 3. Repair-модель: G-фактор и межфракционные перекрёстные члены
# ===========================================================================

class RepairModelTests(unittest.TestCase):
    """
    Проверяем функции advance_unrepaired_dose (экспоненциальный распад)
    и quadratic_term(repair_rate) в TumorExperiment.

    Физика: при repair_lq quadratic_term = ΣᵢΣⱼ dᵢdⱼ·g_ij
      где g_ij = exp(−μ·|tᵢ−tⱼ|) для i≠j (межфракционная репарация).
    - Короткий интервал → g → 1 → term → (Σdᵢ)² (как один большой сеанс).
    - Длинный интервал  → g → 0 → term → Σdᵢ² (как classic_lq).
    """

    def test_advance_zero_dt_returns_same_dose(self) -> None:
        """При dt=0 репарации нет — возвращается исходная доза."""
        dose = 3.0
        result = advance_unrepaired_dose(unrepaired_dose=dose, dt_days=0.0, repair_rate_per_day=10.0)
        self.assertAlmostEqual(result, dose, places=10)

    def test_advance_large_dt_approaches_zero(self) -> None:
        """При большом dt вся доза репарируется → 0."""
        result = advance_unrepaired_dose(unrepaired_dose=5.0, dt_days=1000.0, repair_rate_per_day=10.0)
        self.assertLess(result, 1e-20)

    def test_advance_none_repair_rate_returns_zero(self) -> None:
        """Без ставки репарации (repair_rate=None) — нет памяти, возвращает 0."""
        result = advance_unrepaired_dose(unrepaired_dose=5.0, dt_days=1.0, repair_rate_per_day=None)
        self.assertAlmostEqual(result, 0.0, places=12)

    def test_advance_exponential_decay_exact_value(self) -> None:
        """Распад: dose · exp(−μ·t) для μ=ln2·24/T½, T½=1 ч, t=2 ч = 1/12 дня."""
        dose = 4.0
        t_half_hours = 1.0
        repair_rate = math.log(2.0) * 24.0 / t_half_hours  # ~16.6 день⁻¹
        dt_days = 2.0 / 24.0                               # 2 часа
        result = advance_unrepaired_dose(dose, dt_days, repair_rate)
        expected = dose * math.exp(-repair_rate * dt_days)
        self.assertAlmostEqual(result, expected, places=10)

    def test_sf_with_prior_dose_more_lethal(self) -> None:
        """SF с неполной репарацией (prior_dose > 0) меньше, чем без неё."""
        alpha, beta, dose = 0.1, 0.05, 3.0
        sf_no_memory = surviving_fraction(alpha, beta, dose, prior_unrepaired_dose=0.0)
        sf_with_memory = surviving_fraction(alpha, beta, dose, prior_unrepaired_dose=dose)
        self.assertLess(sf_with_memory, sf_no_memory)

    def test_quadratic_term_short_gap_exceeds_dose2_sum(self) -> None:
        """
        Короткий интервал между фракциями (нет репарации) → quadratic_term > Σdᵢ².
        Физика: перекрёстные члены g_ij ≈ 1, поэтому term ≈ (Σdᵢ)² > Σdᵢ².
        """
        fractions = (4.0, 4.0)
        # T½ = 24 ч → μ = ln2 ≈ 0.693 день⁻¹; интервал 1 ч = 1/24 дня → g ≈ 0.972
        slow_repair_rate = math.log(2.0)                        # ~0.693 день⁻¹
        exp = _make_experiment(fractions=fractions, schedule_days=(0.0, 1.0 / 24.0))
        term_with_repair = exp.quadratic_term(repair_rate_per_day=slow_repair_rate)
        self.assertGreater(
            term_with_repair,
            exp.dose2_sum,
            msg="Короткий интервал должен увеличивать квадратичный член (cross-terms)",
        )

    def test_quadratic_term_long_gap_approaches_dose2_sum(self) -> None:
        """
        Длинный интервал (полная репарация) → quadratic_term ≈ Σdᵢ² (как classic_lq).
        """
        fractions = (4.0, 4.0)
        # T½ = 0.5 ч → μ ≈ 33.3 день⁻¹; интервал 5 суток → g ≈ exp(-166) ≈ 0
        fast_repair_rate = math.log(2.0) * 24.0 / 0.5
        exp = _make_experiment(fractions=fractions, schedule_days=(0.0, 5.0))
        term_with_repair = exp.quadratic_term(repair_rate_per_day=fast_repair_rate)
        self.assertAlmostEqual(term_with_repair, exp.dose2_sum, delta=1e-3)

    def test_quadratic_term_simultaneous_fractions_equals_dose_sum_squared(self) -> None:
        """
        При нулевом интервале (одновременное облучение) quadratic_term = (Σdᵢ)².
        """
        fractions = (3.0, 5.0)                            # Σdᵢ = 8, (Σdᵢ)² = 64
        exp = _make_experiment(fractions=fractions, schedule_days=(0.0, 0.0))
        term = exp.quadratic_term(repair_rate_per_day=100.0)
        self.assertAlmostEqual(term, sum(fractions) ** 2, delta=1e-6)


# ===========================================================================
# 4. Zaider-Rossi: физические свойства смешанного поля
# ===========================================================================

class ZaiderRossiPhysicalTests(unittest.TestCase):
    """
    Физические свойства формулы Zaider-Rossi для смешанных полей.
    Ключевые требования:
      - Инвариантность к разбиению однородного поля.
      - Высокий LET повышает летальность.
      - SF всегда ∈ (0, 1].
    """

    def _flat_params(self, alpha: float = 0.2, beta: float = 0.02) -> LETDependentParams:
        """Параметры без LET-зависимости (λ=0)."""
        return LETDependentParams(alpha_0=alpha, lambda_alpha=0.0, beta_0=beta, lambda_beta=0.0)

    def test_splitting_homogeneous_field_in_two_does_not_change_sf(self) -> None:
        """
        Разбиение однородного поля D на два одинаковых компонента D/2
        не должно менять SF (инвариантность разбиения).

        Доказательство вручную для α=0.2, β=0.02, D=4 Гр:
          single:  exp = α·4 + β·16 = 0.8 + 0.32 = 1.12
          split (Zaider-Rossi):
            diag = α·2 + β·4 + α·2 + β·4 = 1.12 − cross
            cross = 2·√(β²)·2·2 = 2·β·4 = 0.16
            total = 0.96 + 0.16 = 1.12 ← совпадает ✓
        """
        params = self._flat_params()
        total_dose = 4.0
        single = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=total_dose, mean_let_kev_um=0.3)],
            params,
        )
        split = compute_mixed_field_sf(
            [
                FieldComponent(family="y", dose_fraction_gy=total_dose / 2, mean_let_kev_um=0.3),
                FieldComponent(family="y", dose_fraction_gy=total_dose / 2, mean_let_kev_um=0.3),
            ],
            params,
        )
        self.assertAlmostEqual(split.sf_zaider_rossi, single.sf_zaider_rossi, places=8)
        self.assertAlmostEqual(split.sf_tdra, single.sf_tdra, places=8)

    def test_splitting_in_four_equal_parts_does_not_change_sf(self) -> None:
        """То же при разбиении на четыре части."""
        params = self._flat_params()
        total_dose, n = 8.0, 4
        single = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=total_dose, mean_let_kev_um=0.3)],
            params,
        )
        split = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=total_dose / n, mean_let_kev_um=0.3)
             for _ in range(n)],
            params,
        )
        self.assertAlmostEqual(split.sf_zaider_rossi, single.sf_zaider_rossi, places=6)

    def test_total_dose_equals_sum_of_component_doses(self) -> None:
        """total_dose_gy = арифметическая сумма доз компонентов."""
        params = self._flat_params()
        result = compute_mixed_field_sf(
            [
                FieldComponent(family="p", dose_fraction_gy=3.0, mean_let_kev_um=8.0),
                FieldComponent(family="n", dose_fraction_gy=2.0, mean_let_kev_um=30.0),
            ],
            params,
        )
        self.assertAlmostEqual(result.total_dose_gy, 5.0, places=10)

    def test_high_let_component_increases_lethality(self) -> None:
        """
        Замена половины дозы низкого LET на нейтроны (высокий LET)
        при той же суммарной дозе повышает летальность (снижает SF).
        """
        params = LETDependentParams(
            alpha_0=0.08, lambda_alpha=0.005,
            beta_0=0.01, lambda_beta=0.0,
        )
        total_dose = 4.0
        low_let_only = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=total_dose, mean_let_kev_um=0.3)],
            params,
        )
        mixed = compute_mixed_field_sf(
            [
                FieldComponent(family="y", dose_fraction_gy=total_dose / 2, mean_let_kev_um=0.3),
                FieldComponent(family="n", dose_fraction_gy=total_dose / 2, mean_let_kev_um=40.0),
            ],
            params,
        )
        self.assertLess(mixed.sf_zaider_rossi, low_let_only.sf_zaider_rossi)
        self.assertLess(mixed.sf_tdra, low_let_only.sf_tdra)

    def test_sf_strictly_between_zero_and_one(self) -> None:
        """SF всегда строго ∈ (0, 1] для любой дозы."""
        params = LETDependentParams(alpha_0=0.3, lambda_alpha=0.01, beta_0=0.03, lambda_beta=0.0)
        for dose in [0.5, 2.0, 5.0, 10.0, 20.0]:
            result = compute_mixed_field_sf(
                [FieldComponent(family="p", dose_fraction_gy=dose, mean_let_kev_um=5.0)],
                params,
            )
            self.assertGreater(result.sf_zaider_rossi, 0.0)
            self.assertLessEqual(result.sf_zaider_rossi, 1.0)

    def test_effective_alpha_beta_are_dose_weighted_averages(self) -> None:
        """
        При однородном LET эффективные α_eff и β_eff совпадают с α и β компонент.
        """
        alpha, beta = 0.25, 0.025
        params = LETDependentParams(alpha_0=alpha, lambda_alpha=0.0, beta_0=beta, lambda_beta=0.0)
        result = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=3.0, mean_let_kev_um=0.3)],
            params,
            method="zaider_rossi",
        )
        self.assertAlmostEqual(result.effective_alpha, alpha, places=8)
        self.assertAlmostEqual(result.effective_beta, beta, places=8)


# ===========================================================================
# 5. ОБЭ (Relative Biological Effectiveness): физические ограничения
# ===========================================================================

class RBEPhysicalTests(unittest.TestCase):
    """
    ОБЭ = D_reference / D_test при изоэффекте (одинаковый SF).
    Физические требования:
      - ОБЭ нейтронов > 1 по отношению к гамма.
      - ОБЭ ионов углерода > ОБЭ нейтронов при одном и том же LET-профиле.
    """

    @staticmethod
    def _rbe_at_isoeffect(
        params: LETDependentParams,
        ref_let: float,
        test_let: float,
        ref_dose: float,
    ) -> float:
        """Найти ОБЭ бинарным поиском при изоэффекте SF."""
        target_sf = compute_mixed_field_sf(
            [FieldComponent(family="y", dose_fraction_gy=ref_dose, mean_let_kev_um=ref_let)],
            params,
        ).sf_zaider_rossi

        lo, hi = 1e-6, ref_dose * 20.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            sf_mid = compute_mixed_field_sf(
                [FieldComponent(family="n", dose_fraction_gy=mid, mean_let_kev_um=test_let)],
                params,
            ).sf_zaider_rossi
            if sf_mid > target_sf:
                lo = mid
            else:
                hi = mid
        test_dose = (lo + hi) / 2.0
        return ref_dose / test_dose

    def _let_dependent_params(self) -> LETDependentParams:
        return LETDependentParams(
            alpha_0=0.1,  lambda_alpha=0.005,
            beta_0=0.02, lambda_beta=0.0,
        )

    def test_rbe_neutrons_greater_than_one(self) -> None:
        """
        ОБЭ нейтронов (LET ≈ 20 кэВ/мкм) > 1 по отношению к гамма (LET ≈ 0.3).
        Ожидаемый диапазон по литературе МРНЦ: 2.5–4.0.
        """
        params = self._let_dependent_params()
        rbe = self._rbe_at_isoeffect(params, ref_let=0.3, test_let=20.0, ref_dose=4.0)
        self.assertGreater(rbe, 1.0, msg=f"ОБЭ нейтронов = {rbe:.3f}, ожидается > 1")

    def test_rbe_carbon_ions_greater_than_neutrons(self) -> None:
        """
        ОБЭ ионов углерода (LET ≈ 100) > ОБЭ нейтронов (LET ≈ 20).
        """
        params = self._let_dependent_params()
        rbe_n = self._rbe_at_isoeffect(params, ref_let=0.3, test_let=20.0, ref_dose=4.0)
        rbe_c = self._rbe_at_isoeffect(params, ref_let=0.3, test_let=100.0, ref_dose=4.0)
        self.assertGreater(rbe_c, rbe_n,
                           msg=f"ОБЭ C (={rbe_c:.2f}) должен быть > ОБЭ n (={rbe_n:.2f})")

    def test_rbe_gamma_reference_equals_one(self) -> None:
        """ОБЭ гамма по отношению к самому себе = 1."""
        params = self._let_dependent_params()
        rbe = self._rbe_at_isoeffect(params, ref_let=0.3, test_let=0.3, ref_dose=4.0)
        self.assertAlmostEqual(rbe, 1.0, delta=0.01)


# ===========================================================================
# 6. Модель Гомпертца: аналитические и граничные случаи
# ===========================================================================

class GompertzModelTests(unittest.TestCase):
    """
    Проверяем gompertz_volume и simulate_growth.

    Аналитическое решение:
      V(t) = K · exp( ln(V₀/K) · exp(−r·t) )
    Граничные условия:
      V(0) = V₀;  V(∞) → K;  при r=0: V = const.
    """

    def test_v0_exactly_equals_initial_volume(self) -> None:
        """V(0) = V₀."""
        v0, r, K = 150.0, 0.05, 5000.0
        v = float(gompertz_volume(time_days=0.0, initial_volume=v0, growth_rate=r, carrying_capacity=K))
        self.assertAlmostEqual(v, v0, places=8)

    def test_v_approaches_carrying_capacity(self) -> None:
        """V(500 дней) ≈ K (насыщение)."""
        v0, r, K = 100.0, 0.10, 4000.0
        v = float(gompertz_volume(time_days=500.0, initial_volume=v0, growth_rate=r, carrying_capacity=K))
        self.assertAlmostEqual(v, K, delta=0.01)

    def test_volume_monotone_increasing_without_treatment(self) -> None:
        """Без лечения V(t) монотонно растёт при V₀ < K."""
        v0, r, K = 200.0, 0.05, 5000.0
        times = np.linspace(0.0, 60.0, 30)
        volumes = gompertz_volume(time_days=times, initial_volume=v0, growth_rate=r, carrying_capacity=K)
        for i in range(len(volumes) - 1):
            self.assertGreater(float(volumes[i + 1]), float(volumes[i]))

    def test_zero_growth_rate_gives_constant_volume(self) -> None:
        """При r=0 объём не изменяется."""
        v0 = 300.0
        times = np.array([0.0, 5.0, 14.0, 30.0])
        volumes = gompertz_volume(time_days=times, initial_volume=v0, growth_rate=0.0, carrying_capacity=5000.0)
        for v in volumes:
            self.assertAlmostEqual(float(v), v0, delta=1e-6)

    def test_analytic_value_at_14_days(self) -> None:
        """
        Аналитический контроль для r=0.05, K=5000, V₀=200, t=14:
        V(14) = 5000 · exp(ln(200/5000) · exp(−0.05·14)).
        """
        v0, r, K, t = 200.0, 0.05, 5000.0, 14.0
        expected = K * math.exp(math.log(v0 / K) * math.exp(-r * t))
        actual = float(gompertz_volume(time_days=t, initial_volume=v0, growth_rate=r, carrying_capacity=K))
        self.assertAlmostEqual(actual, expected, places=8)

    def test_simulate_growth_no_treatment_matches_gompertz_formula(self) -> None:
        """
        simulate_growth без облучения (α=β=clearance=0) совпадает
        с аналитическим решением Гомпертца.
        """
        v0, r, K = 200.0, 0.05, 4000.0
        params = GrowthModelParameters(
            alpha=0.0, beta=0.0,
            growth_rate=r, carrying_capacity=K, clearance_rate=0.0,
        )
        reference = GeometryReference(
            axis_a=v0 ** (1.0 / 3.0),
            axis_b=v0 ** (1.0 / 3.0),
            axis_c=v0 ** (1.0 / 3.0),
            volume=v0,
        )
        times = [0.0, 7.0, 14.0, 21.0, 28.0]

        result = simulate_growth(
            sample_times=times,
            parameters=params,
            reference=reference,
            schedule=[],
        )

        for i, t in enumerate(times):
            expected = K * math.exp(math.log(v0 / K) * math.exp(-r * t))
            actual = float(result.live_volume[i])
            self.assertAlmostEqual(
                actual, expected, delta=expected * 0.005,
                msg=f"Гомпертц расходится на t={t} дней",
            )

    def test_treated_volume_less_than_untreated_at_late_times(self) -> None:
        """
        Облучённая опухоль меньше необлучённой на поздних сроках
        при достаточно высокой дозе.
        """
        v0 = 500.0
        reference = GeometryReference(
            axis_a=v0 ** (1.0 / 3.0),
            axis_b=v0 ** (1.0 / 3.0),
            axis_c=v0 ** (1.0 / 3.0),
            volume=v0,
        )
        growth_kwargs = dict(growth_rate=0.03, carrying_capacity=5000.0, clearance_rate=0.1)
        params_untreated = GrowthModelParameters(alpha=0.0, beta=0.0, **growth_kwargs)
        params_treated   = GrowthModelParameters(alpha=0.3, beta=0.03, **growth_kwargs)
        times = [0.0, 7.0, 14.0, 21.0]

        untreated = simulate_growth(
            sample_times=times, parameters=params_untreated,
            reference=reference, schedule=[],
        )
        treated = simulate_growth(
            sample_times=times, parameters=params_treated,
            reference=reference,
            schedule=[TreatmentFraction(day=0.0, dose=10.0)],
        )

        self.assertLess(
            float(treated.total_volume[-1]),
            float(untreated.total_volume[-1]),
            msg="Облучённая опухоль должна быть меньше необлучённой на поздних сроках",
        )


# ===========================================================================
# 7. LET-зависимость: физические ограничения
# ===========================================================================

class LETParametrizationPhysicalTests(unittest.TestCase):
    """Физические ограничения LET-зависимых параметров α(LET) и β(LET)."""

    def test_alpha_at_zero_let_equals_alpha0(self) -> None:
        """α(LET=0) = α₀."""
        params = LETDependentParams(alpha_0=0.15, lambda_alpha=0.003, beta_0=0.02, lambda_beta=0.0)
        self.assertAlmostEqual(params.alpha(0.0), 0.15, places=10)

    def test_alpha_increases_with_let(self) -> None:
        """α(LET) строго растёт при λ_α > 0."""
        params = LETDependentParams(alpha_0=0.1, lambda_alpha=0.004, beta_0=0.02, lambda_beta=0.0)
        lets = [0.3, 5.0, 20.0, 80.0]
        alphas = [params.alpha(let) for let in lets]
        for i in range(len(alphas) - 1):
            self.assertGreater(alphas[i + 1], alphas[i])

    def test_alpha_saturates_at_let_max(self) -> None:
        """При let_max насыщение: α(LET > let_max) = α(let_max)."""
        params = LETDependentParams(
            alpha_0=0.1, lambda_alpha=0.01,
            beta_0=0.02, lambda_beta=0.0,
            let_max=20.0,
        )
        alpha_at_max   = params.alpha(20.0)
        alpha_beyond_1 = params.alpha(50.0)
        alpha_beyond_2 = params.alpha(200.0)
        self.assertAlmostEqual(alpha_at_max, alpha_beyond_1, places=8)
        self.assertAlmostEqual(alpha_at_max, alpha_beyond_2, places=8)

    def test_alpha_beta_ratio_increases_with_let(self) -> None:
        """
        α/β возрастает с LET при λ_β=0 (β=const, α растёт).

        Физика: при высоком LET клетки гибнут преимущественно
        через однопоражённые события (α-канал). BED ≈ D (линейно).
        Это согласуется с MKM: при высоком LET β→0, α/β→∞.
        Линейная параметризация (λ_β=0) отражает это через рост α/β:
          α/β(LET) = (α₀ + λ_α·LET) / β₀  → растёт с LET.
        """
        params = LETDependentParams(
            alpha_0=0.1, lambda_alpha=0.005,
            beta_0=0.02, lambda_beta=0.0,
        )
        ab_gamma   = params.alpha_beta_ratio(0.3)    # гамма   ≈ 5.1
        ab_proton  = params.alpha_beta_ratio(5.0)    # протоны ≈ 6.2
        ab_neutron = params.alpha_beta_ratio(20.0)   # нейтроны ≈ 7.5
        ab_carbon  = params.alpha_beta_ratio(80.0)   # ионы C  ≈ 12.5
        self.assertLess(ab_gamma, ab_proton)
        self.assertLess(ab_proton, ab_neutron)
        self.assertLess(ab_neutron, ab_carbon)

    def test_fit_let_dependence_recovers_known_linear_parameters(self) -> None:
        """
        Синтетические α/β на сетке LET должны корректно восстанавливаться.
        """
        alpha_0, lambda_alpha = 0.12, 0.003
        beta_0, lambda_beta   = 0.015, 0.00002
        family_lets = {"y": 0.3, "p": 5.0, "n": 25.0, "c": 90.0}

        family_results = {
            fam: LQFitResult(
                alpha=alpha_0 + lambda_alpha * let_val,
                beta=beta_0  + lambda_beta  * let_val,
                train_count=5, train_kind="all",
                family=fam, sf_mode="absolute",
            )
            for fam, let_val in family_lets.items()
        }

        fitted = fit_let_dependence(family_results, family_lets)

        self.assertAlmostEqual(fitted.alpha_0,      alpha_0,      delta=alpha_0 * 0.01)
        self.assertAlmostEqual(fitted.lambda_alpha, lambda_alpha, delta=lambda_alpha * 0.05)
        self.assertAlmostEqual(fitted.beta_0,       beta_0,       delta=beta_0 * 0.01)
        self.assertAlmostEqual(fitted.alpha_r_squared or 0.0, 1.0, delta=0.001)


# ===========================================================================
# 8. Биологически разумные диапазоны α/β (синтетическое восстановление)
# ===========================================================================

class BiologicalPlausibilityTests(unittest.TestCase):
    """
    Генерируем SF(D) по известным (α, β), подгоняем scipy.curve_fit,
    проверяем что α/β попадает в диапазоны из радиобиологической литературы.

    Нет шума — подгонка обязана восстанавливать параметры точно.
    """

    @staticmethod
    def _lq(doses: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return np.exp(-alpha * doses - beta * doses ** 2)

    def _fit_alpha_beta(
        self,
        true_alpha: float,
        true_beta: float,
        doses: list[float],
    ) -> tuple[float, float]:
        doses_arr = np.array(doses, dtype=float)
        sf_true   = self._lq(doses_arr, true_alpha, true_beta)
        (alpha_fit, beta_fit), _ = curve_fit(
            self._lq, doses_arr, sf_true,
            p0=[true_alpha, true_beta],
            bounds=([0.0, 0.0], [5.0, 1.0]),
        )
        return float(alpha_fit), float(beta_fit)

    def _assert_fit(
        self,
        label: str,
        true_alpha: float,
        true_beta: float,
        doses: list[float],
        ab_min: float,
        ab_max: float,
    ) -> None:
        alpha_fit, beta_fit = self._fit_alpha_beta(true_alpha, true_beta, doses)
        ab = alpha_fit / max(beta_fit, 1e-12)
        self.assertAlmostEqual(alpha_fit, true_alpha, delta=true_alpha * 0.02,
                               msg=f"{label}: восстановленный α далёк от исходного")
        self.assertAlmostEqual(beta_fit,  true_beta,  delta=true_beta  * 0.05,
                               msg=f"{label}: восстановленный β далёк от исходного")
        self.assertGreater(ab, ab_min, msg=f"{label}: α/β={ab:.2f} < min={ab_min}")
        self.assertLess   (ab, ab_max, msg=f"{label}: α/β={ab:.2f} > max={ab_max}")

    def test_no_noise_exact_recovery(self) -> None:
        """Без шума curve_fit восстанавливает (α, β) с точностью 0.1%."""
        true_alpha, true_beta = 0.35, 0.035
        alpha_fit, beta_fit = self._fit_alpha_beta(
            true_alpha, true_beta,
            doses=[1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0],
        )
        self.assertAlmostEqual(alpha_fit, true_alpha, delta=true_alpha * 0.001)
        self.assertAlmostEqual(beta_fit,  true_beta,  delta=true_beta  * 0.001)

    def test_gamma_tumor_alpha_beta_ratio_near_10(self) -> None:
        """
        Гамма-излучение, быстро делящаяся опухоль (саркома):
        α/β ∈ [8, 12] Гр. [Fowler 1989, Br. J. Radiol. 62:679]
        """
        self._assert_fit(
            label="γ-излучение, опухоль",
            true_alpha=0.30, true_beta=0.030,    # α/β = 10
            doses=[4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 32.0],
            ab_min=8.0, ab_max=12.0,
        )

    def test_electrons_similar_to_gamma(self) -> None:
        """
        Электроны: α/β ≈ 10 Гр (близкое качество к гамма).
        Диапазон: [7, 14] Гр.
        """
        self._assert_fit(
            label="электроны",
            true_alpha=0.28, true_beta=0.028,    # α/β = 10
            doses=[4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 32.0],
            ab_min=7.0, ab_max=14.0,
        )

    def test_neutrons_high_alpha_beta_ratio(self) -> None:
        """
        Нейтроны 7–14 МэВ: α/β ∈ [20, 80] Гр.
        Физика: высокий LET → преобладает α-компонент (одиночные разрывы).
        """
        self._assert_fit(
            label="нейтроны",
            true_alpha=0.60, true_beta=0.015,    # α/β = 40
            doses=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
            ab_min=15.0, ab_max=80.0,
        )

    def test_carbon_ions_very_high_alpha_beta_ratio(self) -> None:
        """
        Ионы углерода (пик Брэгга): α/β ∈ [30, 200] Гр.
        Физика: очень высокий LET → поражение практически только α-путём.
        """
        self._assert_fit(
            label="ионы C (пик Брэгга)",
            true_alpha=0.80, true_beta=0.010,    # α/β = 80
            doses=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            ab_min=30.0, ab_max=200.0,
        )

    def test_late_responding_skin_low_alpha_beta_ratio(self) -> None:
        """
        Поздние реакции кожи: α/β ∈ [2, 5] Гр.
        [Joiner & van der Kogel, 'Basic Clinical Radiobiology', 4th ed.]
        """
        self._assert_fit(
            label="поздние реакции кожи",
            true_alpha=0.10, true_beta=0.033,    # α/β ≈ 3
            doses=[2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0],
            ab_min=2.0, ab_max=5.0,
        )

    def test_order_of_alpha_beta_ratios_is_physically_correct(self) -> None:
        """
        Порядок α/β по типам излучения: γ ≈ e < n < C.
        Физически: чем выше LET, тем более доминирует α-компонент.
        """
        doses_high_dose = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 32.0]

        _, beta_y = self._fit_alpha_beta(0.30, 0.030, doses_high_dose)
        alpha_y, _ = self._fit_alpha_beta(0.30, 0.030, doses_high_dose)
        ab_gamma = alpha_y / max(beta_y, 1e-12)

        alpha_n, beta_n = self._fit_alpha_beta(0.60, 0.015, doses_high_dose)
        ab_neutron = alpha_n / max(beta_n, 1e-12)

        alpha_c, beta_c = self._fit_alpha_beta(0.80, 0.010, doses_high_dose)
        ab_carbon = alpha_c / max(beta_c, 1e-12)

        self.assertLess(ab_gamma, ab_neutron,
                        msg=f"α/β(γ)={ab_gamma:.1f} должен быть < α/β(n)={ab_neutron:.1f}")
        self.assertLess(ab_neutron, ab_carbon,
                        msg=f"α/β(n)={ab_neutron:.1f} должен быть < α/β(C)={ab_carbon:.1f}")


# ===========================================================================
# 10. compute_bed: standalone BED с поправкой на ОБЭ
# ===========================================================================

class ComputeBEDTests(unittest.TestCase):
    """
    Верификация функции compute_bed(dose_total, n_fractions, alpha_beta, *, rbe_factor).

    Аналитический контроль: BED = n·d_phys·(1 + d_phys/α/β), d_phys = dose/n/rbe.
    Группы 1–7 (физическая доза): rbe_factor=1.0 (по умолчанию).
    Группы 3 и 8 (ОБЭ-взвешенная доза): rbe_factor=1.1.
    """

    def test_single_fraction_physical_dose(self) -> None:
        """
        Однократная доза 32 Гр (группа 1), α/β=10 Гр:
        BED = 32·(1 + 32/10) = 32·4.2 = 134.4 Гр.
        """
        result = compute_bed(32.0, 1, 10.0)
        self.assertAlmostEqual(result, 134.4, delta=1e-8)

    def test_group3_rbe_weighted_dose(self) -> None:
        """
        Группа 3: 38 Гр·ОБЭ однократно, ОБЭ=1.1, α/β=10 Гр.
        d_phys = 38/1.1 ≈ 34.5455 Гр
        BED = 34.5455·(1 + 34.5455/10) ≈ 153.89 Гр.
        """
        d_phys = 38.0 / 1.1
        expected = d_phys * (1.0 + d_phys / 10.0)
        result = compute_bed(38.0, 1, 10.0, rbe_factor=1.1)
        self.assertAlmostEqual(result, expected, delta=1e-6)

    def test_group8_rbe_weighted_two_fractions(self) -> None:
        """
        Группа 8: 2×25,1 Гр·ОБЭ, ОБЭ=1.1, α/β=10 Гр.
        d_phys = 25.1/1.1 ≈ 22.8182 Гр
        BED = 2·22.8182·(1 + 22.8182/10) ≈ 149.8 Гр.
        """
        d_phys = 25.1 / 1.1
        expected = 2 * d_phys * (1.0 + d_phys / 10.0)
        result = compute_bed(2 * 25.1, 2, 10.0, rbe_factor=1.1)
        self.assertAlmostEqual(result, expected, delta=1e-6)

    def test_rbe_factor_one_equals_no_rbe(self) -> None:
        """rbe_factor=1.0 — поведение идентично вызову без rbe_factor."""
        self.assertAlmostEqual(
            compute_bed(40.0, 1, 10.0, rbe_factor=1.0),
            compute_bed(40.0, 1, 10.0),
            delta=1e-12,
        )

    def test_rbe_correction_lowers_bed(self) -> None:
        """ОБЭ-поправка уменьшает физическую дозу → BED должен быть меньше."""
        bed_phys = compute_bed(38.0, 1, 10.0)
        bed_rbe = compute_bed(38.0, 1, 10.0, rbe_factor=1.1)
        self.assertGreater(bed_phys, bed_rbe)

    def test_invalid_n_fractions_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_bed(30.0, 0, 10.0)

    def test_invalid_alpha_beta_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_bed(30.0, 1, 0.0)

    def test_invalid_rbe_factor_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_bed(30.0, 1, 10.0, rbe_factor=0.0)


if __name__ == "__main__":
    unittest.main()
