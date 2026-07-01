# Verification Tests: Biomodel Validation

> **Статус последнего прогона:** 49/49 OK — 2026-05-29  
> **Файл тестов:** `survival/tests/test_biomodel_verification.py`

Тесты проверяют корректность биомоделей на трёх уровнях:

1. **Аналитический** — результат совпадает с числом, посчитанным вручную по формуле.
2. **Физический** — модель соблюдает инвариантность, монотонность и граничные условия.
3. **Биологически разумный** — подогнанные параметры попадают в диапазоны из опубликованных работ.

---

## Как запустить

```bash
cd C:\dev\neuro_stats
python -m unittest discover \
    -s work_with_prepared_data/radiobioligy_project/survival/tests \
    -p "test_biomodel_verification.py" \
    -v
```

> Запуск должен выполняться из `C:\dev\neuro_stats`, а не из подпапки проекта,
> чтобы пакет `work_with_prepared_data` был виден Python.

Чтобы прогнать все тесты модуля сразу:

```bash
cd C:\dev\neuro_stats
python -m unittest discover \
    -s work_with_prepared_data/radiobioligy_project/survival/tests \
    -v
```

---

## Результаты прогона (2026-05-29)

```
Ran 49 tests in 0.013s

OK
```

Все 49 тестов прошли без ошибок и падений.

---

## Что тестируется

### 1. ЛК-модель (6 тестов) — `LQModelAnalyticTests`

| Тест | Проверяемое утверждение |
|---|---|
| `test_sf_at_zero_dose_equals_one` | SF = 1 при D = 0 |
| `test_sf_exact_known_alpha_beta_and_dose` | α=0.3, β=0.03, D=2 Гр → SF = exp(−0.72) ≈ 0.4868 |
| `test_sf_only_alpha_component_is_exponential` | β=0 → SF = exp(−αD) |
| `test_sf_monotone_decreasing_with_dose` | SF строго убывает при росте D |
| `test_sf_approaches_zero_at_large_dose` | SF < 10⁻⁵⁰ при D = 100 Гр |
| `test_predict_sf_matches_analytic_single_fraction` | `LQFitResult.predict_sf` совпадает с аналитикой |
| `test_predict_sf_multifraction_uses_sum_of_squared_fractions` | classic_lq: SF = exp(−α·ΣDᵢ − β·ΣDᵢ²), без cross-terms |

### 2. BED и EQD2 (6 тестов) — `BEDAndEQD2Tests`

Стандартные формулы: BED = D(1 + d/(α/β)), EQD2 = BED / (1 + 2/(α/β)).

| Тест | Проверяемое значение |
|---|---|
| `test_bed_single_fraction_exact_value` | 1×4 Гр, α/β=10: BED = **5.6 Гр** |
| `test_bed_conventional_5x2gy_equals_12gy` | 5×2 Гр, α/β=10: BED = **12.0 Гр** |
| `test_eqd2_conventional_5x2gy_equals_10gy` | 5×2 Гр, α/β=10: EQD2 = **10.0 Гр** |
| `test_eqd2_single_2gy_fraction_equals_2gy` | 1×2 Гр: EQD2 = **2.0 Гр** (по определению) |
| `test_hypofractionation_gives_higher_bed_than_conventional` | BED(1×10) = 20 Гр > BED(5×2) = 12 Гр |
| `test_bed_increases_with_total_dose` | BED монотонно растёт с дозой |

### 3. Repair-модель (8 тестов) — `RepairModelTests`

Проверяет функцию `advance_unrepaired_dose` (экспоненциальный распад)
и квадратичный член `TumorExperiment.quadratic_term(repair_rate)`.

| Тест | Проверяемое утверждение |
|---|---|
| `test_advance_zero_dt_returns_same_dose` | dt=0: нет репарации, доза не меняется |
| `test_advance_large_dt_approaches_zero` | dt=1000 дней: полная репарация → 0 |
| `test_advance_none_repair_rate_returns_zero` | repair_rate=None: нет памяти → 0 |
| `test_advance_exponential_decay_exact_value` | Точное значение: dose·exp(−μ·t) |
| `test_sf_with_prior_dose_more_lethal` | prior_dose > 0 снижает SF (усиливает гибель) |
| `test_quadratic_term_short_gap_exceeds_dose2_sum` | Короткий интервал → cross-terms активны → term > Σdᵢ² |
| `test_quadratic_term_long_gap_approaches_dose2_sum` | Длинный интервал → полная репарация → term ≈ Σdᵢ² |
| `test_quadratic_term_simultaneous_fractions_equals_dose_sum_squared` | Одновременное облучение → term = (Σdᵢ)² |

### 4. Zaider-Rossi / TDRA (6 тестов) — `ZaiderRossiPhysicalTests`

| Тест | Проверяемое утверждение |
|---|---|
| `test_splitting_homogeneous_field_in_two_does_not_change_sf` | Разбиение D на 2×(D/2) не меняет SF (аналитически доказано) |
| `test_splitting_in_four_equal_parts_does_not_change_sf` | То же для 4 компонентов |
| `test_total_dose_equals_sum_of_component_doses` | `total_dose_gy` = арифметическая сумма |
| `test_high_let_component_increases_lethality` | Высокий LET снижает SF при той же суммарной дозе |
| `test_sf_strictly_between_zero_and_one` | SF ∈ (0, 1] для любой дозы |
| `test_effective_alpha_beta_are_dose_weighted_averages` | При λ=0: α_eff = α, β_eff = β |

### 5. ОБЭ (3 теста) — `RBEPhysicalTests`

ОБЭ = D_γ / D_test при изоэффекте. Находится бинарным поиском.

| Тест | Проверяемое утверждение |
|---|---|
| `test_rbe_neutrons_greater_than_one` | ОБЭ нейтронов (LET≈20) > 1 относительно гамма |
| `test_rbe_carbon_ions_greater_than_neutrons` | ОБЭ ионов C (LET≈100) > ОБЭ нейтронов |
| `test_rbe_gamma_reference_equals_one` | ОБЭ гамма относительно самого себя = 1 |

### 6. Модель Гомпертца (7 тестов) — `GompertzModelTests`

Аналитическое решение: V(t) = K · exp(ln(V₀/K) · exp(−r·t)).

| Тест | Проверяемое утверждение |
|---|---|
| `test_v0_exactly_equals_initial_volume` | V(0) = V₀ точно |
| `test_v_approaches_carrying_capacity` | V(500 дней) ≈ K |
| `test_volume_monotone_increasing_without_treatment` | V(t) монотонно растёт при V₀ < K |
| `test_zero_growth_rate_gives_constant_volume` | r=0 → V = const |
| `test_analytic_value_at_14_days` | Точное число: V(14) при r=0.05, K=5000, V₀=200 |
| `test_simulate_growth_no_treatment_matches_gompertz_formula` | `simulate_growth` без облучения = аналитика (δ < 0.5%) |
| `test_treated_volume_less_than_untreated_at_late_times` | Облучённая опухоль меньше необлучённой на 21 день |

### 7. LET-зависимость (5 тестов) — `LETParametrizationPhysicalTests`

| Тест | Проверяемое утверждение |
|---|---|
| `test_alpha_at_zero_let_equals_alpha0` | α(LET=0) = α₀ |
| `test_alpha_increases_with_let` | α(LET) строго растёт при λ_α > 0 |
| `test_alpha_saturates_at_let_max` | α(LET > let_max) = α(let_max) |
| `test_alpha_beta_ratio_increases_with_let` | α/β растёт с LET при λ_β=0 (высокий LET → α-доминирование) |
| `test_fit_let_dependence_recovers_known_linear_parameters` | Линейная подгонка восстанавливает α₀, λ_α с точностью 1% |

### 8. Биологически разумные диапазоны (8 тестов) — `BiologicalPlausibilityTests`

Синтетические данные SF(D) генерируются с известными (α, β), затем подгоняются
через `scipy.optimize.curve_fit`. Результат сравнивается с опубликованными диапазонами.

| Тест | Тип излучения | Истинный α/β | Ожидаемый диапазон | Источник |
|---|---|---|---|---|
| `test_no_noise_exact_recovery` | — | 10 Гр | восстановление с точностью 0.1% | — |
| `test_gamma_tumor_alpha_beta_ratio_near_10` | γ, опухоль | 10 Гр | **[8, 12] Гр** | Fowler 1989 |
| `test_electrons_similar_to_gamma` | e⁻ | 10 Гр | **[7, 14] Гр** | — |
| `test_neutrons_high_alpha_beta_ratio` | n (7–14 МэВ) | 40 Гр | **[20, 80] Гр** | данные МРНЦ |
| `test_carbon_ions_very_high_alpha_beta_ratio` | C (пик) | 80 Гр | **[30, 200] Гр** | LEM/MKM |
| `test_late_responding_skin_low_alpha_beta_ratio` | γ, кожа (поздние) | 3 Гр | **[2, 5] Гр** | Joiner & van der Kogel |
| `test_order_of_alpha_beta_ratios_is_physically_correct` | γ < n < C | — | α/β монотонно растёт с LET | физика ЛК-модели |

---

## Что не покрыто этими тестами

Следующие аспекты требуют тестов на реальных экспериментальных данных
(те, что в папке «Крысы сканы»), а не синтетических:

- **Сквозная верификация**: подогнанный α/β на реальных данных y32 (гамма 32 Гр)
  попадает в диапазон [8, 12] Гр.
- **Кросс-валидация**: RMSE предсказания объёма при обучении на 2015–2020
  и тестировании на 2021–2024.
- **Кожные реакции**: соответствие шкале RTOG при известных дозах.
- **Смешанное поле N+P**: предсказание Zaider-Rossi совпадает с ТРО из экспериментов 2023–2024.

---

## Связанные файлы

- `survival/tests/test_biomodel_verification.py` — сам файл тестов
- `survival/tests/test_mixed_field_model.py` — тесты формулы Zaider-Rossi
- `survival/tests/test_let_parametrization.py` — тесты LET-подгонки
- `survival/tests/test_voxel_sf_calculator.py` — тесты вокселного SF
- `survival/tests/test_tumor_growth_predictor.py` — тесты геометрии Гомпертца
