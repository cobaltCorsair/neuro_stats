# План реализации расширений предсказательной модели α/β

> **Проект**: `radiobioligy_project/survival/`
> **Дата**: 19.03.2026
> **Цель**: довести модуль α/β до полноты, заявленной в аннотации диссертации

---

## Архитектура проекта (справка)

```
survival/
├── fit_alpha_beta_using_processor.py   (3979 строк) — ядро: Fitter, модели, CLI
├── radiobiology_analysis.py            (577 строк)  — RBE, sensitivity, scenarios
├── tumor_growth_predictor.py           (512 строк)  — Gompertz, kill/clearance
├── fit_alpha_beta_gui.py               (1859 строк) — PyQt6 GUI фиттинга
├── tumor_growth_predictor_gui.py       (2111 строк) — PyQt6 GUI предсказания
└── gui_csv_export.py                   (40 строк)   — CSV-утилиты
```

### Ключевые типы (все в `fit_alpha_beta_using_processor.py`)

```python
ModelKind = Literal["classic_lq", "repair_lq", "glq", "linear", "lq_l", "lq_repop", "repair_repop"]

@dataclass(frozen=True)
class TumorExperiment:
    path: Path
    fractions: Tuple[float, ...]        # дозы каждой фракции
    sf: float                           # surviving fraction (endpoint)
    family: Optional[str]               # "y"|"p"|"p_peak"|"p_through"|"n"|"e"|"c"
    sf_time_day: float                  # день замера SF
    schedule_days: Tuple[float, ...]    # дни фракций (для repair)
    repeat_count: int                   # сколько повторов агрегировано
    sf_std: float                       # SD surviving fraction
    # ... curve_response, control_relative_curve, time_days для curve mode

@dataclass(frozen=True)
class LQFitResult:
    alpha: float
    beta: float
    model_kind: ModelKind
    family: Optional[str]
    sf_mode: str
    repair_half_time_hours: Optional[float]
    # ... transition_dose, saturation_dose, lag_days, repopulation_rate, curve_clearance_rate
    # Свойства: alpha_beta_ratio, repair_rate_per_day
    # Методы: predict_sf(experiment) -> float, predict_curve(experiment) -> ndarray

@dataclass(frozen=True)
class FitMetrics:
    point_count: int
    mae: float
    rmse: float
    mean_abs_log_error: float
    rss: float
    aic: Optional[float] = None
    # ❌ Нет: r_squared, adjusted_r_squared, bic
```

### Ключевые методы Fitter

```python
class Fitter:
    def fit(self, experiments, train_kind, family, sf_mode, response_mode, model_kind) -> LQFitResult
    def compare_models(self, experiments, response_mode, family, sf_mode) -> Tuple[ModelComparisonRow, ...]
    def bootstrap_fit(self, sf_mode, fit_kind, family, repeats, response_mode, model_kind, seed) -> BootstrapSummary
    def select_experiments(self, family, regimen_kind, experiments) -> List[TumorExperiment]
```

### radiobiology_analysis.py

```python
def compute_rbe(reference_result: LQFitResult, test_result: LQFitResult, test_dose: float) -> RBEPoint
def build_rbe_series(reference_result, test_result, doses) -> Tuple[RBEPoint, ...]
def analyze_parameter_sensitivity(...) -> ParameterSensitivityReport
def analyze_interval_sensitivity(...) -> IntervalSensitivityReport
```

---

## Задача 1: BED и EQD2

### Что
BED (Biologically Effective Dose) и EQD2 (Equivalent Dose in 2 Gy fractions) — стандартные клинические метрики, которые должен выдавать любой α/β фиттер.

### Формулы

```
BED = n · d · (1 + d / (α/β))           — для uniform fractionation
BED = D · (1 + D/(n·(α/β)))             — эквивалентная форма через D_total
BED_general = Σᵢ dᵢ · (1 + dᵢ/(α/β))   — для non-uniform fractions

EQD2 = BED / (1 + 2/(α/β))
```

Для repair-aware моделей BED с учётом incomplete repair:

```
BED_repair = D + (1/(α/β)) · Σᵢ Σⱼ dᵢ·dⱼ·exp(-λ·|tᵢ-tⱼ|)
           = D + quadratic_term / (α/β)
```

где `quadratic_term` — уже реализован в `TumorExperiment.quadratic_term(repair_rate_per_day)`.

### Где реализовать

**Файл**: `fit_alpha_beta_using_processor.py`

### Шаг 1.1 — Добавить методы в `LQFitResult` (после строки ~545)

```python
def compute_bed(self, experiment: TumorExperiment) -> float:
    """Biologically Effective Dose для данного эксперимента.

    Для repair-aware моделей использует quadratic_term с учётом
    экспоненциального затухания между фракциями.
    """
    ab = self.alpha_beta_ratio
    if ab is None or ab <= 0.0:
        return float("nan")

    if self.model_kind in ("repair_lq", "repair_repop"):
        qt = experiment.quadratic_term(self.repair_rate_per_day)
    else:
        qt = experiment.dose2_sum

    return experiment.dose_sum + qt / ab


def compute_eqd2(self, experiment: TumorExperiment) -> float:
    """Equivalent Dose in 2 Gy fractions."""
    ab = self.alpha_beta_ratio
    if ab is None or ab <= 0.0:
        return float("nan")
    bed = self.compute_bed(experiment)
    return bed / (1.0 + 2.0 / ab)
```

### Шаг 1.2 — Добавить BED/EQD2 в CSV-вывод

В методе `Fitter.write_analysis_summaries_csv()` (около строки 1194) добавить колонки `BED` и `EQD2` в summary CSV. Для каждого эксперимента вызвать `fit_result.compute_bed(exp)` и `fit_result.compute_eqd2(exp)`.

### Шаг 1.3 — Добавить BED/EQD2 в GUI

В `fit_alpha_beta_gui.py` — в таблицу результатов фита добавить столбцы BED и EQD2. Они вычисляются из уже имеющегося `LQFitResult` + набора экспериментов.

### Тесты

Проверить на простом случае: 5 фракций × 2 Gy, α/β = 10 → BED = 10·(1 + 2/10) = 12 Gy, EQD2 = 12/(1+2/10) = 10 Gy (identity). Для α/β = 3: BED = 10·(1 + 2/3) = 16.67, EQD2 = 16.67/(1+2/3) = 10 Gy.

---

## Задача 2: R² и adjusted R²

### Что
Коэффициент детерминации — стандартная метрика качества фита, которую ожидает любой рецензент.

### Формулы

```
SS_res = Σ (yᵢ - ŷᵢ)²        ← это уже rss в FitMetrics
SS_tot = Σ (yᵢ - ȳ)²
R² = 1 - SS_res / SS_tot
R²_adj = 1 - (1-R²)·(n-1)/(n-p-1)   где p = число параметров модели
```

### Где реализовать

**Файл**: `fit_alpha_beta_using_processor.py`

### Шаг 2.1 — Расширить `FitMetrics` (строка 646)

Добавить поля в dataclass `FitMetrics`:

```python
@dataclass(frozen=True)
class FitMetrics:
    point_count: int
    mae: float
    rmse: float
    mean_abs_log_error: float
    rss: float
    aic: Optional[float] = None
    r_squared: Optional[float] = None         # ← ДОБАВИТЬ
    adjusted_r_squared: Optional[float] = None # ← ДОБАВИТЬ
    bic: Optional[float] = None               # ← ДОБАВИТЬ (бонус)
```

### Шаг 2.2 — Вычислять R² при создании FitMetrics

Найти место, где создаётся `FitMetrics` (метод `Fitter._compute_metrics` или аналогичный). В этом месте уже есть `rss`. Нужно:

1. Вычислить `ss_tot = np.sum((observed - np.mean(observed))**2)`
2. `r_sq = 1.0 - rss / ss_tot if ss_tot > 0 else None`
3. Определить `p` — число свободных параметров модели:
   - `classic_lq`: p=2 (α, β)
   - `linear`: p=1 (α)
   - `glq`: p=3 (α, β, D_sat)
   - `lq_l`: p=3 (α, β, D_t)
   - `lq_repop`: p=4 (α, β, lag, repop_rate)
   - `repair_repop`: p=4 (α, β, lag, repop_rate) (repair_half_time фиксирован)
   - `repair_lq`: p=2 (α, β) (repair_half_time фиксирован)
   - Если `alpha_fixed is not None`: p -= 1
4. `r_sq_adj = 1.0 - (1.0 - r_sq) * (n - 1) / (n - p - 1) if n > p + 1 else None`
5. BIC (бонус): `bic = n * log(rss/n) + p * log(n)`

### Шаг 2.3 — Добавить в CSV и GUI

Аналогично задаче 1: добавить колонки R², R²_adj, BIC в summary CSV и в таблицу GUI.

---

## Задача 3: TCP (Tumor Control Probability)

### Что
Переход от SF к клинически значимому предсказанию: какова вероятность полного контроля опухоли.

### Формула

```
TCP = exp(-N₀ · SF)

N₀ = ρ · V₀    где ρ — плотность клеток (клеток/мм³), V₀ — начальный объём опухоли
```

Типичные значения ρ для солидных опухолей: 10⁶–10⁸ клеток/см³ = 10³–10⁵ клеток/мм³.

Для саркомы М-1 (крысы, данные проекта) можно использовать ρ ≈ 10⁷ клеток/см³ как default с возможностью настройки.

### Где реализовать

**Файл**: `radiobiology_analysis.py` — новая функция.

### Шаг 3.1 — Dataclass для TCP

```python
@dataclass(frozen=True)
class TCPResult:
    dose_total: float
    sf: float
    n_cells: float               # N₀ = ρ·V₀
    tcp: float                   # exp(-N₀·SF)
    cell_density: float          # ρ (клеток/см³)
    initial_volume_cm3: float    # V₀
    family: Optional[str]
    model_kind: str
```

### Шаг 3.2 — Функция compute_tcp

```python
def compute_tcp(
    fit_result: LQFitResult,
    experiment: TumorExperiment,
    initial_volume_cm3: float,
    cell_density: float = 1e7,  # клеток/см³, default для солидных опухолей
) -> TCPResult:
    """Tumor Control Probability из LQ-предсказания SF."""
    sf = fit_result.predict_sf(experiment)
    n_cells = cell_density * initial_volume_cm3
    tcp = math.exp(-n_cells * sf) if n_cells * sf < 700 else 0.0  # overflow protection
    return TCPResult(
        dose_total=experiment.dose_sum,
        sf=sf,
        n_cells=n_cells,
        tcp=tcp,
        cell_density=cell_density,
        initial_volume_cm3=initial_volume_cm3,
        family=experiment.family,
        model_kind=fit_result.model_kind,
    )
```

### Шаг 3.3 — TCP-серия по дозам

```python
def build_tcp_curve(
    fit_result: LQFitResult,
    dose_range: Sequence[float],
    n_fractions: int,
    initial_volume_cm3: float,
    cell_density: float = 1e7,
    schedule_interval_days: float = 1.0,
) -> Tuple[TCPResult, ...]:
    """TCP для диапазона суммарных доз с равномерным фракционированием."""
    results = []
    for d_total in dose_range:
        d_per_fraction = d_total / n_fractions
        fractions = tuple([d_per_fraction] * n_fractions)
        schedule = tuple([i * schedule_interval_days for i in range(n_fractions)])
        exp = TumorExperiment(
            path=Path("synthetic"),
            fractions=fractions,
            sf=0.0,  # placeholder, не используется при predict
            schedule_days=schedule,
            sf_time_day=schedule[-1] + 1.0,
            has_explicit_timing=True,
        )
        results.append(compute_tcp(fit_result, exp, initial_volume_cm3, cell_density))
    return tuple(results)
```

### Шаг 3.4 — Интеграция в GUI

В `fit_alpha_beta_gui.py`: добавить кнопку/вкладку «TCP curve». Параметры: V₀ (из geometry данных, уже есть), ρ (slider 10⁵–10⁸), диапазон доз. Отображение: график TCP vs Dose.

В `tumor_growth_predictor_gui.py`: в панели результатов показывать TCP рядом с объёмом опухоли.

### Шаг 3.5 — CSV экспорт

Добавить TCP-столбец в summary CSV в `Fitter.write_analysis_summaries_csv()`. Потребуется передать `initial_volume_cm3` — можно взять из первого замера объёма опухоли в эксперименте (уже есть в `TumorExperiment.curve_response[0]` для curve mode, или вычислить из geometry).

---

## Задача 4: Cross-validation (LOO)

### Что
Leave-One-Out cross-validation для оценки предсказательной способности модели. Для каждого эксперимента: фит на N-1, предсказание на оставшемся, сбор ошибок.

### Где реализовать

**Файл**: `fit_alpha_beta_using_processor.py` — новый метод в `Fitter`.

### Шаг 4.1 — Dataclass для результата

```python
@dataclass(frozen=True)
class CrossValidationResult:
    n_experiments: int
    n_successful: int
    cv_rmse: float                  # RMSE по LOO-предсказаниям
    cv_mae: float                   # MAE по LOO-предсказаниям
    cv_r_squared: float             # R² по LOO-предсказаниям
    residuals: Tuple[float, ...]    # (predicted - observed) для каждого эксперимента
    model_kind: ModelKind
    family: Optional[str]
    sf_mode: str
```

### Шаг 4.2 — Метод `Fitter.cross_validate_loo`

```python
def cross_validate_loo(
    self,
    experiments: Optional[Sequence[TumorExperiment]] = None,
    family: Optional[str] = None,
    sf_mode: Optional[str] = None,
    response_mode: ResponseMode = "scalar",
    model_kind: RequestedModelKind = "auto",
) -> CrossValidationResult:
    """Leave-One-Out cross-validation.

    Для каждого эксперимента i:
      1. Обучить модель на всех экспериментах кроме i
      2. Предсказать SF для i
      3. Записать residual = predicted_sf - observed_sf
    Вернуть агрегированные метрики.
    """
    selected = self.select_experiments(family=family, experiments=experiments)
    if len(selected) < 3:
        raise ValueError(f"LOO requires at least 3 experiments, got {len(selected)}")

    residuals = []
    observed = []
    predicted = []

    for i, held_out in enumerate(selected):
        train_set = [e for j, e in enumerate(selected) if j != i]
        try:
            # Создать временный Fitter с теми же параметрами
            # Вызвать fit на train_set
            result = self.fit(
                experiments=train_set,
                family=family,
                sf_mode=sf_mode or self.sf_modes[0],
                response_mode=response_mode,
                model_kind=model_kind,
            )
            pred_sf = result.predict_sf(held_out)
            residuals.append(pred_sf - held_out.sf)
            observed.append(held_out.sf)
            predicted.append(pred_sf)
        except Exception:
            continue  # skip failed folds

    obs = np.array(observed)
    pred = np.array(predicted)
    res = np.array(residuals)

    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((obs - np.mean(obs))**2))

    return CrossValidationResult(
        n_experiments=len(selected),
        n_successful=len(residuals),
        cv_rmse=float(np.sqrt(np.mean(res**2))),
        cv_mae=float(np.mean(np.abs(res))),
        cv_r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        residuals=tuple(residuals),
        model_kind=result.model_kind if residuals else "classic_lq",
        family=family,
        sf_mode=sf_mode or self.sf_modes[0],
    )
```

### Шаг 4.3 — CLI-аргумент

Добавить `--cross-validate` flag в argparse секцию CLI. Если указан — после основного фита запустить `cross_validate_loo()` и вывести результаты.

### Шаг 4.4 — GUI

В `fit_alpha_beta_gui.py`: кнопка «Cross-validate (LOO)» рядом с кнопкой Bootstrap. Показать результат в таблице и на графике (predicted vs observed scatter).

---

## Задача 5: LET-зависимость α(LET), β(LET)

### Что
Непрерывное отображение LET → (α, β) вместо независимых фитов по family. Это ключевая новизна для диссертации "излучения разного качества".

### Модель

Стандартная параметризация (Wilkens & Oelfke, 2004):

```
α(LET) = α₀ + λ_α · LET
β(LET) = β₀               (часто β от LET не зависит)
```

Или более гибкая (для тяжёлых ионов):

```
α(LET) = α₀ + λ_α · LET · exp(-LET / LET_sat)   # с насыщением при высоких LET
β(LET) = β₀ · exp(-μ · LET)                       # снижение β при высоких LET
```

### Предварительные условия

Для каждого family нужно задать характерный LET (keV/μm). Предлагаемые значения для проекта:

| Family     | Типичный LET (keV/μm) | Источник                     |
|------------|----------------------|------------------------------|
| y (γ)      | 0.2–0.5              | Co-60, Cs-137                |
| e          | 0.2–0.5              | Электроны ≈ фотоны          |
| p          | 1–5                  | Протоны (плато)              |
| p_peak     | 5–20                 | Протоны (пик Брэгга)        |
| p_through  | 1–3                  | Протоны (за пиком)           |
| n          | 10–30                | Нейтроны (recoil protons)    |
| c          | 50–150               | Углерод-12 (SOBP)            |

### Где реализовать

**Файл**: `fit_alpha_beta_using_processor.py` — новый model_kind и логика.

### Шаг 5.1 — Расширить типы

```python
# Добавить в начало файла:
FAMILY_LET_DEFAULTS: Dict[str, float] = {
    "y": 0.3,
    "e": 0.3,
    "p": 3.0,
    "p_peak": 12.0,
    "p_through": 2.0,
    "n": 20.0,
    "c": 100.0,
}
```

Добавить `"let_dependent"` в `ModelKind` и `RequestedModelKind`.

### Шаг 5.2 — Добавить LET в TumorExperiment

Добавить поле `let_kev_um: Optional[float] = None` в `TumorExperiment`. При загрузке — если LET не указан явно, подставлять из `FAMILY_LET_DEFAULTS[family]`.

### Шаг 5.3 — Новая модель фиттинга

```python
# В классе Fitter, рядом с glq_exponent, lql_exponent:

@staticmethod
def let_dependent_exponent(
    experiment: TumorExperiment,
    alpha_0: float,
    lambda_alpha: float,
    beta_0: float,
) -> float:
    """Exponent для LET-зависимой модели.

    α(LET) = α₀ + λ_α · LET
    β(LET) = β₀
    SF = exp(-α(LET)·D - β(LET)·Σdᵢ²)
    """
    let = experiment.let_kev_um or 0.3  # fallback to photon
    alpha_eff = alpha_0 + lambda_alpha * let
    return alpha_eff * experiment.dose_sum + beta_0 * experiment.dose2_sum
```

### Шаг 5.4 — Фит LET-зависимой модели

В методе `Fitter.fit()`:

- Если `model_kind == "let_dependent"`:
  - Собрать эксперименты **ВСЕХ family** (не фильтровать по family)
  - Параметры фита: `alpha_0`, `lambda_alpha`, `beta_0` (3 параметра)
  - Для каждого эксперимента `let_kev_um` берётся из поля
  - curve_fit: `SF_predicted = exp(-let_dependent_exponent(exp, α₀, λ_α, β₀))`

### Шаг 5.5 — Расширить LQFitResult

Добавить поля:

```python
alpha_0: Optional[float] = None       # ← базовый α при LET=0
lambda_alpha: Optional[float] = None   # ← наклон α(LET)
```

Переопределить `predict_sf` для `model_kind == "let_dependent"`.

### Шаг 5.6 — Визуализация

В GUI: график α(LET) с фитированной прямой и точками per-family. Каждая точка — (LET_family, α_family) из independent per-family fits. Линия — α₀ + λ_α·LET. Это будет ключевой рисунок диссертации.

### Шаг 5.7 — RBE из LET-модели

В `radiobiology_analysis.py`: новая функция `compute_rbe_let(fit_result, test_let, reference_let, dose)` — RBE напрямую из LET, без отдельных per-family фитов.

---

## Задача 6: Би-экспоненциальная репарация

### Что
Сейчас repair_lq использует одну экспоненту `exp(-λ·Δt)`. В реальности ДНК-репарация имеет два компонента: быстрый (NHEJ, T₁/₂ ≈ 15–30 мин) и медленный (HR, T₁/₂ ≈ 2–4 ч).

### Формула

```
G_biexp(Δt) = a · exp(-λ_fast · Δt) + (1-a) · exp(-λ_slow · Δt)

quadratic_term_biexp = Σᵢ Σⱼ dᵢ · dⱼ · G_biexp(|tᵢ - tⱼ|)
```

Где `a` — доля быстрой компоненты (обычно 0.5–0.8).

### Где реализовать

**Файл**: `fit_alpha_beta_using_processor.py`

### Шаг 6.1 — Добавить `"repair_biexp"` в ModelKind

```python
ModelKind = Literal[..., "repair_biexp"]
```

### Шаг 6.2 — Расширить TumorExperiment.quadratic_term

Добавить перегрузку или новый метод:

```python
def quadratic_term_biexp(
    self,
    fast_rate_per_day: float,
    slow_rate_per_day: float,
    fast_fraction: float = 0.6,
) -> float:
    """Квадратичный член с би-экспоненциальной репарацией."""
    if self.fraction_count <= 1:
        return self.dose2_sum
    term = 0.0
    for i, di in enumerate(self.fractions):
        for j, dj in enumerate(self.fractions):
            if i == j:
                term += di * dj  # мгновенный самовклад
            else:
                delta = abs(self.schedule_days[i] - self.schedule_days[j])
                g = fast_fraction * math.exp(-fast_rate_per_day * delta) + \
                    (1.0 - fast_fraction) * math.exp(-slow_rate_per_day * delta)
                term += di * dj * g
    return term
```

### Шаг 6.3 — Параметры фита

Для `repair_biexp`:
- Входные (фиксируемые пользователем): `repair_half_time_fast_hours`, `repair_half_time_slow_hours`, `fast_fraction`
- Или: фитировать `fast_fraction` как свободный параметр (3-параметрическая модель: α, β, a)
- λ_fast и λ_slow конвертируются аналогично текущему: `log(2) * 24 / T_half_hours`

### Шаг 6.4 — LQFitResult

Добавить поля:

```python
repair_half_time_fast_hours: Optional[float] = None
repair_half_time_slow_hours: Optional[float] = None
repair_fast_fraction: Optional[float] = None
```

### Шаг 6.5 — CLI

Добавить аргументы:

```
--repair-half-time-fast HOURS
--repair-half-time-slow HOURS
--repair-fast-fraction FLOAT   (default: 0.6)
```

---

## Задача 7: Lea-Catcheside G-фактор

### Что
Формальное обобщение для протяжённого облучения. Текущий `quadratic_term` — это дискретный аналог G-фактора. Явный G-фактор нужен для теоретической полноты и для случаев непрерывного облучения.

### Формула

Для фракционированного облучения с конечной длительностью фракции t_irr:

```
G = (2/D²) · Σᵢ Σⱼ dᵢ · dⱼ · Gᵢⱼ

Gᵢⱼ = exp(-μ·|Tᵢ - Tⱼ|)                           если i ≠ j  (уже есть)
Gᵢᵢ = (2/μ·tᵢ)·[μ·tᵢ + exp(-μ·tᵢ) - 1] / (μ·tᵢ)  для конечной длительности фракции
```

При мгновенной фракции (tᵢ → 0): Gᵢᵢ = 1, и формула сводится к текущему `quadratic_term / D²`.

### Где реализовать

**Файл**: `fit_alpha_beta_using_processor.py`

### Шаг 7.1 — Добавить поле irradiation_time в TumorExperiment

```python
irradiation_duration_hours: Tuple[float, ...] = ()  # длительность каждой фракции
```

Если не указано — считать мгновенным (current behavior).

### Шаг 7.2 — Метод g_factor

```python
def lea_catcheside_g_factor(
    self,
    repair_rate_per_day: Optional[float] = None,
) -> float:
    """Lea-Catcheside dose-protraction factor G.

    G = 1 для мгновенного однократного облучения.
    G < 1 при конечной длительности фракции или repair между фракциями.
    """
    if repair_rate_per_day is None or repair_rate_per_day <= 0.0:
        return 1.0
    qt = self.quadratic_term(repair_rate_per_day)
    d_total_sq = self.dose_sum ** 2
    if d_total_sq <= 0:
        return 1.0
    return qt / d_total_sq
```

Это вспомогательный метод для отображения и для формулировки в тексте диссертации. Функционально ничего нового к модели не добавляет (quadratic_term уже всё делает), но показывает рецензенту владение формализмом.

### Шаг 7.3 — В CSV/GUI

Добавить колонку G-factor в summary. Значение G ∈ (0, 1] — чем ближе к 1, тем меньше repair между фракциями.

---

## Задача 8: NTCP (бонус, низкий приоритет)

### Что
Normal Tissue Complication Probability — вероятность осложнений нормальных тканей. Для данных проекта — связь с кожными реакциями (шкала RTOG уже реализована).

### Модель Lyman-Kutcher-Burman

```
NTCP = Φ(t)   где Φ — стандартное нормальное CDF

t = (D - TD₅₀) / (m · TD₅₀)

TD₅₀ — доза, дающая 50% осложнений
m — наклон (steepness)
```

### Где реализовать

**Файл**: `radiobiology_analysis.py`

### Шаг 8.1 — Функция

```python
def compute_ntcp_lkb(
    dose_total: float,
    td50: float,
    m: float,
) -> float:
    """Lyman-Kutcher-Burman NTCP.

    Параметры td50 и m фитируются из данных кожных реакций.
    """
    from scipy.stats import norm
    t = (dose_total - td50) / (m * td50)
    return float(norm.cdf(t))
```

### Шаг 8.2 — Фит TD₅₀ и m из данных кожных реакций

Использовать данные RTOG из `from_our_scale_to_rtog.py`. Для каждого эксперимента есть доза и пиковый балл RTOG. Бинаризовать: RTOG ≥ 3 → «осложнение», < 3 → «нет». Фитировать TD₅₀ и m через MLE (scipy.optimize.minimize, negative log-likelihood).

---

## Статус реализации (аудит 19.03.2026)

> Все 8 задач из плана **полностью реализованы**.

| # | Задача | Статус | Где реализовано (строки) |
|---|--------|--------|--------------------------|
| 1 | **R² + R²_adj + BIC** | ✅ Готово | `FitMetrics` (886-897): поля `r_squared`, `adjusted_r_squared`, `bic`. Вычисление в `_compute_metrics()` (3563-3585): `ss_tot`, R², adj.R² с df-поправкой, BIC |
| 2 | **BED + EQD2** | ✅ Готово | `LQFitResult.compute_bed()` (747-752), `compute_eqd2()` (754-759). Свойства `mean_train_bed/eqd2` (1082-1087). CSV-экспорт (4159-4160, 4197-4224). GUI: 3 таблицы (summary/train/val) |
| 3 | **G-фактор (Lea-Catcheside)** | ✅ Готово | `TumorExperiment.lea_catcheside_g_factor()` (605-609). `LQFitResult.compute_g_factor()` (761-776) — одно- и би-экспоненциальная репарация |
| 4 | **TCP** | ✅ Готово | `radiobiology_analysis.py`: `TCPResult` (137-148), `compute_tcp()` (578-610), `build_tcp_curve()` (613-627). TCP = exp(−N₀·SF) с overflow protection |
| 5 | **Cross-validation LOO** | ✅ Готово | `CrossValidationResult` (982-993), `PredictionRow` (920-943), `cross_validate_loo()` (3860-3918). CLI: `--cross-validate-loo` (4727-4732). Интеграция в `analyze_fitter()` (4428-4439) |
| 6 | **LET → α(LET), β(LET)** | ✅ Готово | `FAMILY_LET_DEFAULTS` (101-109), `TumorExperiment.let_kev_um` (479-499), `LQFitResult.alpha_0/lambda_alpha` (680-681), `effective_alpha()` (714-727), `let_dependent_exponent()` (2240-2252), `_fit_curve_let_parameters()` (2399-2430). RBE из LET: `radiobiology_analysis.py` (284-285, 367-370) |
| 7 | **Bi-exp repair** | ✅ Готово | `ModelKind += "repair_biexp"` (47, 58). `quadratic_term_biexp()` (570-603). `LQFitResult`: `repair_half_time_fast/slow_hours`, `repair_fast_fraction` (677-679). Rate properties (701-710) |
| 8 | **NTCP (Lyman-Kutcher-Burman)** | ✅ Готово | `radiobiology_analysis.py`: `NTCPPoint` (150-157), `NTCPFitGroup` (160-171), `NTCPFitResult` (174-183), `compute_ntcp_lkb()` (392-406), `build_ntcp_curve()` (409-429), `fit_ntcp_lkb_from_groups()` (518-575) — бинарный MLE через L-BFGS-B |

### Итоговая архитектура survival/ (после всех реализаций)

```
survival/
├── fit_alpha_beta_using_processor.py   (~4700+ строк)
│   ├── 8 ModelKind: classic_lq, repair_lq, glq, linear, lq_l, lq_repop, repair_repop, repair_biexp, let_dependent
│   ├── TumorExperiment: quadratic_term(), quadratic_term_biexp(), lea_catcheside_g_factor()
│   ├── LQFitResult: compute_bed(), compute_eqd2(), compute_g_factor(), effective_alpha()
│   ├── FitMetrics: R², adj.R², BIC, AIC
│   ├── Fitter: fit(), compare_models(), bootstrap_fit(), cross_validate_loo()
│   └── CLI с --cross-validate-loo, --let-fit, --repair-half-time-fast/slow
│
├── radiobiology_analysis.py            (~630+ строк)
│   ├── RBE: compute_rbe(), build_rbe_series() + LET-dependent RBE
│   ├── TCP: compute_tcp(), build_tcp_curve()
│   ├── NTCP: compute_ntcp_lkb(), build_ntcp_curve(), fit_ntcp_lkb_from_groups()
│   └── Sensitivity: analyze_parameter_sensitivity(), analyze_interval_sensitivity()
│
├── tumor_growth_predictor.py           (512 строк)
│   └── Gompertz + LQ kill + clearance + geometry
│
├── fit_alpha_beta_gui.py               (~2900+ строк) — PyQt6 GUI
├── tumor_growth_predictor_gui.py       (~2100+ строк) — PyQt6 GUI
└── gui_csv_export.py                   (40 строк)
```

---

## Важные ограничения для Codex

1. **Не трогать существующие модели** — все 9 model_kind (включая `repair_biexp` и `let_dependent`) должны продолжать работать без изменений.
2. **Frozen dataclasses** — `LQFitResult`, `FitMetrics`, `TumorExperiment` помечены `frozen=True`. Добавлять новые поля **только** с `default` значениями, чтобы не сломать существующие вызовы.
3. **Обратная совместимость CLI** — новые аргументы должны быть optional с разумными defaults.
4. **Обратная совместимость CSV** — новые колонки добавляются в конец.
5. **Imports** — проект использует `scipy.optimize.curve_fit`, `numpy`, `math`. Новые зависимости (если нужны) — только из `scipy.stats` (уже есть в requirements).
6. **Type hints** — весь проект строго типизирован. Поддерживать тот же стиль.
7. **GUI** — PyQt6 + matplotlib. Не менять layout существующих вкладок, добавлять новые.
