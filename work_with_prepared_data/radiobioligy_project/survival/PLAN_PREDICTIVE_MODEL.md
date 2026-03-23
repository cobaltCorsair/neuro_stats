# План реализации предсказательной модели α/β

> **Дата**: 2026-03-19
> **Автор**: Кизилова Я.В.
> **Цель**: Замкнуть pipeline «CT/MRI → GEANT4 → доза per voxel → LQ-предсказание → динамика опухоли»
> **Формат**: Для Codex-агента — каждый шаг содержит точные пути, функции и критерии готовности

---

## Текущее состояние

### Что ЕСТЬ

| Компонент | Где лежит | Статус |
|-----------|-----------|--------|
| 7 LQ-моделей (classic, repair, linear, glq, lq_l, lq_repop, repair_repop) | `survival/fit_alpha_beta_using_processor.py` | ✅ Готово |
| Bootstrap CI для α, β, α/β | `survival/fit_alpha_beta_using_processor.py` | ✅ Готово |
| RBE через iso-effect (бинарный поиск) | `survival/radiobiology_analysis.py` | ✅ Готово |
| TCP = exp(−N₀·SF) | `survival/radiobiology_analysis.py` | ✅ Готово |
| Sensitivity analysis (OAT, interval, scenario) | `survival/radiobiology_analysis.py` | ✅ Готово |
| Gompertz + LQ kill + clearance | `survival/tumor_growth_predictor.py` | ✅ Готово |
| Геометрия эллипсоида (a-b-c → V) | `survival/tumor_growth_predictor.py` | ✅ Готово |
| GUI fitter + predictor | `survival/fit_alpha_beta_gui.py`, `tumor_growth_predictor_gui.py` | ✅ Готово |
| Model comparison (AIC, RMSE, R²) | `survival/fit_alpha_beta_using_processor.py` | ✅ Готово |
| Выбросы: Mahalanobis, KL, IQR, Grubbs, Isolation Forest | `stats_methods/support_stats_methods.py` | ✅ Готово |
| t-тест Welch (сравнение кривых роста) | `utils/visualizer.py:266` — `prepare_ttest_interpolated()` | ✅ Готово |
| Mann-Whitney (кривые роста + AUC + кожа) | `visualizer.py`, `support_stats_methods.py`, `skin_reactions_base_grapf.py`, `draw_base_graphs.py` — 9 реализаций | ✅ Готово |
| CV, AUC, попарное расхождение кривых | `stats_methods/support_stats_methods.py` | ✅ Готово |
| LONRAT: CT/MRI → NIfTI → landmarks → rigid/affine registration | `C:\dev\lonrat\` | ✅ 42% MVP |
| GEANT4 NPLibrary: вокс. фантом, phase space, dose scoring | `C:\dev\dissertation\vox_project\NPLibrary23\` | ✅ Компилируется |
| Protobuf-схемы: InputVoxelData, PhaseSpaceData, WiseVoxelData | `NPLibrary23/*.proto` | ✅ Готово |
| Воксельная модель крысы из CT | `dcm2materialPBfriendly.py` (внешний) | ✅ Работает |
| Фазовые пространства: нейтроны, протоны, C-12, γ | Бинарные `.pbphsp` файлы | ✅ Есть |

### Статус «Чего НЕТ» — аудит 23.03.2026

| Пробел (из плана) | Статус | Где реализовано |
|--------|----|------|
| Python-десериализация GEANT4 output (protobuf → numpy) | ✅ Готово | `dose_reader.py` (303 стр): `read_dose_map()`, `read_full_dose_map()`, `DoseMap`, `VoxelDose` |
| LET-зависимая параметризация α(LET), β(LET) | ✅ Готово | `let_parametrization.py` (144 стр): `LETDependentParams`, `fit_let_dependence()` |
| Модель смешанных полей (p+n, p+γ, n+γ) | ✅ Готово | `mixed_field_model.py` (137 стр): Zaider-Rossi + TDRA, `compute_mixed_field_sf()` |
| Voxel-level SF → объёмный SF (агрегация) | ✅ Готово | `voxel_sf_calculator.py` (347 стр): `VoxelSF`, `VolumetricSFResult`, D90/D50/V20, EUD |
| End-to-end pipeline GEANT4 → prediction | ✅ Готово | `pipeline_geant4_to_prediction.py` (663 стр): CLI + JSON/CSV export |
| Pipeline GUI | ✅ Готово | `geant4_pipeline_gui.py` (749 стр): PyQt6 GUI |
| Protobuf Python bindings | ✅ Готово | `proto/` (3 .proto + 3 _pb2.py + generate.bat) |
| Тесты pipeline | ✅ Готово | `tests/` (6 модулей, ~34 KB тестового кода) |
| BED/EQD2 экспорт в CSV | ✅ Готово | `fit_alpha_beta_using_processor.py`: CSV (4159+), pipeline CSV export |
| Lea-Catcheside G-фактор | ✅ Готово | `TumorExperiment.lea_catcheside_g_factor()` (605-609) |
| Bi-exponential repair | ✅ Готово | `repair_biexp` ModelKind, `quadratic_term_biexp()` (570-603) |
| **Shapiro-Wilk тест** | ❌ **Не реализован** | Нигде в проекте. Заявлен в аннотации диссертации |

### Что осталось доделать

| Задача | Критичность | Комментарий |
|--------|-------------|-------------|
| **Shapiro-Wilk** | 🟡 Средняя | Добавить `check_normality()` в `support_stats_methods.py`, подключить к auto-выбору теста в `visualizer.py` |
| **Интеграционный прогон на реальных данных** | 🔴 Высокая | Pipeline написан, но не прогнан на реальном GEANT4 output — нужна верификация на `.ivz` файле с дозой |
| **Валидация pipeline на LONRAT контурах** | ✅ Готово | `dose_reader.py` теперь принимает `ContourMeta.pb` и binary `3D Slicer NIfTI` mask, если mask уже приведена к GEANT4 grid |
| **Документация для диссертации** | 🟡 Средняя | Описание pipeline для текста диссертации (алгоритмы, формулы, блок-схемы) |

---

## ЭТАП 1. Protobuf Python bridge (БЛОКИРУЮЩИЙ)

### 1.1 Генерация Python-биндингов из .proto

**Файл**: `survival/proto/` (новая директория)

```
survival/
└── proto/
    ├── __init__.py
    ├── NPInputVoxelData_pb2.py      ← сгенерировать
    ├── NPPhaseSpaceData_pb2.py      ← сгенерировать
    ├── NPWiseVoxelData_pb2.py       ← сгенерировать
    └── generate.sh                  ← скрипт генерации
```

**Действия**:
1. Скопировать 3 `.proto` файла из `C:\dev\dissertation\vox_project\NPLibrary23\` в `survival/proto/`
2. Сгенерировать: `protoc --python_out=. NPWiseVoxelData.proto NPInputVoxelData.proto NPPhaseSpaceData.proto`
3. Написать `generate.sh` (или `.bat` для Windows), чтобы перегенерация была одной командой
4. Добавить `protobuf` в зависимости проекта (pip/poetry)

**Критерий готовности**: `from survival.proto.NPWiseVoxelData_pb2 import totDoseVoxelMap` работает без ошибок.

### 1.2 Десериализатор дозового распределения

**Файл**: `survival/dose_reader.py` (новый, ~150-200 строк)

```python
@dataclass(frozen=True)
class VoxelDose:
    """Одна ячейка дозовой карты."""
    voxel_id: int
    dose_gy: float           # из singleVoxelEntry.dose
    let_kev_um: float        # из singleVoxelEntry.letd
    dep_energy_mev: float    # из singleVoxelEntry.depEnergy
    n_events: int            # из singleVoxelEntry.nEvents
    rel_error: float         # sqrt(depEnergy2/nEvents - (depEnergy/nEvents)²) / (depEnergy/nEvents)
    scaled_dose: float       # из singleVoxelEntry.scaledDose
    eqd_gy: float            # из singleVoxelEntry.doseGyEQD
    mev2gy: float            # из singleVoxelEntry.mev2gy

@dataclass(frozen=True)
class DoseMap:
    """Полная 3D дозовая карта из GEANT4."""
    voxels: Dict[int, VoxelDose]
    grid_shape: Tuple[int, int, int]  # (nx, ny, nz) — из InputVoxelMap
    voxel_size_mm: Tuple[float, float, float]  # (dx, dy, dz)
    structure_ids: Dict[int, str]     # voxelStructureNames из ContourMeta

    def tumor_voxel_ids(self, structure_name: str = "tumor") -> List[int]:
        """ID вокселей, принадлежащих указанной структуре."""
        ...

    def dose_volume_histogram(self, voxel_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """DVH для указанного набора вокселей."""
        ...

    def mean_dose(self, voxel_ids: List[int]) -> float:
        """Средняя доза по набору вокселей."""
        ...

    def mean_let(self, voxel_ids: List[int]) -> float:
        """Средний LET по набору вокселей."""
        ...
```

**Функции**:

```python
def read_dose_map(
    dose_path: Path,           # .pb файл с totDoseVoxelMap или fullVoxelMap
    geometry_path: Path,       # .ivz файл с InputVoxelMap (размеры сетки)
    contour_path: Optional[Path] = None  # ContourMeta (если отдельным файлом)
) -> DoseMap:
    """Десериализация protobuf → DoseMap."""
    ...

def read_full_dose_map(path: Path, geometry_path: Path) -> Dict[str, DoseMap]:
    """Для fullVoxelMap: возвращает totDose, protonDose, midDose, mainDose, stuffDose."""
    ...
```

**Критерий готовности**:
- `read_dose_map("test.pb", "geometry.ivz")` возвращает `DoseMap` с правильными размерами сетки
- `dose_map.mean_dose(dose_map.tumor_voxel_ids())` возвращает число в Gy
- Unit-тест с синтетическим protobuf (10×10×10 сетка, известная доза)

---

## ЭТАП 2. LET-зависимая параметризация α(LET), β(LET)

### 2.1 Модель LET → α/β

**Файл**: `survival/let_parametrization.py` (новый, ~200 строк)

Это ключевой компонент для «разного качества излучения» из аннотации диссертации.

```python
@dataclass(frozen=True)
class LETDependentParams:
    """Параметры α(LET), β(LET) для одного типа ткани."""
    # Линейная модель: α(LET) = α₀ + λ·LET
    alpha_0: float       # α при LET→0 (фотонный предел)
    lambda_alpha: float  # наклон α(LET)

    # β обычно слабо зависит от LET, но можно:
    beta_0: float
    lambda_beta: float   # часто ≈ 0

    # Опционально: сатурация при высоких LET (модель Scholz/Elsässer)
    let_max: Optional[float] = None  # LET, при котором α выходит на плато

    def alpha(self, let_kev_um: float) -> float:
        """α(LET) с опциональной сатурацией."""
        a = self.alpha_0 + self.lambda_alpha * let_kev_um
        if self.let_max is not None:
            a_max = self.alpha_0 + self.lambda_alpha * self.let_max
            a = min(a, a_max)
        return a

    def beta(self, let_kev_um: float) -> float:
        return self.beta_0 + self.lambda_beta * let_kev_um

    def alpha_beta_ratio(self, let_kev_um: float) -> float:
        b = self.beta(let_kev_um)
        return self.alpha(let_kev_um) / b if b > 0 else float('inf')


def fit_let_dependence(
    fit_results: List[LQFitResult],
    mean_lets: Dict[str, float],  # family → средний LET из GEANT4
) -> LETDependentParams:
    """
    Фитирует α(LET), β(LET) по набору LQFitResult из разных family.

    Каждый family (y, p, p_peak, n, c12) имеет свой средний LET.
    Из fit_results берём α, β для каждого family.
    Линейная регрессия: α vs LET, β vs LET.
    """
    ...
```

### 2.2 Подключение к существующему fitter

**Файл**: `survival/fit_alpha_beta_using_processor.py` — модификация

Добавить в `Fitter`:
```python
def fit_let_dependence(
    self,
    family_results: Dict[str, LQFitResult],
    family_lets: Dict[str, float]
) -> LETDependentParams:
    """Вызывает let_parametrization.fit_let_dependence()."""
    ...
```

Добавить в CLI:
```
--let-fit              # Включить фитирование α(LET), β(LET) по всем family
--let-values y=3.5,p=12.0,n=45.0,c=180.0  # Средние LET по family (keV/μm)
```

**Критерий готовности**:
- Фитирование по ≥3 family даёт α₀, λ_α, β₀, λ_β
- R² LET-зависимости > 0.7 (на реальных данных гамма/протоны/нейтроны)
- График α vs LET с точками и линейным фитом экспортируется

---

## ЭТАП 3. Voxel-level предсказание SF

### 3.1 Per-voxel SF расчёт

**Файл**: `survival/voxel_sf_calculator.py` (новый, ~250 строк)

Мост между `dose_reader.DoseMap` и `tumor_growth_predictor.py`.

```python
@dataclass(frozen=True)
class VoxelSF:
    """SF для одного вокселя."""
    voxel_id: int
    dose_gy: float
    let_kev_um: float
    alpha: float          # α(LET) для этого вокселя
    beta: float           # β(LET) для этого вокселя
    sf: float             # exp(-(α·D + β·D²))
    bed: float            # D·(1 + D/(α/β))

@dataclass(frozen=True)
class VolumetricSFResult:
    """Агрегированный результат по объёму опухоли."""
    voxel_results: Tuple[VoxelSF, ...]

    # Агрегаты
    mean_sf: float                 # среднее SF по вокселям
    volume_weighted_sf: float      # SF, взвешенный по объёму
    mean_dose_gy: float
    mean_let_kev_um: float
    d90: float                     # доза, покрывающая 90% объёма
    d50: float
    v20: float                     # доля объёма с дозой > 20 Gy

    # Для передачи в tumor_growth_predictor
    effective_alpha: float         # объёмно-усреднённый α
    effective_beta: float          # объёмно-усреднённый β
    equivalent_uniform_dose: float # EUD


def compute_voxel_sf(
    dose_map: DoseMap,
    let_params: LETDependentParams,
    structure_name: str = "tumor",
    model_kind: str = "classic_lq",
    repair_half_time_hours: Optional[float] = None,
    n_fractions: int = 1,
    schedule_days: Optional[Tuple[float, ...]] = None,
) -> VolumetricSFResult:
    """
    Для каждого вокселя опухоли:
    1. Берём dose и LET из DoseMap
    2. Вычисляем α(LET), β(LET) из LETDependentParams
    3. Вычисляем SF по выбранной модели
    4. Агрегируем в VolumetricSFResult
    """
    ...
```

### 3.2 Подключение к tumor_growth_predictor

**Файл**: `survival/tumor_growth_predictor.py` — модификация

Добавить альтернативный вход в `simulate_growth()`:

```python
def simulate_growth(
    params: GrowthModelParameters,
    schedule: List[FractionEvent],
    observed: Optional[np.ndarray] = None,
    # --- НОВОЕ ---
    volumetric_sf: Optional[VolumetricSFResult] = None,
    # Если передан — использовать effective_alpha, effective_beta
    # вместо params.alpha, params.beta
) -> GrowthSimulationResult:
    ...
```

**Критерий готовности**:
- Синтетический тест: однородная доза 2 Gy по всем вокселям → voxel SF совпадает с аналитическим LQ
- Неоднородная доза → mean_sf ≠ SF(mean_dose) (проверка неравенства Дженсена)
- `effective_alpha`, `effective_beta` передаются в `simulate_growth()` → кривая генерируется

---

## ЭТАП 4. Модель смешанных полей

### 4.1 Комбинированное облучение

**Файл**: `survival/mixed_field_model.py` (новый, ~200 строк)

Для задачи 5 диссертации: сочетанное облучение (p+n, p+γ, n+γ и т.д.).

```python
@dataclass(frozen=True)
class FieldComponent:
    """Один компонент смешанного поля."""
    family: str              # "p", "n", "y", "c12"
    dose_fraction_gy: float  # доза от этого компонента
    mean_let_kev_um: float   # средний LET компонента
    weight: float            # весовой вклад (0..1, сумма = 1)

@dataclass(frozen=True)
class MixedFieldResult:
    """Результат расчёта SF в смешанном поле."""
    components: Tuple[FieldComponent, ...]
    total_dose_gy: float

    # Метод Zaider-Rossi: SF = exp(-Σᵢ (αᵢ·Dᵢ + βᵢ·Dᵢ²))
    sf_zaider_rossi: float

    # Метод TDRA (Theory of Dual Radiation Action):
    # SF = exp(-(ᾱ·D_tot + β̄·D_tot²)), где ᾱ = Σwᵢαᵢ, √β̄ = Σwᵢ√βᵢ
    sf_tdra: float

    effective_alpha: float   # для передачи в predictor
    effective_beta: float


def compute_mixed_field_sf(
    components: List[FieldComponent],
    let_params: LETDependentParams,
    method: str = "zaider_rossi",  # или "tdra"
) -> MixedFieldResult:
    """
    Расчёт SF для комбинированного облучения.

    Метод Zaider-Rossi (independent action):
      SF = Π SF_i = exp(-Σ(αᵢDᵢ + βᵢDᵢ²))
      где αᵢ = α(LETᵢ), βᵢ = β(LETᵢ)

    Метод TDRA (dose-averaged):
      ᾱ = Σ(Dᵢ/D_tot)·αᵢ
      √β̄ = Σ(Dᵢ/D_tot)·√βᵢ
      SF = exp(-(ᾱ·D + β̄·D²))
    """
    ...
```

### 4.2 Подключение к fullVoxelMap

Из `NPWiseVoxelData.proto` уже есть `fullVoxelMap` с полями `protonDose`, `midDose`, `mainDose`, `stuffDose`. Каждое поле — отдельный компонент поля.

В `dose_reader.py` → `read_full_dose_map()` уже спроектировано. Нужно:
1. Для каждого вокселя собрать `List[FieldComponent]` из нескольких dose maps
2. Вызвать `compute_mixed_field_sf()` для каждого вокселя
3. Агрегировать в `VolumetricSFResult`

**Критерий готовности**:
- Тест: чистое γ-поле → Zaider-Rossi и TDRA дают одинаковый SF
- Тест: p+n поле → SF_mixed < SF_p и SF_mixed < SF_n (синергия)
- Разница между Zaider-Rossi и TDRA < 15% на типичных схемах

---

## ЭТАП 5. Shapiro-Wilk + BED/EQD2

> **Примечание**: t-тест и Mann-Whitney **уже реализованы** в основном проекте
> (`visualizer.py:prepare_ttest_interpolated`, `visualizer.py:prepare_mann_whitney_test_interpolated`,
> `support_stats_methods.py:apply_mann_whitney_test`, `skin_reactions_base_grapf.py`,
> `draw_base_graphs.py` — всего 9 реализаций). Они работают в контексте сравнения
> кривых роста опухоли между группами крыс. Подключать их к survival/ **не нужно** —
> это разные задачи.

### 5.1 Критерий Шапиро-Уилка

**Шапиро-Уилка нет нигде в проекте.** Заявлен в аннотации диссертации.

Логичное место: `utils/visualizer.py` или `stats_methods/support_stats_methods.py` —
рядом с существующими t-тестом и Mann-Whitney, **не в survival/**.

**Файл**: `stats_methods/support_stats_methods.py` — добавить

```python
from scipy.stats import shapiro

def check_normality(data: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Критерий Шапиро-Уилка для проверки нормальности распределения.

    Используется перед выбором теста:
      - p > alpha → нормальность не отвергнута → t-тест
      - p ≤ alpha → нормальность отвергнута → Mann-Whitney

    Returns:
        {"statistic": W, "p_value": p, "is_normal": bool,
         "recommended_test": "t_test" | "mann_whitney",
         "n": len(data)}
    """
    if len(data) < 3:
        return {"statistic": None, "p_value": None, "is_normal": None,
                "recommended_test": "mann_whitney", "n": len(data)}
    w, p = shapiro(data)
    return {
        "statistic": w, "p_value": p,
        "is_normal": p > alpha,
        "recommended_test": "t_test" if p > alpha else "mann_whitney",
        "n": len(data)
    }
```

**Подключение**: вызывать из `visualizer.py:prepare_and_add_data_to_graph()` перед
выбором между `prepare_ttest_interpolated()` и `prepare_mann_whitney_test_interpolated()`.
Сейчас выбор теста зависит от параметра `stat_test` в GUI — добавить опцию `"auto"`,
которая сначала проверяет нормальность обеих групп через `check_normality()`.

**Критерий готовности**:
- `check_normality()` работает и возвращает правильную рекомендацию
- В GUI (`stat_test = "auto"`) автоматически выбирается t-тест или MW
- Результат проверки нормальности отображается на графике (текстовая аннотация)

### 5.2 BED/EQD2 экспорт

**Файл**: `survival/radiobiology_analysis.py` — добавить

```python
def export_bed_eqd2_table(
    fit_results: Dict[str, LQFitResult],
    dose_grid: np.ndarray = np.arange(0.5, 25.1, 0.5),
    fractions: List[int] = [1, 3, 5, 10, 20, 30],
    reference_ab: float = 2.0,  # α/β для EQD2
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Таблица BED и EQD2 для каждой комбинации (dose, n_fractions, family).

    BED = n·d·(1 + d/(α/β))
    EQD2 = BED / (1 + 2/(α/β_ref))
    """
    ...
```

**Критерий готовности**:
- BED/EQD2 таблица экспортируется в CSV для ≥3 family

---

## ЭТАП 6. Улучшение моделей репарации

### 6.1 Bi-exponential repair

**Файл**: `survival/fit_alpha_beta_using_processor.py` — новый `ModelKind`

```python
# Добавить в ModelKind:
"repair_biexp"  # Би-экспоненциальная репарация

# Квадратичный терм:
# Q = Σᵢ Σⱼ dᵢ·dⱼ · [f·exp(-λ_fast·|tᵢ-tⱼ|) + (1-f)·exp(-λ_slow·|tᵢ-tⱼ|)]
#
# Параметры: α, β, f (доля быстрого), t_half_fast (~15 мин), t_half_slow (~2-4 ч)
# Итого 5 параметров → нужно ≥6 экспериментов с разным timing
```

В `LQFitResult` добавить:
```python
repair_half_time_fast_hours: Optional[float] = None  # ~0.25 ч
repair_half_time_slow_hours: Optional[float] = None  # ~2-4 ч
repair_fast_fraction: Optional[float] = None          # f ∈ (0, 1)
```

### 6.2 Lea-Catcheside G-фактор

**Файл**: `survival/fit_alpha_beta_using_processor.py` — утилитная функция

```python
def lea_catcheside_g(
    dose_rate_gy_min: float,
    exposure_time_min: float,
    repair_rate: float  # ln(2) / t_half
) -> float:
    """
    G = 2·(μ·T − 1 + exp(−μ·T)) / (μ·T)²

    При G=1 (мгновенное облучение) → обычный LQ.
    При G→0 (очень длинное облучение) → только линейный терм.
    """
    mu_T = repair_rate * exposure_time_min
    if mu_T < 1e-6:
        return 1.0
    return 2.0 * (mu_T - 1.0 + math.exp(-mu_T)) / (mu_T ** 2)
```

В модели `repair_lq` заменить мгновенный β·d² на β·G·d² когда известен dose rate.

**Критерий готовности**:
- `repair_biexp` фитируется с synthetic data → параметры восстанавливаются
- G-фактор: при dose_rate → ∞, G → 1.0; при dose_rate → 0, G → 0.0
- Inventory heuristic: `repair_biexp` рекомендуется только если ≥3 разных inter-fraction interval

---

## ЭТАП 7. Интеграционный pipeline

### 7.1 End-to-end скрипт

**Файл**: `survival/pipeline_geant4_to_prediction.py` (новый, ~300 строк)

```python
def run_prediction_pipeline(
    # GEANT4 output
    dose_pb_path: Path,
    geometry_ivz_path: Path,

    # LQ параметры (откуда брать)
    fit_results_csv: Optional[Path] = None,   # из fitter
    let_params: Optional[LETDependentParams] = None,  # из LET-фита
    manual_alpha_beta: Optional[Tuple[float, float]] = None,

    # Tumor
    structure_name: str = "tumor",
    initial_volume_mm3: Optional[float] = None,

    # Growth model
    growth_rate: float = 0.05,
    carrying_capacity: float = 5000.0,
    clearance_rate: float = 0.1,

    # Treatment schedule (для multi-fraction)
    schedule_days: Optional[List[float]] = None,

    # Mixed field
    mixed_field: bool = False,

    # Output
    output_dir: Path = Path("./prediction_output"),
) -> dict:
    """
    Полный pipeline:
    1. Читает дозу из protobuf (dose_reader)
    2. Определяет α(LET), β(LET) для каждого вокселя (let_parametrization)
    3. Считает per-voxel SF (voxel_sf_calculator)
    4. Агрегирует в effective α, β
    5. Запускает tumor_growth_predictor
    6. Экспортирует: DVH, SF-map, кривую роста, BED/EQD2

    Returns: {
        "volumetric_sf": VolumetricSFResult,
        "growth_prediction": GrowthSimulationResult,
        "dvh": (doses, volumes),
        "output_files": List[Path]
    }
    """
    ...
```

### 7.2 Выходные файлы pipeline

```
prediction_output/
├── dvh_tumor.csv           # Dose-Volume Histogram
├── sf_per_voxel.csv        # voxel_id, dose, LET, alpha, beta, SF
├── aggregated_params.json  # effective_alpha, effective_beta, mean_dose, EUD
├── growth_curve.csv        # time, live_vol, dead_vol, total_vol
├── bed_eqd2_table.csv      # BED и EQD2 для разных схем
└── summary.json            # Всё в одном JSON для дальнейшей обработки
```

**Критерий готовности**:
- Весь pipeline проходит на тестовых данных (синтетический protobuf + реальные α/β из фиттера)
- Output файлы создаются, JSON парсится
- Кривая роста опухоли визуально адекватна

---

## ЭТАП 8. Тесты и валидация

### 8.1 Unit-тесты

**Файл**: `survival/tests/` (новая директория)

```
tests/
├── test_dose_reader.py             # Синтетический protobuf → DoseMap
├── test_let_parametrization.py     # Фит α(LET) на 3-4 точках
├── test_voxel_sf_calculator.py     # Однородная/неоднородная доза
├── test_mixed_field_model.py       # Zaider-Rossi vs TDRA
├── test_pipeline.py                # End-to-end с мок-данными
└── conftest.py                     # Фикстуры: sample DoseMap, LQFitResult
```

### 8.2 Валидация на реальных данных

1. Взять данные γ-облучения (есть в `survival/*.xlsx`)
2. Фитировать α, β → `LQFitResult`
3. Промоделировать то же облучение в GEANT4 (фотонный пучок → воксельная крыса)
4. Прогнать через pipeline → предсказанный SF
5. Сравнить предсказанный SF с экспериментальным → валидация pipeline

---

## Порядок реализации и зависимости

```
ЭТАП 1 (protobuf bridge)
    │
    ├──→ ЭТАП 2 (LET parametrization) ──→ ЭТАП 3 (voxel SF)
    │                                           │
    │                                           ├──→ ЭТАП 4 (mixed field)
    │                                           │
    │                                           └──→ ЭТАП 7 (pipeline)
    │                                                    │
    │                                                    └──→ ЭТАП 8 (тесты)
    │
    └──→ ЭТАП 5 (статметоды) — параллельно, не блокирует
    └──→ ЭТАП 6 (repair models) — параллельно, не блокирует
```

## Оценка трудозатрат

| Этап | Сложность | Оценка | Зависит от |
|------|-----------|--------|------------|
| 1. Protobuf bridge | Низкая | 1-2 дня | — |
| 2. LET параметризация | Средняя | 2-3 дня | Этап 1 |
| 3. Voxel SF | Средняя | 2-3 дня | Этап 1, 2 |
| 4. Mixed field | Средняя | 2-3 дня | Этап 3 |
| 5. Статметоды | Низкая | 1 день | — |
| 6. Repair models | Средняя | 2-3 дня | — |
| 7. Pipeline | Средняя | 2-3 дня | Этап 1-4 |
| 8. Тесты | Средняя | 2-3 дня | Этап 7 |
| **ИТОГО** | | **~15-20 дней** | |

---

## Примечания для Codex

- **Protobuf**: ориентация на `proto3`, пакет `NPLibrary`. Python-биндинги генерить через `grpcio-tools` или `protobuf` (pip).
- **Не трогать**: существующие 7 моделей в `fit_alpha_beta_using_processor.py` — они работают. Только добавлять.
- **Тесты**: каждый новый модуль — с `pytest`. Фикстуры для protobuf — синтетические (не нужен реальный GEANT4 run).
- **GUI**: пока не нужно. Pipeline работает из CLI и programmatic API. GUI-обёртка — позже.
- **Формат вывода**: CSV + JSON. Protobuf только на входе (из GEANT4).
