# `fit_alpha_beta_using_processor.py`

Автономный скрипт для подбора параметров α и β линейно-квадратичной модели (Linear-Quadratic model) на основе экспериментальных данных об объёмах опухолей, считанных функциями `process_tumor_data_excel` и `TumorDataProcessor` из модулей `data_processing`.

## Модель

```math
S(F) = \exp(-\alpha \cdot D - \beta \cdot D^2)
```

где:
- `S(F)` — surviving fraction (доля выживших клеток),
- `D = \sum_i d_i` — суммарная доза всех фракций,
- `D² = \sum_i d_i²` — сумма квадратов доз,
- `α` (Gy⁻¹) и `β` (Gy⁻²) — искомые параметры.

---

## Алгоритм

### 1. Сбор Excel-файлов
- Если задано `--files`, берёт только указанные файлы;
- Иначе — все `.xlsx` в текущей папке.

### 2. Для каждого файла

#### a) `process_tumor_data_excel(path)` возвращает:
- `experiment_params`: список строк с описанием опыта (первая строка Excel),
- `time_data`: метки времени (даты или дни),
- `rat_labels`: метки животных,
- `tumor_volumes`: матрица объёмов [n_animals × n_times].

#### b) `parse_fractions(experiment_params)` извлекает дозы:
- Все числа с суффиксом «Гр» или «Gy» → `[d1, d2, …]`.

#### c) `TumorDataProcessor` строит:
- `mean_rel = ⟨V(t)/V₀⟩` — средняя относительная величина,
- `mean_abs = ⟨V(t)⟩` — средний абсолютный объём (см³).

#### d) Вычисление surviving fraction `S(F)` по `--sf`:
- `relative`: `S(F) = min(mean_rel[1:])`
- `absolute`: `S(F) = min(mean_abs[1:]) / mean_abs[0]`
- `index:N`: `S(F) = mean_rel[N]`
- `absindex:N`: `S(F) = mean_abs[N] / mean_abs[0]`

#### e) Фильтрация:
- Пропускаем, если `S(F) ≥ --min-sf` (по умолчанию 1.0)

---

### 3. По всем отобранным экспериментам строим систему уравнений:

```math
-\ln S(F)_j = \alpha D_j + \beta D_j^2, \quad j = 1..N_{exp}
```

- Где `D_j` и `D_j²` считаются из списка фракций.

---

### 4. Решение

- Без фиксации `α`: NNLS (ограничения `α, β ≥ 0`)
- При `--alpha VALUE`: фиксируем `α = VALUE`, подбираем `β`:
  - `y' = y - α·D`
  - `β = max(0, (D² · y') / (D² · D²))`

---

### 5. Вывод

- Параметры `α`, `β`, `α/β`
- Отчёт по каждому эксперименту: список фракций, `D`, `D²`, `S(F)`

---

## Запуск

```bash
python fit_alpha_beta_using_processor.py [--files a.xlsx b.xlsx ...]
      [--sf relative|absolute|index:N|absindex:N]
      [--alpha VALUE]
      [--min-sf VALUE]
      [--verbose]
```

---
