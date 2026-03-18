# `fit_alpha_beta_using_processor.py`

Скрипт подбирает параметры `alpha` и `beta` линейно-квадратичной модели по in vivo данным роста опухолей из Excel-файлов.

Он рассчитан на серии, где:
- есть `control`-файлы для нормировки;
- дозовые фракции можно извлечь из параметров эксперимента;
- отклик оценивается через динамику объёма опухоли, а не через clonogenic assay.

## Что умеет

- читать `.xlsx` с опухолевыми объёмами;
- строить усреднённую control-кривую;
- нормировать экспериментальные кривые на контроль;
- считать `SF` несколькими способами;
- фитить `alpha` и `beta` по LQ-модели;
- анализировать семейства излучения `y`, `p`, `n`, `e` по отдельности;
- разделять `single` и `fractionated` режимы на train и holdout;
- делать bootstrap по животным и control-группам;
- агрегировать повторные режимы `(family, D, D2)` в один weighted regimen;
- сохранять batch-summary в CSV;
- отдавать структурированные результаты в GUI и тесты.

## Модель

Используется модель:

```text
SF = exp(-alpha * D - beta * D2)
```

где:
- `D = sum(d_i)` — суммарная доза;
- `D2 = sum(d_i^2)` — сумма квадратов фракций;
- `SF` — effective surviving fraction, полученный из нормированных кривых роста.

Если задан `--alpha`, `alpha` фиксируется, а фитится только `beta`.

## Как формируется `SF`

Поддерживаются две основные метрики:

- `absolute`
  - `SF = min(mean_norm[1:]) / mean_norm[0]`
- `absindex:N`
  - `SF = mean_norm[N] / mean_norm[0]`

Можно передавать несколько режимов сразу:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --sf absolute --sf absindex:1 --sf absindex:2
```

## Response modes

Теперь fitter поддерживает два режима отклика:

- `scalar`
  - старый режим;
  - для каждого режима облучения строится один `SF`, и fit идет по набору точек `SF(D)`.
- `curve`
  - новый режим;
  - вместо одного `SF` используется вся нормированная кривая ответа опухоли;
  - fitter дополнительно оценивает `curve_clearance_rate`, чтобы описать возврат от раннего ответа к более поздней динамике.

CLI:

```bash
--response-mode scalar
--response-mode curve
```

Замечание:
- при `curve` fitter использует только первый `--sf`, потому что остальные `SF`-метрики относятся к scalar-постановке.

## Model selection and comparison

Доступные семейства моделей:

- `--model-kind auto`
- `--model-kind classic_lq`
- `--model-kind repair_lq`
- `--model-kind glq`
- `--model-kind repair_repop`
- `--model-kind lq_l`
- `--model-kind lq_repop`
- `--model-kind linear`

Смысл:

- `classic_lq`
  - `SF = exp(-alpha * D - beta * sum(d_i^2))`
- `repair_lq`
  - тот же LQ, но с учетом `t=` интервалов и репарации через `--repair-half-time-hours`
- `glq`
  - generalized high-dose LQ variant with saturating quadratic term;
  - fitter оценивает `saturation_dose`
- `repair_repop`
  - repair-aware LQ plus delayed repopulation;
  - fitter использует `t=` интервалы, `lag_days` и `repopulation_rate`
- `lq_l`
  - LQ-L with transition dose;
  - до переходной дозы используется обычный квадратичный член, а выше включается линейный хвост
- `lq_repop`
  - классический LQ с задержкой и репопуляцией;
  - fitter оценивает `lag_days` и `repopulation_rate`
- `linear`
  - частный случай без квадратичного члена, то есть `beta = 0`

Для явного сравнения моделей:

```bash
--compare-models
```

В этом режиме fitter считает несколько кандидатов и ранжирует их по ошибке на train-наборе (`MAE`, `RMSE`, `mean_abs_log_error`, `AIC`).

Для расширенных моделей fitter дополнительно выводит:

- `saturation_dose` для `gLQ`
- `transition_dose` для `LQ-L`
- `lag_days` и `repopulation_rate` для `LQ + repopulation`
- `lag_days` и `repopulation_rate` для `repair + repopulation`

## Family

Скрипт пытается автоматически определить family по имени файла:

- `y` — гамма / фотонные серии в текущем naming convention;
- `p` — протоны;
- `n` — нейтроны;
- `e` — электроны.

Примеры:
- `19.03.2025_y_40.xlsx` -> `y`
- `22.10.2025_p40_in_peak.xlsx` -> `p`
- `02.02.2023_n_12.xlsx` -> `n`
- `15.01.2026_e_18.xlsx` -> `e`

Разные family не стоит смешивать в одном fit без отдельного радиобиологического обоснования.

## Train / Validation

Current family mapping used by the code:

- `y` -> gamma / photon series
- `p` -> proton series without an explicit beam-position marker
- `p_peak` -> proton series in peak (`in_peak`, `в_пике`)
- `p_through` -> proton series in shoot-through / pass-through (`прострел`)
- `n` -> neutron series
- `e` -> electron series
- `c` -> carbon-ion C-12 series (`c` or Cyrillic `с` in the file name)

Examples:
- `22.10.2025_p40_in_peak.xlsx` -> `p_peak`
- `08.10.2021_p_32_прострел.xlsx` -> `p_through`
- `05.12.2018_c_12.xlsx` -> `c`

Режимы можно делить на:

- `all`
- `single`
- `fractionated`

Типовой сценарий:
- обучить модель на `single`;
- проверить предсказание на `fractionated`.

Пример:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --fit-kind single ^
  --validate-kind fractionated ^
  --sf absolute
```

## Bootstrap

Включается через `--bootstrap N`.

Ресэмплируются:
- животные внутри experimental groups;
- животные внутри control groups.

На выходе:
- mean;
- std;
- median;
- `95%` bootstrap interval.

Пример:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --sf absolute ^
  --bootstrap 500 ^
  --bootstrap-seed 7
```

Если bootstrap часто уводит `beta` к нулю, интерпретация `alpha/beta` становится неустойчивой.

## Повторы режимов

Если в одной family есть несколько файлов с одинаковым `(D, D2)`, доступны два режима работы.

### Агрегация повторов

```bash
--aggregate-regimens
```

Повторы сворачиваются в один режим со:
- средним `SF`;
- числом повторов `repeats`;
- разбросом `sf_std`.

При fit используется variance-aware weighting:
- если для агрегированного режима есть `sf_std`, fitter использует стандартную ошибку среднего `sf_std / sqrt(repeats)`;
- если разброс неизвестен или равен нулю, fitter использует мягкий fallback, совместимый с прежней логикой weighting по `repeats`.

### Удаление дублей

```bash
--dedupe-regimens
```

Оставляет только один файл на `(family, D, D2)`.

## Batch-режим

Флаг:

```bash
--by-family
```

Скрипт проходит по каждому найденному family отдельно и строит summary по каждой паре `(sf_mode, family)`.

Пример:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --by-family ^
  --fit-kind single ^
  --validate-kind fractionated ^
  --sf absolute ^
  --aggregate-regimens ^
  --bootstrap 200
```

## Summary CSV

Для экспорта summary:

```bash
--summary-csv C:\dev\neuro_stats\family_summary.csv
```

CSV содержит:
- `sf_mode`
- `response_mode`
- `model_kind`
- `family`
- `status`
- `total_count`
- `single_count`
- `fractionated_count`
- `train_count`
- `validation_count`
- `alpha`
- `beta`
- `alpha_beta_ratio`
- `reason`

## Основные примеры CLI

Простой fit по `.xlsx` в текущей папке:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor
```

Fit только для одной family:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --sf absolute
```

Несколько `SF`-метрик за один запуск:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --sf absolute ^
  --sf absindex:1
```

Full-curve fit:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --response-mode curve ^
  --model-kind classic_lq
```

Сравнение моделей на одном train-наборе:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --family y ^
  --response-mode curve ^
  --repair-half-time-hours 1.0 ^
  --compare-models
```

Batch по всем family + CSV:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor ^
  --files *.xlsx ^
  --by-family ^
  --fit-kind single ^
  --validate-kind fractionated ^
  --sf absolute ^
  --aggregate-regimens ^
  --summary-csv C:\dev\neuro_stats\family_summary.csv
```

## GUI

Для fitter есть отдельный простой интерфейс на `PyQt6`.

Запуск:

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_gui
```

Что умеет GUI:
- перетаскивать `.xlsx` файлы мышью;
- добавлять отдельные файлы или целую папку;
- задавать явное сопоставление `experiment -> control`, если загружено несколько control-серий;
- настраивать `SF modes`, `family`, `fit-kind`, `validate-kind`, `bootstrap`;
- фиксировать `alpha` при необходимости;
- включать `aggregate-regimens` и `dedupe-regimens`;
- сохранять summary в CSV;
- показывать таблицу summary по всем run;
- показывать отдельные таблицы train, validation и bootstrap;
- выводить текстовую сводку по выбранному run.

Типовой сценарий:

1. Добавить `control` и экспериментальные `.xlsx`.
2. Оставить `SF modes = absolute`.
3. Если загружено несколько control-файлов, заполнить таблицу `Experiment to control mapping`.
4. Выбрать `family`, например `y`.
5. При необходимости включить `fit-kind = single` и `validate-kind = fractionated`.
6. Нажать `Run fit`.
7. Посмотреть summary-таблицу и детали выбранного run.

### Выбор control в GUI

Если в окне загружено несколько файлов с `control` в имени:
- GUI показывает отдельную таблицу `Experiment to control mapping`;
- для каждого экспериментального файла можно выбрать свой control-файл;
- есть `Default control` и кнопка `Apply to all experiments`, чтобы быстро заполнить таблицу;
- опция `Use all controls (average)` оставлена как явный осознанный режим, а не поведение по умолчанию.

Это сделано потому, что control-файлы из разных серий не должны автоматически смешиваться.

Важно: в CLI поведение по умолчанию осталось прежним. Если передать несколько control-файлов напрямую в `fit_alpha_beta_using_processor.py`, они будут усреднены в одну общую control-кривую.

## Что сильнее всего улучшает точность alpha/beta

GUI inventory mode:

- `Scan inventory` inspects the loaded folder before fitting
- the `Inventory` tab shows file-by-file `family`, `kind`, `fractions`, `schedule`, `control`, `fit ready`, and `notes`
- proton files are now separated into `p_peak` and `p_through`
- carbon-ion C-12 files are tracked as family `c`

На практике самый большой прирост точности дают не косметические изменения fit, а следующие шаги:

1. Учить модель на однократных дозах одного типа излучения и проверять на фракционированных режимах того же family.
2. Не смешивать разные family и разные control-серии в одном fit.
3. Добавлять больше режимов с одинаковой `D`, но разной `D2`, потому что именно они лучше всего идентифицируют `beta`.
4. Использовать bootstrap и смотреть не только точечные `alpha/beta`, но и интервалы неопределённости.
5. Сравнивать несколько определений `SF`, чтобы проверять устойчивость параметров к выбору endpoint.
6. По возможности держать отдельный holdout-набор, а не фитить все режимы сразу.

Если нужен следующий уровень улучшения модели, самые полезные направления такие:
- учёт внутри-режимной дисперсии в самом fit, а не только через веса повторов;
- вариант модели с временным фактором репарации, если интервалы между фракциями различаются;
- сравнение `LQ` с альтернативами для очень больших разовых доз.

## Ограничения

- интервалы между фракциями пока явно не входят в модель;
- внутри-режимная дисперсия пока не моделируется отдельно, кроме weighting по числу повторов;
- в GUI доступна привязка `experiment -> control`, но в CLI поведение по умолчанию всё ещё усредняет все переданные control-файлы, если не передавать явную карту соответствий через API;
- для очень больших разовых доз интерпретация LQ-параметров должна быть осторожной;
- `SF` здесь effective in vivo metric, а не классический clonogenic endpoint.

## Быстрая проверка опций CLI

```bash
python -m work_with_prepared_data.radiobioligy_project.survival.fit_alpha_beta_using_processor --help
```
