### Радиобиологический модуль

Этот раздел проекта предназначен для обработки экспериментальных данных по опухолевому росту и кожным реакциям у животных, визуализации, статистического анализа и оценки радиочувствительности (модель LQ: подгонка α, β).

Основные возможности:
- Визуализация объёмов опухолей (индивидуальные кривые, средние, относительные, сравнение AUC) — `draw_base_graphs.py`, `draw_base_graphs_compare.py`.
- Сравнение нескольких экспериментов и продвинутые сравнения абсолютных/относительных кривых — `draw_abs_rel_graph_compare.py`.
- Анализ кожных реакций, в т.ч. приведение к шкале RTOG и комбинированная визуализация — `stats_methods/from_our_scale_to_rtog.py`, `skin_reactions_base_grapf.py`.
- GUI для интерактивной работы с графиками, выбором файлов и управлением обработкой — `gui/main_window.py`.
- Оценка параметров α и β по данным объёмов опухоли — `survival/fit_alpha_beta_using_processor.py` (подробности в `survival/README.md`).

---

### Структура каталогов

- `data_processing/` — загрузка и предобработка Excel: `excel_data_processor.py`, менеджмент меток крыс `rat_manager.py`.
- `gui/` — PyQt6 GUI: основное окно `main_window.py`, утилиты отрисовки `graph_manager.py`, `checkable_combobox.py`.
- `utils/` — вспомогательные функции визуализации и сохранения графиков.
- `stats_methods/` — поддерживающие статистические методы (выбросы, тесты Манна–Уитни и др.).
- `survival/` — инструменты подгонки LQ-модели (α, β), примеры Excel и отдельный `README.md`.
- Корневые скрипты визуализации: `draw_base_graphs.py`, `draw_base_graphs_compare.py`, `draw_abs_rel_graph_compare.py`, `skin_reactions_base_grapf.py`, а также `controls.py` для control-групп.

---

### Установка

Требуется Python 3.10+.

Вариант A (Poetry, рекомендуемый):
```bash
poetry install
poetry run python -V
```

Вариант B (pip):
```bash
pip install -U pip
pip install -e .
```

Примечания по зависимостям:
- Используются: PyQt6, matplotlib, numpy, pandas, scipy, scikit-learn, fastdtw и др. Установка через Poetry подтянет версии из `pyproject.toml` / `poetry.lock` на уровне репозитория.

---

### Формат входных данных (Excel)

Стандартизированный формат таблиц (пример для опухолей/кожных реакций):
- Первая строка — параметры эксперимента (включая дозовые фракции; парсятся из текста вида «10 Гр + 5 Гр»).
- Далее — строки по животным; один из столбцов — индекс/метка животного (часто колонка с именем `Метка`).
- Столбцы с временными точками содержат значения (объёмы или баллы реакции).

Минимальные ожидания по парсингу:
- `excel_data_processor.py` извлекает `experiment_params`, `time_data`, `rat_labels`, матрицу значений.
- Для некоторых анализов первый столбец индексируется по `Метка`.

Расположение примеров данных см. в `work_with_prepared_data/datas/` и подкаталогах.

---

### Запуск GUI

Windows PowerShell (из корня репозитория):
```powershell
python -m work_with_prepared_data.radiobioligy_project.gui.main_window
```

Альтернативно, можно запустить напрямую (если возникают проблемы с импортами при запуске модулем):
```powershell
python .\work_with_prepared_data\radiobioligy_project\gui\main_window.py
```

GUI позволяет:
- Открывать Excel-файлы с данными по опухолям/реакциям кожи;
- Переключать режимы графиков (индивидуальные, средние, относительные, сравнения);
- Исключать выбросы с помощью встроенных методов (`stats_methods/support_stats_methods.py`).

Если наблюдаются ошибки бэкенда Matplotlib, убедитесь, что установлен Qt и PyQt6, и у бэкенда `Qt5Agg/QtAgg` есть корректная связка (в проекте явно устанавливается `QT5Agg`).

---

### Скриншот GUI

Ниже пример основного окна приложения с загруженными экспериментами и сравнением кривых.

![Главное окно GUI](gui/screenshot_main_window.png)

> Если изображение не отображается, убедитесь, что файл сохранён по пути `work_with_prepared_data/radiobioligy_project/gui/screenshot_main_window.png`.

---

### Скрипты визуализации (CLI)

1) Базовые графики по одному эксперименту — средние/индивидуальные/относительные, AUC:
```powershell
python .\work_with_prepared_data\radiobioligy_project\draw_base_graphs.py
```
Отредактируйте внутри файла переменную `file_path` (внизу), чтобы указать ваш Excel.

2) Сравнение нескольких экспериментов:
```powershell
python .\work_with_prepared_data\radiobioligy_project\draw_base_graphs_compare.py
```
Укажите `file_path1`, `file_path2`, … в блоке `if __name__ == "__main__":`.

3) Продвинутые сравнения абсолютных/относительных кривых:
```powershell
python .\work_with_prepared_data\radiobioligy_project\draw_abs_rel_graph_compare.py
```

4) Кожные реакции и объединённые шкалы (в т.ч. RTOG):
```powershell
python .\work_with_prepared_data\radiobioligy_project\skin_reactions_base_grapf.py
python .\work_with_prepared_data\radiobioligy_project\stats_methods\from_our_scale_to_rtog.py
```

---

### Оценка α и β (LQ-модель)

Используйте скрипт в `survival/` (подробности и примеры — в отдельном README):
```powershell
cd .\work_with_prepared_data\radiobioligy_project\survival
python .\fit_alpha_beta_using_processor.py
python .\fit_alpha_beta_using_processor.py --alpha 0.3
python .\fit_alpha_beta_using_processor.py --sf absindex:2 --verbose
python .\fit_alpha_beta_using_processor.py --files a.xlsx b.xlsx
```

Коротко об алгоритме:
- Собирает контрольные кривые, строит среднюю контрольную;
- Нормирует экспериментальные кривые на контроль;
- Извлекает фракции доз из строки параметров;
- Считает surviving fraction (SF), фильтрует по порогу;
- Подгоняет α и β по кривой `SF = exp(-α·D - β·D²)`.

См. `survival/README.md` для детальной инструкции и формул.

---

### Советы по данным и качеству

- Следите за единицами измерения доз (Гр) и корректностью записи фракций.
- Для сравнений групп используйте согласованные временные точки.
- Методы удаления выбросов доступны в `stats_methods/support_stats_methods.py` (EllipticEnvelope, IsolationForest, IQR, Граббс, Махаланобис и др.).

---

### Частые проблемы

- Импорты в GUI: при запуске модулем используйте корень репозитория в `PYTHONPATH` или команду `python -m ...` как указано выше. Если возникают ошибки импорта `gui`, запустите файл напрямую.
- Отрисовка Matplotlib/Qt: убедитесь, что установлен PyQt6 и подходящий бэкенд Matplotlib. На Windows запуск из PowerShell обычно работает без дополнительных настроек.

---

### Лицензия и назначение

Код предназначен для исследовательских целей в области радиобиологии. Убедитесь, что публикации и отчёты содержат корректные ссылки на методики и источники данных.


