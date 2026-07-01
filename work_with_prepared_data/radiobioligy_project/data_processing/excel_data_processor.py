# файл excel_data_processor.py

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime, date, timedelta
from work_with_prepared_data.radiobioligy_project.data_processing.rat_manager import register_rat_labels


def _extract_t_values_hours(t_str: str) -> List[float]:
    """
    Парсит строку t-ячейки заголовка Excel в список интервалов (в часах).

    Примеры входа:
        't = 2.5 ч / 1 ч'       → [2.5, 1.0]
        't1 = 2ч/24ч/1ч 45мин'  → [2.0, 24.0, 1.75]
        't = 2 ч / 2 ч. / 48 ч. / 2 ч / 2 ч. /'  → [2.0, 2.0, 48.0, 2.0, 2.0]
        't = 30 мин.'            → [0.5]
        't = 2,5 ч.'             → [2.5]
        't = 5 сут'              → [120.0]
    """
    s = re.sub(r'^t\d*\s*=\s*', '', str(t_str).strip(), flags=re.IGNORECASE)
    tokens = [tok.strip().rstrip('.') for tok in s.split('/') if tok.strip().rstrip('.')]
    values = []
    for tok in tokens:
        h_match = re.search(r'(\d+[.,]\d*|\d+)\s*ч', tok)
        m_match = re.search(r'(\d+[.,]\d*|\d+)\s*мин', tok)
        d_match = re.search(r'(\d+[.,]\d*|\d+)\s*сут', tok)
        hours = 0.0
        if h_match:
            hours += float(h_match.group(1).replace(',', '.'))
        if m_match:
            hours += float(m_match.group(1).replace(',', '.')) / 60.0
        if d_match:
            hours += float(d_match.group(1).replace(',', '.')) * 24.0
        if h_match or m_match or d_match:
            values.append(round(hours, 6))
    return values


def parse_irradiation_schedule(t_str: str, dose_keys: List[str]) -> List[float]:
    """
    Возвращает список N-1 интервалов между фракциями (в часах).

    Правила:
      1 значение в t          → все интервалы равны этому значению
      N-1 значений в t        → позиционно: intervals[i] = values[i]
      2 значения, N-1 > 2     → блочный разбор:
          k = длина первого блока одного типа слева;
          threshold = max(k-1, 1);
          intervals[i] = A  при i < threshold, иначе B

    Args:
        t_str:     сырая строка t-ячейки (например 't = 2.5 ч / 1 ч')
        dose_keys: список типов излучения по порядку (например ['p','n','n','n','n'])

    Returns:
        Список float (часы), длина = len(dose_keys) - 1. Пустой список, если не удалось.
    """
    N = len(dose_keys)
    if N <= 1:
        return []
    values = _extract_t_values_hours(t_str)
    n_gaps = N - 1
    if not values:
        return []
    if len(values) == 1:
        return [values[0]] * n_gaps
    if len(values) == n_gaps:
        return list(values)
    if len(values) == 2 and n_gaps > 2:
        A, B = values
        k = 1
        while k < N and dose_keys[k] == dose_keys[0]:
            k += 1
        threshold = max(k - 1, 1)
        return [A if i < threshold else B for i in range(n_gaps)]
    # fallback: повторить первое значение
    return [values[0]] * n_gaps


def get_schedule_from_params(experiment_params: List[str]) -> List[float]:
    """
    Извлекает распарсенные интервалы из experiment_params (в часах).
    Возвращает пустой список, если Schedule= не найден.
    """
    for p in experiment_params:
        if p.startswith("Schedule="):
            raw = p.split("=", 1)[1]
            return [float(v) for v in raw.split(",") if v.strip()]
    return []


def _process_header_row(raw_params: List[str]) -> Tuple[List[str], List[float]]:
    """
    Обрабатывает сырой список ячеек первой строки Excel:
    - Нормализует t/t1/t2-ячейки → 'Irradiation Time=...' (независимо от позиции)
    - Вычисляет расписание фракций и добавляет 'Schedule=...' (часы через запятую)

    Returns:
        (обновлённый список params, список интервалов в часах)
    """
    params = list(raw_params)
    t_str = None
    for i, p in enumerate(params):
        if re.match(r'^t\d*\s*=', p.strip(), re.IGNORECASE):
            t_str = p.strip()
            params[i] = f"Irradiation Time={t_str.split('=', 1)[1].strip()}"
    dose_keys = []
    for p in params:
        m = re.match(r'^([a-zA-Z])\s*=\s*[\d.,]', p.strip())
        if m:
            dose_keys.append(m.group(1).lower())
    schedule: List[float] = []
    if t_str and dose_keys:
        schedule = parse_irradiation_schedule(t_str, dose_keys)
        if schedule:
            params.append(f"Schedule={','.join(str(v) for v in schedule)}")
    return params, schedule


def _cumulative_days_from_gaps_hours(gaps_hours: List[float]) -> List[float]:
    """
    Превращает список интервалов между фракциями (часы, длина N-1) в список
    абсолютных дней начала каждой фракции (длина N, начиная с 0.0).
    Пустой вход → [0.0] (только день первой фракции).
    """
    days = [0.0]
    current = 0.0
    for gap_h in gaps_hours:
        current += float(gap_h) / 24.0
        days.append(current)
    return days


def _infer_reference_year(file_path: str, raw_labels: List[str]) -> int:
    """
    Год для дат без явного указания года в подписях ('V исх. - 21.10' без года).

    Приоритет: год, явно указанный в любой подписи ('...- 20.05.26') > год из имени
    файла > текущий год (крайний случай, если ничего не найдено).
    """
    for label in raw_labels:
        m = _DATE_IN_TEXT_RE.search(str(label))
        if m and m.group(3):
            year = int(m.group(3))
            return year + 2000 if year < 100 else year
    filename_date = extract_date_from_filename(file_path)
    if filename_date:
        try:
            return int(filename_date.split('.')[-1])
        except (ValueError, IndexError):
            pass
    return datetime.now().year


def _label_calendar_dates(labels: List[str], ref_year: int) -> List[Optional[date]]:
    """
    Парсит календарную дату каждой подписи по порядку, увеличивая текущий год при
    обнаружении перехода через Новый год (дата "уходит назад" более чем на полгода
    относительно предыдущей — явный признак того, что подпись без года относится
    уже к следующему году).
    """
    dates: List[Optional[date]] = []
    current_year = ref_year
    previous_date: Optional[date] = None
    for label in labels:
        parsed = _parse_calendar_date(str(label), current_year)
        if parsed is not None and previous_date is not None and (previous_date - parsed).days > 180:
            current_year += 1
            parsed = _parse_calendar_date(str(label), current_year)
        if parsed is not None:
            previous_date = parsed
        dates.append(parsed)
    return dates


# Минимальное преимущество (в сутках) календарной точности более позднего якоря фракции
# над текущим, при котором считаем переразметку реальной, а не шумом. Многие файлы несут
# устойчивое расхождение в ~1 сутки между номинальной подписью 'N сут.' и календарной датой
# (привычная конвенция лаборатории) — этого недостаточно для переключения базовой точки.
_REBASE_SWITCH_THRESHOLD_DAYS = 2


def _rebase_time_point_labels(labels: List[str], fraction_days: List[float], file_path: str = "") -> List[str]:
    """
    Преобразует подписи временных точек (вторая строка Excel) в абсолютные дни
    от дня первой фракции.

    Базовый случай — один маркер 'V исх.' в начале, 'N сут.' считается от него
    (номинальная цифра используется как есть, без проверки по дате: в подписях
    есть устойчивое расхождение в ~1 сутки между номинальной цифрой и календарной
    датой — это давняя конвенция, а не ошибка, трогать её не нужно).

    Протоколы с многосуточным перерывом между фракциями переразмечают волюметрию
    заново от очередной фракции. Явный признак — строка 'V промежут.': она и раньше
    привязывалась к очередному дню из fraction_days по порядку появления.

    Но иногда эта строка отсутствует (например, в файле кожных реакций той же серии
    измерений, где её просто не стали дублировать), а счёт суток всё равно идёт от
    последней фракции. Чтобы поймать и такой случай без побочных эффектов на обычные
    файлы, при обработке каждой подписи 'N сут.' проверяем: не предсказывает ли
    календарная дата ЭТОЙ подписи день много точнее (на ≥ _REBASE_SWITCH_THRESHOLD_DAYS
    суток), если считать её от ещё не использованной более поздней фракции, а не от
    текущей базовой точки. Если да — неявно переключаемся на неё (как будто там стояла
    своя 'V промежут.'). Небольшие фоновые расхождения (~1 сутки) такой порог не пройдут.

    Если дату подписи не удалось распарсить — просто используем текущую базовую точку,
    без проверки переключения (старое поведение).

    Args:
        labels:        сырые подписи второй строки (например 'V исх. - 20.05.26',
                       'V промежут. - 25.05.2026', '2 сут. - 27.05', ...)
        fraction_days: абсолютные дни начала каждой фракции, см. _cumulative_days_from_gaps_hours
        file_path:     путь к файлу (для вывода года из имени файла, если в подписях его нет)

    Returns:
        Список строк с абсолютными днями (например ['0', '5', '7', '9', ...])
    """
    ref_year = _infer_reference_year(file_path, labels)
    parsed_dates = _label_calendar_dates(labels, ref_year)
    day0_date = parsed_dates[0] if parsed_dates else None

    result: List[str] = []
    v_seen = 0
    baseline_day = 0.0

    for label, label_date in zip(labels, parsed_dates):
        token = str(label).strip().split(' ')[0]

        if token.upper().startswith('V'):
            if v_seen < len(fraction_days):
                baseline_day = fraction_days[v_seen]
            v_seen += 1
            result.append(str(int(round(baseline_day))))
            continue

        try:
            nominal = int(token)
        except ValueError:
            nominal = 0

        if day0_date is not None and label_date is not None and v_seen < len(fraction_days):
            current_mismatch = abs((label_date - (day0_date + timedelta(days=baseline_day + nominal))).days)
            best_idx, best_baseline, best_mismatch = None, baseline_day, current_mismatch
            for idx in range(v_seen, len(fraction_days)):
                candidate_baseline = fraction_days[idx]
                candidate_mismatch = abs(
                    (label_date - (day0_date + timedelta(days=candidate_baseline + nominal))).days)
                if candidate_mismatch < best_mismatch:
                    best_idx, best_baseline, best_mismatch = idx, candidate_baseline, candidate_mismatch
            if best_idx is not None and (current_mismatch - best_mismatch) >= _REBASE_SWITCH_THRESHOLD_DAYS:
                baseline_day = best_baseline
                v_seen = best_idx + 1

        result.append(str(int(round(baseline_day + nominal))))

    return result


@dataclass(frozen=True)
class RatSurvivalEvent:
    """
    Событие конца наблюдения для одной крысы: подтверждённая смерть или цензурирование
    (потеря из-под наблюдения по иной причине, например потеря бирки, либо конец эксперимента).

    Attributes:
        label:          метка крысы
        day:            абсолютный день события (от дня первой фракции); None если день не определён
        event_observed: True = подтверждённая смерть, False = цензурировано
        reason:         исходный текст маркера ('⊗ 29.03', 'death', 'выгрызла', ...);
                       '' если крыса жива до конца таблицы (censored at last observation)
        source_file:    имя файла-источника
    """
    label: str
    day: Optional[float]
    event_observed: bool
    reason: str
    source_file: str = ""


# Маркеры смерти. Символ '⊗' (падёж) — основной принятый в лаборатории способ записи,
# может сопровождаться точной календарной датой смерти ('⊗ 29.03'). Слова — запасной вариант.
_DEATH_SYMBOL_RE = re.compile(r'^[⊗†✝]\s*(.*)$')
_DEATH_WORD_RE = re.compile(r'^(death|смерть|пал[аои]|падеж|погиб\w*)$', re.IGNORECASE)
_DATE_IN_TEXT_RE = re.compile(r'(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?')


def _classify_marker(raw_text: str) -> Tuple[bool, str]:
    """
    Классифицирует нечисловое содержимое ячейки объёма опухоли.

    Returns:
        (is_death, date_text): is_death=True, если маркер означает подтверждённую смерть;
        date_text — дата, найденная внутри маркера (например '29.03'), или '' если её нет.
        Любой иной непустой нечисловой текст ('выгрызла' и т.п.) считается цензурированием
        (is_death=False) — само наличие текста уже сохраняется как reason вызывающей стороной.
    """
    text = raw_text.strip()
    symbol_match = _DEATH_SYMBOL_RE.match(text)
    if symbol_match:
        return True, symbol_match.group(1).strip()
    if _DEATH_WORD_RE.match(text):
        return True, ''
    return False, ''


def _parse_calendar_date(date_str: str, ref_year: int) -> Optional[date]:
    """Парсит дату вида 'D.MM', 'D.MM.YY' или 'D.MM.YYYY'. Без года -> используется ref_year."""
    m = _DATE_IN_TEXT_RE.search(date_str)
    if not m:
        return None
    day_s, month_s, year_s = m.groups()
    if year_s is None:
        year = ref_year
    else:
        year = int(year_s)
        if year < 100:
            year += 2000
    try:
        return date(year, int(month_s), int(day_s))
    except ValueError:
        return None


def _is_numeric_tumor_cell(text: str) -> bool:
    """True, если текст — обычное измерение (число или a-b-c триплет), а не маркер события."""
    if "-" in text and len(text.split("-")) == 3:
        return True
    return text.replace(".", "").isdigit()


def extract_survival_events(file_path: str) -> List[RatSurvivalEvent]:
    """
    Извлекает события смерти/цензурирования крыс из файла объёмов опухолей.

    Распознаёт:
      - подтверждённую смерть: символ '⊗' (опционально с точной датой смерти, например
        '⊗ 29.03') или слова death/смерть/падёж/погибла/пала;
      - цензурирование: любой другой непустой нечисловой текст (например 'выгрызла' —
        потеряна бирка) — животное не считается умершим, но дальнейших измерений нет.

    Если в маркере смерти есть точная дата, день события вычисляется по календарной дате
    из подписи столбца, а не по номинальному дню столбца — в лаборатории дату гибели
    регистрируют отдельно, и она может на 1-2 дня отличаться от дня плановой волюметрии.

    Если в строке есть и цензурирующая пометка, и более поздний подтверждённый маркер смерти
    (например сначала 'выгрызла', затем '⊗ <дата>' в следующих измеренных столбцах),
    итоговым событием считается смерть — она более информативна.

    Крысы без какого-либо маркера до конца таблицы считаются цензурированными на момент
    последнего измеренного столбца (event_observed=False, reason='').

    Returns:
        Список RatSurvivalEvent, один на крысу, в порядке появления в файле.
    """
    data = pd.read_excel(file_path, header=None)
    raw_params = data.iloc[0, :].dropna().astype(str).tolist()
    _, schedule_hours = _process_header_row(raw_params)
    fraction_days = _cumulative_days_from_gaps_hours(schedule_hours)

    raw_labels = [str(item) for item in data.iloc[1, 1:]]
    rebased_days = [int(v) for v in _rebase_time_point_labels(raw_labels, fraction_days, file_path)]

    ref_year = _infer_reference_year(file_path, raw_labels)
    column_dates = _label_calendar_dates(raw_labels, ref_year)

    file_name = os.path.basename(file_path)
    tumor_data = data.iloc[2:, :].copy()
    tumor_data = tumor_data.apply(lambda column: column.map(_normalize_tumor_cell))

    events: List[RatSurvivalEvent] = []
    for _, row in tumor_data.iterrows():
        label = str(row.iloc[0])
        cells = list(row.iloc[1:])
        death_event: Optional[RatSurvivalEvent] = None
        first_other_marker: Optional[Tuple[int, str]] = None
        last_valid_col_idx: Optional[int] = None

        for col_idx, item in enumerate(cells):
            text = str(item).strip()
            if text in ("", "NA", "nan", "None"):
                continue
            if _is_numeric_tumor_cell(text):
                last_valid_col_idx = col_idx
                continue
            is_death, date_text = _classify_marker(text)
            if is_death and death_event is None:
                day = float(rebased_days[col_idx]) if col_idx < len(rebased_days) else None
                if date_text and col_idx < len(column_dates) and column_dates[col_idx] is not None:
                    marker_date = _parse_calendar_date(date_text, ref_year)
                    if marker_date is not None and day is not None:
                        day = float(rebased_days[col_idx] - (column_dates[col_idx] - marker_date).days)
                death_event = RatSurvivalEvent(label, day, True, text, file_name)
            elif not is_death and first_other_marker is None:
                first_other_marker = (col_idx, text)

        if death_event is not None:
            events.append(death_event)
        elif first_other_marker is not None:
            col_idx, text = first_other_marker
            day = float(rebased_days[col_idx]) if col_idx < len(rebased_days) else None
            events.append(RatSurvivalEvent(label, day, False, text, file_name))
        else:
            if last_valid_col_idx is not None and last_valid_col_idx < len(rebased_days):
                day = float(rebased_days[last_valid_col_idx])
            else:
                day = float(rebased_days[-1]) if rebased_days else None
            events.append(RatSurvivalEvent(label, day, False, "", file_name))

    return events


def find_rats_that_died_before_experiment_end(file_path: str) -> List[str]:
    """
    Возвращает метки животных с подтверждённой смертью (event_observed=True) СТРОГО ДО
    последнего наблюдённого дня в этом же файле — то есть тех, чья кривая объёма опухоли
    обрывается раньше остальных из-за гибели, а не обычного цензурирования (потеря бирки
    и т.п.) или дожития до конца наблюдения.

    Сравнение идёт с максимальным днём внутри того же списка событий (а не с отдельно
    посчитанной сеткой time_data), чтобы не зависеть от согласованности двух независимых
    расчётов дней.
    """
    events = extract_survival_events(file_path)
    days = [event.day for event in events if event.day is not None]
    if not days:
        return []
    last_day = max(days)
    return [
        event.label for event in events
        if event.event_observed and event.day is not None and event.day < last_day
    ]


def _normalize_tumor_cell(value) -> str:
    """Normalize raw Excel cell contents before tumor-volume parsing."""
    if pd.isna(value):
        return "NA"
    return str(value).strip().replace(',', '.').replace(' -', '-')


def _to_skin_reaction_value(value) -> float:
    """
    Преобразует сырую ячейку кожной реакции в float, NaN для пустых ячеек и любых
    маркеров события ('⊗', 'death', 'выгрызла' и т.п.) — иначе при попадании
    нечислового текста в матрицу numpy приводит ВЕСЬ массив к строковому dtype, и
    дальнейшая арифметика (np.nanmean и т.п.) падает с UFuncTypeError. Сами маркеры
    разбираются отдельно через extract_survival_events, читающий исходный файл заново.
    """
    if pd.isna(value):
        return float('nan')
    text = str(value).strip().replace(',', '.')
    try:
        return float(text)
    except ValueError:
        return float('nan')


def process_skin_data_excel(file_path) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
    """
    Обрабатывает данные из указанного файла Excel, содержащего информацию о реакциях кожи на эксперименты.

    Args:
        file_path (str): Путь к файлу Excel с данными о реакциях кожи.

    Returns:
        Tuple[List[str], List[str], List[str], List[List[float]]]:
            - experiment_params (List[str]): Список, содержащий параметры эксперимента, извлеченные из первой строки файла.
            - time_data (List[str]): Список меток времени для каждого измерения, преобразованный из строк в числовой формат.
            - rat_labels (List[str]): Список меток (идентификаторов) крыс, участвовавших в эксперименте.
            - skin_reactions (List[List[float]]): Список списков с данными о реакциях кожи для каждой крысы на каждом
            временном интервале.
    """
    data = pd.read_excel(file_path, header=None)
    raw_params = data.iloc[0, :].dropna().astype(str).tolist()
    experiment_params, schedule_hours = _process_header_row(raw_params)
    fraction_days = _cumulative_days_from_gaps_hours(schedule_hours)

    skin_data = data.iloc[2:, :].copy()  # Копируем данные, начиная с третьей строки
    time_data = _rebase_time_point_labels(
        [str(item) for item in data.iloc[1, 1:]], fraction_days, file_path)  # Преобразуем метки времени
    rat_labels = skin_data.iloc[:, 0].tolist()  # Извлекаем метки крыс из первого столбца
    skin_reactions = skin_data.iloc[:, 1:].apply(
        lambda column: column.map(_to_skin_reaction_value)).to_numpy().tolist()

    # Извлечение и форматирование даты из имени файла
    formatted_date = extract_date_from_filename(file_path)
    if formatted_date:
        experiment_params.append(f"Date={formatted_date}")  # Добавляем дату как параметр

    # Регистрируем метки крыс для кожных реакций
    file_name = os.path.basename(file_path)
    register_rat_labels(rat_labels, file_name)
    
    return experiment_params, time_data, rat_labels, skin_reactions


def extract_date_from_filename(file_path: str) -> str:
    """
    Извлекает дату из имени файла и форматирует ее как 'месяц.день.год'.

    Args:
        file_path (str): Путь к файлу.

    Returns:
        str: Отформатированная дата, например, '5.12.2023'.
             Возвращает пустую строку, если дата не найдена.
    """
    filename = os.path.basename(file_path)
    # Регулярное выражение для поиска даты формата dd.mm.yyyy, mm.dd.yyyy, dd-mm-yyyy и т.д.
    match = re.search(r'(\d{1,2})[.\-_](\d{1,2})[.\-_](\d{4})', filename)
    if match:
        part1, part2, year = match.groups()
        # Всегда предполагаем формат ДД.ММ.ГГГГ
        day, month = part1, part2
        try:
            # Проверяем корректность даты
            date_obj = datetime(int(year), int(month), int(day))
            formatted_date = f"{date_obj.day}.{date_obj.month}.{date_obj.year}"
            return formatted_date
        except ValueError:
            # Некорректная дата
            return ""
    else:
        # Дата не найдена
        return ""


def process_tumor_data_excel(file_path) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
    """
    Обрабатывает данные из указанного файла Excel, содержащего объемы опухолей и извлекает необходимые данные для анализа.

    Args:
        file_path (str): Путь к файлу Excel с данными об объемах опухолей.

    Returns:
        Tuple[List[str], List[str], List[str], List[List[float]]]:
            - experiment_params (List[str]): Параметры эксперимента, извлеченные из первой строки файла.
            - time_data (List[str]): Список меток времени для каждого измерения, преобразованный из строк в числовой формат.
            - rat_labels (List[str]): Список меток (идентификаторов) крыс, участвовавших в эксперименте.
            - tumor_volumes (List[List[float]]): Список списков с объемами опухолей для каждой крысы на каждом временном интервале.
    """
    data = pd.read_excel(file_path, header=None)
    raw_params = data.iloc[0, :].dropna().astype(str).tolist()
    experiment_params, schedule_hours = _process_header_row(raw_params)
    fraction_days = _cumulative_days_from_gaps_hours(schedule_hours)

    tumor_data = data.iloc[2:, :].copy()
    time_data = _rebase_time_point_labels([str(item) for item in data.iloc[1, 1:]], fraction_days, file_path)

    # Преобразование данных об объемах опухолей
    tumor_data = tumor_data.apply(lambda column: column.map(_normalize_tumor_cell))
    rat_labels = tumor_data.iloc[:, 0].tolist()

    # Преобразование объемов опухолей в числовой формат
    tumor_volumes = []
    for _, row in tumor_data.iterrows():
        rat_volumes = []
        for item in row[1:]:
            if "-" in item:
                parts = item.split("-")
                if len(parts) == 3:
                    try:
                        a, b, c = map(float, parts)
                        volume = (np.pi * a * b * c) / 6
                    except ValueError:
                        volume = np.nan
                else:
                    volume = np.nan
            elif item.replace(".", "").isdigit():
                volume = float(item)
            else:
                volume = np.nan
            rat_volumes.append(volume)
        tumor_volumes.append(rat_volumes)

    # Извлечение и форматирование даты из имени файла
    formatted_date = extract_date_from_filename(file_path)
    if formatted_date:
        experiment_params.append(f"Date={formatted_date}")  # Добавляем дату как параметр

    # Сохраняем данные в датакласс
    file_name = os.path.basename(file_path)
    register_rat_labels(rat_labels, file_name)  # Регистрируем метки с указанием файла
    return experiment_params, time_data, rat_labels, tumor_volumes
