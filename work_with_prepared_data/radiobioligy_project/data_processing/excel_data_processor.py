# файл excel_data_processor.py

from typing import List, Tuple
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
from work_with_prepared_data.radiobioligy_project.data_processing.rat_manager import register_rat_labels


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
    experiment_params = data.iloc[0, :].dropna().astype(str).tolist()

    # Проверка на наличие интервала времени облучения в последней ячейке первой строки
    if experiment_params and 'ч' in experiment_params[-1]:
        irradiation_time = experiment_params.pop().strip()
        experiment_params.append(
            f"Irradiation Time={irradiation_time}")  # Добавляем интервал времени облучения как параметр
        print(irradiation_time)

    skin_data = data.iloc[2:, :].copy()  # Копируем данные, начиная с третьей строки
    time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in
                 data.iloc[1, 1:]]  # Преобразуем метки времени
    rat_labels = skin_data.iloc[:, 0].tolist()  # Извлекаем метки крыс из первого столбца
    skin_reactions = skin_data.iloc[:, 1:].to_numpy().tolist()  # Преобразуем оставшиеся данные в список списков

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
    # Извлечение всех непустых значений из первой строки как параметры эксперимента
    experiment_params = data.iloc[0, :].dropna().astype(str).tolist()

    # Проверка на наличие интервала времени облучения в последней ячейке первой строки
    if experiment_params and 'ч' in experiment_params[-1]:
        irradiation_time = experiment_params.pop().strip()
        experiment_params.append(f"Irradiation Time={irradiation_time}")  # Добавляем интервал времени облучения как параметр

    tumor_data = data.iloc[2:, :].copy()
    time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in data.iloc[1, 1:]]

    # Преобразование данных об объемах опухолей
    tumor_data = tumor_data.applymap(
        lambda x: str(x).strip().replace(',', '.').replace(' -', '-') if pd.notna(x) else "NA")
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
