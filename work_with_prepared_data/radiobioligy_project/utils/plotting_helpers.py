# файл plotting_helpers.py
from typing import List

import matplotlib.pyplot as plt


class MatplotlibConfigurator:
    """
    Класс для конфигурации и восстановления настроек визуализации Matplotlib.

    Методы:
    - apply_custom_styles(): Применяет пользовательские стили к графикам Matplotlib.
    - restore_original_styles(): Восстанавливает оригинальные стили графиков Matplotlib.
    """

    def __init__(self):
        """
        Инициализирует экземпляр класса MatplotlibConfigurator, сохраняя оригинальные настройки стилей.
        """
        self.original_rcParams = plt.rcParams.copy()

    def apply_custom_styles(self):
        """
        Применяет пользовательские стили к графикам Matplotlib.

        Устанавливает семейство шрифтов, размеры шрифтов и другие параметры визуализации.
        """
        plt.rcParams.update({
            'font.family': 'Times New Roman',  # Семейство шрифтов
            'font.size': 22,  # Размер основного текста
            'axes.titlesize': 24,  # Размер заголовков осей
            'axes.labelsize': 24,  # Размер меток осей
            'xtick.labelsize': 20,  # Размер меток делений на оси X
            'ytick.labelsize': 20,  # Размер меток делений на оси Y
            'legend.fontsize': 25  # Размер текста в легенде
        })

    def restore_original_styles(self):
        """
        Восстанавливает оригинальные настройки стилей графиков Matplotlib.
        """
        plt.rcParams.update(self.original_rcParams)


def subscriptify(text):
    """
    Преобразует текст в подстрочный формат, используя символы Unicode.

    Параметры:
        text (str): Текст, который нужно преобразовать.

    Возвращает:
        str: Текст в подстрочном формате.

    Пример:
        >>> subscriptify("H2O")
        'H₂O'
    """
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        'n': 'ₙ', 'p': 'ₚ', 'e': 'ₑ', 'a': 'ₐ', 'b': 'ᵦ', 'y': 'ᵧ'
    }
    return ''.join(subscript_map.get(char, char) for char in text)


def format_experiment_params(params: List[str]) -> str:
    """
    Форматирует параметры эксперимента для отображения в легенде.

    Args:
        params (List[str]): Список параметров эксперимента.

    Returns:
        str: Отформатированная строка параметров эксперимента с датой и временем облучения.
    """
    # Удаляем пустые строки и значения 'nan'
    cleaned_params = [str(param).replace('nan', '').strip() for param in params if str(param).strip()]

    # Разбиваем параметры на ключ и значение
    rad_values = {}
    sequence = []  # Сохраняем порядок ключей
    date_str = ""
    irradiation_time_str = ""
    for param in cleaned_params:
        if param.startswith("Date="):
            date_str = param.split("=", 1)[1].strip()
        elif param.startswith("Irradiation Time="):
            irradiation_time_str = param.split("=", 1)[1].strip()
        elif '=' in param and not param.startswith('t'):
            key, value = param.split('=', 1)
            key = key.strip()
            value = value.split()[0]  # Берём только первую часть, исключая "Гр."
            rad_values[key] = value.strip()

            sequence.append(key)

    # Формирование строки для легенды
    formatted_params = []

    # Добавление стрелок, если есть более одного типа излучения
    if len(sequence) > 1:
        arrows = ' → '.join(sequence)
        formatted_params.append(arrows)

    for key in sequence:
        if key in rad_values:
            formatted_params.append(f"D{subscriptify(key.lower())} = {rad_values[key]} Гр")

    # Добавление времени облучения, если оно присутствует
    if irradiation_time_str:
        formatted_params.append(f"Интервал: {irradiation_time_str}")

    # Добавление даты, если она присутствует
    if date_str:
        formatted_params.append(f"Дата: {date_str}")

    return ', '.join(formatted_params)


def custom_fill_between(x, y1, y2=0, color=None, alpha=None, **kwargs):
    """
        Реализует пользовательскую версию функции заполнения между двумя линиями на графике с добавлением
        горизонтальных линий на концах каждого заполненного сегмента.

        Args:
            x (List[float]): Список координат по оси X.
            y1 (List[float]): Список значений первой линии (верхней границы) для заполнения.
            y2 (Union[List[float], int]): Список значений второй линии (нижней границы) для заполнения или одно
            значение, если оно одинаково для всех x.
            color (Optional[str]): Цвет заполнения. По умолчанию используется 'blue'.
            alpha (Optional[float]): Прозрачность заполнения. По умолчанию прозрачность не устанавливается.
            **kwargs: Дополнительные аргументы для plt.plot().

        Returns:
            None: Функция ничего не возвращает, но отображает график.

        Пример использования:
            >>> custom_fill_between([1, 2, 3], [1, 2, 3], [0, 1, 2], color='red', alpha=0.5)
        """
    horizontal_line_length = 0.2  # Длина горизонтальных линий на концах
    line_color = color if color is not None else 'blue'  # Используйте заданный цвет, если он предоставлен

    for xi, y1i, y2i in zip(x, y1, y2):
        # Вертикальные линии
        plt.plot([xi, xi], [y1i, y2i], color=line_color, alpha=1, zorder=1)

        # Горизонтальные линии на концах
        plt.plot([xi - horizontal_line_length / 2, xi + horizontal_line_length / 2], [y1i, y1i], color=line_color,
                 alpha=1, zorder=1)
        plt.plot([xi - horizontal_line_length / 2, xi + horizontal_line_length / 2], [y2i, y2i], color=line_color,
                 alpha=1, zorder=1)
