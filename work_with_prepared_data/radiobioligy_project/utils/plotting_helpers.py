# файл plotting_helpers.py

import matplotlib.pyplot as plt


class MatplotlibConfigurator:
    def __init__(self):
        self.original_rcParams = plt.rcParams.copy()

    def apply_custom_styles(self):
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 22,
            'axes.titlesize': 24,
            'axes.labelsize': 24,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 25
        })

    def restore_original_styles(self):
        plt.rcParams = self.original_rcParams



def subscriptify(text):
    """
        Converts text to subscript format using Unicode characters.

        Parameters:
            text (str): Text to be converted.

        Returns:
            str: Text in subscript format.
        """
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        'n': 'ₙ', 'p': 'ₚ', 'e': 'ₑ', 'a': 'ₐ', 'b': 'ᵦ', 'y': 'ᵧ'
        # Add more if available
    }
    return ''.join(subscript_map.get(char, char) for char in text)

def format_experiment_params(params):
    """
        Форматирует параметры эксперимента для отображения в легенде.

        Parameters:
            params (list): Список параметров эксперимента.

        Returns:
            str: Отформатированная строка параметров эксперимента.
        """
    # Удаляем пустые строки и значения 'nan'
    cleaned_params = [str(param).replace('nan', '').strip() for param in params if str(param).strip()]

    # Разбиваем параметры на ключ и значение
    rad_values = {}
    sequence = []  # Сохраняем порядок ключей
    for param in cleaned_params:
        if '=' in param and not param.startswith('t'):
            key, value = param.split('=')
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

    return ', '.join(formatted_params)

def custom_fill_between(x, y1, y2=0, color=None, alpha=None, **kwargs):
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
