import matplotlib.pyplot as plt
import os


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


def save_plot(file_path, plot_title, comparison_type=None, file_suffix=None, set_limits=True):
    """
    Сохраняет текущий график в файл PNG с учетом различных параметров.

    Parameters:
        file_path (str): Путь к файлу данных, используемый для создания части имени файла.
        plot_title (str): Основа для имени файла, обычно название графика.
        comparison_type (str, optional): Тип сравнения, если применимо, для дополнения имени файла.
        file_suffix (str, optional): Суффикс для имени файла для уточнения типа графика.
        set_limits (bool, optional): Флаг для установки минимальных значений осей X и Y равными 0.
    """
    # Базовое имя файла из пути к файлу данных
    file_name_base = os.path.splitext(os.path.basename(file_path))[0]

    # Составление имени файла из предоставленных компонентов
    components = [component for component in [file_name_base, plot_title, comparison_type, file_suffix] if component]
    file_name = f"{'_'.join(components).replace(' ', '_')}.png"

    if set_limits:
        plt.xlim(left=0)  # Установка минимального значения для оси X равным 0
        plt.ylim(bottom=0)  # Установка минимального значения для оси Y равным 0

    plt.savefig(file_name, format='png', dpi=300)
    print(f"Plot saved as {file_name}")
