import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from draw_base_grapfs import TumorDataVisualizer
from work_with_prepared_data.controls import ControlGroupVisualizer
from work_with_prepared_data.support_stats_methods import SupportingFunctions

# Сохраняем оригинальную функцию в другой переменной, на случай, если она понадобится
original_fill_between = plt.fill_between


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


# Переопределяем функцию
plt.fill_between = custom_fill_between

# Увеличение размера фигуры
plt.figure(figsize=(15, 8))  # Увеличение размера фигуры

# Глобальное изменение размеров шрифтов
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Установка семейства шрифтов
    'font.size': 22,  # Размер основного шрифта
    'axes.titlesize': 24,  # Размер заголовка
    'axes.labelsize': 24,  # Размер подписей осей
    'xtick.labelsize': 20,  # Размер меток на оси X
    'ytick.labelsize': 20,  # Размер меток на оси Y
    'legend.fontsize': 25  # Размер шрифта в легенде
})


class TumorDataComparatorAdvanced:
    def __init__(self, *visualizers: List[TumorDataVisualizer]):
        """
        Инициализатор класса для сравнения данных произвольного количества экспериментов.

        Parameters:
            *visualizers (List[TumorDataVisualizer]): Произвольное количество объектов TumorDataVisualizer.
        """
        self.visualizers = visualizers

    def save_plot(self, comparison_type: str):
        """
        Сохраняет график в файл PNG.

        Parameters:
            comparison_type (str): Тип сравнения, используется для формирования имени файла.
        """
        file_names = [os.path.splitext(os.path.basename(v.file_path))[0] for v in self.visualizers]
        file_name = f"{comparison_type}_{'_vs_'.join(file_names)}.png"
        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def normalize_time_data(self):
        """
        Нормализует временные данные для всех объектов визуализатора.
        """
        # Находим минимальную начальную точку времени среди всех экспериментов
        min_start_time = min([int(v.time_data[0]) for v in self.visualizers])

        # Выравниваем все временные ряды, вычитая минимальную начальную точку
        for visualizer in self.visualizers:
            visualizer.time_data = [int(time) - min_start_time for time in visualizer.time_data]

    def subscriptify(self, text):
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

    def format_experiment_params(self, params):
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
        for key in sequence:
            if key in rad_values:
                formatted_params.append(f"D{self.subscriptify(key.lower())} = {rad_values[key]} Гр")

        # Добавление стрелок, если есть более одного типа излучения
        if len(sequence) > 1:
            arrows = ' → '.join(sequence)
            formatted_params.append(arrows)

        return ', '.join(formatted_params)

    def compare_mean_volumes(self):
        """
        Сравнивает средние абсолютные объемы опухолей для всех экспериментов и строит график.
        """
        self.normalize_time_data()
        plt.figure(figsize=(15, 8))

        # Список маркеров
        markers = ['o', 's', '^', 'x', '*', 'D', 'h', '+', 'p']
        marker_index = 0
        marker_size = 12  # Установка размера маркера

        for visualizer in self.visualizers:
            mean_volumes = visualizer.get_mean_tumor_volumes()
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_volumes)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            # Использование format_experiment_params для форматирования параметров эксперимента
            formatted_params = self.format_experiment_params(visualizer.experiment_params)

            line, = plt.plot(
                visualizer.time_data,
                mean_volumes,
                marker=markers[marker_index % len(markers)],
                markersize=marker_size,
                linestyle='-',
                zorder=2,
                label=f"{formatted_params}: M/V абс."
            )
            line_color = line.get_color()
            custom_fill_between(visualizer.time_data,
                                [mean - err for mean, err in zip(mean_volumes, error_margin)],
                                [mean + err for mean, err in zip(mean_volumes, error_margin)],
                                color=line_color, alpha=0.2)

            marker_index += 1

        plt.title("Сравнение среднего объема опухолей")
        max_time = max([max(v.time_data) for v in self.visualizers])
        plt.xticks(ticks=range(0, max_time + 1, 3), rotation=0)
        plt.xlabel("Время, сут.")
        plt.ylabel("Средний объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot("compare_mean_volumes")
        plt.show()

    def compare_relative_volumes(self):
        """
        Сравнивает средние относительные объемы опухолей для всех экспериментов и строит график.
        """
        self.normalize_time_data()
        plt.figure(figsize=(12, 7))

        # Список маркеров
        markers = ['o', 's', '^', 'x', '*', 'D', 'h', '+', 'p']
        marker_index = 0
        marker_size = 12  # Установка размера маркера

        for visualizer in self.visualizers:
            mean_rel_volumes = visualizer.get_mean_relative_tumor_volumes()
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_rel_volumes)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            # Использование format_experiment_params для форматирования параметров эксперимента
            formatted_params = self.format_experiment_params(visualizer.experiment_params)

            line, = plt.plot(
                visualizer.time_data,
                mean_rel_volumes,
                marker=markers[marker_index % len(markers)],
                markersize=marker_size,
                linestyle='-',
                zorder=2,
                label=formatted_params
            )
            line_color = line.get_color()
            custom_fill_between(visualizer.time_data,
                                [mean - err for mean, err in zip(mean_rel_volumes, error_margin)],
                                [mean + err for mean, err in zip(mean_rel_volumes, error_margin)],
                                color=line_color, alpha=0.2)

            marker_index += 1

        max_time = max([max(v.time_data) for v in self.visualizers])
        plt.xticks(ticks=range(0, max_time + 1, 3), rotation=0)
        plt.xlabel("Время, сут.")
        plt.ylabel("Относительный объем опухоли, отн. ед.")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot("compare_relative_volumes")
        plt.show()

    def compare_control_and_experiment(self, control_visualizers):
        """
        Сравнивает средние относительные объемы опухолей между контрольными и экспериментальными группами и строит график.

        Parameters:
            control_visualizers (list): Список визуализаторов для контрольных групп.
        """
        all_visualizers = list(self.visualizers) + control_visualizers
        for viz in all_visualizers:
            viz.time_data = [int(time) - int(viz.time_data[0]) for time in viz.time_data]

        plt.figure(figsize=(12, 7))

        # Список маркеров
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        marker_index = 0
        marker_size = 12  # Установка размера маркера

        # Списки для хранения объектов линий и значений AUC
        lines = []
        aucs = []

        # Визуализация для контрольных групп
        for visualizer in control_visualizers:
            mean_rel_volumes = visualizer.get_mean_relative_tumor_volumes()
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_rel_volumes)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]
            formatted_params = self.format_experiment_params(visualizer.experiment_params)

            line, = plt.plot(
                visualizer.time_data,
                mean_rel_volumes,
                marker=markers[marker_index % len(markers)],
                linestyle='-',
                markersize=marker_size,
                zorder=2,
                label=f"Контроль: {'без облучения'}",
            )
            line_color = line.get_color()
            custom_fill_between(visualizer.time_data,
                                [mean - err for mean, err in zip(mean_rel_volumes, error_margin)],
                                [mean + err for mean, err in zip(mean_rel_volumes, error_margin)],
                                color=line_color, alpha=0.2)
            lines.append(line)
            aucs.append(np.trapz(mean_rel_volumes, visualizer.time_data))

            marker_index += 1

            # Визуализация для экспериментальных групп
            for visualizer in self.visualizers:
                mean_rel_volumes = visualizer.get_mean_relative_tumor_volumes()
                std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                           for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_rel_volumes)]
                error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                                for std in std_dev]
                formatted_params = self.format_experiment_params(visualizer.experiment_params)

                line, = plt.plot(
                    visualizer.time_data,
                    mean_rel_volumes,
                    marker=markers[marker_index % len(markers)],
                    linestyle='-',
                    markersize=marker_size,
                    zorder=2,
                    label=f"Эксперимент: {''.join(formatted_params)}",
                )
                line_color = line.get_color()
                custom_fill_between(visualizer.time_data,
                                    [mean - err for mean, err in zip(mean_rel_volumes, error_margin)],
                                    [mean + err for mean, err in zip(mean_rel_volumes, error_margin)],
                                    color=line_color, alpha=0.2)

                lines.append(line)
                aucs.append(np.trapz(mean_rel_volumes, visualizer.time_data))

                marker_index += 1

        # plt.title("Сравнение контрольных и экспериментальных групп")
        # Установка меток на оси X
        max_time = max([max(v.time_data) for v in self.visualizers])  # Находим максимальное время из всех экспериментов
        plt.xticks(ticks=range(0, max_time + 1, 3), rotation=0)  # Устанавливаем метки каждые 3 дня, без поворота
        plt.xlabel("Время, сут.")
        plt.ylabel("Относительный объем опухоли, отн. ед.")
        plt.grid(True)

        # Добавление первой легенды с параметрами экспериментов
        # first_legend = plt.legend(handles=lines, title="Параметры эксперимента", loc='upper left')
        first_legend = plt.legend(handles=lines, title="", loc='upper left')
        plt.gca().add_artist(first_legend)  # Добавление первой легенды на график

        # Добавление второй легенды с AUC
        # auc_labels = [f"AUC: {auc:.2f}" for auc in aucs]
        # plt.legend(lines, auc_labels, title="Площадь под кривой", loc='upper right')

        plt.tight_layout()
        self.save_plot("compare_control_and_experiment")
        plt.show()


if __name__ == "__main__":
    # # Используем с файлом данных
    # file_path1 = './datas/n_7.2_p_25.2_2023_2.xlsx'
    # file_path2 = './datas/p_25.2_n_7.2_2023_2.xlsx'

    # Используем с файлом данных
    # file_path1 = './datas/n_7.2_p_25.2_2023.xlsx'
    # file_path2 = './datas/p_25.2_n_7.2_2023.xlsx'

    # Используем с файлом данных
    # file_path1 = './datas/n_2.56_p_25.6_2019.xlsx'
    # file_path2 = './datas/p_25.6_n_2.56_2019.xlsx'

    # Контроль
    # control_path = './datas/control/16.03.2023_e_36.xlsx'
    # control_path = './datas/control/08.10.2021_p32_прострел.xlsx'

    # Пути к файлам данных для контрольных и экспериментальных групп
    control_paths = [
        './datas/control/control.xlsx',
    ]
    experiment_paths = [
        # './datas/control/02.02.2023_n_12.xlsx',
        # './datas/control/02.02.2023_n_18.xlsx',
        # './datas/control/16.03.2023_n_22.xlsx',
        # './datas/control/30.03.2022_p_36_прострел.xlsx',
        './datas/n_7.2_p_25.2_2023.xlsx',
        './datas/p_25.2_n_7.2_2023.xlsx',
        # './datas/n_7.2_p_25.2_2023_2.xlsx',
        # './datas/p_25.2_n_7.2_2023_2.xlsx',
    ]

    # Создание объектов визуализатора для контрольных групп
    control_visualizers = [ControlGroupVisualizer(path) for path in control_paths]

    # Создание объектов визуализатора для экспериментальных групп
    experiment_visualizers = [TumorDataVisualizer(path) for path in experiment_paths]

    # Создание объекта сравнителя
    comparator = TumorDataComparatorAdvanced(*experiment_visualizers)

    #comparator.compare_mean_volumes()  # Сравниваем средние абсолютные объемы
    #comparator.compare_relative_volumes()  # Сравниваем средние относительные объемы

    # Сравнение контрольных и экспериментальных групп
    comparator.compare_control_and_experiment(control_visualizers)
