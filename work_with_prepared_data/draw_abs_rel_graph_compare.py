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
    # Перерисовывает доверительный интервал на чёрточки
    horizontal_line_length = 0.2  # Длина горизонтальных линий на концах
    for xi, y1i, y2i in zip(x, y1, y2):
        # Вертикальные линии
        plt.plot([xi, xi], [y1i, y2i], color='grey', alpha=1, zorder=1)

        # Горизонтальные линии на концах
        plt.plot([xi - horizontal_line_length / 2, xi + horizontal_line_length / 2], [y1i, y1i], color='grey', alpha=1, zorder=1)
        plt.plot([xi - horizontal_line_length / 2, xi + horizontal_line_length / 2], [y2i, y2i], color='grey', alpha=1, zorder=1)


# Переопределяем функцию
plt.fill_between = custom_fill_between


# Увеличение размера фигуры
plt.figure(figsize=(15, 8))  # Увеличение размера фигуры

# Глобальное изменение размеров шрифтов
plt.rcParams.update({
    'font.size': 16,           # Размер основного шрифта
    'axes.titlesize': 18,      # Размер заголовка
    'axes.labelsize': 16,      # Размер подписей осей
    'xtick.labelsize': 14,     # Размер меток на оси X
    'ytick.labelsize': 14,     # Размер меток на оси Y
    'legend.fontsize': 14      # Размер шрифта в легенде
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
        # Преобразуем список в строку, исключая квадратные скобки
        return ', '.join(cleaned_params)

    def compare_mean_volumes(self):
        """
        Сравнивает средние объемы опухолей для всех экспериментов и строит график.
        """
        self.normalize_time_data()
        plt.figure(figsize=(15, 8))

        for visualizer in self.visualizers:
            mean_volumes = visualizer.get_mean_tumor_volumes()
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_volumes)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            # Использование format_experiment_params для форматирования параметров эксперимента
            formatted_params = self.format_experiment_params(visualizer.experiment_params)

            plt.plot(
                visualizer.time_data,
                mean_volumes,
                marker='o',
                linestyle='-',
                zorder=2,
                label=f"{formatted_params}: M/V абс."
            )
            plt.fill_between(visualizer.time_data,
                             [mean - err for mean, err in zip(mean_volumes, error_margin)],
                             [mean + err for mean, err in zip(mean_volumes, error_margin)], alpha=0.2)

        plt.title("Сравнение среднего объема опухолей")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
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
        plt.figure(figsize=(15, 8))

        # Инициализация списка для хранения объектов линий и AUC
        lines = []
        aucs = []

        for visualizer in self.visualizers:
            mean_rel_volumes = visualizer.get_mean_relative_tumor_volumes()
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_rel_volumes)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            # Использование format_experiment_params для форматирования параметров эксперимента
            formatted_params = self.format_experiment_params(visualizer.experiment_params)

            # Создание графика и сохранение объекта линии для каждого эксперимента
            line, = plt.plot(
                visualizer.time_data,
                mean_rel_volumes,
                marker='o',
                linestyle='-',
                zorder=2,
                label=formatted_params
            )
            plt.fill_between(visualizer.time_data,
                             [mean - err for mean, err in zip(mean_rel_volumes, error_margin)],
                             [mean + err for mean, err in zip(mean_rel_volumes, error_margin)], alpha=0.2)
            lines.append(line)
            aucs.append(np.trapz(mean_rel_volumes, visualizer.time_data))

        plt.title("Сравнение среднего относительного объема опухолей")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний относительный объем опухоли")
        plt.grid(True)

        # Добавление первой легенды с параметрами экспериментов
        first_legend = plt.legend(handles=lines, title="Параметры эксперимента", loc='upper left')
        plt.gca().add_artist(first_legend)  # Добавление первой легенды на график

        # Добавление второй легенды с AUC
        auc_labels = [f"AUC: {auc:.2f}" for auc in aucs]
        plt.legend(lines, auc_labels, title="Площадь под кривой", loc='upper right')

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

        plt.figure(figsize=(15, 8))

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
                marker='o',
                linestyle='-',
                zorder=2,
                label=f"Контроль: {''.join(formatted_params)}",
            )
            plt.fill_between(visualizer.time_data,
                             [mean - err for mean, err in zip(mean_rel_volumes, error_margin)],
                             [mean + err for mean, err in zip(mean_rel_volumes, error_margin)], alpha=0.2)
            lines.append(line)
            aucs.append(np.trapz(mean_rel_volumes, visualizer.time_data))

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
                marker='o',
                linestyle='-',
                zorder=2,
                label=f"Эксперимент: {''.join(formatted_params)}",
            )
            plt.fill_between(visualizer.time_data,
                             [mean - err for mean, err in zip(mean_rel_volumes, error_margin)],
                             [mean + err for mean, err in zip(mean_rel_volumes, error_margin)], alpha=0.2)
            lines.append(line)
            aucs.append(np.trapz(mean_rel_volumes, visualizer.time_data))

        plt.title("Сравнение контрольных и экспериментальных групп")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний относительный объем опухоли")
        plt.grid(True)

        # Добавление первой легенды с параметрами экспериментов
        first_legend = plt.legend(handles=lines, title="Параметры эксперимента", loc='upper left')
        plt.gca().add_artist(first_legend)  # Добавление первой легенды на график

        # Добавление второй легенды с AUC
        auc_labels = [f"AUC: {auc:.2f}" for auc in aucs]
        plt.legend(lines, auc_labels, title="Площадь под кривой", loc='upper right')

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
        './datas/control/02.02.2023_n_12.xlsx',
        './datas/control/02.02.2023_n_18.xlsx',
        './datas/control/16.03.2023_n_22.xlsx',
        './datas/control/30.03.2022_p_36_прострел.xlsx',
    ]

    # Создание объектов визуализатора для контрольных групп
    control_visualizers = [ControlGroupVisualizer(path) for path in control_paths]

    # Создание объектов визуализатора для экспериментальных групп
    experiment_visualizers = [TumorDataVisualizer(path) for path in experiment_paths]

    # Создание объекта сравнителя
    comparator = TumorDataComparatorAdvanced(*experiment_visualizers)

    # comparator.compare_mean_volumes()  # Сравниваем средние абсолютные объемы
    comparator.compare_relative_volumes()  # Сравниваем средние относительные объемы

    # Сравнение контрольных и экспериментальных групп
    # comparator.compare_control_and_experiment(control_visualizers)
