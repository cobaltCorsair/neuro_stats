# файл draw_abs_rel_graph_compare.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from draw_base_grapfs import TumorDataVisualizer
from controls import ControlGroupVisualizer
from utils.plotting_helpers import custom_fill_between, format_experiment_params
from utils.plot_saver import save_plot
from stats_methods.support_stats_methods import SupportingFunctions
from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer

# Сохраняем оригинальную функцию в другой переменной, на случай, если она понадобится
original_fill_between = plt.fill_between

# Переопределяем функцию
plt.fill_between = custom_fill_between

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

    def compare_mean_volumes(self):
        """
        Сравнивает средние абсолютные объемы опухолей для всех экспериментов и строит график.
        """
        SupportingFunctions.normalize_time_data_min(self.visualizers)
        drawgraph = GraphVisualizer("Сравнение среднего объема опухолей", "Время, сут.",
                                    "Средний объем опухоли")
        drawgraph.setup_figure()

        x_data_lists = []
        for visualizer in self.visualizers:
            mean_volumes = visualizer.get_mean_tumor_volumes()
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_volumes)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            drawgraph.add_plot(visualizer.time_data, mean_volumes, visualizer.experiment_params, ": M/V абс.", error_margin, False)
            x_data_lists.append(visualizer.time_data)  # Добавляем данные по оси X для каждого визуализатора

        # Обновляем максимальное значение по оси X и настраиваем деления после добавления всех графиков
        drawgraph.update_axes_limits(x_data_lists)
        drawgraph.finalize_figure('')

    def compare_relative_volumes(self):
        """
        Сравнивает средние относительные объемы опухолей для всех экспериментов и строит график.
        """
        SupportingFunctions.normalize_time_data_min(self.visualizers)
        drawgraph = GraphVisualizer("Сравнение среднего относительного объема опухолей", "Время, сут.",
                                    "Относительный объем опухоли, отн. ед.")
        drawgraph.setup_figure()

        time_ = []
        x_data_lists = []

        for visualizer in self.visualizers:
            mean_rel_volumes = visualizer.get_mean_relative_tumor_volumes()
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), mean_rel_volumes)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            time_.append(visualizer.experiment_params[-1])
            drawgraph.add_plot(visualizer.time_data, mean_rel_volumes, visualizer.experiment_params, "", error_margin, False)
            x_data_lists.append(visualizer.time_data)  # Добавляем данные по оси X для каждого визуализатора

        # Обновляем максимальное значение по оси X и настраиваем деления после добавления всех графиков
        drawgraph.update_axes_limits(x_data_lists)

        # Добавляем дополнительные легенды
        #time_labels = [f"Интервал: {time}" for time in time_]
        #drawgraph.add_legend(time_labels, "Интервалы между облучениями", "lower right")
        drawgraph.finalize_figure('')

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
            formatted_params = format_experiment_params(visualizer.experiment_params)

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
                formatted_params = format_experiment_params(visualizer.experiment_params)

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
        auc_labels = [f"AUC: {auc:.2f}" for auc in aucs]
        plt.legend(lines, auc_labels, title="Площадь под кривой", loc='center left')

        plt.tight_layout()
        save_plot('', "compare_control_and_experiment")
        plt.xlim(left=0)  # Установка минимального значения для оси X равным 0
        plt.ylim(bottom=0)  # Установка минимального значения для оси Y равным 0
        plt.show()

    def compare_tumor_growth_inhibition_with_multiple_experiments(self, control_visualizer, experiment_visualizers):
        """
        Сравнивает торможение роста опухоли между одной контрольной и несколькими экспериментальными группами.
        """
        plt.figure(figsize=(12, 7))

        markers = ['o', 's', '^', 'x', '*', 'D', 'h', '+', 'p']
        marker_index = 0
        marker_size = 12

        # Нормализация временных рядов
        all_visualizers = [control_visualizer] + experiment_visualizers

        # Преобразуем строки в целые числа и находим минимальное начальное время
        min_start_time = min([min([int(time) for time in visualizer.time_data]) for visualizer in all_visualizers])

        # Находим минимальную длину временного ряда
        min_length = min([len(visualizer.time_data) for visualizer in all_visualizers])

        for visualizer in all_visualizers:
            # Преобразуем строки в целые числа и вычитаем минимальное начальное время
            visualizer.time_data = [int(time) - min_start_time for time in visualizer.time_data][:min_length]
            # Обрезаем данные объема опухоли до минимальной длины
            visualizer.mean_tumor_volumes = visualizer.get_mean_tumor_volumes()[:min_length]

        # Расчет среднего объема опухоли для контрольной группы
        control_mean_volumes = control_visualizer.get_mean_tumor_volumes()

        for experiment_visualizer in experiment_visualizers:
            # Расчет среднего объема опухоли для экспериментальной группы
            experiment_mean_volumes = experiment_visualizer.get_mean_tumor_volumes()

            # Расчет ТРО для каждого временного интервала
            tumor_growth_inhibition = [(Vk - Vo) / Vk * 100 for Vk, Vo in
                                       zip(control_mean_volumes, experiment_mean_volumes)]

            # Использование format_experiment_params для форматирования параметров эксперимента
            formatted_params = format_experiment_params(experiment_visualizer.experiment_params)

            plt.plot(
                experiment_visualizer.time_data,
                tumor_growth_inhibition,
                marker=markers[marker_index % len(markers)],
                markersize=marker_size,
                linestyle='-',
                label=f"{formatted_params}"
            )

            marker_index += 1
        # Установка меток на оси X
        max_time = max([max(visualizer.time_data) for visualizer in all_visualizers])
        plt.xticks(ticks=range(0, max_time + 1, 3))

        # Для отладки: вывод длин временных рядов и объемов опухоли
        print("Контрольная группа:", len(control_visualizer.time_data), len(control_mean_volumes))
        for experiment_visualizer in experiment_visualizers:
            print("Экспериментальная группа:", len(experiment_visualizer.time_data),
                  len(experiment_visualizer.get_mean_tumor_volumes()))

        plt.xlabel("Время, сут.")
        plt.ylabel("Торможение роста опухоли, %")
        # plt.title("Сравнение торможения роста опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_plot('', "compare_tumor_growth_inhibition_multiple_experiments")
        plt.show()

    def create_tumor_growth_inhibition_table(self, control_visualizer, experiment_visualizers):
        # Получение временных данных и среднего объема опухоли для контрольной группы
        control_time_data = control_visualizer.time_data
        control_mean_volumes = control_visualizer.get_mean_tumor_volumes()

        # Создание словаря для последующего создания DataFrame
        data = {'Время (сут)': control_time_data}

        for experiment_visualizer in experiment_visualizers:
            # Получение среднего объема опухоли для экспериментальной группы
            experiment_mean_volumes = experiment_visualizer.get_mean_tumor_volumes()

            # Расчет ТРО для каждого временного интервала
            tumor_growth_inhibition = [(Vk - Vo) / Vk * 100 for Vk, Vo in
                                       zip(control_mean_volumes, experiment_mean_volumes)]

            # Добавление данных в словарь
            experiment_name = format_experiment_params(experiment_visualizer.experiment_params)
            data[experiment_name] = tumor_growth_inhibition

        # Создание DataFrame из словаря
        df = pd.DataFrame(data)

        # Преобразование 'Время (сут)' в числовой тип данных для возможности фильтрации
        df['Время (сут)'] = pd.to_numeric(df['Время (сут)'])

        # Фильтрация DataFrame для времени начиная с 9-го дня
        df_filtered = df.loc[df['Время (сут)'] >= 9].copy()

        # Расчет абсолютной разницы эффективности ТРО и относительной эффективности
        df_filtered['Absolute Difference (%)'] = df_filtered.iloc[:, 1] - df_filtered.iloc[:, 2]
        df_filtered['Relative Difference (%)'] = (df_filtered['Absolute Difference (%)'] / df_filtered.iloc[:, 1] * 100)

        # Установка формата чисел
        pd.set_option('display.float_format', '{:.2f}'.format)
        pd.set_option('display.max_rows', None)  # Для показа всех строк
        pd.set_option('display.max_columns', None)  # Для показа всех столбцов

        # Изменение индекса
        df.set_index('Время (сут)', inplace=True)

        # Печать отфильтрованной таблицы и среднего значения относительной разницы
        print(df_filtered)

        # Расчет среднего значения относительной разницы
        # average_relative_difference = df_filtered['Relative Difference (%)'].mean()
        # print(f"Средний процент отличия от Dp = 36 Гр начиная с 9-го дня: {average_relative_difference:.2f}%")

        # Расчет среднего абсолютного значения относительных различий
        average_absolute_relative_difference = df_filtered['Relative Difference (%)'].abs().mean()

        print(f"Среднее абсолютное значение относительного различия начиная с 9-го дня: "
              f"{average_absolute_relative_difference:.2f}%")


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
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\control.xlsx',
    ]
    experiment_paths = [
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\30.03.2022_p_36_прострел.xlsx',
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_12.xlsx',
        # r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_18.xlsx',
        # r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\16.03.2023_n_22.xlsx'
    ]

    # Создание объектов визуализатора для контрольных групп
    control_visualizers = [ControlGroupVisualizer(path) for path in control_paths]
    # Создание объекта визуализатора для контрольной группы
    control_visualizer = ControlGroupVisualizer(control_paths[0])

    # Создание объектов визуализатора для экспериментальных групп
    experiment_visualizers = [TumorDataVisualizer(path) for path in experiment_paths]

    # Создание объекта сравнителя
    comparator = TumorDataComparatorAdvanced(*experiment_visualizers)

    #comparator.compare_mean_volumes()  # Сравниваем средние абсолютные объемы
    #comparator.compare_relative_volumes()  # Сравниваем средние относительные объемы

    # Сравнение контрольных и экспериментальных групп
    #comparator.compare_control_and_experiment(control_visualizers)

    # Сравнение торможения роста опухоли между контрольной и несколькими экспериментальными группами
    #comparator.compare_tumor_growth_inhibition_with_multiple_experiments(control_visualizer, experiment_visualizers)
    #comparator.create_tumor_growth_inhibition_table(control_visualizer, experiment_visualizers)
