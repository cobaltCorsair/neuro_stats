# файл draw_abs_rel_graph_compare.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from draw_base_grapfs import TumorDataVisualizer
from controls import ControlGroupVisualizer
from utils.plotting_helpers import custom_fill_between, format_experiment_params, MatplotlibConfigurator
from stats_methods.support_stats_methods import SupportingFunctions
from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import process_tumor_data_excel
from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
configurator.restore_original_styles()


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
        drawgraph = GraphVisualizer("Сравнение среднего объема опухолей", "Время, сут.", "Средний объем опухоли, абс. ед.")
        drawgraph.setup_figure()
        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            lambda visualizer: visualizer.data_processor.get_mean_tumor_volumes(),
            drawgraph,
            "M/V абс.: ")

        drawgraph.finalize_figure('')

    def compare_relative_volumes(self):
        """
        Сравнивает средние относительные объемы опухолей для всех экспериментов и строит график.
        """
        SupportingFunctions.normalize_time_data_min(self.visualizers)
        drawgraph = GraphVisualizer("Сравнение среднего относительного объема опухолей", "Время, сут.",
                                    "Относительный объем опухоли, отн. ед.")
        drawgraph.setup_figure()

        # Собираем информацию об интервалах
        time_intervals = [visualizer.experiment_params[-1] for visualizer in self.visualizers]

        # Используем лямбда-функцию для извлечения значений
        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            lambda visualizer: visualizer.data_processor.get_mean_relative_tumor_volumes(),  # Лямбда-функция
            drawgraph,
            ""
        )
        # Добавляем легенду с интервалами
        time_labels = [f"Интервал: {interval}" for interval in time_intervals]
        #drawgraph.add_legend(time_labels, "Интервалы между облучениями", "lower right")

        drawgraph.finalize_figure('')

    def compare_control_and_experiment(self, control_visualizers):
        """
        Сравнивает средние относительные объемы опухолей между контрольными и экспериментальными группами и строит график.

        Parameters:
            control_visualizers (list): Список визуализаторов для контрольных групп.
        """
        SupportingFunctions.normalize_time_data_min(list(self.visualizers) + control_visualizers)
        drawgraph = GraphVisualizer("Сравнение контрольных и экспериментальных групп", "Время, сут.",
                                    "Относительный объем опухоли, отн. ед.")
        drawgraph.setup_figure()

        # Функция для извлечения значений средних относительных объемов из визуализатора
        value_extractor = lambda visualizer: visualizer.data_processor.get_mean_relative_tumor_volumes()

        # Добавляем данные контрольных групп
        GraphVisualizer.prepare_and_add_data_to_graph(
            control_visualizers,
            value_extractor,
            drawgraph,
            "Контроль: без облучения",
            calculate_auc=True  # Указываем, что нужно рассчитать AUC
        )

        # Добавляем данные экспериментальных групп
        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            value_extractor,
            drawgraph,
            "Эксперимент: ",
            calculate_auc=True  # Указываем, что нужно рассчитать AUC
        )

        drawgraph.finalize_figure('')

    def compare_tumor_growth_inhibition_with_multiple_experiments(self, control_visualizer, experiment_visualizers):
        """
        Сравнивает торможение роста опухоли между одной контрольной и несколькими экспериментальными группами.
        """
        SupportingFunctions.normalize_time_data_min([control_visualizer] + experiment_visualizers)
        drawgraph = GraphVisualizer("Сравнение торможения роста опухоли", "Время, сут.", "Торможение роста опухоли, %")
        drawgraph.setup_figure()

        # Подготовка данных
        control_mean_volumes = control_visualizer.data_processor.get_mean_tumor_volumes()
        x_data_lists = []
        for experiment_visualizer in experiment_visualizers:
            experiment_mean_volumes = experiment_visualizer.data_processor.get_mean_tumor_volumes()
            tumor_growth_inhibition = SupportingFunctions.calculate_tumor_growth_inhibition(control_mean_volumes,
                                                                                            experiment_mean_volumes)
            label = ''

            # Добавление данных на график
            drawgraph.add_plot(experiment_visualizer.time_data, tumor_growth_inhibition,
                               experiment_visualizer.experiment_params, label, calculate_auc=False)

            x_data_lists.append(experiment_visualizer.time_data)  # Добавляем данные по оси X для каждого визуализатора

        drawgraph.update_axes_limits(x_data_lists)
        drawgraph.finalize_figure('')

    def create_tumor_growth_inhibition_table(self, control_visualizer, experiment_visualizers):
        SupportingFunctions.normalize_time_data_min([control_visualizer] + experiment_visualizers)

        # Находим минимальную длину временного ряда среди всех визуализаторов
        min_length = min(len(viz.time_data) for viz in [control_visualizer] + experiment_visualizers)

        # Обрезаем данные до минимальной длины
        control_time_data = control_visualizer.time_data[:min_length]
        control_mean_volumes = control_visualizer.data_processor.get_mean_tumor_volumes()[:min_length]

        data = {'Время (сут)': control_time_data}

        for experiment_visualizer in experiment_visualizers:
            experiment_mean_volumes = experiment_visualizer.data_processor.get_mean_tumor_volumes()[:min_length]

            tumor_growth_inhibition = SupportingFunctions.calculate_tumor_growth_inhibition(
                control_mean_volumes,
                experiment_mean_volumes)

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

        # Печать отфильтрованной таблицы с использованием to_string()
        print(df_filtered.to_string())

        # Расчет среднего абсолютного значения относительных различий
        average_absolute_relative_difference = df_filtered['Relative Difference (%)'].abs().mean()
        print(
            f"Среднее абсолютное значение относительного различия начиная с 9-го дня: {average_absolute_relative_difference:.2f}%")


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

    comparator.compare_mean_volumes()  # Сравниваем средние абсолютные объемы
    comparator.compare_relative_volumes()  # Сравниваем средние относительные объемы

    # Сравнение контрольных и экспериментальных групп
    comparator.compare_control_and_experiment(control_visualizers)

    # Сравнение торможения роста опухоли между контрольной и несколькими экспериментальными группами
    comparator.compare_tumor_growth_inhibition_with_multiple_experiments(control_visualizer, experiment_visualizers)
    comparator.create_tumor_growth_inhibition_table(control_visualizer, experiment_visualizers)
