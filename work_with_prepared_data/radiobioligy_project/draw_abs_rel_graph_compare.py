# файл draw_abs_rel_graph_compare.py

import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from draw_base_graphs import TumorDataVisualizer
from controls import ControlGroupVisualizer
from utils.plotting_helpers import custom_fill_between, format_experiment_params, MatplotlibConfigurator
from stats_methods.support_stats_methods import SupportingFunctions
from utils.visualizer import GraphVisualizer

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
configurator.restore_original_styles()


class TumorDataComparatorAdvanced:
    """
    Класс для сравнения данных произвольного количества экспериментов, представленных экземплярами TumorDataVisualizer.

    Позволяет агрегировать и анализировать данные из различных источников, сравнивая ключевые метрики и визуализируя
    результаты.

    Attributes: visualizers (List[TumorDataVisualizer]): Список визуализаторов, каждый из которых представляет собой
    эксперимент или группу экспериментов.

    Args: *visualizers (TumorDataVisualizer): Произвольное количество объектов TumorDataVisualizer, каждый из которых
    представляет данные одного эксперимента.
    """

    def __init__(self, *visualizers: TumorDataVisualizer):
        self.visualizers = visualizers
        self._perform_stat_test = False  # Значение по умолчанию
        self._annotation_multiplier = 0
        self._use_ttest = False
        self._use_AUC = False

    @property
    def perform_stat_test(self):
        return self._perform_stat_test

    @perform_stat_test.setter
    def perform_stat_test(self, value: bool):
        self._perform_stat_test = value

    @property
    def annotation_multiplier(self):
        return self._annotation_multiplier

    @annotation_multiplier.setter
    def annotation_multiplier(self, value: int):
        self._annotation_multiplier = value

    @property
    def use_ttest(self):
        return self._use_ttest

    @use_ttest.setter
    def use_ttest(self, value: int):
        self._use_ttest = value

    @property
    def use_AUC(self):
        return self._use_AUC

    @use_AUC.setter
    def use_AUC(self, value: int):
        self._use_AUC = value

    def compare_mean_volumes(self):
        """
        Сравнивает средние абсолютные объемы опухолей для всех экспериментов и визуализирует результаты на графике.

        Этот метод агрегирует данные о средних объемах опухолей из всех предоставленных экспериментов, нормализует
        временные данные и строит обобщённый график сравнения, чтобы понять общие тенденции и различия между
        экспериментальными группами.

        Применяется нормализация временных данных для обеспечения корректного сравнения между экспериментами,
        которые могли начинаться в разные моменты времени или иметь различную продолжительность.

        Использует внешнюю функцию `normalize_time_data_min` для нормализации временных данных и
        `prepare_and_add_data_to_graph` для подготовки и добавления данных на график. Результатом является график с
        линиями, каждая из которых представляет средний объем опухоли по времени для каждого эксперимента.

        Args:
            Нет аргументов.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график.

        Использует:
            - `normalize_time_data_min(self.visualizers)` для нормализации временных данных всех экспериментов.
            - `GraphVisualizer` для создания и настройки объекта визуализации графика.
            - `prepare_and_add_data_to_graph` для добавления данных о средних объемах опухолей на график.
            - `finalize_figure` для финализации и отображения графика.
        """
        SupportingFunctions.normalize_time_data_min(self.visualizers)
        drawgraph = GraphVisualizer("Сравнение среднего объема опухолей", "Время, сут.",
                                    "Средний объем опухоли, абс. ед.")
        drawgraph.setup_figure()
        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            lambda visualizer: visualizer.data_processor.get_mean_tumor_volumes(),
            drawgraph,
            "M/V абс.: ",
            self._use_AUC,
            self.perform_stat_test,
            [self.visualizers[0], self.visualizers[1]],
            'up',
            self.annotation_multiplier,
            self.use_ttest
            )

        drawgraph.finalize_figure('')

    def compare_relative_volumes(self):
        """
        Сравнивает средние относительные объемы опухолей для всех экспериментов и визуализирует результаты на графике.

        Этот метод агрегирует данные о средних относительных объемах опухолей из всех предоставленных экспериментов,
        нормализует временные данные и строит обобщенный график сравнения. Целью является понимание общих тенденций и
        различий между экспериментальными группами в контексте изменения объема опухолей относительно их начального
        размера.

        Для обеспечения корректного сравнения временные данные нормализуются, что позволяет сравнивать эксперименты с
        различной продолжительностью и начальным временем. Относительный объем опухоли выражается как отношение текущего
        объема к начальному, что позволяет оценить динамику роста или уменьшения опухоли.

        Args:
            Нет аргументов.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график.

        Использует:
            - `normalize_time_data_min(self.visualizers)` для нормализации временных данных всех экспериментов.
            - `GraphVisualizer` для создания и настройки объекта визуализации графика.
            - `prepare_and_add_data_to_graph` для добавления данных о средних относительных объемах опухолей на график.
            - `finalize_figure` для финализации и отображения графика.

        Примечание: - В текущей реализации закомментированная строка `drawgraph.add_legend(time_labels, "Интервалы
        между облучениями", "lower right")` предполагает возможность добавления легенды с интервалами между
        облучениями. Эту возможность можно восстановить или модифицировать в соответствии с требованиями к визуализации.
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
            "",
            self._use_AUC,
            self.perform_stat_test,
            [self.visualizers[0], self.visualizers[1]],
            'down',
            self.annotation_multiplier,
            self.use_ttest
        )
        # Добавляем легенду с интервалами (при необходимости)
        time_labels = [f"Интервал: {interval}" for interval in time_intervals]
        # drawgraph.add_legend(time_labels, "Интервалы между облучениями", "lower right")

        drawgraph.finalize_figure('')

    def compare_control_and_experiment(self, control_visualizers: List[TumorDataVisualizer]):
        """
        Сравнивает средние относительные объемы опухолей между контрольными и экспериментальными группами.

        Этот метод анализирует и сравнивает динамику изменения объема опухолей в контрольных и экспериментальных группах
        на протяжении всего эксперимента. Относительные объемы опухолей вычисляются как отношение текущего объема к
        начальному, что позволяет оценить эффективность терапии или воздействия в экспериментальной группе по сравнению
        с контролем.

        Метод нормализует временные данные, чтобы обеспечить корректное сравнение экспериментов с различными начальными
        условиями и продолжительностью. Затем строит совместный график, отображающий динамику изменения относительных
        объемов опухолей во времени для обеих групп, позволяя визуально оценить различия между ними.

        Args: control_visualizers (List[TumorDataVisualizer]): Список объектов `TumorDataVisualizer` для контрольных
        групп.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график сравнения.

        Использует: - `normalize_time_data_min` для нормализации временных данных всех экспериментов. -
        `GraphVisualizer` для создания объекта визуализации и настройки параметров графика. -
        `prepare_and_add_data_to_graph` для агрегации и добавления данных о средних относительных объемах на график.
        - `finalize_figure` для финализации графика и его отображения.

        Примечание:
            - Важно правильно выбрать и подготовить контрольные группы, чтобы сравнение было корректным и показательным.
            - Метод позволяет визуализировать влияние экспериментальных условий на динамику роста опухолей.
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
            self._use_AUC,  # Указываем, что нужно рассчитать AUC
        )

        # Добавляем данные экспериментальных групп
        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            value_extractor,
            drawgraph,
            "Эксперимент: ",
            self._use_AUC,  # Указываем, что нужно рассчитать AUC
            self.perform_stat_test,
            [self.visualizers[0], self.visualizers[1]],
            # TODO: Необходимо предусмотреть, что группа может быть одна
            'up',
            self.annotation_multiplier,
            self.use_ttest
        )

        drawgraph.finalize_figure('')

    def compare_tumor_growth_inhibition_with_multiple_experiments(self, control_visualizer: TumorDataVisualizer,
                                                                  experiment_visualizers: List[TumorDataVisualizer]):
        """
        Сравнивает торможение роста опухоли между одной контрольной и несколькими экспериментальными группами.

        Метод вычисляет и сравнивает процент торможения роста опухоли между контрольной группой и несколькими
        экспериментальными группами на протяжении времени эксперимента. Торможение роста опухоли выражается как
        процентное снижение объема опухоли в экспериментальной группе по сравнению с контрольной группой, что
        позволяет оценить эффективность экспериментального воздействия.

        Args:
            control_visualizer (TumorDataVisualizer): Визуализатор для контрольной группы.
            experiment_visualizers (List[TumorDataVisualizer]): Список визуализаторов для экспериментальных групп.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график сравнения.

        Использует:
            - `normalize_time_data_min` для нормализации временных данных всех экспериментов.
            - `GraphVisualizer` для создания объекта визуализации и настройки параметров графика.
            - `calculate_tumor_growth_inhibition` для вычисления процентного торможения роста опухоли.
            - `finalize_figure` для финализации графика и его отображения.

        Примечание:
            - Важно выбрать адекватную контрольную группу для корректного сравнения и интерпретации результатов.
            - Этот метод позволяет исследователям визуально сравнивать эффективность различных экспериментальных
              условий или терапий на основе их способности тормозить рост опухолей.
        """
        SupportingFunctions.normalize_time_data_min([control_visualizer] + experiment_visualizers)
        # Обрезка данных до общей минимальной длины
        SupportingFunctions.trim_data_to_common_length([control_visualizer] + experiment_visualizers)
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
                               experiment_visualizer.experiment_params, label, self._use_AUC)

            x_data_lists.append(experiment_visualizer.time_data)  # Добавляем данные по оси X для каждого визуализатора

        drawgraph.update_axes_limits(x_data_lists)
        drawgraph.finalize_figure('')

    def create_tumor_growth_inhibition_table(self, control_visualizer, experiment_visualizers):
        """
        Создает таблицу сравнения торможения роста опухоли между контрольной и экспериментальными группами.

        Метод сначала нормализует временные данные, обрезает их до минимальной длины временного ряда среди всех
        визуализаторов, и затем вычисляет процент торможения роста опухоли (ТРО) для каждой экспериментальной группы
        по сравнению с контрольной группой. Результаты представляются в виде таблицы, где также рассчитывается и
        отображается абсолютная и относительная разница в эффективности ТРО между группами начиная с 9-го дня.

        Args:
            control_visualizer (TumorDataVisualizer): Визуализатор данных для контрольной группы.
            experiment_visualizers (List[TumorDataVisualizer]): Список визуализаторов данных для экспериментальных групп.

        Returns:
            None: Функция не возвращает значения, но выводит таблицу с результатами сравнения и аналитические
                  показатели по ней в консоль.

        Использует:
            - `normalize_time_data_min` для нормализации временных данных.
            - `calculate_tumor_growth_inhibition` для расчета торможения роста опухоли.
            - `pd.DataFrame` для создания и обработки таблицы результатов.
            - `pd.set_option` для настройки отображения таблицы в консоли.

        Примечание:
            - Метод важен для количественной оценки и сравнения эффективности различных терапевтических подходов
              в контексте торможения роста опухолей.
            - Анализ абсолютной и относительной разницы в эффективности ТРО помогает глубже понять степень
              влияния экспериментальных условий на динамику роста опухолей.
        """
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

        # # Печать отфильтрованной таблицы с использованием to_string()
        # print(df_filtered.to_string())
        #
        # # Расчет среднего абсолютного значения относительных различий
        # average_absolute_relative_difference = df_filtered['Relative Difference (%)'].abs().mean()
        # print(
        #     f"Среднее абсолютное значение относительного различия начиная с 9-го дня: {average_absolute_relative_difference:.2f}%")

        return df_filtered


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
        #r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_12.xlsx',
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_18.xlsx',
        #r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\16.03.2023_n_22.xlsx'
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
    #comparator.compare_control_and_experiment(control_visualizers)

    # Сравнение торможения роста опухоли между контрольной и несколькими экспериментальными группами
    comparator.compare_tumor_growth_inhibition_with_multiple_experiments(control_visualizer, experiment_visualizers)
    comparator.create_tumor_growth_inhibition_table(control_visualizer, experiment_visualizers)
