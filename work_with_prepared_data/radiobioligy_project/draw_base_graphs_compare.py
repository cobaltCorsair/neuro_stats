# файл draw_base_graphs_compare.py

import matplotlib.pyplot as plt
from draw_base_graphs import TumorDataVisualizer
from utils.plotting_helpers import custom_fill_between, format_experiment_params, MatplotlibConfigurator
from stats_methods.support_stats_methods import SupportingFunctions
from utils.visualizer import GraphVisualizer

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
configurator.restore_original_styles()


class TumorDataComparator:
    def __init__(self, *visualizers):
        """
        Инициализирует объект для сравнения данных о опухолях между несколькими экспериментами.

        Этот конструктор принимает переменное количество аргументов, каждый из которых является объектом TumorDataVisualizer,
        содержащим данные одного эксперимента. Позволяет сравнивать данные, полученные в различных экспериментальных условиях.

        Args:
            *visualizers (TumorDataVisualizer): Произвольное количество объектов TumorDataVisualizer,
                                                 каждый из которых представляет данные одного эксперимента.

        Пример использования:
            >>> visualizer1 = TumorDataVisualizer('experiment1.xlsx')
            >>> visualizer2 = TumorDataVisualizer('experiment2.xlsx')
            >>> comparator = TumorDataComparator(visualizer1, visualizer2)

        В результате создается экземпляр компаратора, который может быть использован для сравнения данных,
        предоставленных визуализаторами, например, для построения сравнительных графиков.
        """
        self.visualizers = visualizers

    def compare_tumor_volumes(self):
        """
        Сравнивает абсолютные объемы опухолей между экспериментами и строит соответствующий график.

        Этот метод нормализует временные данные всех экспериментов, чтобы выровнять начальные точки,
        и строит график, демонстрирующий изменения объемов опухолей во времени для каждой крысы в каждом эксперименте.
        На графике для каждой крысы используется уникальная комбинация метки и стиля линии,
        позволяющая визуально сравнивать результаты между экспериментальными группами.

        Данные для графика извлекаются из атрибутов `tumor_volumes` и `time_data` объектов `TumorDataVisualizer`,
        переданных в конструктор `TumorDataComparator`.

        Args:
            Нет аргументов.

        Примеры:
            Допустим, есть два эксперимента с различными параметрами обработки. После инициализации `TumorDataComparator`
            с визуализаторами этих экспериментов, вызов `compare_tumor_volumes` построит сравнительный график,
            показывающий, как обработка влияет на рост опухоли.

        Особенности:
            - График включает в себя данные всех крыс из всех переданных экспериментов.
            - Не требует предварительной обработки данных пользователями.
            - Автоматически нормализует временные метки для согласованности представления данных.
        """
        # Нормализация временных меток всех визуализаторов
        SupportingFunctions.normalize_time_data(self.visualizers)

        drawgraph = GraphVisualizer("Сравнение экспериментов", "Время, сут.)", "Объем опухоли, абс. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        for visualizer in self.visualizers:
            formatted_params = format_experiment_params(visualizer.experiment_params)
            for label, volumes in zip(visualizer.rat_labels, visualizer.tumor_volumes):
                # Создаем полный label, включающий параметры эксперимента и метку крысы
                full_label = f"{formatted_params}: {label}"
                # Нет необходимости в error_margin и calculate_auc для этого графика
                drawgraph.add_plot(visualizer.time_data, volumes, {}, full_label)

        # Можем добавить дополнительную легенду, если нужно. В этом случае просто используем finalize_figure
        drawgraph.finalize_figure("compare_tumor_volumes", 'Метка крысы', 2, 18)

    def compare_relative_tumor_volumes(self):
        """
          Сравнивает относительные объемы опухолей между экспериментами и строит соответствующий график.

          Этот метод анализирует изменения объема опухолей в относительных единицах, позволяя сравнивать динамику роста
          или уменьшения опухолей между различными экспериментальными условиями независимо от исходного размера опухоли.
          График иллюстрирует, как различные обработки влияют на прогрессирование болезни во времени.

          Args:
            Нет аргументов.

          Примеры:
              Если эксперименты включают разные методы лечения, этот график поможет визуализировать,
              какой метод более эффективен в снижении относительного объема опухоли со временем.

          Особенности:
              - Визуализация данных в относительных единицах облегчает сравнение между группами с различным начальным объемом опухоли.
              - Нормализует временные метки для всех экспериментов, улучшая сопоставимость данных.
              - Поддерживает визуализацию множественных экспериментов на одном графике для удобства сравнения.
          """
        # Нормализация временных меток всех визуализаторов
        SupportingFunctions.normalize_time_data(self.visualizers)

        drawgraph = GraphVisualizer("Сравнение относительных объемов опухолей", "Время, сут.",
                                    "Объем опухоли, отн. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        for visualizer in self.visualizers:
            formatted_params = format_experiment_params(visualizer.experiment_params)
            relative_volumes = visualizer.data_processor.get_relative_tumor_volumes()
            for label, volumes in zip(visualizer.rat_labels, relative_volumes):
                # Создаем полный label, включающий параметры эксперимента и метку крысы
                full_label = f"{formatted_params}: {label}"
                # Нет необходимости в error_margin и calculate_auc для этого графика
                drawgraph.add_plot(visualizer.time_data, volumes, {}, full_label)

        # Можем добавить дополнительную легенду, если нужно. В этом случае просто используем finalize_figure
        drawgraph.finalize_figure("compare_relative_tumor_volumes", 'Метка крысы', 2, 18)


if __name__ == "__main__":
    # Используем с файлом данных
    file_path1 = r'C:\dev\neuro_stats\work_with_prepared_data\datas\y_32_2023.xlsx'
    file_path2 = r'C:\dev\neuro_stats\work_with_prepared_data\datas\y_36_2023.xlsx'

    # Используем с файлом данных
    # file_path1 = './datas/control/02.02.2023_n_12.xlsx'
    # file_path2 = './datas/control/02.02.2023_n_18.xlsx'
    # file_path3 = './datas/control/16.03.2023_n_22.xlsx'

    # Создаем объекты визуализатора для каждого файла данных
    #visualizers = [TumorDataVisualizer(file_path) for file_path in [file_path1, file_path2, file_path3]]

    visualizers = [TumorDataVisualizer(file_path) for file_path in [file_path1, file_path2]]

    # Создаем объект сравнителя и сравниваем данные из всех экспериментов
    comparator = TumorDataComparator(*visualizers)
    comparator.compare_tumor_volumes()  # Сравниваем абсолютные объемы
    comparator.compare_relative_tumor_volumes()  # Сравниваем относительные объемы
