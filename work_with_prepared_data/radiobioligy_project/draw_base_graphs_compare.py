# файл draw_base_graphs_compare.py

import matplotlib.pyplot as plt
from draw_base_graphs import TumorDataVisualizer
from utils.plotting_helpers import custom_fill_between, format_experiment_params, MatplotlibConfigurator
from stats_methods.support_stats_methods import SupportingFunctions
from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
configurator.restore_original_styles()


class TumorDataComparator:
    def __init__(self, *visualizers):
        """
        Инициализатор класса сравнителя данных о опухолях.

        Parameters:
            visualizers (list of TumorDataVisualizer): Список визуализаторов данных экспериментов.
        """
        self.visualizers = visualizers

    def compare_tumor_volumes(self):
        """
        Сравнивает абсолютные объемы опухолей между экспериментами и строит соответствующий график.
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
