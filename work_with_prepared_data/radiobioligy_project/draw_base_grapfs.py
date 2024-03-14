# файл draw_base_grapfs.py

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting_helpers import custom_fill_between, format_experiment_params, MatplotlibConfigurator
from stats_methods.support_stats_methods import SupportingFunctions
from data_processing.excel_data_processor import process_tumor_data_excel
from work_with_prepared_data.radiobioligy_project.data_processing.data_processing import TumorDataProcessor
from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
configurator.restore_original_styles()


class TumorDataVisualizer:
    def __init__(self, file_path: str):
        """
        Инициализация визуализатора данных опухоли.

        Параметры:
            file_path (str): Путь к файлу Excel с данными.
        """
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.tumor_volumes = process_tumor_data_excel(file_path)
        self.data_processor = TumorDataProcessor(self.tumor_volumes)  # Создаем экземпляр TumorDataProcessor

    def plot_tumor_volumes_single_graph(self):
        """
        Построение графика объемов опухолей для каждой крысы на одном графике.
        """
        drawgraph = GraphVisualizer("Абсолютные объемы опухоли", "Время, сут.", "Объем опухоли, абс. ед.",
                                    figsize=(12, 7))
        drawgraph.setup_figure()
        drawgraph.add_individual_plots(self.rat_labels, self.tumor_volumes, self.time_data)

        formatted_params = format_experiment_params(self.experiment_params)
        drawgraph.add_legend([formatted_params], "Параметры эксперимента", "upper center", display_marker=False)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_absolute_volumes", 'Метка крысы')

    def plot_relative_tumor_volumes_single_graph(self):
        """
        Построение графика относительных объемов опухолей для каждой крысы на одном графике.
        """
        drawgraph = GraphVisualizer("Относительные объемы опухоли", "Время, сут.", "Объем опухоли, отн. ед.",
                                    figsize=(12, 7))
        drawgraph.setup_figure()
        drawgraph.add_individual_plots(self.rat_labels, self.data_processor.get_relative_tumor_volumes(), self.time_data)
        formatted_params = format_experiment_params(self.experiment_params)
        # Добавление легенды с параметрами эксперимента
        drawgraph.add_legend([formatted_params], "Параметры эксперимента", "upper center", display_marker=False)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_relative_volumes", 'Метка крысы')

    def plot_mean_tumor_volume(self):
        """
        Построение графика среднего объема опухоли со всеми крысами.
        """
        drawgraph = GraphVisualizer("", "Время, сут.", "Объем опухоли, абс. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        mean_volumes = self.data_processor.get_mean_tumor_volumes()
        std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                   zip(np.transpose(self.tumor_volumes), mean_volumes)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.tumor_volumes)) for std in std_dev]

        # Добавление данных на график
        drawgraph.add_plot(self.time_data, mean_volumes, self.experiment_params, "M/V абс.: ", error_margin)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_mean_volumes")

    def plot_average_relative_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли со всеми крысами.
        """
        drawgraph = GraphVisualizer("(V отн.)", "Время, сут.", "Относительный объем опухоли, отн. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        relative_tumor_volumes = self.data_processor.get_relative_tumor_volumes()
        mean_relative_volumes = self.data_processor.get_mean_relative_tumor_volumes()
        std_dev_rel = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                       zip(np.transpose(relative_tumor_volumes), mean_relative_volumes)]
        error_margin_rel = [SupportingFunctions.calculate_error_margin(std, len(relative_tumor_volumes)) for std in
                            std_dev_rel]

        # Добавление данных на график
        drawgraph.add_plot(self.time_data, mean_relative_volumes, self.experiment_params, "M/V отн.: ",
                           error_margin_rel)

        formatted_params = format_experiment_params(self.experiment_params)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_average_relative_volumes")

    def plot_mean_relative_mean_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли, усредненного по всем крысам.
        """
        drawgraph = GraphVisualizer("Средний относительный объем опухоли (V отн. ср.)", "Время, сут.",
                                    "Относительный объем опухоли, отн. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        # Вычисление среднего относительного объема опухоли и его доверительного интервала
        relative_mean_volumes = self.data_processor.get_mean_relative_tumor_volumes()
        mean_volumes = self.data_processor.get_mean_tumor_volumes(relative_mean_volumes)
        std_dev_rel_mean = SupportingFunctions.calculate_std_dev(relative_mean_volumes, mean_volumes)
        error_margin_rel_mean = SupportingFunctions.calculate_error_margin(std_dev_rel_mean, len(relative_mean_volumes))

        # Добавление данных на график
        drawgraph.add_plot(self.time_data, relative_mean_volumes, self.experiment_params, "M/V отн. ср.: ",
                           error_margin_rel_mean)

        formatted_params = format_experiment_params(self.experiment_params)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_mean_relative_mean_volumes")



if __name__ == '__main__':
    # Используем с файлом данных
    file_path = r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\16.03.2023_n_22.xlsx'

    visualizer = TumorDataVisualizer(file_path)
    # ExtractOutliers(visualizer).exclude_rats(['пл', 'г'], 'tumor_volumes')  # for p_25.2_n_7.2_2023.xlsx
    # ExtractOutliers(visualizer).exclude_rats(['г- пл'], 'tumor_volumes')  # for n_7.2_p_25.2_2023_2.xlsx

    # Сохраняем график для каждой крысы
    visualizer.plot_tumor_volumes_single_graph()

    # Сохраняем график относительных объемов для каждой крысы
    visualizer.plot_relative_tumor_volumes_single_graph()

    # Сохраняем график средних значений
    visualizer.plot_mean_tumor_volume()

    # Сохраняем график среднего относительного объема опухоли
    visualizer.plot_average_relative_tumor_volume()

    # Сохраняем график среднего относительного усреднённого объема опухоли
    visualizer.plot_mean_relative_mean_tumor_volume()
