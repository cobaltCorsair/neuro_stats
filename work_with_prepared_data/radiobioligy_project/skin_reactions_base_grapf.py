# файл skin_reactions_base_grapf.py

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from utils.plotting_helpers import format_experiment_params, MatplotlibConfigurator, custom_fill_between
from stats_methods.support_stats_methods import SupportingFunctions
from data_processing.excel_data_processor import process_skin_data_excel
from work_with_prepared_data.radiobioligy_project.data_processing.data_processing import SkinReactionsDataProcessor
from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
configurator.restore_original_styles()

class SkinReactionsVisualizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.skin_reactions = process_skin_data_excel(
            file_path)
        self.data_processor = SkinReactionsDataProcessor(self.skin_reactions)

    def plot_skin_reactions(self):
        drawgraph = GraphVisualizer(
            f"Кожные реакции, Параметры эксперимента: {format_experiment_params(self.experiment_params)}",
            "Время, сут.",
            "Кожные реакции, абс. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        # Итерация по крысам и их кожным реакциям для добавления на график
        for label, reactions in zip(self.rat_labels, self.skin_reactions):
            clean_reactions = np.array(reactions)[~np.isnan(reactions)]
            clean_time_data = np.array(self.time_data)[~np.isnan(reactions)]

            if not list(clean_reactions):
                continue

            # Добавление данных на график
            drawgraph.add_plot(clean_time_data, clean_reactions, {}, label)
        drawgraph.finalize_figure(self.file_path, 'Метки крыс', 2, 25)

    def plot_mean_skin_reactions(self):
        drawgraph = GraphVisualizer(
            f"Средние кожные реакции, Параметры эксперимента: {format_experiment_params(self.experiment_params)}",
            "Время, сут.",
            "Средние кожные реакции, абс. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()
        # Получение средних кожных реакций и их статистических характеристик
        mean_reactions, std_dev, error_margin = self.data_processor.get_mean_skin_reactions()
        # Добавление данных на график
        drawgraph.add_plot(self.time_data, mean_reactions, self.experiment_params, "", error_margin)
        drawgraph.finalize_figure('', '', 1, 25)

    @staticmethod
    def plot_multiple_experiments(file_paths: List[str]):
        drawgraph = GraphVisualizer(
            "Сравнение кожных реакций между экспериментами",
            "Время, сут.",
            "Кожные реакции, усл. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()
        common_timepoints = list(range(0, 25))

        for file_path in file_paths:
            visualizer = SkinReactionsVisualizer(file_path)
            mean_reactions, std_dev, _ = visualizer.data_processor.get_mean_skin_reactions()

            interpolated_values = SupportingFunctions.interpolate_data_to_common_timepoints(visualizer.time_data,
                                                                                            mean_reactions,
                                                                                            common_timepoints)
            interpolated_std_dev = SupportingFunctions.interpolate_data_to_common_timepoints(visualizer.time_data,
                                                                                             std_dev,
                                                                                             common_timepoints)
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(file_paths)) for std in
                            interpolated_std_dev]
            label = format_experiment_params(visualizer.experiment_params)

            # Добавление данных на график с автоматическим выбором стиля линии и маркера
            drawgraph.add_plot(common_timepoints, interpolated_values, {}, label, error_margin, calculate_auc=True)

        base_file_name = '_'.join([os.path.splitext(os.path.basename(fp))[0] for fp in file_paths])
        drawgraph.finalize_figure(base_file_name, ncol=1, legend_fontsize='20')


if __name__ == '__main__':
    # Пример использования
    file_path = r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_3.xlsx'
    visualizer = SkinReactionsVisualizer(file_path)

    # Удаление выбросов
    # ExtractOutliers(visualizer).remove_local_outliers()

    # Удаление точек
    # ExtractOutliers(visualizer).exclude_rats(['б/м'], 'tumor_volumes')  # for skin_reactions_p_25,2_n_7,2_2023_2.xlsx
    # ExtractOutliers(visualizer).exclude_rats(['г', 'х'], 'tumor_volumes')  # for skin_reactions_n_7.2_p_25.2_2023_2.xlsx

    visualizer.plot_skin_reactions()  # Визуализация индивидуальных кожных реакций
    visualizer.plot_mean_skin_reactions()  # Визуализация средних кожных реакций

    # Отображения средних кожных реакций для нескольких экспериментов
    file_paths = [
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_2.xlsx',
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_3.xlsx',
    ]
    SkinReactionsVisualizer.plot_multiple_experiments(file_paths)
