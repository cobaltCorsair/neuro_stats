# файл skin_reactions_base_grapf.py

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from utils.plotting_helpers import custom_fill_between, subscriptify, format_experiment_params
from utils.plot_saver import save_plot
from stats_methods.support_stats_methods import SupportingFunctions
from data_processing.excel_data_processor import process_skin_data_excel
from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer

# Сохраняем оригинальную функцию в другой переменной, на случай, если она понадобится
original_fill_between = plt.fill_between

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


class SkinReactionsVisualizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.skin_reactions = process_skin_data_excel(file_path)

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

        # Устанавливаем тики по оси X с шагом в 3 дня и поворачиваем их на 45 градусов
        if drawgraph.max_x is not None:
            plt.xticks(ticks=np.arange(0, int(drawgraph.max_x) + 1, 3), rotation=45)

        drawgraph.finalize_figure(self.file_path, 'Метки крыс', 2, 25)

    def plot_mean_skin_reactions(self):
        # Преобразование self.time_data в числовые значения
        self.time_data = [float(x) for x in self.time_data]
        plt.figure(figsize=(15, 8))
        formatted_params = format_experiment_params(self.experiment_params)
        plt.title(f"Средние кожные реакции, Параметры эксперимента: {formatted_params}", fontsize=24, y=1.02)

        mean_reactions, std_dev, error_margin = self.get_mean_skin_reactions()

        # Используем первый маркер из списка для единообразия
        marker = 'o'
        marker_size = 12  # Установка размера маркера

        # Отрисовка линии и сохранение её цвета
        line, = plt.plot(self.time_data, mean_reactions, marker=marker, markersize=marker_size, linestyle='-',
                         label='Среднее')
        line_color = line.get_color()  # Получение цвета линии

        # Использование цвета линии для доверительных интервалов
        plt.fill_between(self.time_data, mean_reactions - error_margin, mean_reactions + error_margin, color=line_color,
                         alpha=0.2)

        # Настройка тиков оси X для отображения каждые 3 дня
        min_day = min(self.time_data)
        max_day = max(self.time_data)
        plt.xticks(np.arange(min_day, max_day + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Средние кожные реакции", fontsize=24)
        plt.grid(True)
        plt.legend(fontsize=25)
        plt.tight_layout()
        save_plot('', f"Mean_Skin_Reactions_{formatted_params}")
        plt.show()

    def get_mean_skin_reactions(self):
        mean_reactions = np.nanmean(self.skin_reactions, axis=0)
        std_dev = [SupportingFunctions.calculate_std_dev(values, mean_value) for values, mean_value in zip(np.transpose(self.skin_reactions), mean_reactions)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.skin_reactions)) for std in std_dev]
        return mean_reactions, np.array(std_dev), np.array(error_margin)

    @staticmethod
    def plot_multiple_experiments(file_paths: List[str]):
        plt.figure(figsize=(12, 7))
        common_timepoints = list(range(0, 25))
        aucs = []
        lines = []
        time_ = []

        # Список маркеров
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        marker_size = 12  # Установка размера маркера

        for file_path, marker in zip(file_paths, markers):
            visualizer = SkinReactionsVisualizer(file_path)
            mean_reactions, std_dev, _ = visualizer.get_mean_skin_reactions()
            mean_reactions_interp = SupportingFunctions.interpolate_data_to_common_timepoints(
                visualizer.time_data, mean_reactions, common_timepoints
            )
            std_dev_interp = SupportingFunctions.interpolate_data_to_common_timepoints(
                visualizer.time_data, std_dev, common_timepoints  # Предполагаемая интерполяция стандартного отклонения
            )
            error_margin = std_dev_interp / np.sqrt(len(file_paths))  # Предполагаемый расчет доверительного интервала

            auc = SupportingFunctions.calculate_auc(common_timepoints, mean_reactions_interp)
            aucs.append(auc)
            label = format_experiment_params(visualizer.experiment_params)
            time_.append(visualizer.experiment_params[-1])
            line, = plt.plot(common_timepoints,
                             mean_reactions_interp,
                             marker=marker,
                             linestyle='-',
                             markersize=marker_size,
                             label=label)
            lines.append(line)
            plt.fill_between(common_timepoints,
                             mean_reactions_interp - error_margin,
                             mean_reactions_interp + error_margin,
                             alpha=0.2, color=line.get_color())

        first_legend = plt.legend(title="", loc='lower right')
        plt.gca().add_artist(first_legend)
        labels = [f'{auc:.2e} тыс.' for auc in aucs]
        plt.xticks(np.arange(25)[::3], rotation=0)
        plt.xlabel("Время, сут.")
        plt.ylabel("Кожные реакции, усл. ед.")
        plt.grid(True)
        plt.tight_layout()

        # Вторая легенда с AUC
        auc_labels = [f"AUC: {auc:.2f}" for auc in aucs]
        plt.legend(lines, auc_labels, title="Площадь под кривой", loc='upper left')

        # Вторая легенда с интервалом облучения
        #time_labels = [f"{time1}" for time1 in time_]
        #plt.legend(lines, time_labels, title="Интервал между \n облучениями", loc='upper left')

        # Сбор частей имен файлов
        file_name_parts = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]
        base_file_name = '_'.join(file_name_parts)

        # Использование статического метода для сохранения графика
        save_plot(base_file_name, 'multiple_experiments')

        plt.xlim(left=0)  # Установка минимального значения для оси X равным 0
        plt.ylim(bottom=0)  # Установка минимального значения для оси Y равным 0
        plt.show()

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
    # visualizer.plot_mean_skin_reactions()  # Визуализация средних кожных реакций

    # Отображения средних кожных реакций для нескольких экспериментов
    file_paths = [
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_2.xlsx',
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_3.xlsx',
    ]
    SkinReactionsVisualizer.plot_multiple_experiments(file_paths)
