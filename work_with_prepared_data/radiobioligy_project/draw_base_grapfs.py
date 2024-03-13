# файл draw_base_grapfs.py
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting_helpers import custom_fill_between, subscriptify, format_experiment_params
from utils.plot_saver import save_plot
from stats_methods.support_stats_methods import SupportingFunctions
from data_processing.excel_data_processor import process_tumor_data_excel
from work_with_prepared_data.radiobioligy_project.utils.visualizer import GraphVisualizer

# Сохраняем оригинальную функцию в другой переменной, на случай, если она понадобится
original_fill_between = plt.fill_between

# Переопределяем функцию
plt.fill_between = custom_fill_between

# Глобальное изменение размеров шрифтов
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 25
})


class TumorDataVisualizer:
    def __init__(self, file_path: str):
        """
        Инициализация визуализатора данных опухоли.

        Параметры:
            file_path (str): Путь к файлу Excel с данными.
        """
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.tumor_volumes = process_tumor_data_excel(file_path)

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
        drawgraph.add_individual_plots(self.rat_labels, self.get_relative_tumor_volumes(), self.time_data)
        formatted_params = format_experiment_params(self.experiment_params)
        # Добавление легенды с параметрами эксперимента
        drawgraph.add_legend([formatted_params], "Параметры эксперимента", "upper center", display_marker=False)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_relative_volumes", 'Метка крысы')

    def plot_mean_tumor_volume(self):
        """
        Построение графика среднего объема опухоли со всеми крысами.
        """
        plt.figure(figsize=(12, 7))
        self.time_data = [float(x) for x in self.time_data]

        # Используем функцию для форматирования параметров эксперимента
        formatted_params = format_experiment_params(self.experiment_params)

        plt.title("(M/V абс.)", fontsize=24)

        mean_volumes = self.get_mean_tumor_volumes()
        std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                   zip(np.transpose(self.tumor_volumes), mean_volumes)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.tumor_volumes)) for std in std_dev]

        marker = 'o'
        marker_size = 12  # Установка размера маркера

        plt.plot(self.time_data, mean_volumes, marker=marker, markersize=marker_size, linestyle='-', color='b',
                 label='M/V абс.')
        plt.fill_between(self.time_data, mean_volumes - error_margin, mean_volumes + error_margin, color='b', alpha=0.2)

        plt.xticks(np.arange(min(self.time_data), max(self.time_data) + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Объем опухоли", fontsize=24)
        plt.grid(True)

        # Создание "пустых" линий для легенды с параметрами эксперимента
        custom_lines = [plt.Line2D([0], [0], color="none", marker="None", label=formatted_params)]
        first_legend = plt.legend(handles=custom_lines, loc='upper left', fontsize=24, handlelength=0, handletextpad=0)
        plt.gca().add_artist(first_legend)  # Добавляем первую легенду обратно на график

        # Основная легенда с меткой "M/V абс."
        plt.legend(fontsize=25, loc='lower right')
        plt.tight_layout()

        save_plot(f"{', '.join(self.experiment_params)}_mean_volumes", "mean_volume")
        plt.show()

    def plot_average_relative_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли со всеми крысами.
        """
        plt.figure(figsize=(12, 7))
        self.time_data = [float(x) for x in self.time_data]

        # Используем функцию для форматирования параметров эксперимента
        formatted_params = format_experiment_params(self.experiment_params)

        plt.title("(V отн.)", fontsize=24)

        relative_tumor_volumes = self.get_relative_tumor_volumes()
        mean_relative_volumes = np.nanmean(relative_tumor_volumes, axis=0)
        std_dev_rel = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                       zip(np.transpose(relative_tumor_volumes), mean_relative_volumes)]
        error_margin_rel = [SupportingFunctions.calculate_error_margin(std, len(relative_tumor_volumes)) for std in
                            std_dev_rel]

        marker = 'o'
        marker_size = 12

        plt.plot(self.time_data, mean_relative_volumes, marker=marker, markersize=marker_size, linestyle='-', color='b',
                 label='M/V отн.')
        plt.fill_between(self.time_data, mean_relative_volumes - error_margin_rel,
                         mean_relative_volumes + error_margin_rel, color='b', alpha=0.2)

        plt.xticks(np.arange(min(self.time_data), max(self.time_data) + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Объем опухоли", fontsize=24)
        plt.grid(True)

        # Создание "пустых" линий для легенды с параметрами эксперимента
        custom_lines = [plt.Line2D([0], [0], color="none", marker="None", label=formatted_params)]
        first_legend = plt.legend(handles=custom_lines, loc='upper left', fontsize=24,
                                  handlelength=0, handletextpad=0)
        plt.gca().add_artist(first_legend)

        # Основная легенда с меткой "M/V отн."
        plt.legend(fontsize=25, loc='lower right')
        plt.tight_layout()

        save_plot(f"{', '.join(self.experiment_params)}_average_relative_volumes", "mean_relative_volume")
        plt.show()

    def plot_mean_relative_mean_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли, усредненного по всем крысам.
        """
        plt.figure(figsize=(12, 7))
        self.time_data = [float(x) for x in self.time_data]

        # Используем функцию для форматирования параметров эксперимента
        formatted_params = format_experiment_params(self.experiment_params)

        # plt.title("Средний относительный объем опухоли (V отн. ср.)", fontsize=24)

        # Вычисление среднего относительного объема опухоли
        relative_mean_volumes = self.get_mean_relative_tumor_volumes()

        # Расчет стандартного отклонения и доверительного интервала
        std_dev_rel_mean = SupportingFunctions.calculate_std_dev(relative_mean_volumes,
                                                                 np.nanmean(relative_mean_volumes))
        error_margin_rel_mean = SupportingFunctions.calculate_error_margin(std_dev_rel_mean, len(relative_mean_volumes))

        marker = 'o'
        marker_size = 12

        plot_line = plt.plot(self.time_data, relative_mean_volumes, marker=marker, markersize=marker_size,
                             linestyle='-', color='b', label='M/V отн. ср.')
        plt.fill_between(self.time_data, relative_mean_volumes - error_margin_rel_mean,
                         relative_mean_volumes + error_margin_rel_mean, color='b', alpha=0.2)

        plt.xticks(np.arange(min(self.time_data), max(self.time_data) + 1, 3), fontsize=20)
        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Объем опухоли", fontsize=24)
        plt.grid(True)

        # Создание "пустых" линий для легенды с параметрами эксперимента
        custom_lines = [plt.Line2D([0], [0], color="none", marker="None", label=formatted_params)]
        first_legend = plt.legend(handles=custom_lines, loc='upper left', fontsize=24, handlelength=0, handletextpad=0)
        plt.gca().add_artist(first_legend)  # Добавляем первую легенду обратно на график

        # Основная легенда с меткой "M/V отн. ср.", созданная после первой легенды
        plt.legend(handles=plot_line, fontsize=25, loc='lower right')  # Исправление: handles должно быть списком

        plt.tight_layout()
        save_plot(f"{', '.join(self.experiment_params)}_mean_relative_mean_volumes", "mean_relative_mean_volume")
        plt.show()

    def get_mean_tumor_volumes(self) -> np.ndarray:
        """
        Вычисляет средний объем опухоли для всех крыс на каждом временном интервале.

        Возвращает:
            np.ndarray: Массив средних объемов опухоли.
        """
        return np.nanmean(self.tumor_volumes, axis=0)

    def get_relative_tumor_volumes(self) -> np.ndarray:
        """
        Вычисляет относительные объемы опухолей для каждой крысы.

        Возвращает:
            np.ndarray: Массив относительных объемов опухолей.
        """
        return np.array([[vol / volumes[0] for vol in volumes] for volumes in self.tumor_volumes])

    def get_mean_relative_tumor_volumes(self) -> np.ndarray:
        """
        Вычисляет средний относительный усреднённый объем опухоли для всех крыс.

        Возвращает:
            np.ndarray: Массив средних относительных объемов опухоли.
        """
        # Получение средних объемов опухоли
        mean_volumes = self.get_mean_tumor_volumes()

        # Вычисление среднего относительного объема опухоли
        mean_rel_volumes = mean_volumes / mean_volumes[0]

        return mean_rel_volumes


if __name__ == '__main__':
    # Используем с файлом данных
    file_path = r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\16.03.2023_n_22.xlsx'

    visualizer = TumorDataVisualizer(file_path)
    # ExtractOutliers(visualizer).exclude_rats(['пл', 'г'], 'tumor_volumes')  # for p_25.2_n_7.2_2023.xlsx
    # ExtractOutliers(visualizer).exclude_rats(['г- пл'], 'tumor_volumes')  # for n_7.2_p_25.2_2023_2.xlsx

    # Сохраняем график для каждой крысы
    #visualizer.plot_tumor_volumes_single_graph()

    # Сохраняем график относительных объемов для каждой крысы
    visualizer.plot_relative_tumor_volumes_single_graph()

    # Сохраняем график средних значений
    #visualizer.plot_mean_tumor_volume()

    # Сохраняем график среднего относительного объема опухоли
    #visualizer.plot_average_relative_tumor_volume()

    # Сохраняем график среднего относительного усреднённого объема опухоли
    #visualizer.plot_mean_relative_mean_tumor_volume()
