import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from work_with_prepared_data.support_stats_methods import SupportingFunctions, ExtractOutliers

# Сохраняем оригинальную функцию в другой переменной, на случай, если она понадобится
original_fill_between = plt.fill_between


# Custom fill_between function
def custom_fill_between(x, y1, y2=0, color=None, alpha=None, **kwargs):
    horizontal_line_length = 0.2  # Длина горизонтальных линий на концах
    line_color = color if color is not None else 'blue'  # Используйте заданный цвет, если он предоставлен

    for xi, y1i, y2i in zip(x, y1, y2):
        # Вертикальные линии
        plt.plot([xi, xi], [y1i, y2i], color=line_color, alpha=1, zorder=1)

        # Горизонтальные линии на концах
        plt.plot([xi - horizontal_line_length / 2, xi + horizontal_line_length / 2], [y1i, y1i], color=line_color,
                 alpha=1, zorder=1)
        plt.plot([xi - horizontal_line_length / 2, xi + horizontal_line_length / 2], [y2i, y2i], color=line_color,
                 alpha=1, zorder=1)


# Overriding the function
plt.fill_between = custom_fill_between

# Global matplotlib parameters for consistent visual styling
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
        self.experiment_params, self.time_data, self.rat_labels, self.tumor_volumes = self.process_excel()

    def save_plot(self, plot_title: str, file_suffix: str):
        """
        Сохраняет текущий график в файл с заданным именем и суффиксом.

        Параметры:
            plot_title (str): Название графика, используемое для создания имени файла.
            file_suffix (str): Суффикс для имени файла для уточнения типа графика.

        Примечание:
            Имя файла формируется с использованием базового имени файла данных,
            plot_title и file_suffix.
        """
        # Извлечение имени файла без расширения и пути
        file_name_base = os.path.splitext(os.path.basename(self.file_path))[0]

        # Сборка окончательного имени файла
        file_name = f"{file_name_base}_{plot_title.replace(' ', '_')}_{file_suffix}.png"

        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
        """
        Обрабатывает данные из файла Excel и извлекает необходимые данные.

        Возвращает:
            tuple: Кортеж, содержащий:
                - experiment_params (List[str]): Параметры эксперимента.
                - time_data (List[str]): Метки времени для каждого измерения.
                - rat_labels (List[str]): Метки крыс.
                - tumor_volumes (List[List[float]]): Объемы опухолей для каждой крысы на каждом временном интервале.
        """
        data = pd.read_excel(self.file_path, header=None)
        experiment_params = data.iloc[0, :3].astype(str).replace('nan', '').tolist()
        tumor_data = data.iloc[2:, :].copy()
        time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in data.iloc[1, 1:]]

        tumor_data = tumor_data.applymap(
            lambda x: str(x).strip().replace(',', '.').replace(' -', '-') if pd.notna(x) else "NA")
        rat_labels = tumor_data.iloc[:, 0].tolist()

        tumor_volumes = []
        for _, row in tumor_data.iterrows():
            rat_volumes = []
            for item in row[1:]:
                if "-" in item:
                    a, b, c = map(float, item.split("-"))
                    volume = (np.pi * a * b * c) / 6
                elif item.replace(".", "").isdigit():
                    volume = float(item)
                else:
                    volume = np.nan
                rat_volumes.append(volume)
            tumor_volumes.append(rat_volumes)

        return experiment_params, time_data, rat_labels, tumor_volumes

    def plot_tumor_volumes_single_graph(self):
        """
        Построение графика объемов опухолей для каждой крысы на одном графике.
        """
        plt.figure(figsize=(15, 8))
        self.time_data = [float(x) for x in self.time_data]

        # Список маркеров
        markers = ['o', 's', '^', 'x', '*', 'D', 'h', '+', 'p']
        marker_index = 0
        marker_size = 12  # Установка размера маркера

        # Очистка списка experiment_params от пустых строк и строк, состоящих только из пробелов
        cleaned_experiment_params = [param for param in self.experiment_params if param.strip()]

        plt.title(f"Абсолютные объемы опухоли, Параметры эксперимента: {', '.join(cleaned_experiment_params)}",
                  fontsize=24, y=1.02)

        for label, volumes in zip(self.rat_labels, self.tumor_volumes):
            clean_volumes = np.array(volumes)[~np.isnan(volumes)]
            clean_time_data = np.array(self.time_data)[~np.isnan(volumes)]

            if not list(clean_volumes):
                continue

            mean_volume = np.mean(clean_volumes)
            std_dev = SupportingFunctions.calculate_std_dev(clean_volumes, mean_volume)
            error_margin = SupportingFunctions.calculate_error_margin(std_dev, len(clean_volumes))

            line, = plt.plot(clean_time_data, clean_volumes, marker=markers[marker_index % len(markers)],
                             markersize=marker_size, linestyle='-', label=label)
            marker_index += 1
            line_color = line.get_color()

            custom_fill_between(clean_time_data,
                                clean_volumes - error_margin,
                                clean_volumes + error_margin,
                                color=line_color, alpha=0.2)

        # Настройка тиков оси X
        min_day = min(self.time_data)
        max_day = max(self.time_data)
        plt.xticks(np.arange(min_day, max_day + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Объем опухоли", fontsize=24)
        plt.grid(True)
        plt.legend(title="Метка крысы", fontsize=25)
        plt.tight_layout()
        self.save_plot(f"{', '.join(self.experiment_params)}_absolute_volumes", "single_graph")
        plt.show()

    def plot_relative_tumor_volumes_single_graph(self):
        """
        Построение графика относительных объемов опухолей для каждой крысы на одном графике.
        """
        relative_volumes = self.get_relative_tumor_volumes()

        plt.figure(figsize=(15, 8))
        self.time_data = [float(x) for x in self.time_data]

        # Список маркеров
        markers = ['o', 's', '^', 'x', '*', 'D', 'h', '+', 'p']
        marker_index = 0
        marker_size = 12

        # Очистка списка experiment_params от пустых строк и строк, состоящих только из пробелов
        cleaned_experiment_params = [param for param in self.experiment_params if param.strip()]

        plt.title(f"Относительные объемы опухоли, Параметры эксперимента: {', '.join(cleaned_experiment_params)}",
                  fontsize=24, y=1.02)

        relative_volumes = self.get_relative_tumor_volumes()

        for label, volumes in zip(self.rat_labels, relative_volumes):
            clean_volumes = np.array(volumes)[~np.isnan(volumes)]
            clean_time_data = np.array(self.time_data)[~np.isnan(volumes)]

            if not list(clean_volumes):
                continue

            mean_volume = np.mean(clean_volumes)
            std_dev = SupportingFunctions.calculate_std_dev(clean_volumes, mean_volume)
            error_margin = SupportingFunctions.calculate_error_margin(std_dev, len(clean_volumes))

            line, = plt.plot(clean_time_data, clean_volumes, marker=markers[marker_index % len(markers)],
                             markersize=marker_size, linestyle='-', label=label)
            marker_index += 1  # Переход к следующему маркеру для следующей крысы
            line_color = line.get_color()

            # Настройка тиков оси X
            plt.fill_between(clean_time_data,
                             clean_volumes - error_margin,
                             clean_volumes + error_margin,
                             color=line_color, alpha=0.2)

        min_day = min(self.time_data)
        max_day = max(self.time_data)
        plt.xticks(np.arange(min_day, max_day + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Относительный объем опухоли", fontsize=24)
        plt.grid(True)
        plt.legend(title="Метка крысы", fontsize=25)
        plt.tight_layout()
        self.save_plot(f"{', '.join(self.experiment_params)}_relative_volumes", "single_graph_rel")
        plt.show()

    def plot_mean_tumor_volume(self):
        """
        Построение графика среднего объема опухоли со всеми крысами.
        """
        plt.figure(figsize=(15, 8))

        # Очистка списка experiment_params от пустых строк и строк, состоящих только из пробелов
        cleaned_experiment_params = [param for param in self.experiment_params if param.strip()]

        plt.title(f"(M/V абс.), Параметры эксперимента: {', '.join(cleaned_experiment_params)}", fontsize=24)

        mean_volumes = self.get_mean_tumor_volumes()
        std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                   zip(np.transpose(self.tumor_volumes), mean_volumes)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.tumor_volumes)) for std in std_dev]

        self.time_data = [float(x) for x in self.time_data]
        # Используем первый маркер из списка для единообразия
        marker = 'o'
        marker_size = 12  # Установка размера маркера

        plt.plot(self.time_data, mean_volumes, marker=marker, markersize=marker_size, linestyle='-', color='b',
                 label='M/V абс.')
        plt.fill_between(self.time_data, mean_volumes - error_margin, mean_volumes + error_margin, color='b', alpha=0.2)

        # Настройка тиков оси X с шагом в 3 дня
        min_day = min(self.time_data)
        max_day = max(self.time_data)
        plt.xticks(np.arange(min_day, max_day + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Объем опухоли", fontsize=24)
        plt.grid(True)
        plt.legend(fontsize=25)
        plt.tight_layout()
        self.save_plot(f"{', '.join(cleaned_experiment_params)}_mean_volumes", "mean_volume")
        plt.show()

    def plot_average_relative_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли со всеми крысами.
        """
        plt.figure(figsize=(15, 8))

        # Очистка списка experiment_params от пустых строк и строк, состоящих только из пробелов
        cleaned_experiment_params = [param for param in self.experiment_params if param.strip()]

        plt.title(f"(V отн.), Параметры эксперимента: {', '.join(cleaned_experiment_params)}", fontsize=24)

        relative_tumor_volumes = self.get_relative_tumor_volumes()
        mean_relative_volumes = np.nanmean(relative_tumor_volumes, axis=0)
        std_dev_rel = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                       zip(np.transpose(relative_tumor_volumes), mean_relative_volumes)]
        error_margin_rel = [SupportingFunctions.calculate_error_margin(std, len(relative_tumor_volumes)) for std in
                            std_dev_rel]

        self.time_data = [float(x) for x in self.time_data]
        # Используем первый маркер из списка для единообразия
        marker = 'o'
        marker_size = 12  # Установка размера маркера

        plt.plot(self.time_data, mean_relative_volumes, marker=marker, markersize=marker_size, linestyle='-', color='b',
                 label='M/V отн.')
        plt.fill_between(self.time_data, mean_relative_volumes - error_margin_rel,
                         mean_relative_volumes + error_margin_rel, color='b', alpha=0.2)

        # Настройка тиков оси X с шагом в 3 дня
        min_day = min(self.time_data)
        max_day = max(self.time_data)
        plt.xticks(np.arange(min_day, max_day + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Относительный объем опухоли", fontsize=24)
        plt.grid(True)
        plt.legend(fontsize=25)
        plt.tight_layout()
        self.save_plot(f"{', '.join(cleaned_experiment_params)}_average_relative_volumes", "mean_relative_volume")
        plt.show()

    def plot_mean_relative_mean_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли, усредненного по всем крысам.
        """
        plt.figure(figsize=(15, 8))

        # Очистка списка experiment_params от пустых строк и строк, состоящих только из пробелов
        cleaned_experiment_params = [param for param in self.experiment_params if param.strip()]

        plt.title(f"(V отн. ср.), Параметры эксперимента: {', '.join(cleaned_experiment_params)}", fontsize=24)

        # Вычисление среднего относительного объема опухоли
        relative_mean_volumes = self.get_mean_relative_tumor_volumes()

        # Расчет стандартного отклонения и доверительного интервала
        std_dev_rel_mean = SupportingFunctions.calculate_std_dev(relative_mean_volumes,
                                                                 np.nanmean(relative_mean_volumes))
        error_margin_rel_mean = SupportingFunctions.calculate_error_margin(std_dev_rel_mean, len(relative_mean_volumes))

        self.time_data = [float(x) for x in self.time_data]
        # Используем первый маркер из списка для единообразия
        marker = 'o'
        marker_size = 12  # Установка размера маркера

        plt.plot(self.time_data, relative_mean_volumes, marker=marker, markersize=marker_size, linestyle='-', color='b',
                 label='M/V отн. ср.')
        plt.fill_between(self.time_data,
                         relative_mean_volumes - error_margin_rel_mean,
                         relative_mean_volumes + error_margin_rel_mean, color='b', alpha=0.2)

        # Настройка тиков оси X с шагом в 3 дня
        min_day = min(self.time_data)
        max_day = max(self.time_data)
        plt.xticks(np.arange(min_day, max_day + 1, 3), fontsize=20)

        plt.xlabel("Время, сут.", fontsize=24)
        plt.ylabel("Относительный объем опухоли", fontsize=24)
        plt.grid(True)
        plt.legend(fontsize=25)
        plt.tight_layout()
        self.save_plot(f"{', '.join(cleaned_experiment_params)}_mean_relative_mean_volumes",
                       "mean_relative_mean_volume")
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
    # file_path = './datas/n_7.2_p_25.2_2023.xlsx'
    # file_path = './datas/p_25.2_n_7.2_2023.xlsx'
    # file_path = './datas/p_25.2_n_7.2_2023_2.xlsx'
    #file_path = './datas/n_7.2_p_25.2_2023_2.xlsx'
    # file_path = './datas/n_2.56_p_25.6_2019.xlsx'
    # file_path = './datas/p_25.6_n_2.56_2019.xlsx'
    #file_path = './datas/y_32_2023.xlsx'
    #file_path ='./datas/y_36_2023.xlsx'
    file_path = './datas/control/02.02.2023_n_12.xlsx'
    #file_path = './datas/control/02.02.2023_n_18.xlsx'
    #file_path = './datas/control/16.03.2023_n_22.xlsx'

    visualizer = TumorDataVisualizer(file_path)
    # ExtractOutliers(visualizer).exclude_rats(['пл', 'г'], 'tumor_volumes')  # for p_25.2_n_7.2_2023.xlsx
    ExtractOutliers(visualizer).exclude_rats(['х'], 'tumor_volumes')  # for n_7.2_p_25.2_2023_2.xlsx

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
