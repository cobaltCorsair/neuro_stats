import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from work_with_prepared_data.support_stats_methods import SupportingFunctions


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
        experiment_params = data.iloc[0, :3].tolist()
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
        plt.title(f"Абсолютные объемы опухоли, Параметры эксперимента: {', '.join(self.experiment_params)}",
                  fontsize=16, y=1.02)

        for label, volumes in zip(self.rat_labels, self.tumor_volumes):
            plt.plot(self.time_data, volumes, marker='o', linestyle='-', label=label)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        self.save_plot(f"{', '.join(self.experiment_params)}_absolute_volumes", "single_graph")
        plt.show()

    def plot_relative_tumor_volumes_single_graph(self):
        """
        Построение графика относительных объемов опухолей для каждой крысы на одном графике.
        """
        relative_volumes = self.get_relative_tumor_volumes()

        plt.figure(figsize=(15, 8))
        plt.title(f"Относительные объемы опухоли, Параметры эксперимента: {', '.join(self.experiment_params)}",
                  fontsize=16, y=1.02)

        for label, volumes in zip(self.rat_labels, relative_volumes):
            plt.plot(self.time_data, volumes, marker='o', linestyle='-', label=label)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        self.save_plot(f"{', '.join(self.experiment_params)}_relative_volumes", "single_graph_rel")
        plt.show()

    def plot_mean_tumor_volume(self):
        """
        Построение графика среднего объема опухоли со всеми крысами.
        """
        plt.figure(figsize=(15, 8))
        plt.title(f"(M/V абс.), Параметры эксперимента: {', '.join(self.experiment_params)}", fontsize=16)

        mean_volumes = self.get_mean_tumor_volumes()
        std_dev = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume) for volumes, mean_volume in
                   zip(np.transpose(self.tumor_volumes), mean_volumes)]
        error_margin = [SupportingFunctions.calculate_error_margin(self, std, len(self.tumor_volumes)) for std in
                        std_dev]

        plt.plot(self.time_data, mean_volumes, marker='o', linestyle='-', color='b', label='M/V абс.')
        plt.fill_between(self.time_data, mean_volumes - error_margin, mean_volumes + error_margin, color='b', alpha=0.2)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot(f"{', '.join(self.experiment_params)}_mean_volumes", "mean_volume")
        plt.show()

    def plot_average_relative_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли со всеми крысами.
        """
        plt.figure(figsize=(15, 8))
        plt.title(f"(V отн.), Параметры эксперимента: {', '.join(self.experiment_params)}", fontsize=16)

        relative_tumor_volumes = self.get_relative_tumor_volumes()
        mean_relative_volumes = np.nanmean(relative_tumor_volumes, axis=0)
        std_dev_rel = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume) for volumes, mean_volume in
                       zip(np.transpose(relative_tumor_volumes), mean_relative_volumes)]
        error_margin_rel = [SupportingFunctions.calculate_error_margin(self, std, len(relative_tumor_volumes)) for std
                            in std_dev_rel]

        plt.plot(self.time_data, mean_relative_volumes, marker='o', linestyle='-', color='b', label='M/V отн.')
        plt.fill_between(self.time_data, mean_relative_volumes - error_margin_rel,
                         mean_relative_volumes + error_margin_rel, color='b', alpha=0.2)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot(f"{', '.join(self.experiment_params)}_average_relative_volumes", "mean_relative_volume")
        plt.show()

    def plot_mean_relative_mean_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли, усредненного по всем крысам.
        """
        plt.figure(figsize=(15, 8))
        plt.title(f"(V отн. ср.), Параметры эксперимента: {', '.join(self.experiment_params)}", fontsize=16)

        # Вычисление среднего относительного объема опухоли
        relative_mean_volumes = self.get_mean_relative_tumor_volumes()

        # Расчет стандартного отклонения
        std_dev_rel_mean = SupportingFunctions.calculate_std_dev(self, relative_mean_volumes,
                                                                 np.nanmean(relative_mean_volumes))

        # Расчет доверительного интервала
        error_margin_rel_mean = SupportingFunctions.calculate_error_margin(self, std_dev_rel_mean,
                                                                           len(relative_mean_volumes))

        plt.plot(self.time_data, relative_mean_volumes, marker='o', linestyle='-', color='b', label='M/V отн. ср.')
        plt.fill_between(self.time_data,
                         relative_mean_volumes - error_margin_rel_mean,
                         relative_mean_volumes + error_margin_rel_mean, color='b', alpha=0.2)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot(f"{', '.join(self.experiment_params)}_mean_relative_mean_volumes", "mean_relative_mean_volume")
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


# Используем с файлом данных
file_path = './datas/n_7.2_p_25.2_2023.xlsx'
# file_path = './datas/p_25.2_n_7.2_2023.xlsx'
# file_path = './datas/p_25.2_n_7.2_2023_2.xlsx'
# file_path = './datas/n_7.2_p_25.2_2023_2.xlsx'
# file_path = './datas/n_2.56_p_25.6_2019.xlsx'
# file_path = './datas/p_25.6_n_2.56_2019.xlsx'

visualizer = TumorDataVisualizer(file_path)

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
