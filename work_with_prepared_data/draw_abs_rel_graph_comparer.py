import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from draw_base_grapfs import TumorDataVisualizer, SupportingFunctions


class TumorDataComparatorAdvanced:
    def __init__(self, visualizer1: TumorDataVisualizer, visualizer2: TumorDataVisualizer):
        self.visualizer1 = visualizer1
        self.visualizer2 = visualizer2

    def save_plot(self, comparison_type: str):
        # Извлечение имен файлов без расширения и пути
        file_name1 = os.path.splitext(os.path.basename(self.visualizer1.file_path))[0]
        file_name2 = os.path.splitext(os.path.basename(self.visualizer2.file_path))[0]

        # Сборка окончательного имени файла
        file_name = f"{comparison_type}_{file_name1}_vs_{file_name2}.png"

        # Сохранение изображения
        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def normalize_time_data(self):
        # Перевести временные метки в числовой формат и нормализовать их
        self.visualizer1.time_data = [int(time) - int(self.visualizer1.time_data[0]) for time in
                                      self.visualizer1.time_data]
        self.visualizer2.time_data = [int(time) - int(self.visualizer2.time_data[0]) for time in
                                      self.visualizer2.time_data]

    def compare_mean_volumes(self):
        self.normalize_time_data()
        plt.figure(figsize=(15, 8))

        # Для первого визуализатора
        mean_volumes1 = self.visualizer1.get_mean_tumor_volumes()
        std_dev1 = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume)
                    for volumes, mean_volume in zip(np.transpose(self.visualizer1.tumor_volumes), mean_volumes1)]
        error_margin1 = [SupportingFunctions.calculate_error_margin(self, std, len(self.visualizer1.tumor_volumes))
                         for std in std_dev1]

        plt.plot(self.visualizer1.time_data, mean_volumes1, marker='o', linestyle='-', label=f"Exp1: M/V абс.")
        plt.fill_between(self.visualizer1.time_data,
                         mean_volumes1 - error_margin1,
                         mean_volumes1 + error_margin1, color='b', alpha=0.2)

        # Для второго визуализатора
        mean_volumes2 = self.visualizer2.get_mean_tumor_volumes()
        std_dev2 = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume)
                    for volumes, mean_volume in zip(np.transpose(self.visualizer2.tumor_volumes), mean_volumes2)]
        error_margin2 = [SupportingFunctions.calculate_error_margin(self, std, len(self.visualizer2.tumor_volumes))
                         for std in std_dev2]

        plt.plot(self.visualizer2.time_data, mean_volumes2, marker='o', linestyle='-', label=f"Exp2: M/V абс.")
        plt.fill_between(self.visualizer2.time_data,
                         mean_volumes2 - error_margin2,
                         mean_volumes2 + error_margin2, color='g', alpha=0.2)

        plt.title(
            f"Сравнение среднего объема опухоли\nExp1: {', '.join(self.visualizer1.experiment_params)} vs Exp2: {', '.join(self.visualizer2.experiment_params)}")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot("compare_mean_volumes")
        plt.show()

    def compare_relative_volumes(self):
        self.normalize_time_data()
        plt.figure(figsize=(15, 8))

        # Для первого визуализатора
        mean_rel_volumes1 = self.visualizer1.get_mean_relative_tumor_volumes()
        std_dev1 = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume)
                    for volumes, mean_volume in zip(np.transpose(self.visualizer1.tumor_volumes), mean_rel_volumes1)]
        error_margin1 = [SupportingFunctions.calculate_error_margin(self, std, len(self.visualizer1.tumor_volumes))
                         for std in std_dev1]

        plt.plot(self.visualizer1.time_data, mean_rel_volumes1,
                 marker='o', linestyle='-', label=f"Exp1: M/V отн.")
        plt.fill_between(self.visualizer1.time_data,
                         mean_rel_volumes1 - error_margin1,
                         mean_rel_volumes1 + error_margin1, color='b', alpha=0.2)

        # Для второго визуализатора
        mean_rel_volumes2 = self.visualizer2.get_mean_relative_tumor_volumes()
        std_dev2 = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume)
                    for volumes, mean_volume in zip(np.transpose(self.visualizer2.tumor_volumes), mean_rel_volumes2)]
        error_margin2 = [SupportingFunctions.calculate_error_margin(self, std, len(self.visualizer2.tumor_volumes))
                         for std in std_dev2]

        plt.plot(self.visualizer2.time_data, mean_rel_volumes2,
                 marker='o', linestyle='--', label=f"Exp2: M/V отн.")
        plt.fill_between(self.visualizer2.time_data,
                         mean_rel_volumes2 - error_margin2,
                         mean_rel_volumes2 + error_margin2, color='g', alpha=0.2)

        plt.title(
            f"Сравнение среднего относительного объема опухоли\nExp1: {', '.join(self.visualizer1.experiment_params)} vs Exp2: {', '.join(self.visualizer2.experiment_params)}")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний относительный объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot("compare_relative_volumes")
        plt.show()

# # Используем с файлом данных
# file_path1 = './datas/n_7.2_p_25.2_2023_2.xlsx'
# file_path2 = './datas/p_25.2_n_7.2_2023_2.xlsx'

# Используем с файлом данных
# file_path1 = './datas/n_7.2_p_25.2_2023.xlsx'
# file_path2 = './datas/p_25.2_n_7.2_2023.xlsx'

# Используем с файлом данных
file_path1 = './datas/n_2.56_p_25.6_2019.xlsx'
file_path2 = './datas/p_25.6_n_2.56_2019.xlsx'


# Создаем объекты визуализатора для каждого файла данных
visualizer1 = TumorDataVisualizer(file_path1)
visualizer2 = TumorDataVisualizer(file_path2)

# Создаем объект сравнителя и сравниваем данные из двух экспериментов
comparator = TumorDataComparatorAdvanced(visualizer1, visualizer2)
comparator.compare_mean_volumes()  # Сравниваем средние абсолютные объемы
comparator.compare_relative_volumes()  # Сравниваем средние относительные объемы
