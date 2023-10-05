import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from draw_base_grapf import TumorDataVisualizer


class TumorDataComparatorAdvanced:
    def __init__(self, visualizer1: TumorDataVisualizer, visualizer2: TumorDataVisualizer):
        self.visualizer1 = visualizer1
        self.visualizer2 = visualizer2

    def normalize_time_data(self):
        # Перевести временные метки в числовой формат и нормализовать их
        self.visualizer1.time_data = [int(time) - int(self.visualizer1.time_data[0]) for time in
                                      self.visualizer1.time_data]
        self.visualizer2.time_data = [int(time) - int(self.visualizer2.time_data[0]) for time in
                                      self.visualizer2.time_data]

    def compare_mean_volumes(self):
        self.normalize_time_data()
        plt.figure(figsize=(15, 8))

        plt.plot(self.visualizer1.time_data, self.visualizer1.get_mean_tumor_volumes(),
                 marker='o', linestyle='-', label=f"Exp1: M/V абс.")
        plt.plot(self.visualizer2.time_data, self.visualizer2.get_mean_tumor_volumes(),
                 marker='o', linestyle='--', label=f"Exp2: M/V абс.")

        plt.title(
            f"Сравнение среднего объема опухоли\nExp1: {', '.join(self.visualizer1.experiment_params)} vs Exp2: {', '.join(self.visualizer2.experiment_params)}")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни, нормализованные)")
        plt.ylabel("Средний объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_relative_volumes(self):
        self.normalize_time_data()
        plt.figure(figsize=(15, 8))

        plt.plot(self.visualizer1.time_data, self.visualizer1.get_mean_relative_tumor_volumes(),
                 marker='o', linestyle='-', label=f"Exp1: M/V отн.")
        plt.plot(self.visualizer2.time_data, self.visualizer2.get_mean_relative_tumor_volumes(),
                 marker='o', linestyle='--', label=f"Exp2: M/V отн.")

        plt.title(
            f"Сравнение среднего относительного объема опухоли\nExp1: {', '.join(self.visualizer1.experiment_params)} vs Exp2: {', '.join(self.visualizer2.experiment_params)}")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни, нормализованные)")
        plt.ylabel("Средний относительный объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Используем с файлом данных
file_path1 = './datas/n_7.2_p_25.2_2023.xlsx'
file_path2 = './datas/p_25.2_n_7.2_2023_2.xlsx'

# Создаем объекты визуализатора для каждого файла данных
visualizer1 = TumorDataVisualizer(file_path1)
visualizer2 = TumorDataVisualizer(file_path2)

# Создаем объект сравнителя и сравниваем данные из двух экспериментов
comparator = TumorDataComparatorAdvanced(visualizer1, visualizer2)
comparator.compare_mean_volumes()
comparator.compare_relative_volumes()
