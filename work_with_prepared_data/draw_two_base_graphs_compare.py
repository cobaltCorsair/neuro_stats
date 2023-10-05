import matplotlib.pyplot as plt
from draw_base_grapf import TumorDataVisualizer
import numpy as np


class TumorDataComparator:
    def __init__(self, visualizer1: TumorDataVisualizer, visualizer2: TumorDataVisualizer):
        self.visualizer1 = visualizer1
        self.visualizer2 = visualizer2

    def normalize_time_data(self):
        # Перевести временные метки в числовой формат и нормализовать их
        self.visualizer1.time_data = [int(time) - int(self.visualizer1.time_data[0]) for time in
                                      self.visualizer1.time_data]
        self.visualizer2.time_data = [int(time) - int(self.visualizer2.time_data[0]) for time in
                                      self.visualizer2.time_data]

    def compare_tumor_volumes(self):
        # Нормализовать временные метки
        self.normalize_time_data()

        plt.figure(figsize=(15, 8))

        # Построение графиков для первого эксперимента
        for label, volumes in zip(self.visualizer1.rat_labels, self.visualizer1.tumor_volumes):
            plt.plot(self.visualizer1.time_data, volumes, marker='o', linestyle='-', label=f"Exp1: {label}")

        # Построение графиков для второго эксперимента
        for label, volumes in zip(self.visualizer2.rat_labels, self.visualizer2.tumor_volumes):
            plt.plot(self.visualizer2.time_data, volumes, marker='o', linestyle='--', label=f"Exp2: {label}")

        plt.title(
            f"Сравнение экспериментов \n\n Exp1: {', '.join(self.visualizer1.experiment_params)} vs Exp2: {', '.join(self.visualizer2.experiment_params)}",
            fontsize=16, y=1.02)
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни, нормализованные)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        plt.show()


# Используем с файлом данных
file_path1 = './datas/n_7.2_p_25.2_2023.xlsx'
file_path2 = './datas/p_25.2_n_7.2_2023_2.xlsx'

# Создаем объекты визуализатора для каждого файла данных
visualizer1 = TumorDataVisualizer(file_path1)
visualizer2 = TumorDataVisualizer(file_path2)

# Создаем объект сравнителя и сравниваем данные из двух экспериментов
comparator = TumorDataComparator(visualizer1, visualizer2)
comparator.compare_tumor_volumes()
