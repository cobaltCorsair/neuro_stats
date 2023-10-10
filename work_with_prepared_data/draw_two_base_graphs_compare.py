import os
import matplotlib.pyplot as plt
from draw_base_grapfs import TumorDataVisualizer
import numpy as np


class TumorDataComparator:
    def __init__(self, visualizer1: TumorDataVisualizer, visualizer2: TumorDataVisualizer):
        """
        Инициализатор класса сравнителя данных о опухолях.

        Parameters:
            visualizer1 (TumorDataVisualizer): Визуализатор данных первого эксперимента.
            visualizer2 (TumorDataVisualizer): Визуализатор данных второго эксперимента.
        """
        self.visualizer1 = visualizer1
        self.visualizer2 = visualizer2

    def save_plot(self, comparison_type: str):
        """
        Сохраняет текущий график в файл PNG.

        Parameters:
            comparison_type (str): Тип сравнения, используется в имени файла.
        """
        # Извлечение имен файлов без расширения и пути
        file_name1 = os.path.splitext(os.path.basename(self.visualizer1.file_path))[0]
        file_name2 = os.path.splitext(os.path.basename(self.visualizer2.file_path))[0]

        # Сборка окончательного имени файла и сохранение графика
        file_name = f"{comparison_type}_{file_name1}_vs_{file_name2}.png"
        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def normalize_time_data(self):
        """
        Нормализует временные метки, приводя их к числовому формату и вычитая начальное время.
        """
        self.visualizer1.time_data = [int(time) - int(self.visualizer1.time_data[0]) for time in
                                      self.visualizer1.time_data]
        self.visualizer2.time_data = [int(time) - int(self.visualizer2.time_data[0]) for time in
                                      self.visualizer2.time_data]

    def compare_tumor_volumes(self):
        """
        Сравнивает абсолютные объемы опухолей между двумя экспериментами и строит соответствующий график.
        """
        # Нормализовать временные метки
        self.normalize_time_data()

        # Инициализация графика и построение данных для каждого эксперимента
        plt.figure(figsize=(15, 8))
        for label, volumes in zip(self.visualizer1.rat_labels, self.visualizer1.tumor_volumes):
            plt.plot(self.visualizer1.time_data, volumes, marker='o', linestyle='-', label=f"Exp1: {label}")
        for label, volumes in zip(self.visualizer2.rat_labels, self.visualizer2.tumor_volumes):
            plt.plot(self.visualizer2.time_data, volumes, marker='o', linestyle='--', label=f"Exp2: {label}")

        # Добавление меток, заголовка и сетки на график
        plt.title(
            f"Сравнение экспериментов \n\n Exp1: {', '.join(self.visualizer1.experiment_params)} vs Exp2: {', '.join(self.visualizer2.experiment_params)}",
            fontsize=16, y=1.02)
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()

        # Сохранение графика
        self.save_plot("compare_tumor_volumes")
        plt.show()

    def compare_relative_tumor_volumes(self):
        """
        Сравнивает относительные объемы опухолей между двумя экспериментами и строит соответствующий график.
        """
        # Нормализовать временные метки
        self.normalize_time_data()

        # Инициализация графика и построение данных для каждого эксперимента
        plt.figure(figsize=(15, 8))
        for label, volumes in zip(self.visualizer1.rat_labels, self.visualizer1.get_relative_tumor_volumes()):
            plt.plot(self.visualizer1.time_data, volumes, marker='o', linestyle='-', label=f"Exp1: {label}")
        for label, volumes in zip(self.visualizer2.rat_labels, self.visualizer2.get_relative_tumor_volumes()):
            plt.plot(self.visualizer2.time_data, volumes, marker='o', linestyle='--', label=f"Exp2: {label}")

        # Добавление меток, заголовка и сетки на график
        plt.title(
            f"Сравнение относительных объемов опухолей \n\n Exp1: {', '.join(self.visualizer1.experiment_params)} vs Exp2: {', '.join(self.visualizer2.experiment_params)}",
            fontsize=16, y=1.02)
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Относительный объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()

        # Сохранение графика
        self.save_plot("compare_relative_tumor_volumes")
        plt.show()


# # Используем с файлом данных
file_path1 = './datas/n_7.2_p_25.2_2023_2.xlsx'
file_path2 = './datas/p_25.2_n_7.2_2023_2.xlsx'

# Используем с файлом данных
# file_path1 = './datas/n_7.2_p_25.2_2023.xlsx'
# file_path2 = './datas/p_25.2_n_7.2_2023.xlsx'

# Используем с файлом данных
# file_path1 = './datas/n_2.56_p_25.6_2019.xlsx'
# file_path2 = './datas/p_25.6_n_2.56_2019.xlsx'

# Создаем объекты визуализатора для каждого файла данных
visualizer1 = TumorDataVisualizer(file_path1)
visualizer2 = TumorDataVisualizer(file_path2)

# Создаем объект сравнителя и сравниваем данные из двух экспериментов
comparator = TumorDataComparator(visualizer1, visualizer2)
comparator.compare_tumor_volumes()  # Сравниваем абсолютные объемы
comparator.compare_relative_tumor_volumes()  # Сравниваем относительные объемы
