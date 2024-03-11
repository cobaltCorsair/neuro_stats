import os
import matplotlib.pyplot as plt
from draw_base_grapfs import TumorDataVisualizer
from utils.plotting_helpers import custom_fill_between, format_experiment_params, save_plot

# Сохраняем оригинальную функцию в другой переменной, на случай, если она понадобится
original_fill_between = plt.fill_between

# Переопределяем функцию
plt.fill_between = custom_fill_between

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


class TumorDataComparator:
    def __init__(self, *visualizers):
        """
        Инициализатор класса сравнителя данных о опухолях.

        Parameters:
            visualizers (list of TumorDataVisualizer): Список визуализаторов данных экспериментов.
        """
        self.visualizers = visualizers

    def normalize_time_data(self):
        """
        Нормализует временные метки всех экспериментов, приводя их к числовому формату и вычитая начальное время.
        """
        for visualizer in self.visualizers:
            visualizer.time_data = [int(time) - int(visualizer.time_data[0]) for time in visualizer.time_data]

    def compare_tumor_volumes(self):
        """
        Сравнивает абсолютные объемы опухолей между экспериментами и строит соответствующий график.
        """
        # Нормализовать временные метки
        self.normalize_time_data()

        plt.figure(figsize=(12, 7))
        linestyles = ['-', '--', '-.', ':']
        for visualizer, linestyle in zip(self.visualizers, linestyles[:len(self.visualizers)]):
            formatted_params = format_experiment_params(visualizer.experiment_params)
            for label, volumes in zip(visualizer.rat_labels, visualizer.tumor_volumes):
                plt.plot(visualizer.time_data, volumes, marker='o', linestyle=linestyle,
                         label=f"{formatted_params}: {label}")

        plt.title("Сравнение экспериментов")
        plt.xticks()
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        save_plot('', "compare_tumor_volumes")
        plt.show()

    def compare_relative_tumor_volumes(self):
        """
        Сравнивает относительные объемы опухолей между экспериментами и строит соответствующий график.
        """
        # Нормализовать временные метки
        self.normalize_time_data()

        plt.figure(figsize=(12, 7))
        linestyles = ['-', '--', '-.', ':']
        for visualizer, linestyle in zip(self.visualizers, linestyles[:len(self.visualizers)]):
            formatted_params = format_experiment_params(visualizer.experiment_params)
            for label, volumes in zip(visualizer.rat_labels, visualizer.get_relative_tumor_volumes()):
                plt.plot(visualizer.time_data, volumes, marker='o', linestyle=linestyle,
                         label=f"{formatted_params}: {label}")

        plt.title("Сравнение относительных объемов опухолей")
        plt.xticks()
        plt.xlabel("Время (дни)")
        plt.ylabel("Относительный объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()

        # Сохранение графика
        save_plot('', "compare_relative_tumor_volumes")
        plt.show()


if __name__ == "__main__":
    # Используем с файлом данных
    file_path1 = r'C:\dev\neuro_stats\work_with_prepared_data\datas\y_32_2023.xlsx'
    file_path2 = r'C:\dev\neuro_stats\work_with_prepared_data\datas\y_36_2023.xlsx'

    # Используем с файлом данных
    # file_path1 = './datas/control/02.02.2023_n_12.xlsx'
    # file_path2 = './datas/control/02.02.2023_n_18.xlsx'
    # file_path3 = './datas/control/16.03.2023_n_22.xlsx'

    # Создаем объекты визуализатора для каждого файла данных
    #visualizers = [TumorDataVisualizer(file_path) for file_path in [file_path1, file_path2, file_path3]]

    visualizers = [TumorDataVisualizer(file_path) for file_path in [file_path1, file_path2]]

    # Создаем объект сравнителя и сравниваем данные из всех экспериментов
    comparator = TumorDataComparator(*visualizers)
    comparator.compare_tumor_volumes()  # Сравниваем абсолютные объемы
    comparator.compare_relative_tumor_volumes()  # Сравниваем относительные объемы
