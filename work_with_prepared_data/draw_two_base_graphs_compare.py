import os
import matplotlib.pyplot as plt
from draw_base_grapfs import TumorDataVisualizer
import numpy as np

from work_with_prepared_data.support_stats_methods import SupportingFunctions


class TumorDataComparator:
    def __init__(self, visualizer1: TumorDataVisualizer, visualizer2: TumorDataVisualizer, visualizer3=None):
        """
        Инициализатор класса сравнителя данных о опухолях.

        Parameters:
            visualizer1 (TumorDataVisualizer): Визуализатор данных первого эксперимента.
            visualizer2 (TumorDataVisualizer): Визуализатор данных второго эксперимента.
        """
        self.visualizer1 = visualizer1
        self.visualizer2 = visualizer2
        self.visualizer3 = visualizer3  # дополнительный график

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
        if self.visualizer3:
            # для трёх графиков
            for visualizer in [self.visualizer1, self.visualizer2, self.visualizer3]:
                visualizer.time_data = [int(time) - int(visualizer.time_data[0]) for time in visualizer.time_data]
        else:
            # для двух графиков
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

    def compare_relative_tumor_volumes_tree_data(self):
        """
        Сравнивает средние относительные объемы опухолей для трех экспериментов и строит график.
        """
        self.normalize_time_data()
        plt.figure(figsize=(15, 8))

        # Определение минимальной последней временной точки среди всех экспериментов
        min_last_timepoint = min(int(self.visualizer1.time_data[-1]),
                                 int(self.visualizer2.time_data[-1]),
                                 int(self.visualizer3.time_data[-1]))

        # Обрезка данных до минимальной последней временной точки
        time_data_1, mean_rel_volumes1 = SupportingFunctions.trim_data_to_timepoint(
            self.visualizer1.time_data,
            self.visualizer1.get_mean_relative_tumor_volumes(),
            min_last_timepoint
        )

        time_data_2, mean_rel_volumes2 = SupportingFunctions.trim_data_to_timepoint(
            self.visualizer2.time_data,
            self.visualizer2.get_mean_relative_tumor_volumes(),
            min_last_timepoint
        )

        time_data_3, mean_rel_volumes3 = SupportingFunctions.trim_data_to_timepoint(
            self.visualizer3.time_data,
            self.visualizer3.get_mean_relative_tumor_volumes(),
            min_last_timepoint
        )

        # Для первого визуализатора
        std_dev1 = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                    for volumes, mean_volume in zip(np.transpose(self.visualizer1.tumor_volumes), mean_rel_volumes1)]
        error_margin1 = [SupportingFunctions.calculate_error_margin(std, len(self.visualizer1.tumor_volumes))
                         for std in std_dev1]

        # Для второго визуализатора
        std_dev2 = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                    for volumes, mean_volume in zip(np.transpose(self.visualizer2.tumor_volumes), mean_rel_volumes2)]
        error_margin2 = [SupportingFunctions.calculate_error_margin(std, len(self.visualizer2.tumor_volumes))
                         for std in std_dev2]

        # Для третьего визуализатора
        std_dev3 = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                    for volumes, mean_volume in zip(np.transpose(self.visualizer3.tumor_volumes), mean_rel_volumes3)]
        error_margin3 = [SupportingFunctions.calculate_error_margin(std, len(self.visualizer3.tumor_volumes))
                         for std in std_dev3]

        # Построение графиков
        line1, = plt.plot(time_data_1, mean_rel_volumes1, marker='o', linestyle='-',
                          label=f"Exp1: {', '.join(map(str, self.visualizer1.experiment_params))}")
        plt.fill_between(time_data_1, [mean - err for mean, err in zip(mean_rel_volumes1, error_margin1)],
                         [mean + err for mean, err in zip(mean_rel_volumes1, error_margin1)], color='b', alpha=0.2)

        line2, = plt.plot(time_data_2, mean_rel_volumes2, marker='o', linestyle='--',
                          label=f"Exp2: {', '.join(map(str, self.visualizer2.experiment_params))}")
        plt.fill_between(time_data_2, [mean - err for mean, err in zip(mean_rel_volumes2, error_margin2)],
                         [mean + err for mean, err in zip(mean_rel_volumes2, error_margin2)], color='g', alpha=0.2)

        line3, = plt.plot(time_data_3, mean_rel_volumes3, marker='o', linestyle=':',
                          label=f"Exp3: {', '.join(map(str, self.visualizer3.experiment_params))}")
        plt.fill_between(time_data_3, [mean - err for mean, err in zip(mean_rel_volumes3, error_margin3)],
                         [mean + err for mean, err in zip(mean_rel_volumes3, error_margin3)], color='r', alpha=0.2)

        # Вычисление AUC
        auc1 = np.trapz(mean_rel_volumes1, time_data_1)
        auc2 = np.trapz(mean_rel_volumes2, time_data_2)
        auc3 = np.trapz(mean_rel_volumes3, [int(x) for x in time_data_3])

        # Оформление графика
        plt.title(f"Сравнение среднего относительного объема опухоли\n"
                  f"Exp1: {SupportingFunctions.join_params(self.visualizer1.experiment_params)} vs "
                  f"Exp2: {SupportingFunctions.join_params(self.visualizer2.experiment_params)} vs "
                  f"Exp3: {SupportingFunctions.join_params(self.visualizer3.experiment_params)}")

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний относительный объем опухоли")
        plt.grid(True)

        # Легенда

        line1, = plt.plot(time_data_1, mean_rel_volumes1, marker='o', linestyle='-',
                          label=f"Exp1: {SupportingFunctions.join_params(self.visualizer1.experiment_params)}")

        line2, = plt.plot(time_data_2, mean_rel_volumes2, marker='o', linestyle='--',
                          label=f"Exp2: {SupportingFunctions.join_params(self.visualizer2.experiment_params)}")

        line3, = plt.plot(time_data_3, mean_rel_volumes3, marker='o', linestyle=':',
                          label=f"Exp3: {SupportingFunctions.join_params(self.visualizer3.experiment_params)}")

        first_legend = plt.legend(handles=[line1, line2, line3], title="Параметры эксперимента", loc='upper left')
        plt.gca().add_artist(first_legend)

        labels = [f"ур. знач.: {auc:.2f}" for auc in [auc1, auc2, auc3]]
        plt.legend([line1, line2, line3], labels, title="Площадь под кривой", loc='upper right')

        plt.tight_layout()
        plt.autoscale()
        self.save_plot("compare_relative_volumes")
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

# Используем с файлом данных (три сравнения)
# file_path1 = './datas/control/16.03.2023_n_22.xlsx'
# file_path2 = './datas/control/30.03.2022_p_36_прострел.xlsx'
# file_path3 = './datas/control/02.02.2023_n_18.xlsx'

# Создаем объекты визуализатора для каждого файла данных
visualizer1 = TumorDataVisualizer(file_path1)
visualizer2 = TumorDataVisualizer(file_path2)
# visualizer3 = TumorDataVisualizer(file_path3)

# Создаем объект сравнителя и сравниваем данные из двух экспериментов
comparator = TumorDataComparator(visualizer1, visualizer2)
comparator.compare_tumor_volumes()  # Сравниваем абсолютные объемы
comparator.compare_relative_tumor_volumes()  # Сравниваем относительные объемы

# comparator = TumorDataComparator(visualizer1, visualizer2, visualizer3)
# comparator.compare_relative_tumor_volumes_tree_data()  # Сравниваем три относительных уср. объёмов за три эксперимента
