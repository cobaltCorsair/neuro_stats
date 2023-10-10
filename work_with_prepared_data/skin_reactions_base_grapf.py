import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.lines import Line2D

from work_with_prepared_data.support_stats_methods import ExtractOutliers, SupportingFunctions


class SkinReactionsVisualizer:
    def __init__(self, file_path: str):
        """
        Инициализатор объекта визуализатора реакций кожи.

        Parameters:
            file_path (str): Путь к файлу Excel с данными эксперимента.
        """
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.skin_reactions = self.process_excel()

    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
        """
        Обрабатывает файл Excel и извлекает параметры эксперимента, временные данные,
        метки крыс и данные о реакциях кожи.

        Returns:
            Tuple: Кортеж, содержащий параметры эксперимента, временные данные,
                   метки крыс и данные о реакциях кожи.
        """
        # Загрузка данных из Excel файла
        data = pd.read_excel(self.file_path, header=None)

        # Извлечение параметров эксперимента
        experiment_params = data.iloc[0, :3].tolist()

        # Копирование данных о реакциях кожи
        skin_data = data.iloc[2:, :].copy()

        # Извлечение временных данных, преобразование строк в целые числа
        time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in data.iloc[1, 1:]]

        # Извлечение меток крыс
        rat_labels = skin_data.iloc[:, 0].tolist()

        # Извлечение и конвертация данных о реакциях кожи в список
        skin_reactions = skin_data.iloc[:, 1:].to_numpy().tolist()

        return experiment_params, time_data, rat_labels, skin_reactions

    def save_plot(self, plot_title: str):
        """
        Сохраняет текущий график в файл PNG.

        Parameters:
            plot_title (str): Заголовок графика, используемый для создания имени файла.
        """
        # Извлечение имени файла без расширения и пути
        file_name_stub = self.file_path.split("/")[-1].replace(".xlsx", "")

        # Формирование имени файла и сохранение графика
        file_name = f"{plot_title.replace(' ', '_')}_{file_name_stub}.png"
        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def plot_skin_reactions(self):
        """
        Визуализация данных о кожных реакциях для каждой крысы.
        """
        # Инициализация фигуры и добавление заголовка графика
        plt.figure(figsize=(15, 8))
        plt.title(f"Кожные реакции, Параметры эксперимента: {', '.join(self.experiment_params)}",
                  fontsize=16, y=1.02)

        # Построение графиков для каждой крысы
        for label, reactions in zip(self.rat_labels, self.skin_reactions):
            plt.plot(self.time_data, reactions, marker='o', linestyle='-', label=label)

        # Добавление меток и сетки на график
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Кожные реакции")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()

        # Сохранение графика
        self.save_plot(f"Skin_Reactions_{', '.join(self.experiment_params)}")
        plt.show()

    def plot_mean_skin_reactions(self):
        """
        Визуализация средних кожных реакций.
        """
        plt.figure(figsize=(15, 8))
        plt.title(f"Средние кожные реакции, Параметры эксперимента: {', '.join(self.experiment_params)}",
                  fontsize=16, y=1.02)

        mean_reactions, std_dev, error_margin = self.get_mean_skin_reactions()

        plt.plot(self.time_data, mean_reactions, marker='o', linestyle='-', label='Среднее')
        plt.fill_between(self.time_data,
                         mean_reactions - error_margin,
                         mean_reactions + error_margin, alpha=0.2)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средние кожные реакции")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        self.save_plot(f"Mean_Skin_Reactions_{', '.join(self.experiment_params)}")
        plt.show()

    def get_mean_skin_reactions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает средние кожные реакции, их стандартное отклонение и ошибку среднего.
        """
        mean_reactions = np.nanmean(self.skin_reactions, axis=0)
        std_dev = [SupportingFunctions.calculate_std_dev(values, mean_value)
                   for values, mean_value in zip(np.transpose(self.skin_reactions), mean_reactions)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.skin_reactions))
                        for std in std_dev]

        return mean_reactions, np.array(std_dev), np.array(error_margin)

    @staticmethod
    def plot_multiple_experiments(file_paths: List[str]):
        plt.figure(figsize=(15, 8))
        plt.title("Средние кожные реакции для множества экспериментов", fontsize=16, y=1.02)

        common_timepoints = list(range(0, 25))

        aucs = []  # Список для хранения значений AUC
        lines = []  # Список для хранения объектов Line2D

        for file_path in file_paths:
            visualizer = SkinReactionsVisualizer(file_path)
            mean_reactions, _, _ = visualizer.get_mean_skin_reactions()

            mean_reactions_interp = SupportingFunctions.interpolate_data_to_common_timepoints(
                visualizer.time_data, mean_reactions, common_timepoints
            )

            auc = SupportingFunctions.calculate_auc(common_timepoints, mean_reactions_interp)
            aucs.append(auc)  # Добавление AUC в список

            label = ', '.join(visualizer.experiment_params)
            line, = plt.plot(common_timepoints, mean_reactions_interp, marker='o', linestyle='-',
                             label=label)  # Запоминаем объект Line2D
            lines.append(line)  # Добавление объекта Line2D в список

        # Первая легенда
        first_legend = plt.legend(title="Параметры эксперимента", loc='upper left')
        plt.gca().add_artist(first_legend)  # Добавление первой легенды

        # Вторая легенда
        labels = [f'{auc:.2e} тыс.' for auc in aucs]
        plt.legend(lines, labels, title="Общий уровень реакции", loc='upper right')

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средние кожные реакции")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig("multiple_experiments_mean_reactions.png", format='png', dpi=300)
        print("Plot saved as multiple_experiments_mean_reactions.png")
        plt.show()


# Пример использования
# file_path = 'datas/skin_reactions/skin_reactions_n_7.2_p_25.2_2023.xlsx'
# file_path = 'datas/skin_reactions/skin_reactions_p_25,2_n_7,2_2023.xlsx'
# file_path = 'datas/skin_reactions/skin_reactions_n_7.2_p_25.2_2023_2.xlsx'
# file_path = 'datas/skin_reactions/skin_reactions_p_25,2_n_7,2_2023_2.xlsx'

# visualizer = SkinReactionsVisualizer(file_path)
#
# # Удаление выбросов
# # ExtractOutliers(visualizer).remove_local_outliers()
# Удаление точек
# ExtractOutliers(visualizer).exclude_rats(['б/м'], 'tumor_volumes')  # for skin_reactions_p_25,2_n_7,2_2023_2.xlsx
# ExtractOutliers(visualizer).exclude_rats(['г', 'х'], 'tumor_volumes')  # for skin_reactions_n_7.2_p_25.2_2023_2.xlsx

# visualizer.plot_skin_reactions()  # Визуализация индивидуальных кожных реакций
# visualizer.plot_mean_skin_reactions()  # Визуализация средних кожных реакций

# Отображения средних кожных реакций для нескольких экспериментов
file_paths = [
    'datas/skin_reactions/skin_reactions_n_7.2_p_25.2_2023.xlsx',
    'datas/skin_reactions/skin_reactions_p_25,2_n_7,2_2023.xlsx',
    'datas/skin_reactions/skin_reactions_n_7.2_p_25.2_2023_2.xlsx',
    'datas/skin_reactions/skin_reactions_p_25,2_n_7,2_2023_2.xlsx'
]
SkinReactionsVisualizer.plot_multiple_experiments(file_paths)
