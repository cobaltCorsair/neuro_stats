import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


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


# Пример использования
file_path = 'datas/skin_reactions/skin_reactions_n_7.2_p_25.2_2023.xlsx'
visualizer = SkinReactionsVisualizer(file_path)
visualizer.plot_skin_reactions()
