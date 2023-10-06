import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


class SupportingFunctions:

    @staticmethod
    def calculate_std_dev(self, values, mean_value):
        """
        Расчет стандартного отклонения по заданной формуле.
        """
        n = len(values)
        sum_squared_deviations = sum((val - mean_value) ** 2 for val in values if not np.isnan(val))
        return np.sqrt(sum_squared_deviations / (n - 1))

    @staticmethod
    def calculate_error_margin(self, std_dev, n):
        """
        Расчет предела погрешности.
        """
        return std_dev / np.sqrt(n)


class TumorDataVisualizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.tumor_volumes = self.process_excel()

    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
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
        plt.show()

    def plot_relative_tumor_volumes_single_graph(self):
        relative_volumes = self.get_relative_tumor_volumes()

        plt.figure(figsize=(15, 8))
        plt.title(f"Относительные объемы опухоли, Параметры эксперимента: {', '.join(self.experiment_params)}",
                  fontsize=16, y=1.02)

        for label, volumes in zip(self.rat_labels, relative_volumes):
            plt.plot(self.time_data, volumes, marker='o', linestyle='-', label=label)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Относительный объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        plt.show()

    def plot_mean_tumor_volume(self):
        plt.figure(figsize=(15, 8))
        plt.title(f"(M/V абс.), Параметры эксперимента: {', '.join(self.experiment_params)}", fontsize=16)

        mean_volumes = np.nanmean(self.tumor_volumes, axis=0)
        std_dev = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume) for volumes, mean_volume in
                   zip(np.transpose(self.tumor_volumes), mean_volumes)]
        error_margin = [SupportingFunctions.calculate_error_margin(self, std, len(self.tumor_volumes)) for std in std_dev]

        plt.plot(self.time_data, mean_volumes, marker='o', linestyle='-', color='b', label='M/V абс.')
        plt.fill_between(self.time_data, mean_volumes - error_margin, mean_volumes + error_margin, color='b', alpha=0.2)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_average_relative_tumor_volume(self):
        plt.figure(figsize=(15, 8))
        plt.title(f"(V отн.), Параметры эксперимента: {', '.join(self.experiment_params)}", fontsize=16)

        relative_tumor_volumes = np.array([[vol / volumes[0] for vol in volumes] for volumes in self.tumor_volumes])
        mean_relative_volumes = np.nanmean(relative_tumor_volumes, axis=0)
        std_dev_rel = [SupportingFunctions.calculate_std_dev(self, volumes, mean_volume) for volumes, mean_volume in
                       zip(np.transpose(relative_tumor_volumes), mean_relative_volumes)]
        error_margin_rel = [SupportingFunctions.calculate_error_margin(self, std, len(relative_tumor_volumes)) for std in std_dev_rel]

        plt.plot(self.time_data, mean_relative_volumes, marker='o', linestyle='-', color='b', label='M/V отн.')
        plt.fill_between(self.time_data, mean_relative_volumes - error_margin_rel,
                         mean_relative_volumes + error_margin_rel, color='b', alpha=0.2)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний относительный объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_mean_tumor_volumes(self) -> np.ndarray:
        return np.nanmean(self.tumor_volumes, axis=0)

    def get_relative_tumor_volumes(self) -> np.ndarray:
        return np.array([[vol / volumes[0] for vol in volumes] for volumes in self.tumor_volumes])

    def get_mean_relative_tumor_volumes(self) -> np.ndarray:
        relative_volumes = self.get_relative_tumor_volumes()
        return np.nanmean(relative_volumes, axis=0)


# Используем с файлом данных
# file_path = './datas/n_7.2_p_25.2_2023.xlsx'
file_path = './datas/p_25.2_n_7.2_2023.xlsx'
visualizer = TumorDataVisualizer(file_path)

# Сохраняем график для каждой крысы
visualizer.plot_tumor_volumes_single_graph()

# Сохраняем график относительных объемов для каждой крысы
visualizer.plot_relative_tumor_volumes_single_graph()

# Сохраняем график средних значений
visualizer.plot_mean_tumor_volume()

# Сохраняем график среднего относительного объема опухоли
visualizer.plot_average_relative_tumor_volume()
