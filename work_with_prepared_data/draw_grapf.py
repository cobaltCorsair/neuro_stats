import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

class TumorDataVisualizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.tumor_volumes = self.process_excel()

    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
        data = pd.read_excel(self.file_path, header=None)
        experiment_params = data.iloc[0, :3].tolist()
        tumor_data = data.iloc[2:, :].copy()
        time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in data.iloc[1, 1:]]

        tumor_data = tumor_data.applymap(lambda x: str(x).strip().replace(',', '.').replace(' -', '-') if pd.notna(x) else "NA")
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
        plt.title(f"Параметры эксперимента: {', '.join(self.experiment_params)}", fontsize=16, y=1.02)

        for label, volumes in zip(self.rat_labels, self.tumor_volumes):
            plt.plot(self.time_data, volumes, marker='o', linestyle='-', label=label)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        plt.show()

    def plot_mean_tumor_volume(self):
        plt.figure(figsize=(15, 8))
        plt.title(f"(M/V абс.), Параметры эксперимента: {', '.join(self.experiment_params)}", fontsize=16)

        mean_volumes = np.nanmean(self.tumor_volumes, axis=0)
        plt.plot(self.time_data, mean_volumes, marker='o', linestyle='-', color='b', label='M/V абс.')

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

        plt.plot(self.time_data, mean_relative_volumes, marker='o', linestyle='-', color='b', label='M/V отн.')
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средний относительный объем опухоли")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Используем с файлом данных
file_path = './datas/n_7.2_p_25.2_2023.xlsx'
visualizer = TumorDataVisualizer(file_path)

# Сохраняем график для каждой крысы
visualizer.plot_tumor_volumes_single_graph()

# Сохраняем график средних значений
visualizer.plot_mean_tumor_volume()

# Сохраняем график среднего относительного объема опухоли
visualizer.plot_average_relative_tumor_volume()
