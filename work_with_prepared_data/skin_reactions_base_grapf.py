import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class SkinReactionsVisualizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.skin_reactions = self.process_excel()

    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
        data = pd.read_excel(self.file_path, header=None)
        experiment_params = data.iloc[0, :3].tolist()
        skin_data = data.iloc[2:, :].copy()
        time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in data.iloc[1, 1:]]

        rat_labels = skin_data.iloc[:, 0].tolist()
        skin_reactions = skin_data.iloc[:, 1:].to_numpy().tolist()

        return experiment_params, time_data, rat_labels, skin_reactions

    def save_plot(self, plot_title: str):
        # Extracting the file name (without extension) from the file path
        file_name_stub = self.file_path.split("/")[-1].replace(".xlsx", "")
        file_name = f"{plot_title.replace(' ', '_')}_{file_name_stub}.png"
        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def plot_skin_reactions(self):
        plt.figure(figsize=(15, 8))
        plt.title(f"Кожные реакции, Параметры эксперимента: {', '.join(self.experiment_params)}",
                  fontsize=16, y=1.02)

        for label, reactions in zip(self.rat_labels, self.skin_reactions):
            plt.plot(self.time_data, reactions, marker='o', linestyle='-', label=label)

        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Кожные реакции")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        self.save_plot(f"Skin_Reactions_{', '.join(self.experiment_params)}")
        plt.show()


# Пример использования
file_path = 'datas/skin_reactions/skin_reactions_n_7.2_p_25.2_2023.xlsx'
visualizer = SkinReactionsVisualizer(file_path)
visualizer.plot_skin_reactions()
