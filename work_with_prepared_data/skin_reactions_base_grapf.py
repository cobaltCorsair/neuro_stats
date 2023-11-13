import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.lines import Line2D

from work_with_prepared_data.support_stats_methods import ExtractOutliers, SupportingFunctions

# Глобальное изменение размеров шрифтов и стиля
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Установка семейства шрифтов
    'font.size': 22,                   # Размер основного шрифта
    'axes.titlesize': 24,              # Размер заголовка
    'axes.labelsize': 24,              # Размер подписей осей
    'xtick.labelsize': 20,             # Размер меток на оси X
    'ytick.labelsize': 20,             # Размер меток на оси Y
    'legend.fontsize': 25              # Размер шрифта в легенде
})

class SkinReactionsVisualizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.skin_reactions = self.process_excel()

    def subscriptify(self, text):
        subscript_map = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
            '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
            'n': 'ₙ', 'p': 'ₚ', 'e': 'ₑ', 'a': 'ₐ', 'b': 'ᵦ', 'y': 'ᵧ'
            # Add more if available
        }
        return ''.join(subscript_map.get(char, char) for char in text)

    def format_experiment_params(self, params):
        cleaned_params = [str(param).replace('nan', '').strip() for param in params if str(param).strip()]
        rad_values = {}
        sequence = []
        for param in cleaned_params:
            if '=' in param and not param.startswith('t'):
                key, value = param.split('=')
                key = key.strip()
                value = value.split()[0]
                rad_values[key] = value.strip()
                sequence.append(key)
        formatted_params = []
        for key in sequence:
            if key in rad_values:
                formatted_params.append(f"D{self.subscriptify(key.lower())} = {rad_values[key]} Гр")
        if len(sequence) > 1:
            arrows = ' → '.join(sequence)
            formatted_params.append(arrows)
        return ', '.join(formatted_params)

    def process_excel(self):
        data = pd.read_excel(self.file_path, header=None)
        experiment_params = data.iloc[0, :3].tolist()
        skin_data = data.iloc[2:, :].copy()
        time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in data.iloc[1, 1:]]
        rat_labels = skin_data.iloc[:, 0].tolist()
        skin_reactions = skin_data.iloc[:, 1:].to_numpy().tolist()
        return experiment_params, time_data, rat_labels, skin_reactions

    def save_plot(self, plot_title: str):
        file_name_stub = self.file_path.split("/")[-1].replace(".xlsx", "")
        file_name = f"{plot_title.replace(' ', '_')}_{file_name_stub}.png"
        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def plot_skin_reactions(self):
        plt.figure(figsize=(15, 8))
        formatted_params = self.format_experiment_params(self.experiment_params)
        plt.title(f"Кожные реакции, Параметры эксперимента: {formatted_params}", fontsize=24, y=1.02)

        # Список маркеров
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        marker_size = 12  # Установка размера маркера

        for label, reactions, marker in zip(self.rat_labels, self.skin_reactions, markers):
            plt.plot(self.time_data, reactions, marker=marker, linestyle='-', markersize=marker_size, label=label)

        plt.xticks(np.arange(len(self.time_data))[::3], rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Кожные реакции")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        self.save_plot(f"Skin_Reactions_{formatted_params}")
        plt.show()

    def plot_mean_skin_reactions(self):
        plt.figure(figsize=(15, 8))
        formatted_params = self.format_experiment_params(self.experiment_params)
        plt.title(f"Средние кожные реакции, Параметры эксперимента: {formatted_params}", fontsize=24, y=1.02)
        mean_reactions, std_dev, error_margin = self.get_mean_skin_reactions()
        plt.plot(self.time_data, mean_reactions, marker='o', linestyle='-', label='Среднее')
        plt.fill_between(self.time_data, mean_reactions - error_margin, mean_reactions + error_margin, alpha=0.2)
        plt.xticks(np.arange(len(self.time_data))[::3], rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Средние кожные реакции")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self.save_plot(f"Mean_Skin_Reactions_{formatted_params}")
        plt.show()

    def get_mean_skin_reactions(self):
        mean_reactions = np.nanmean(self.skin_reactions, axis=0)
        std_dev = [SupportingFunctions.calculate_std_dev(values, mean_value) for values, mean_value in zip(np.transpose(self.skin_reactions), mean_reactions)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.skin_reactions)) for std in std_dev]
        return mean_reactions, np.array(std_dev), np.array(error_margin)

    @staticmethod
    def plot_multiple_experiments(file_paths: List[str]):
        plt.figure(figsize=(12, 7))
        common_timepoints = list(range(0, 25))
        aucs = []
        lines = []

        # Список маркеров
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        marker_size = 12  # Установка размера маркера

        for file_path, marker in zip(file_paths, markers):
            visualizer = SkinReactionsVisualizer(file_path)
            mean_reactions, _, _ = visualizer.get_mean_skin_reactions()
            mean_reactions_interp = SupportingFunctions.interpolate_data_to_common_timepoints(
                visualizer.time_data, mean_reactions, common_timepoints
            )
            auc = SupportingFunctions.calculate_auc(common_timepoints, mean_reactions_interp)
            aucs.append(auc)
            label = visualizer.format_experiment_params(visualizer.experiment_params)
            line, = plt.plot(common_timepoints,
                             mean_reactions_interp,
                             marker=marker,
                             linestyle='-',
                             markersize=marker_size,
                             label=label)
            lines.append(line)

        first_legend = plt.legend(title="", loc='lower center')
        plt.gca().add_artist(first_legend)
        labels = [f'{auc:.2e} тыс.' for auc in aucs]
        plt.xticks(np.arange(25)[::3], rotation=0)
        plt.xlabel("Время (сут.)")
        plt.ylabel("Кожные реакции, отн. ед.")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("multiple_experiments_mean_reactions.png", format='png', dpi=300)
        print("Plot saved as multiple_experiments_mean_reactions.png")
        plt.show()

if __name__ == '__main__':
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
        # 'datas/skin_reactions/skin_reactions_n_7.2_p_25.2_2023_2.xlsx',
        # 'datas/skin_reactions/skin_reactions_p_25,2_n_7,2_2023_2.xlsx'
    ]
    SkinReactionsVisualizer.plot_multiple_experiments(file_paths)
