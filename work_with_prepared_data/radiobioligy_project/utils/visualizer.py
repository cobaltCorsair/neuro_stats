import numpy as np
from matplotlib import pyplot as plt

from work_with_prepared_data.radiobioligy_project.stats_methods.support_stats_methods import SupportingFunctions
from work_with_prepared_data.radiobioligy_project.utils.plot_saver import save_plot
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import format_experiment_params


class GraphVisualizer:
    def __init__(self, title, x_label, y_label, figsize=(12, 7)):
        self.figsize = figsize
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.markers = ['o', 's', '^', 'x', '*', 'D', 'h', '+', 'p']
        self.marker_index = 0
        self.marker_size = 12
        self.lines = []
        self.aucs = []
        self.max_x = None  # Добавлено для хранения максимального значения по оси X
        self.max_y = None  # Дополнительно, можно добавить для Y

    def setup_figure(self):
        plt.figure(figsize=self.figsize)
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid(True)

    def add_plot(self, x_data, y_data, params, label, error_margin=None, calculate_auc=False, fill_alpha=0.2):
        """
        Добавляет линейный график с опциональными доверительными интервалами и расчётом AUC.

        Parameters:
        - x_data: данные по оси X
        - y_data: данные по оси Y
        - label: метка для графика
        - error_margin: доверительные интервалы или погрешности для заполнения (опционально)
        - calculate_auc: флаг для расчёта площади под кривой (AUC)
        - fill_alpha: прозрачность заполнения доверительных интервалов
        """
        line, = plt.plot(
            x_data,
            y_data,
            marker=self.markers[self.marker_index % len(self.markers)],
            markersize=self.marker_size,
            linestyle='-',
            zorder=2,
            label= f"{format_experiment_params(params)}: {label}"
        )
        if calculate_auc:
            auc_value = SupportingFunctions.calculate_auc(y_data, x_data)
            self.aucs.append(auc_value)

        self.lines.append(line)
        self.marker_index += 1

        if error_margin is not None:
            line_color = line.get_color()
            plt.fill_between(x_data,
                             [y - e for y, e in zip(y_data, error_margin)],
                             [y + e for y, e in zip(y_data, error_margin)],
                             color=line_color, alpha=fill_alpha)

        # Метод для обновления границ осей, вызываемый после добавления всех графиков

    def update_axes_limits(self, x_data_lists):
        # Изменено на правильное вычисление максимального значения из списка списков
        self.max_x = max(max(x_data) for x_data in x_data_lists) if x_data_lists else self.max_x

    def finalize_figure(self, file_path):
        # Установка делений оси X должна быть здесь, после установки границ осей
        if self.max_x is not None:
            plt.xticks(ticks=range(0, self.max_x + 1, 3), rotation=0)

        if self.lines:
            first_legend = plt.legend(handles=self.lines, loc='upper left')
            plt.gca().add_artist(first_legend)

        if self.aucs:
            auc_labels = [f"AUC: {auc:.2f}" for auc in self.aucs]
            plt.legend(self.lines, auc_labels, title="Площадь под кривой", loc='upper center')

        plt.tight_layout()
        save_plot(file_path, self.title)
        plt.show()