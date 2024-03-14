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
        self.markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        self.linestyles = ['-', '--', '-.', ':']
        self.linestyle_index = 0
        self.marker_index = 0
        self.marker_size = 12
        self.lines = []
        self.aucs = []
        self.max_x = None  # Добавлено для хранения максимального значения по оси X
        self.max_y = None  # Дополнительно, можно добавить для Y
        self.legend_info = []  # Список для хранения информации для дополнительных легенд

    def setup_figure(self):
        plt.figure(figsize=self.figsize)
        #plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid(True)

    @staticmethod
    def prepare_and_add_data_to_graph(visualizers, value_extractor_func, graph_visualizer, label_prefix, calculate_auc=False):
        """
        Подготавливает и добавляет данные к объекту GraphVisualizer.

        Parameters:
            visualizers (list): Список объектов визуализатора.
            value_extractor_func (function): Функция для извлечения значений из визуализатора.
            graph_visualizer (GraphVisualizer): Объект GraphVisualizer для добавления данных.
            label_prefix (str): Префикс для метки легенды графика.
            :param calculate_auc:
        """
        x_data_lists = []
        for visualizer in visualizers:
            values = value_extractor_func(visualizer)
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), values)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            graph_visualizer.add_plot(visualizer.time_data, values, visualizer.experiment_params, f"{label_prefix}",
                                      error_margin, calculate_auc)
            x_data_lists.append(visualizer.time_data)  # Добавляем данные по оси X для каждого визуализатора

        # Обновляем максимальное значение по оси X и настраиваем деления после добавления всех графиков
        graph_visualizer.update_axes_limits(x_data_lists)

    def add_individual_plots(self, labels, volumes_data, time_data):
        """
        Добавляет индивидуальные графики для каждой единицы данных (например, для каждой крысы).

        Parameters:
        - labels: метки для каждого графика (например, метки крыс).
        - volumes_data: данные объемов для каждой метки.
        - time_data: временные данные, общие для всех графиков.
        """
        for label, volumes in zip(labels, volumes_data):
            clean_volumes = np.array(volumes)[~np.isnan(volumes)]
            clean_time_data = np.array(time_data)[~np.isnan(volumes)]

            if not list(clean_volumes):
                continue

            mean_volume = np.mean(clean_volumes)
            std_dev = SupportingFunctions.calculate_std_dev(clean_volumes, mean_volume)
            error_margin = SupportingFunctions.calculate_error_margin(std_dev, len(clean_volumes))

            # Добавление данных на график
            self.add_plot(clean_time_data, clean_volumes, {}, label, error_margin)

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
        - linestyle: стиль линии графика.
        """
        # Выбор стиля линии и инкремент индекса
        current_linestyle = self.linestyles[self.linestyle_index % len(self.linestyles)]
        self.linestyle_index += 1

        x_data = np.array(x_data, dtype=float)
        line, = plt.plot(
            x_data,
            y_data,
            marker=self.markers[self.marker_index % len(self.markers)],
            markersize=self.marker_size,
            linestyle=current_linestyle,
            zorder=2,
            label=f"{label}{format_experiment_params(params)}"
        )
        if calculate_auc:
            auc_value = SupportingFunctions.calculate_auc(y_data, x_data)
            self.aucs.append(auc_value)

        self.lines.append(line)
        self.marker_index += 1

        # Проверяем, является ли error_margin итерируемым объектом, и если нет, преобразуем его
        if error_margin is not None and not hasattr(error_margin, '__iter__'):
            error_margin = [error_margin] * len(y_data)  # Создаем список с повторяющимся значением error_margin

        if error_margin is not None:
            line_color = line.get_color()
            plt.fill_between(x_data,
                             [y - e for y, e in zip(y_data, error_margin)],
                             [y + e for y, e in zip(y_data, error_margin)],
                             color=line_color, alpha=fill_alpha)

        # Метод для обновления границ осей, вызываемый после добавления всех графиков

    def add_legend(self, labels, title="", loc="upper left", display_marker=True):
        """Добавляет информацию для создания легенды. Поддерживает как одиночное значение, так и список."""
        if not isinstance(labels, list):  # Если labels не список, преобразуем в список
            labels = [labels]
        self.legend_info.append((labels, title, loc, display_marker))

    def update_axes_limits(self, x_data_lists):
        # Изменено на правильное вычисление максимального значения из списка списков
        self.max_x = max(max(x_data) for x_data in x_data_lists) if x_data_lists else self.max_x

    def finalize_figure(self, file_path, main_legend_title="", ncol=1, legend_fontsize='medium'):
        ax = plt.gca()  # Получаем текущий объект Axes

        # Устанавливаем деления оси X
        if not self.max_x and self.lines:
            # Предполагаем, что все линии используют одинаковый набор данных x, поэтому берем максимум из первой
            self.max_x = max(self.lines[0].get_xdata())
            # Устанавливаем деления оси X
        if self.max_x is not None:
            plt.xticks(ticks=range(0, int(self.max_x) + 1, 3), rotation=0)

        # Создаем и добавляем основную легенду
        if self.lines:
            first_legend = plt.legend(handles=self.lines, loc='upper left', title=main_legend_title, ncol=ncol, fontsize=legend_fontsize)
            ax.add_artist(first_legend)  # Важно использовать add_artist для сохранения основной легенды

        # Создаем и добавляем легенду AUC, если есть значения AUC
        if self.aucs:
            auc_labels = [f"AUC: {auc:.2f}" for auc in self.aucs]
            # Создаем объекты легенды AUC. Важно передать 'handles=self.lines', если стили линий важны
            auc_legend = plt.legend(handles=self.lines, labels=auc_labels, title="Площадь под кривой",
                                    loc='center left', fontsize=legend_fontsize)
            ax.add_artist(auc_legend)  # Добавляем легенду AUC

        for extra_legend_data in self.legend_info:
            labels, title, loc, display_marker = extra_legend_data
            if display_marker:
                extra_handles = [plt.Line2D([], [], color=line.get_color(), marker=line.get_marker()) for line in
                                 self.lines[:len(labels)]]
            else:
                # Если маркер не нужен, создаем элементы легенды без маркера
                extra_handles = [plt.Line2D([], [], color="none", marker=None, linestyle="None", label=label) for label
                                 in labels]
            extra_legend = plt.legend(handles=extra_handles, title=title, loc=loc, fontsize=legend_fontsize)
            ax.add_artist(extra_legend)

        plt.tight_layout()
        save_plot(file_path, self.title)
        plt.show()