# файл draw_base_graphs.py

import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
import pandas as pd
import re

from utils.plotting_helpers import custom_fill_between, format_experiment_params, MatplotlibConfigurator
from stats_methods.support_stats_methods import SupportingFunctions, ExtractOutliers
from data_processing.excel_data_processor import process_tumor_data_excel
from data_processing.data_processing import TumorDataProcessor
from utils.visualizer import GraphVisualizer, plot_group_auc_barplot

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
# После создания и сохранения всех графиков восстанавливаем оригинальные стили
# configurator.restore_original_styles()


class TumorDataVisualizer:
    def __init__(self, file_path: str):
        """
        Инициализирует объект визуализатора данных по опухолям, загружая данные из Excel файла и подготавливая их к анализу.

        Args:
            file_path (str): Полный путь к Excel файлу, содержащему данные эксперимента, включая временные точки,
                             метки крыс и объемы опухолей.

        Атрибуты:
            file_path (str): Сохраняет путь к файлу с данными.
            experiment_params (List[str]): Список параметров эксперимента, извлеченный из файла.
            time_data (List[float]): Список временных точек измерений.
            rat_labels (List[str]): Список меток крыс, участвующих в эксперименте.
            tumor_volumes (List[List[float]]): Двумерный список, содержащий объемы опухолей каждой крысы на различных
            временных точках.
            data_processor (TumorDataProcessor): Объект для обработки и анализа данных об объемах опухолей.

        Описание:
            Конструктор сначала вызывает функцию process_tumor_data_excel для извлечения данных из указанного Excel файла.
            Полученные данные включают в себя параметры эксперимента, временные точки измерений, метки крыс и соответствующие
            объемы опухолей. Затем создается объект TumorDataProcessor, который будет использоваться для дальнейшей
            обработки и анализа данных о росте опухолей.

        Примечание:
            Для работы конструктора необходимо, чтобы формат Excel файла соответствовал ожидаемому шаблону данных,
            где первые строки и столбцы содержат необходимую метаинформацию об эксперименте, а остальные данные
            представляют измерения объемов опухолей.
        """
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.tumor_volumes = process_tumor_data_excel(file_path)
        self.data_processor = TumorDataProcessor(self.tumor_volumes)  # Создаем экземпляр TumorDataProcessor для обработки данных

    def plot_tumor_volumes_single_graph(self):
        """
        Визуализирует объемы опухолей всех крыс в рамках одного эксперимента на общем графике.

        Этот метод создает график, на котором каждая крыса представлена отдельной линией, отражающей динамику
        изменения объема ее опухоли в зависимости от времени. Для каждой крысы используется уникальный маркер и стиль
        линии.

        Args:
            Нет аргументов.

        Атрибуты:
            rat_labels (List[str]): Список меток крыс, участвующих в эксперименте.
            tumor_volumes (List[List[float]]): Двумерный список, содержащий объемы опухолей каждой крысы на различных
            временных точках.
            time_data (List[float]): Список временных точек измерений.
            experiment_params (List[str]): Список параметров эксперимента, извлеченный из файла.

        Использует:
            GraphVisualizer: Класс для упрощения процесса создания и настройки графиков.
            format_experiment_params: Функция для форматирования параметров эксперимента в строку
            для отображения в легенде.

        Результат: Создает и отображает график с динамикой изменения объемов опухолей всех крыс из эксперимента,
        добавляя легенду с параметрами эксперимента и уникальные метки для каждой крысы. График сохраняется в файл с
        именем, соответствующим параметрам эксперимента.
        """
        drawgraph = GraphVisualizer("Абсолютные объемы опухоли", "Время, сут.", "Объем опухоли, абс. ед.",
                                    figsize=(12, 7))
        drawgraph.setup_figure()
        drawgraph.add_individual_plots(self.rat_labels, self.tumor_volumes, self.time_data)

        formatted_params = format_experiment_params(self.experiment_params)
        drawgraph.add_legend([formatted_params], "Параметры эксперимента", "upper center", display_marker=False)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_absolute_volumes", 'Метка крысы')

    def plot_relative_tumor_volumes_single_graph(self):
        """
        Визуализирует относительные объемы опухолей всех крыс в одном эксперименте на одном графике.

        Этот метод генерирует график, где для каждой крысы отображается изменение относительного объема ее
        опухоли относительно начального объема в зависимости от времени. Относительный объем опухоли помогает
        оценить динамику роста или уменьшения опухоли в процентном соотношении.

        Args:
            Нет аргументов.

        Атрибуты:
            rat_labels (List[str]): Список меток крыс, участвующих в эксперименте.
            data_processor (TumorDataProcessor): Экземпляр класса для обработки данных опухоли, включая расчет
            относительных объемов.
            time_data (List[float]): Список временных точек измерений.
            experiment_params (List[str]): Список параметров эксперимента, извлеченный из файла.

        Использует:
            GraphVisualizer: Класс для создания и настройки графиков.
            format_experiment_params: Функция для форматирования параметров эксперимента в строку,
            отображаемую в легенде.

        Результат: Генерирует и отображает график с динамикой изменения относительных объемов опухолей всех крыс
            эксперимента. На графике добавляется легенда с параметрами эксперимента и уникальными метками для каждой
            крысы. График сохраняется в файл, имя которого соответствует параметрам эксперимента.
        """
        drawgraph = GraphVisualizer("Относительные объемы опухоли", "Время, сут.", "Объем опухоли, отн. ед.",
                                    figsize=(12, 7))
        drawgraph.setup_figure()
        drawgraph.add_individual_plots(self.rat_labels, self.data_processor.get_relative_tumor_volumes(), self.time_data)
        formatted_params = format_experiment_params(self.experiment_params)
        # Добавление легенды с параметрами эксперимента
        drawgraph.add_legend([formatted_params], "Параметры эксперимента", "upper center", display_marker=False)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_relative_volumes", 'Метка крысы')

    def plot_mean_tumor_volume(self):
        """
        Визуализирует средние объемы опухолей всех крыс в рамках одного эксперимента на общем графике,
        включая доверительные интервалы.

        Этот метод вычисляет средние объемы опухолей на каждой временной точке и визуализирует эти данные на графике.
        Для каждой временной точки также рассчитываются стандартные отклонения и доверительные интервалы, которые
        отображаются на графике в виде зон окрашенных прозрачным цветом.

        Args:
            Нет аргументов.

        Атрибуты:
            time_data (List[float]): Список временных точек измерений.
            tumor_volumes (List[List[float]]): Двумерный список, содержащий объемы опухолей каждой крысы на различных
            временных точках.
            experiment_params (List[str]): Список параметров эксперимента, извлеченный из файла.

        Использует:
            GraphVisualizer: Класс для упрощения процесса создания и настройки графиков.
            get_mean_tumor_volumes: Метод для вычисления средних значений объемов опухолей.
            SupportingFunctions.calculate_std_dev и calculate_error_margin: Функции для расчета стандартного отклонения
            и доверительного интервала.

        Результат:
            Создает и отображает график с динамикой изменения средних объемов опухолей всех крыс из эксперимента,
            добавляя зоны доверительных интервалов. График сохраняется в файл с именем, соответствующим параметрам
            эксперимента.
        """
        drawgraph = GraphVisualizer("", "Время, сут.", "Объем опухоли, абс. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        mean_volumes = self.data_processor.get_mean_tumor_volumes()
        std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                   zip(np.transpose(self.tumor_volumes), mean_volumes)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.tumor_volumes)) for std in std_dev]

        # Форматируем параметры эксперимента для уникальной метки
        formatted_params = format_experiment_params(self.experiment_params)
        # Создаем уникальную метку для этого эксперимента
        unique_label = f"M/V абс.: {formatted_params}"
        
        # Добавление данных на график с уникальной меткой
        drawgraph.add_plot(self.time_data, mean_volumes, {}, unique_label, error_margin)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_mean_volumes")

    def plot_average_relative_tumor_volume(self):
        """
        Визуализирует средние относительные объемы опухолей всех крыс в рамках одного эксперимента на общем графике,
        включая доверительные интервалы.

        Этот метод вычисляет средние относительные объемы опухолей на каждой временной точке и визуализирует эти данные на графике.
        Для каждой временной точки также рассчитываются стандартные отклонения и доверительные интервалы, которые
        отображаются на графике в виде зон окрашенных прозрачным цветом.

        Args:
            Нет аргументов.

        Атрибуты:
            time_data (List[float]): Список временных точек измерений.
            relative_tumor_volumes (List[List[float]]): Двумерный список, содержащий относительные объемы опухолей
            каждой крысы на различных временных точках.
            experiment_params (List[str]): Список параметров эксперимента, извлеченный из файла.

        Использует:
            GraphVisualizer: Класс для упрощения процесса создания и настройки графиков.
            get_relative_tumor_volumes и get_mean_relative_tumor_volumes: Методы для вычисления относительных и средних
            относительных значений объемов опухолей.
            SupportingFunctions.calculate_std_dev и calculate_error_margin: Функции для расчета стандартного отклонения
            и доверительного интервала.

        Результат:
            Создает и отображает график с динамикой изменения средних относительных объемов опухолей всех крыс из эксперимента,
            добавляя зоны доверительных интервалов. График сохраняется в файл с именем, соответствующим параметрам эксперимента.
        """
        drawgraph = GraphVisualizer("(V отн.)", "Время, сут.", "Относительный объем опухоли, отн. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        relative_tumor_volumes = self.data_processor.get_relative_tumor_volumes()
        mean_relative_volumes = self.data_processor.get_mean_relative_tumor_volumes()
        std_dev_rel = [SupportingFunctions.calculate_std_dev(volumes, mean_volume) for volumes, mean_volume in
                       zip(np.transpose(relative_tumor_volumes), mean_relative_volumes)]
        error_margin_rel = [SupportingFunctions.calculate_error_margin(std, len(relative_tumor_volumes)) for std in
                            std_dev_rel]

        # Форматируем параметры эксперимента для уникальной метки
        formatted_params = format_experiment_params(self.experiment_params)
        # Создаем уникальную метку для этого эксперимента
        unique_label = f"M/V отн.: {formatted_params}"

        # Добавление данных на график с уникальной меткой
        drawgraph.add_plot(self.time_data, mean_relative_volumes, {}, unique_label,
                           error_margin_rel)

        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_average_relative_volumes")

    def plot_mean_relative_mean_tumor_volume(self):
        """
        Построение графика среднего относительного объема опухоли, усредненного по всем крысам, с отображением доверительных
        интервалов.

        Этот метод вычисляет средний относительный объем опухоли на каждой временной точке, основываясь на данных всех
        крыс в эксперименте.
        Также рассчитывает стандартное отклонение и доверительные интервалы для этих средних значений.

        Атрибуты:
            time_data (List[float]): Список временных точек измерений.
            relative_mean_volumes (List[float]): Список средних относительных объемов опухолей на различных временных точках.
            experiment_params (List[str]): Список параметров эксперимента, извлеченный из файла.

        Использует:
            GraphVisualizer: Класс для упрощения процесса создания и настройки графиков.
            get_mean_relative_tumor_volumes: Метод для вычисления средних относительных значений объемов опухолей.
            SupportingFunctions.calculate_std_dev и calculate_error_margin: Функции для расчета стандартного отклонения
            и доверительного интервала.

        Результат:
            Создает и отображает график с динамикой изменения среднего относительного объема опухолей со всеми крысами,
            добавляя зоны доверительных интервалов. График сохраняется в файл с именем, соответствующим параметрам эксперимента.
        """
        drawgraph = GraphVisualizer("Средний относительный усреднённый объем опухоли (V отн. ср.)", "Время, сут.",
                                    "Средний отн. объем опухоли, отн. ед.", figsize=(12, 7))
        drawgraph.setup_figure()

        # Вычисление среднего относительного объема опухоли и его доверительного интервала
        relative_mean_volumes = self.data_processor.get_mean_relative_tumor_volumes()
        mean_volumes = self.data_processor.get_mean_tumor_volumes(relative_mean_volumes)
        std_dev_rel_mean = SupportingFunctions.calculate_std_dev(relative_mean_volumes, mean_volumes)
        error_margin_rel_mean = SupportingFunctions.calculate_error_margin(std_dev_rel_mean, len(relative_mean_volumes))

        # Форматируем параметры эксперимента для уникальной метки
        formatted_params = format_experiment_params(self.experiment_params)
        # Создаем уникальную метку для этого эксперимента
        unique_label = f"M/V отн. ср.: {formatted_params}"

        # Добавление данных на график с уникальной меткой
        drawgraph.add_plot(self.time_data, relative_mean_volumes, {}, unique_label,
                           error_margin_rel_mean)

        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_mean_relative_mean_volumes")

    @staticmethod
    def plot_auc_comparison(file_paths: List[str],
                             title="Сравнение AUC объёмов опухоли",
                             x_label="",
                             y_label="AUC (абс. ед.)"):
        """
        Построение столбчатого графика для сравнения площади под кривой
        объёмов опухоли между экспериментами.

        Args:
            file_paths: Пути к файлам с данными экспериментов.
            title: Заголовок графика.
            x_label: Подпись оси X.
            y_label: Подпись оси Y.
        """
        with sns.axes_style("whitegrid"):
            def extract_total_dose(experiment_params):
                total = 0.0
                for p in experiment_params:
                    if '=' in p and ('Гр' in p or 'Gy' in p or 'гр' in p or 'gy' in p):
                        try:
                            value = re.findall(r'[-+]?\d*\.\d+|\d+', p)
                            if value:
                                total += float(value[0].replace(',', '.'))
                        except Exception:
                            continue
                return total if total > 0 else None
            doses = []
            aucs_for_fit = []
            errors_for_fit = []
            labels_for_legend = []
            auc_values = []
            colors = sns.color_palette("Set3", n_colors=len(file_paths))
            for file_path, color in zip(file_paths, colors):
                visualizer = TumorDataVisualizer(file_path)
                try:
                    # Индивидуальные кривые по животным
                    individual_volumes = visualizer.tumor_volumes
                    time_data = visualizer.time_data
                    aucs_individual = []
                    for rat_curve in individual_volumes:
                        interpolated = SupportingFunctions.interpolate_data_to_common_timepoints(
                            time_data, rat_curve, list(range(0, 25))
                        )
                        aucs_individual.append(SupportingFunctions.calculate_auc(interpolated, list(range(0, 25))))
                    aucs_individual = np.array(aucs_individual)
                    auc_mean = np.mean(aucs_individual)
                    auc_sem = np.std(aucs_individual, ddof=1) / np.sqrt(len(aucs_individual))
                    auc_values.append(auc_mean)
                    labels_for_legend.append(format_experiment_params(visualizer.experiment_params))
                    dose = extract_total_dose(visualizer.experiment_params)
                    if dose is not None:
                        doses.append(dose)
                        aucs_for_fit.append(auc_mean)
                        errors_for_fit.append(auc_sem)
                except ValueError as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")
                    continue
            # --- Barplot по числовой оси X (doses) ---
            plt.figure(figsize=(12, 8))
            bar_width = 2.5 if len(doses) < 10 else 0.8
            bars = plt.bar(doses, aucs_for_fit, yerr=errors_for_fit, width=bar_width, color=colors[:len(doses)], edgecolor="black", zorder=2, capsize=8)
            plt.xlabel("Суммарная доза, Гр", fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.title(title, fontsize=16)
            # Подписи над столбиками
            # Удаляю старую подпись над столбиком (оставляю только под error bar)
            # for i, (bar, auc) in enumerate(zip(bars, aucs_for_fit)):
            #     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{auc:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
            # Подписи с дозами на тиках оси X
            plt.xticks(doses, [str(d) for d in doses], fontsize=12)
            # Подписи под error bar
            for i, (bar, auc, err) in enumerate(zip(bars, aucs_for_fit, errors_for_fit)):
                y_text = bar.get_height() - err - 0.03 * max(aucs_for_fit)
                # Не уводим подпись ниже нуля
                y_text = max(0, y_text)
                plt.text(bar.get_x() + bar.get_width()/2, y_text, f"{auc:.2f}", ha='center', va='top', fontsize=12, fontweight='bold', color='black')
            # Легенда как раньше
            legend_patches = [mpatches.Patch(color=col, label=lab) for col, lab in zip(colors[:len(labels_for_legend)], labels_for_legend)]
            ncol = math.ceil(len(labels_for_legend) / 2) if len(labels_for_legend) > 4 else len(labels_for_legend)
            plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=ncol, fontsize=12, frameon=False, handletextpad=0.5, columnspacing=2.5)
            plt.tight_layout()
            # plt.savefig("tumor_auc_comparison_plot.png")


def plot_auc_comparison_absolute_tumor(file_paths, title="Сравнение AUC объёмов опухоли (абс.)", x_label="Суммарная доза, Гр", y_label="AUC (абс. ед.)"):
    from utils.plotting_helpers import format_experiment_params
    import re
    def extract_total_dose(experiment_params):
        total = 0.0
        for p in experiment_params:
            if '=' in p and ('Гр' in p or 'Gy' in p or 'гр' in p or 'gy' in p):
                try:
                    value = re.findall(r'[-+]?\d*\.\d+|\d+', p)
                    if value:
                        total += float(value[0].replace(',', '.'))
                except Exception:
                    continue
        return total if total > 0 else None
    plot_group_auc_barplot(
        file_paths,
        TumorDataVisualizer,
        lambda vis: vis.tumor_volumes,
        lambda vis: vis.time_data,
        lambda vis: format_experiment_params(vis.experiment_params),
        lambda vis: extract_total_dose(vis.experiment_params),
        title=title,
        y_label=y_label,
        x_label=x_label,
        relative=False
    )

def plot_auc_comparison_relative_tumor(file_paths, title="Сравнение AUC объёмов опухоли (отн.)", x_label="Суммарная доза, Гр", y_label="AUC (отн. ед.)"):
    from utils.plotting_helpers import format_experiment_params
    import re
    def extract_total_dose(experiment_params):
        total = 0.0
        for p in experiment_params:
            if '=' in p and ('Гр' in p or 'Gy' in p or 'гр' in p or 'gy' in p):
                try:
                    value = re.findall(r'[-+]?\d*\.\d+|\d+', p)
                    if value:
                        total += float(value[0].replace(',', '.'))
                except Exception:
                    continue
        return total if total > 0 else None
    plot_group_auc_barplot(
        file_paths,
        TumorDataVisualizer,
        lambda vis: vis.data_processor.get_relative_tumor_volumes(),
        lambda vis: vis.time_data,
        lambda vis: format_experiment_params(vis.experiment_params),
        lambda vis: extract_total_dose(vis.experiment_params),
        title=title,
        y_label=y_label,
        x_label=x_label,
        relative=False # относительность уже учтена в данных
    )


if __name__ == '__main__':
    # Используем с файлом данных
    #file_path = r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\16.03.2023_n_22.xlsx'
    file_path = r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_12.xlsx'
    #file_path = r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_18.xlsx'

    visualizer = TumorDataVisualizer(file_path)
    # ExtractOutliers(visualizer).exclude_rats(['пл', 'г'], 'tumor_volumes')  # for p_25.2_n_7.2_2023.xlsx
    # ExtractOutliers(visualizer).exclude_rats(['г- пл'], 'tumor_volumes')  # for n_7.2_p_25.2_2023_2.xlsx
    outlier_extractor = ExtractOutliers(visualizer)
    #outlier_extractor.remove_outliers_elliptic_envelope(contamination=0.05)  # Применение метода Гаусса
    #outlier_extractor.remove_outliers_isolation_forest(contamination=0.05)  # Применение метода изоляции леса
    #outlier_extractor.remove_outliers()
    #outlier_extractor.remove_outliers_iqr()
    #outlier_extractor.remove_outliers_grubbs()
    #outlier_extractor.remove_outliers_mahalanobis(alpha=0.05)
    # Сохраняем график для каждой крысы
    visualizer.plot_tumor_volumes_single_graph()

    # Сохраняем график относительных объемов для каждой крысы
    visualizer.plot_relative_tumor_volumes_single_graph()

    # Сохраняем график средних значений
    visualizer.plot_mean_tumor_volume()

    # Сохраняем график среднего относительного объема опухоли
    visualizer.plot_average_relative_tumor_volume()

    # Сохраняем график среднего относительного усреднённого объема опухоли
    #visualizer.plot_mean_relative_mean_tumor_volume()
