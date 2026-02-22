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
from utils.visualizer import GraphVisualizer

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

        # Проверяем, нужно ли вычислять AUC
        calculate_auc = getattr(self, 'use_AUC', False)

        # Добавление данных на график с уникальной меткой
        drawgraph.add_plot(self.time_data, mean_volumes, {}, unique_label, error_margin, calculate_auc=calculate_auc)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_mean_volumes", legend_fontsize=18)

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

        # Проверяем, нужно ли вычислять AUC
        calculate_auc = getattr(self, 'use_AUC', False)

        # Добавление данных на график с уникальной меткой
        drawgraph.add_plot(self.time_data, mean_relative_volumes, {}, unique_label,
                           error_margin_rel, calculate_auc=calculate_auc)

        # Увеличиваем размер шрифта легенды для относительных графиков
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_average_relative_volumes", legend_fontsize=18)

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
                             y_label="AUC (абс. ед.)",
                             perform_stat_test: bool = False,
                             control_index: int = 0,
                             control_groups_info: dict = None,
                             show_separate_legend: bool = False):
        """
        Построение столбчатого графика для сравнения площади под кривой
        объёмов опухоли между экспериментами.

        Args:
            file_paths: Пути к файлам с данными экспериментов.
            title: Заголовок графика.
            x_label: Подпись оси X.
            y_label: Подпись оси Y.
            perform_stat_test: Если True, применяется критерий Манна-Уитни.
            control_index: Индекс контрольной группы в списке file_paths (устарел, используйте control_groups_info).
            control_groups_info: Словарь {control_type: [indices]} для множественных контролей.
            show_separate_legend: Если True, легенда выводится в отдельное окно предпросмотра.
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
            # Сначала собираем все данные по файлам
            file_data = []  # [(dose, auc_mean, auc_sem, label, individual_aucs, color), ...]
            colors = sns.color_palette("Set3", n_colors=len(file_paths))

            for file_path, color in zip(file_paths, colors):
                visualizer = TumorDataVisualizer(file_path)
                try:
                    individual_volumes = visualizer.tumor_volumes
                    time_data = list(map(int, visualizer.time_data))
                    # Собираем пересечение временных точек для всех животных
                    all_time_points = [set([int(t) for t in time_data]) for _ in individual_volumes]
                    common_time_points = sorted(set.intersection(*all_time_points))
                    # Для каждого животного: получить значения только в этих точках
                    aligned_curves = []
                    for rat_curve in individual_volumes:
                        aligned_curve = []
                        for t in common_time_points:
                            if t in time_data:
                                idx = time_data.index(t)
                                aligned_curve.append(rat_curve[idx])
                            else:
                                aligned_curve.append(np.nan)
                        aligned_curves.append(aligned_curve)
                    aligned_curves = np.array(aligned_curves)
                    mean_curve = np.nanmean(aligned_curves, axis=0)
                    auc_mean = SupportingFunctions.calculate_auc(mean_curve, common_time_points)
                    individual_aucs = [SupportingFunctions.calculate_auc(curve, common_time_points) for curve in aligned_curves]
                    auc_sem = np.std(individual_aucs, ddof=1) / np.sqrt(len(aligned_curves))
                    label = format_experiment_params(visualizer.experiment_params)
                    dose = extract_total_dose(visualizer.experiment_params)
                    # Если доза не найдена (контрольная группа), используем 0
                    if dose is None:
                        dose = 0

                    file_data.append({
                        'dose': dose,
                        'auc_mean': auc_mean,
                        'auc_sem': auc_sem,
                        'label': label,
                        'individual_aucs': individual_aucs,
                        'color': color,
                        'file_index': len(file_data)  # Индекс для Mann-Whitney
                    })
                except ValueError as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")
                    continue

            # Группируем файлы по дозе
            from collections import defaultdict
            dose_groups = defaultdict(list)
            for data in file_data:
                dose_groups[data['dose']].append(data)

            # Подготовка данных для отрисовки
            unique_doses = sorted(dose_groups.keys())
            all_individual_aucs = [d['individual_aucs'] for d in file_data]  # Для Mann-Whitney

            # --- Barplot по числовой оси X (doses) ---
            plt.figure(figsize=(12, 8))
            bar_width = 2.5 if len(unique_doses) < 10 else 0.8

            # Паттерны для разделения файлов в одной дозе
            hatches = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo']

            # Рисуем столбцы для каждой дозы
            bars = []
            dose_positions = []
            for dose in unique_doses:
                files_in_dose = dose_groups[dose]

                if len(files_in_dose) == 1:
                    # Один файл - простой столбец
                    data = files_in_dose[0]
                    bar = plt.bar(dose, data['auc_mean'], yerr=data['auc_sem'],
                                 width=bar_width, color=data['color'],
                                 edgecolor="black", zorder=2, capsize=8)

                    # Подпись AUC НАД столбцом (над error bar)
                    label_y = data['auc_mean'] + data['auc_sem'] + 0.02 * data['auc_mean']
                    plt.text(dose, label_y, f"{data['auc_mean']:.2f}",
                            ha='center', va='bottom', fontsize=12,
                            fontweight='bold', color='black')

                    bars.append((bar, dose, data['auc_mean'], data['auc_sem']))
                else:
                    # Несколько файлов - отдельные столбцы с промежутками (stacked с gap)
                    # Находим максимальный AUC для расчета промежутка
                    max_auc_in_group = max([d['auc_mean'] for d in files_in_dose])
                    gap = max_auc_in_group * 0.5  # 50% от максимального AUC как промежуток

                    # Рисуем столбцы с промежутками
                    bottom = 0
                    max_height_with_error = 0
                    for idx, data in enumerate(files_in_dose):
                        hatch = hatches[idx % len(hatches)]
                        segment_height = data['auc_mean']

                        # Рисуем столбец с error bar
                        plt.bar(dose, segment_height, bottom=bottom,
                               width=bar_width, color=data['color'],
                               hatch=hatch, edgecolor="black", linewidth=1.5,
                               yerr=data['auc_sem'], capsize=8, zorder=2,
                               error_kw={'ecolor': 'black', 'linewidth': 2, 'zorder': 3})

                        # Подпись AUC НАД столбцом (над error bar)
                        label_y = bottom + segment_height + data['auc_sem'] + 0.02 * max_auc_in_group
                        plt.text(dose, label_y, f"{data['auc_mean']:.2f}",
                                ha='center', va='bottom', fontsize=12,
                                fontweight='bold', color='black', zorder=10)

                        # Обновляем максимальную высоту для Mann-Whitney
                        current_top = bottom + segment_height + data['auc_sem']
                        if current_top > max_height_with_error:
                            max_height_with_error = current_top

                        # Добавляем промежуток после каждого столбца
                        bottom += segment_height + gap

                    # Сохраняем максимальную высоту для размещения символов
                    bars.append((None, dose, max_height_with_error, 0))

                dose_positions.append(dose)

            plt.xlabel("Суммарная доза, Гр", fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.title(title, fontsize=16)

            # Подписи с дозами на тиках оси X
            dose_labels = ["Контроль" if d == 0 else str(d) for d in unique_doses]
            plt.xticks(unique_doses, dose_labels, fontsize=12)

            # Легенда
            legend_patches = []
            for data in file_data:
                legend_patches.append(mpatches.Patch(color=data["color"], label=data["label"]))

            if show_separate_legend:
                # Легенда в отдельном окне - скрываем на графике
                legend = plt.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0.0, -0.05),
                               ncol=1, fontsize=11, frameon=False, handletextpad=0.8)
                legend.set_visible(False)
            else:
                # Легенда под графиком - выравнивание по левому краю (где начало оси X)
                legend = plt.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0.0, -0.05),
                               ncol=1, fontsize=11, frameon=False, handletextpad=0.8)


            # Критерий Манна-Уитни
            if perform_stat_test and len(all_individual_aucs) > 1:
                from scipy.stats import mannwhitneyu
                y_max = max([b[2] for b in bars]) if bars else 0
                y_offset = y_max * 0.05

                # Определяем символы для контролей
                control_symbols = {1: '*', 2: '^', 3: '#'}

                # Если указаны множественные контроли, используем их
                if control_groups_info:
                    for file_idx, data in enumerate(file_data):
                        # Проверяем, не является ли текущий файл контрольным
                        is_control = any(file_idx in indices for indices in control_groups_info.values())
                        if is_control:
                            continue

                        # Сравниваем с каждой контрольной группой
                        symbols_to_add = []
                        for control_type, control_indices in sorted(control_groups_info.items()):
                            for control_idx in control_indices:
                                if control_idx < len(all_individual_aucs):
                                    control_aucs = all_individual_aucs[control_idx]
                                    exp_aucs = all_individual_aucs[file_idx]
                                    try:
                                        _, p_value = mannwhitneyu(control_aucs, exp_aucs, alternative='two-sided')
                                        if p_value < 0.05:
                                            symbol = control_symbols.get(control_type, '*')
                                            if symbol not in symbols_to_add:
                                                symbols_to_add.append(symbol)
                                    except Exception as e:
                                        print(f"Ошибка при выполнении теста для файла {file_idx} с контролем {control_type}: {e}")

                        # Добавляем символы над столбцом
                        if symbols_to_add:
                            dose = data['dose']
                            # Находим столбец для этой дозы
                            for bar_tuple in bars:
                                if bar_tuple[1] == dose:
                                    y_position = bar_tuple[2] + bar_tuple[3] + y_offset
                                    symbols_text = ''.join(symbols_to_add)
                                    plt.text(dose, y_position, symbols_text,
                                           ha='center', va='bottom',
                                           fontsize=20, color='black', fontweight='bold')
                                    break

                # Если используется старый формат (один контроль)
                elif control_index < len(all_individual_aucs):
                    control_aucs = all_individual_aucs[control_index]
                    for file_idx, data in enumerate(file_data):
                        if file_idx == control_index:
                            continue

                        if file_idx < len(all_individual_aucs):
                            exp_aucs = all_individual_aucs[file_idx]
                            try:
                                _, p_value = mannwhitneyu(control_aucs, exp_aucs, alternative='two-sided')
                                if p_value < 0.05:
                                    dose = data['dose']
                                    # Находим столбец для этой дозы
                                    for bar_tuple in bars:
                                        if bar_tuple[1] == dose:
                                            y_position = bar_tuple[2] + bar_tuple[3] + y_offset
                                            plt.text(dose, y_position, '*',
                                                   ha='center', va='bottom',
                                                   fontsize=20, color='black', fontweight='bold')
                                            break
                            except Exception as e:
                                print(f"Ошибка при выполнении теста для файла {file_idx}: {e}")

                # Добавляем пояснение символов, если используются множественные контроли
                if control_groups_info and len(control_groups_info) > 0:
                    explanation_text = "Статистическая значимость (p<0.05): "
                    symbols_used = []
                    control_symbols = {1: '*', 2: '^', 3: '#'}
                    control_names = {1: 'Контроль 1', 2: 'Контроль 2', 3: 'Контроль 3'}
                    for control_type in sorted(control_groups_info.keys()):
                        symbol = control_symbols.get(control_type, '*')
                        name = control_names.get(control_type, f'Контроль {control_type}')
                        symbols_used.append(f"{symbol} - {name}")
                    explanation_text += ", ".join(symbols_used)
                    plt.figtext(0.5, 0.02, explanation_text, ha='center', fontsize=10, style='italic')

            if show_separate_legend:
                plt.tight_layout()  # Легенда отдельно - не нужно дополнительное место
            else:
                # Вычисляем нужное место в зависимости от количества элементов легенды
                legend_space = 0.05 + len(file_data) * 0.03  # Базовый отступ + по 3% на элемент
                plt.tight_layout(rect=[0, legend_space, 1, 1])  # Больше места снизу для легенды



    def plot_pairwise_divergence_individual(self):
        """
        Визуализирует попарное расхождение d(t) для каждой пары крыс — индивидуальные кривые.

        Модернизированная формула (Кизилова, 2026):
            d(t) = 2 * |V₁(t) − V₂(t)| / (V₁(t) + V₂(t))

        Для каждой пары (i, j) строится отдельная кривая на одном графике. Данные предварительно
        нормируются на начальный объём (V/V₀). Позволяет оценить межособевую вариабельность
        в контрольной (необлучённой) группе крыс.
        """
        relative_volumes = self.data_processor.get_relative_tumor_volumes()
        pairs = SupportingFunctions.calculate_pairwise_divergence(relative_volumes, self.rat_labels)

        drawgraph = GraphVisualizer(
            "Попарное расхождение d(t) (индивидуальные пары)",
            "Временная точка, сут.",
            "d(t), отн. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        pair_labels = [f"{li}–{lj}" for li, lj, _ in pairs]
        pair_curves = [d_vals for _, _, d_vals in pairs]
        drawgraph.add_individual_plots(pair_labels, pair_curves, self.time_data)

        formatted_params = format_experiment_params(self.experiment_params)
        drawgraph.add_legend([formatted_params], "Параметры эксперимента", "upper center", display_marker=False)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_pairwise_divergence_individual", "Пара крыс")

    def plot_pairwise_divergence_mean(self):
        """
        Визуализирует среднее попарное расхождение d(t) по всем парам крыс с доверительным интервалом.

        Модернизированная формула (Кизилова, 2026):
            d(t) = 2 * |V₁(t) − V₂(t)| / (V₁(t) + V₂(t))

        Усредняется по всем C(N,2) парам. Отображается одна кривая среднего d(t) ± SEM.
        Позволяет в едином числе оценить уровень межособевой вариабельности группы.
        """
        relative_volumes = self.data_processor.get_relative_tumor_volumes()
        pairs = SupportingFunctions.calculate_pairwise_divergence(relative_volumes, self.rat_labels)

        if not pairs:
            return

        # Матрица d(t): (n_пар, n_точек)
        d_matrix = np.array([d_vals for _, _, d_vals in pairs], dtype=float)
        mean_d = np.nanmean(d_matrix, axis=0)
        n_pairs = d_matrix.shape[0]
        std_d = [SupportingFunctions.calculate_std_dev(
            [d_matrix[p, t] for p in range(n_pairs) if not np.isnan(d_matrix[p, t])],
            mean_d[t]
        ) if n_pairs > 1 else 0.0 for t in range(d_matrix.shape[1])]
        sem_d = [SupportingFunctions.calculate_error_margin(s, n_pairs) for s in std_d]

        drawgraph = GraphVisualizer(
            "Среднее попарное расхождение d(t)",
            "Временная точка, сут.",
            "d(t), отн. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        formatted_params = format_experiment_params(self.experiment_params)
        unique_label = f"Среднее d(t): {formatted_params}"
        drawgraph.add_plot(self.time_data, mean_d.tolist(), {}, unique_label, sem_d)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_pairwise_divergence_mean", legend_fontsize=18)

    def plot_cv(self):
        """
        Визуализирует коэффициент вариации CV(t) по группе крыс.

        Стандартная формула:
            CV(t) = σ(t) / μ(t) × 100%

        Данные предварительно нормируются (V/V₀). Одна кривая CV в % от времени.
        Применяется при N > 2 как обобщённая мера межособевой вариабельности группы.
        """
        relative_volumes = self.data_processor.get_relative_tumor_volumes()
        cv_values, _ = SupportingFunctions.calculate_cv(relative_volumes)

        drawgraph = GraphVisualizer(
            "Коэффициент вариации CV(t)",
            "Временная точка, сут.",
            "CV(t), %",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        formatted_params = format_experiment_params(self.experiment_params)
        unique_label = f"CV(t): {formatted_params}"
        drawgraph.add_plot(self.time_data, cv_values, {}, unique_label, None)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_cv", legend_fontsize=18)

    def plot_relative_divergence_per_rat(self):
        """
        Визуализирует вариабельность группы в зависимости от числа крыс N.

        N=2: одна кривая попарного расхождения
            d(1,2,t) = 2|V₁(t) − V₂(t)| / (V₁(t) + V₂(t)) × 100%
            Формула симметрична, поэтому D₁ ≡ D₂ — рисуется одна линия.

        N>2: одна кривая коэффициента вариации по группе
            CV(t) = σ(t) / μ(t) × 100%
            где σ(t) — СКО, μ(t) — среднее нормированных объёмов в точке t.

        На графике всегда отображаются:
        - Одна кривая вариабельности
        - Точечная горизонтальная линия — среднее за период с подписью «Ср. за период: X%»
        """
        from matplotlib.lines import Line2D
        relative_volumes = self.data_processor.get_relative_tumor_volumes()

        if len(self.rat_labels) < 2:
            return

        if len(self.rat_labels) == 2:
            # N=2: попарное расхождение d(1,2,t)
            pairs = SupportingFunctions.calculate_pairwise_divergence(relative_volumes, self.rat_labels)
            if not pairs:
                return
            li, lj, d_vals = pairs[0]
            curve = [v * 100.0 for v in d_vals]
            curve_label = "Крыса 1–Крыса 2"
            legend_title = (
                "$d_{1,2}=\\dfrac{2|V_1-V_2|}{V_1+V_2}$\n"
                "попарное расхождение двух крыс"
            )
            y_label = "d(t), %"
        else:
            # N>2: коэффициент вариации по группе CV(t)
            cv_values, _ = SupportingFunctions.calculate_cv(relative_volumes)
            curve = cv_values
            curve_label = "CV(t)"
            legend_title = (
                "$CV(t)=\\sigma(t)/\\mu(t)\\times100\\%$\n"
                "коэффициент вариации группы"
            )
            y_label = "CV(t), %"

        # Глобальное среднее за период
        overall_mean = float(np.nanmean(np.array(curve, dtype=float)))

        drawgraph = GraphVisualizer(
            "Вариабельность группы по временны\u0301м точкам",
            "Временная точка, сут.",
            y_label,
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        drawgraph.add_individual_plots([curve_label], [curve], self.time_data)

        # Горизонталь среднего за период
        plt.axhline(y=overall_mean, linestyle=':', color='#7b68ee', linewidth=1.5, zorder=1)
        horiz_label = f"Ср. за период: {overall_mean:.1f}%"
        horiz_proxy = Line2D([0], [0], linestyle=':', color='#7b68ee', linewidth=1.5,
                             label=horiz_label)
        drawgraph.lines.append(horiz_proxy)

        drawgraph.finalize_figure(
            f"{', '.join(self.experiment_params)}_relative_divergence_per_rat",
            legend_title,
            legend_fontsize=18
        )


    @staticmethod
    def plot_measurement_divergence(paths_a: list, paths_b: list):
        """
        Строит график расхождения замеров между двумя группами (A и B).

        Для каждой пары файл_A[i] ↔ файл_B[i] вычисляет кривую d(t):
            d(t) = 2|V_A(t) − V_B(t)| / (V_A(t) + V_B(t)) × 100%
        где V — нормированный объём (V/V₀).

        Дополнительно отображает:
        - Среднюю кривую d(t) по всем парам (чёрный пунктир)
        - Горизонтальную линию глобального среднего (фиолетовая точечная)

        Args:
            paths_a: Пути к файлам группы A (замер 1, например ваши измерения).
            paths_b: Пути к файлам группы B (замер 2, например МРТ студентки).
        """
        import os
        n_pairs = min(len(paths_a), len(paths_b))
        if n_pairs == 0:
            return

        all_d_curves = []
        pair_labels = []

        for i in range(n_pairs):
            vis_a = TumorDataVisualizer(paths_a[i])
            vis_b = TumorDataVisualizer(paths_b[i])

            # Используем первую (и обычно единственную) крысу в каждом файле
            vols_a = np.array(vis_a.data_processor.get_relative_tumor_volumes(), dtype=float)
            vols_b = np.array(vis_b.data_processor.get_relative_tumor_volumes(), dtype=float)

            # Усредняем по крысам внутри файла (если их несколько)
            mean_a = np.nanmean(vols_a, axis=0)
            mean_b = np.nanmean(vols_b, axis=0)

            denom = mean_a + mean_b
            with np.errstate(invalid='ignore', divide='ignore'):
                d = np.where(denom == 0, np.nan, 2.0 * np.abs(mean_a - mean_b) / denom * 100.0)

            all_d_curves.append(d.tolist())

            pair_labels.append(f"Пара {i + 1}")

        time_data = TumorDataVisualizer(paths_a[0]).time_data

        d_matrix = np.array(all_d_curves, dtype=float)
        mean_d = np.nanmean(d_matrix, axis=0).tolist()
        overall_mean = float(np.nanmean(np.array(mean_d, dtype=float)))

        drawgraph = GraphVisualizer(
            "Расхождение замеров между группами A и B",
            "Временная точка, сут.",
            "d(t), %",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        # Рисуем каждую пару с is_individual_rat=False — без обращения к кэшу label_styles.
        time_floats = [float(t) for t in time_data]
        for lbl, d_curve in zip(pair_labels, all_d_curves):
            d_arr = np.array(d_curve, dtype=float)
            clean_t = np.array(time_floats)[~np.isnan(d_arr)]
            clean_d = d_arr[~np.isnan(d_arr)]
            drawgraph.add_plot(clean_t, clean_d, {}, lbl, error_margin=None, is_individual_rat=False)

        # Горизонтальная линия глобального среднего
        horiz_label = f"Ср. за период: {overall_mean:.1f}%"
        plt.axhline(y=overall_mean, linestyle=':', color='#7b68ee', linewidth=1.5, zorder=1)
        from matplotlib.lines import Line2D
        horiz_proxy = Line2D([0], [0], linestyle=':', color='#7b68ee', linewidth=1.5,
                             label=horiz_label)
        drawgraph.lines.append(horiz_proxy)

        # Средняя линия нужна только при n_pairs > 1 (при одной паре среднее = та же кривая)
        if n_pairs > 1:
            mean_line, = plt.plot(
                time_floats, mean_d,
                linestyle='--', color='black', linewidth=2,
                label='Среднее', zorder=3
            )
            drawgraph.lines.append(mean_line)

        drawgraph.finalize_figure(
            "measurement_divergence_A_vs_B",
            "$d_{A,B}=\\dfrac{2|V_A-V_B|}{V_A+V_B}$\nрасхождение замеров A и B",
            legend_fontsize=18
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
