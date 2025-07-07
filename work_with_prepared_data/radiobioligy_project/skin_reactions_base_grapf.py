# файл skin_reactions_base_grapf.py

import os
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
import pandas as pd
from utils.plotting_helpers import format_experiment_params, MatplotlibConfigurator, custom_fill_between
from stats_methods.support_stats_methods import SupportingFunctions
from data_processing.excel_data_processor import process_skin_data_excel
from data_processing.data_processing import SkinReactionsDataProcessor
from utils.visualizer import GraphVisualizer

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()

class SkinReactionsVisualizer:
    def __init__(self, file_path: str):
        """
        Инициализирует визуализатор кожных реакций на основе данных из файла Excel.

        Этот конструктор загружает данные о кожных реакциях из указанного Excel файла и инициализирует обработчик данных
        для дальнейшей работы с этими данными.

        Args:
            file_path (str): Путь к файлу Excel с данными о кожных реакциях. Файл должен содержать информацию
                             об экспериментальных параметрах, временных точках, метках крыс и данных о кожных реакциях.

        Attributes:
            file_path (str): Хранит путь к исходному файлу данных.
            experiment_params (list): Список параметров эксперимента.
            time_data (list): Список временных точек измерения.
            rat_labels (list): Список меток крыс, участвующих в эксперименте.
            skin_reactions (list): Список данных о кожных реакциях для каждой крысы.
            data_processor (SkinReactionsDataProcessor): Объект для обработки данных о кожных реакциях.

        Пример использования:
            visualizer = SkinReactionsVisualizer("путь/к/файлу.xlsx")
            visualizer.plot_skin_reactions()  # Визуализация данных о кожных реакциях
        """
        self.file_path = file_path
        self.experiment_params, self.time_data, self.rat_labels, self.skin_reactions = process_skin_data_excel(
            file_path)
        self.data_processor = SkinReactionsDataProcessor(self.skin_reactions)

    def plot_skin_reactions(self):
        """
        Визуализирует кожные реакции для каждой крысы в виде графика.

        Данный метод создает график, на котором для каждой крысы из эксперимента отображается изменение кожных реакций
        во времени. График включает в себя данные о кожных реакциях, а также метки времени измерений.

        Визуализация помогает в анализе динамики кожных реакций в рамках проведенного эксперимента, а также позволяет
        сравнить реакции между разными крысами.

        Args:
            Не принимает аргументов.

        Returns:
            Ничего не возвращает. Результатом выполнения является отображение графика с кожными реакциями.

        Пример использования:
            visualizer = SkinReactionsVisualizer("путь/к/файлу.xlsx")
            visualizer.plot_skin_reactions()  # Визуализация кожных реакций
        """
        drawgraph = GraphVisualizer(
            f"Кожные реакции, Параметры эксперимента: {format_experiment_params(self.experiment_params)}",
            "Время, сут.",
            "Кожные реакции, абс. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        # Итерация по крысам и их кожным реакциям для добавления на график
        for label, reactions in zip(self.rat_labels, self.skin_reactions):
            clean_reactions = np.array(reactions)[~np.isnan(reactions)]
            clean_time_data = np.array(self.time_data)[~np.isnan(reactions)]

            if not list(clean_reactions):
                continue

            # Добавление данных на график
            drawgraph.add_plot(clean_time_data, clean_reactions, {}, label)
        drawgraph.finalize_figure(self.file_path, 'Метки крыс', 2, 25)

    def plot_mean_skin_reactions(self):
        """
            Визуализация средних кожных реакций всех крыс на одном графике.

            Этот метод строит график, который демонстрирует средние значения кожных реакций для группы крыс в рамках
            эксперимента по времени. На графике также отображаются доверительные интервалы для оценки разброса значений
            кожных реакций в группе.

            График средних кожных реакций позволяет оценить общую тенденцию изменения кожных реакций в зависимости от времени
            и может быть использован для сравнения с другими экспериментальными группами или условиями.

            Args:
                Не принимает аргументов.

            Returns:
                Ничего не возвращает. Результатом выполнения является отображение графика со средними кожными реакциями.

            Пример использования:
                visualizer = SkinReactionsVisualizer("путь/к/файлу.xlsx")
                visualizer.plot_mean_skin_reactions()  # Визуализация средних кожных реакций
            """
        drawgraph = GraphVisualizer(
            f"Средние кожные реакции, Параметры эксперимента: {format_experiment_params(self.experiment_params)}",
            "Время, сут.",
            "Средние кожные реакции, абс. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()
        # Получение средних кожных реакций и их статистических характеристик
        mean_reactions, std_dev, error_margin = self.data_processor.get_mean_skin_reactions()
        # Добавление данных на график
        drawgraph.add_plot(self.time_data, mean_reactions, self.experiment_params, "", None)
        drawgraph.finalize_figure('', '', 1, 25)

    @staticmethod
    def plot_multiple_experiments(file_paths: List[str], use_AUC: bool = False, apply_statistical_test=False):
        """
        Построение графика для сравнения кожных реакций между экспериментами
        с использованием интерполяции, отображением AUC и опционального применения статистических тестов Манна-Уитни.

        Args:
            file_paths (List[str]): Пути к файлам данных экспериментов.
            use_AUC (bool): Если True, вычисляется и отображается площадь под кривой (AUC).
            apply_statistical_test (bool): Флаг для выполнения теста Манна-Уитни.
                                           Если True, тест выполняется и значимые различия отображаются на графике.
        """
        # Инициализация объекта GraphVisualizer с согласованным стилем
        drawgraph = GraphVisualizer(
            "Сравнение кожных реакций между экспериментами",
            "Время, сут.",
            "Кожные реакции, усл. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        # Подготовка данных для каждого эксперимента
        all_reactions = []
        common_timepoints = list(range(0, 25))  # Временные точки от 0 до 24 с шагом 1

        # Создаем словарь для хранения верхних границ доверительных интервалов
        upper_bounds_by_time = {}

        for file_path in file_paths:
            visualizer = SkinReactionsVisualizer(file_path)

            try:
                # Получаем индивидуальные данные реакций кожи
                individual_skin_reactions = visualizer.skin_reactions
                time_data = np.array(visualizer.time_data, dtype=float)

                # Интерполяция индивидуальных данных на общие временные точки
                interpolated_skin_reactions = []
                for reaction in individual_skin_reactions:
                    interpolated_reaction = SupportingFunctions.interpolate_data_to_common_timepoints(
                        time_data, reaction, common_timepoints
                    )
                    interpolated_skin_reactions.append(interpolated_reaction)

                # Вычисляем среднюю реакцию и SEM
                reactions = np.array(interpolated_skin_reactions)
                mean_reaction = np.nanmean(reactions, axis=0)
                std_reaction = np.nanstd(reactions, axis=0)
                sem_reaction = std_reaction / np.sqrt(len(reactions))

                # Сохраняем верхние границы доверительных интервалов для каждой временной точки
                for t_idx, t in enumerate(common_timepoints):
                    upper_bound = mean_reaction[t_idx] + sem_reaction[t_idx]
                    if t in upper_bounds_by_time:
                        # Обновляем максимальное значение, если текущая верхняя граница больше
                        upper_bounds_by_time[t] = max(upper_bounds_by_time[t], upper_bound)
                    else:
                        upper_bounds_by_time[t] = upper_bound

                # Рассчитываем error_margin
                error_margin = [SupportingFunctions.calculate_error_margin(std, len(reactions)) for std in
                                std_reaction]

                # Сохраняем данные для дальнейшего анализа
                all_reactions.append({
                    'reactions': reactions,  # Индивидуальные реакции
                    'mean_reaction': mean_reaction,
                    'sem_reaction': sem_reaction,
                    'label': format_experiment_params(visualizer.experiment_params)
                })

                # Добавляем график с использованием GraphVisualizer
                drawgraph.add_plot(
                    common_timepoints,
                    mean_reaction,
                    params={},
                    label=format_experiment_params(visualizer.experiment_params),
                    error_margin=error_margin,
                    calculate_auc=use_AUC
                )

            except ValueError as e:
                print(f"Ошибка при обработке файла {file_path}: {e}")
                continue

        # Если тест Манна-Уитни включен
        if apply_statistical_test:
            # Передаем upper_bounds_by_time в функцию
            SupportingFunctions.apply_mann_whitney_test(all_reactions, common_timepoints, upper_bounds_by_time,
                                                        offset_ratio=0.00, annotation_fontsize=18)

        # Финализация и сохранение графика
        base_file_name = '_'.join([os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]) + "_comparison.png"
        drawgraph.finalize_figure(base_file_name, ncol=1, legend_fontsize=20)


    @staticmethod
    def plot_auc_comparison(file_paths: List[str], title="Сравнение AUC кожных реакций", x_label="",
                            y_label="AUC (усл. ед.)"):
        """
        Построение столбчатого графика для сравнения AUC кожных реакций между экспериментами с легендой.

        Args:
            file_paths (List[str]): Список путей к файлам с данными экспериментов.
            title (str): Заголовок графика.
            x_label (str): Подпись оси X.
            y_label (str): Подпись оси Y.

        Returns:
            Ничего не возвращает. Результатом является отображение и сохранение столбчатого графика.
        """

        with sns.axes_style("whitegrid"):
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
            doses = []
            aucs_for_fit = []
            labels_for_legend = []
            auc_values = []
            colors = sns.color_palette("Set3", n_colors=len(file_paths))
            for file_path, color in zip(file_paths, colors):
                visualizer = SkinReactionsVisualizer(file_path)
                try:
                    mean_reactions, std_dev, _ = visualizer.data_processor.get_mean_skin_reactions()
                    interpolated_values = SupportingFunctions.interpolate_data_to_common_timepoints(
                        visualizer.time_data,
                        mean_reactions,
                        list(range(0, 25))
                    )
                    auc = SupportingFunctions.calculate_auc(interpolated_values, list(range(0, 25)))
                    auc_values.append(auc)
                    experiment_label = format_experiment_params(visualizer.experiment_params)
                    labels_for_legend.append(experiment_label)
                    dose = extract_total_dose(visualizer.experiment_params)
                    if dose is not None:
                        doses.append(dose)
                        aucs_for_fit.append(auc)
                except ValueError as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")
                    continue
            # --- Barplot по числовой оси X (doses) ---
            plt.figure(figsize=(12, 8))
            bar_width = 2.5 if len(doses) < 10 else 0.8
            bars = plt.bar(doses, aucs_for_fit, width=bar_width, color=colors[:len(doses)], edgecolor="black", zorder=2)
            plt.xlabel("Суммарная доза, Гр", fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.title(title, fontsize=16)
            # Подписи над столбиками
            for i, (bar, auc) in enumerate(zip(bars, aucs_for_fit)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{auc:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
            # Подписи с дозами на тиках оси X
            plt.xticks(doses, [str(d) for d in doses], fontsize=12)
            # Легенда как раньше
            legend_patches = [mpatches.Patch(color=col, label=lab) for col, lab in zip(colors[:len(labels_for_legend)], labels_for_legend)]
            ncol = math.ceil(len(labels_for_legend) / 2) if len(labels_for_legend) > 4 else len(labels_for_legend)
            plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=ncol, fontsize=12, frameon=False)
            plt.tight_layout()
            # plt.savefig("auc_comparison_plot.png")


if __name__ == '__main__':
    # Пример использования
    file_path = r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_3.xlsx'
    visualizer = SkinReactionsVisualizer(file_path)

    # Удаление выбросов
    # ExtractOutliers(visualizer).remove_local_outliers()

    # Удаление точек
    # ExtractOutliers(visualizer).exclude_rats(['б/м'], 'tumor_volumes')  # for skin_reactions_p_25,2_n_7,2_2023_2.xlsx
    # ExtractOutliers(visualizer).exclude_rats(['г', 'х'], 'tumor_volumes')  # for skin_reactions_n_7.2_p_25.2_2023_2.xlsx

    visualizer.plot_skin_reactions()  # Визуализация индивидуальных кожных реакций
    visualizer.plot_mean_skin_reactions()  # Визуализация средних кожных реакций

    # Отображения средних кожных реакций для нескольких экспериментов
    file_paths = [
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_2.xlsx',
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\skin_reactions\skin_reactions_p_25,2_n_7,2_2023_3.xlsx',
    ]
    SkinReactionsVisualizer.plot_multiple_experiments(file_paths)
