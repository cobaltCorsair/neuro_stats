# файл skin_reactions_base_grapf.py

import os
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
import pandas as pd
from utils.plotting_helpers import format_experiment_params, MatplotlibConfigurator, custom_fill_between, PLOT_FONT_FAMILY
from stats_methods.support_stats_methods import SupportingFunctions
from data_processing.excel_data_processor import process_skin_data_excel
from data_processing.data_processing import SkinReactionsDataProcessor
from utils.visualizer import GraphVisualizer
from work_with_prepared_data.radiobioligy_project.gui import graph_manager

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
        drawgraph.finalize_figure(self.file_path, 'Метки крыс', 2, 18)

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

        # Проверяем, нужно ли вычислять AUC
        calculate_auc = getattr(self, 'use_AUC', False)

        # Добавление данных на график
        drawgraph.add_plot(self.time_data, mean_reactions, self.experiment_params, "", None, calculate_auc=calculate_auc)
        drawgraph.finalize_figure('', '', 1, 18)

    @staticmethod
    def plot_multiple_experiments_from_visualizers(visualizers: List['SkinReactionsVisualizer'], use_AUC: bool = False, apply_statistical_test: bool = False):
        """
        Сравнение кожных реакций между экспериментами используя уже созданные и модифицированные визуализаторы.
        Этот метод позволяет использовать визуализаторы с уже применёнными исключениями крыс.

        Args:
            visualizers (List[SkinReactionsVisualizer]): Список визуализаторов (могут быть модифицированы).
            use_AUC (bool): Если True, вычисляется и отображается площадь под кривой (AUC).
            apply_statistical_test (bool): Флаг для выполнения теста Манна-Уитни.
        """
        # Инициализация объекта GraphVisualizer
        drawgraph = GraphVisualizer(
            "Сравнение кожных реакций между экспериментами",
            "Время, сут.",
            "Кожные реакции, усл. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        def _to_float_list(seq):
            out = []
            for x in seq:
                s = str(x).strip()
                if s == "" or s.lower() in ("nan", "none"):
                    out.append(float('nan'))
                else:
                    try:
                        out.append(float(s.replace(',', '.')))
                    except Exception:
                        out.append(float('nan'))
            return out

        all_experiments = []
        x_data_lists = []

        for vis in visualizers:
            # дни эксперимента (как есть из Excel) и индивидуальные реакции
            time_data = _to_float_list(vis.time_data)
            individual = [_to_float_list(row) for row in vis.skin_reactions]

            # среднее/стд/SEM ПОВЕРХ родной сетки времени
            reactions_arr = np.array(individual, dtype=float)
            mean_reaction = np.nanmean(reactions_arr, axis=0)
            std_reaction = np.nanstd(reactions_arr, axis=0)
            sem_reaction = std_reaction / np.sqrt(len(reactions_arr))

            # error bars через функцию погрешности
            error_margin = [SupportingFunctions.calculate_error_margin(s, len(reactions_arr)) for s in std_reaction]

            label_text = format_experiment_params(vis.experiment_params)

            # рисуем на СВОИХ днях
            drawgraph.add_plot(
                time_data,
                mean_reaction.tolist(),
                params={},
                label=label_text,
                error_margin=error_margin,
                calculate_auc=use_AUC
            )

            all_experiments.append({
                "time_data": time_data,
                "reactions": reactions_arr,
                "mean_reaction": mean_reaction,
                "sem_reaction": sem_reaction,
                "label": label_text
            })
            x_data_lists.append(time_data)

        # авто-границы осей с учётом всех X
        drawgraph.update_axes_limits(x_data_lists)

        # опциональная статистика Манна–Уитни
        if apply_statistical_test and len(all_experiments) >= 2:
            ref_time = all_experiments[0]["time_data"]
            upper_bounds_by_time = {}
            aligned_for_test = []

            for exp in all_experiments:
                aligned_individual = [
                    SupportingFunctions.interpolate_data_to_common_timepoints(
                        exp["time_data"], row.tolist(), ref_time
                    )
                    for row in exp["reactions"]
                ]
                aligned_individual = np.array(aligned_individual, dtype=float)

                mean_aligned = np.nanmean(aligned_individual, axis=0)
                std_aligned = np.nanstd(aligned_individual, axis=0)
                sem_aligned = std_aligned / np.sqrt(len(aligned_individual))

                for i, t in enumerate(ref_time):
                    ub = mean_aligned[i] + sem_aligned[i]
                    upper_bounds_by_time[t] = max(upper_bounds_by_time.get(t, -np.inf), ub)

                aligned_for_test.append({
                    "reactions": aligned_individual,
                    "mean_reaction": mean_aligned,
                    "sem_reaction": sem_aligned,
                    "label": exp["label"]
                })

            SupportingFunctions.apply_mann_whitney_test(
                aligned_for_test,
                ref_time,
                upper_bounds_by_time,
                offset_ratio=0.00,
                annotation_fontsize=18
            )

        # финализация
        base = "skin_reactions_comparison.png"
        drawgraph.finalize_figure(base, ncol=1, legend_fontsize=18)

    @staticmethod
    def plot_multiple_experiments(file_paths: List[str], use_AUC: bool = False, apply_statistical_test: bool = False):
        """
        Сравнение кожных реакций между экспериментами без общей сетки времени.
        Каждая кривая рисуется на СВОИХ временных точках. При apply_statistical_test=True
        для теста данные временно выравниваются на сетку 1-го эксперимента.
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

        def _to_float_list(seq):
            out = []
            for x in seq:
                s = str(x).strip()
                if s == "" or s.lower() in ("nan", "none"):
                    out.append(float('nan'))
                else:
                    try:
                        out.append(float(s.replace(',', '.')))
                    except Exception:
                        out.append(float('nan'))
            return out

        all_experiments = []
        x_data_lists = []

        for file_path in file_paths:
            vis = SkinReactionsVisualizer(file_path)

            # дни эксперимента (как есть из Excel) и индивидуальные реакции
            time_data = _to_float_list(vis.time_data)
            individual = [_to_float_list(row) for row in vis.skin_reactions]

            # среднее/стд/SEM ПОВЕРХ родной сетки времени
            reactions_arr = np.array(individual, dtype=float)
            mean_reaction = np.nanmean(reactions_arr, axis=0)
            std_reaction = np.nanstd(reactions_arr, axis=0)
            sem_reaction = std_reaction / np.sqrt(len(reactions_arr))

            # error bars через твою функцию погрешности
            error_margin = [SupportingFunctions.calculate_error_margin(s, len(reactions_arr)) for s in std_reaction]

            label_text = format_experiment_params(vis.experiment_params)

            # рисуем на СВОИХ днях
            drawgraph.add_plot(
                time_data,
                mean_reaction.tolist(),
                params={},
                label=label_text,
                error_margin=error_margin,
                calculate_auc=use_AUC
            )

            all_experiments.append({
                "time_data": time_data,
                "reactions": reactions_arr,  # индивидуальные кривые (на своей сетке)
                "mean_reaction": mean_reaction,  # средняя (на своей сетке)
                "sem_reaction": sem_reaction,
                "label": label_text
            })
            x_data_lists.append(time_data)

        # авто-границы осей с учётом всех X
        drawgraph.update_axes_limits(x_data_lists)

        # ----- опциональная статистика Манна–Уитни -----
        if apply_statistical_test and len(all_experiments) >= 2:
            # опорная сетка — дни 1-го эксперимента
            ref_time = all_experiments[0]["time_data"]

            # верхние границы для размещения аннотаций p-value
            upper_bounds_by_time = {}
            aligned_for_test = []

            for exp in all_experiments:
                # интерполируем каждую индивидуальную кривую на ref_time
                aligned_individual = [
                    SupportingFunctions.interpolate_data_to_common_timepoints(
                        exp["time_data"], row.tolist(), ref_time
                    )
                    for row in exp["reactions"]
                ]
                aligned_individual = np.array(aligned_individual, dtype=float)

                mean_aligned = np.nanmean(aligned_individual, axis=0)
                std_aligned = np.nanstd(aligned_individual, axis=0)
                sem_aligned = std_aligned / np.sqrt(len(aligned_individual))

                # копим верхние границы, чтобы p-аннотации не налезали
                for i, t in enumerate(ref_time):
                    ub = mean_aligned[i] + sem_aligned[i]
                    upper_bounds_by_time[t] = max(upper_bounds_by_time.get(t, -np.inf), ub)

                aligned_for_test.append({
                    "reactions": aligned_individual,  # уже на ref_time
                    "mean_reaction": mean_aligned,
                    "sem_reaction": sem_aligned,
                    "label": exp["label"]
                })

            # функция аннотирования теста по общей сетке
            SupportingFunctions.apply_mann_whitney_test(
                aligned_for_test,
                ref_time,
                upper_bounds_by_time,
                offset_ratio=0.00,
                annotation_fontsize=18
            )

        # финализация
        base = '_'.join([os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]) + "_comparison.png"
        drawgraph.finalize_figure(base, ncol=1, legend_fontsize=18)

    @staticmethod
    def plot_all_individual_curves_from_visualizers(visualizers: List['SkinReactionsVisualizer']):
        """
        Отображает все индивидуальные кривые крыс из одного или нескольких экспериментов.

        Args:
            visualizers (List[SkinReactionsVisualizer]): Список визуализаторов (может быть один).
        """
        # Определяем заголовок в зависимости от количества экспериментов
        if len(visualizers) == 1:
            title = f"Кожные реакции, Параметры эксперимента: {format_experiment_params(visualizers[0].experiment_params)}"
        else:
            title = "Индивидуальные кожные реакции из всех экспериментов"

        drawgraph = GraphVisualizer(
            title,
            "Время, сут.",
            "Кожные реакции, абс. ед.",
            figsize=(12, 7)
        )
        drawgraph.setup_figure()

        # Итерация по всем экспериментам
        for vis in visualizers:
            # Получаем метку эксперимента (если больше одного)
            exp_label = format_experiment_params(vis.experiment_params) if len(visualizers) > 1 else ""

            # Итерация по крысам в текущем эксперименте
            for rat_label, reactions in zip(vis.rat_labels, vis.skin_reactions):
                # Очистка NaN значений
                clean_reactions = np.array(reactions)[~np.isnan(reactions)]
                clean_time_data = np.array(vis.time_data)[~np.isnan(reactions)]

                if not list(clean_reactions):
                    continue

                # Формируем полную метку
                if len(visualizers) > 1:
                    full_label = f"{exp_label} - {rat_label}"
                else:
                    full_label = rat_label

                # Добавление данных на график используя стандартный метод
                drawgraph.add_plot(clean_time_data, clean_reactions, {}, full_label)

        # Финализация с корректными параметрами
        if len(visualizers) == 1:
            drawgraph.finalize_figure(visualizers[0].file_path, 'Метки крыс', 2, 18)
        else:
            base = "all_individual_skin_reactions_comparison.png"
            drawgraph.finalize_figure(base, 'Метки крыс', 2, 18)

    @staticmethod
    def plot_all_individual_curves(file_paths: List[str]):
        """
        Отображает все индивидуальные кривые крыс из множественных экспериментов.

        Args:
            file_paths (List[str]): Пути к файлам данных экспериментов.
        """
        visualizers = [SkinReactionsVisualizer(path) for path in file_paths]
        SkinReactionsVisualizer.plot_all_individual_curves_from_visualizers(visualizers)

    @staticmethod
    def plot_auc_comparison_from_visualizers(visualizers: List['SkinReactionsVisualizer'],
                                            title="Сравнение AUC кожных реакций",
                                            x_label="",
                                            y_label="AUC (усл. ед.)",
                                            perform_stat_test: bool = False,
                                            control_index: int = 0,
                                            control_groups_info: dict = None):
        """
        Сравнение AUC кожных реакций используя уже созданные и модифицированные визуализаторы.

        Args:
            visualizers (List[SkinReactionsVisualizer]): Список визуализаторов (могут быть модифицированы).
            perform_stat_test: Если True, применяется критерий Манна-Уитни.
            control_index: Индекс контрольной группы в списке visualizers (устарел).
            control_groups_info: Словарь {control_type: [indices]} для множественных контролей.
        """
        import re
        with sns.axes_style("whitegrid", rc={'font.family': PLOT_FONT_FAMILY}):
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

            common_timepoints = list(range(0, 25))
            doses, aucs_for_fit, errors_for_fit, labels_for_legend = [], [], [], []
            all_individual_aucs = []  # Для критерия Манна-Уитни
            colors = sns.color_palette("Set3", n_colors=len(visualizers))

            for visualizer, color in zip(visualizers, colors):
                try:
                    individual_skin_reactions = visualizer.skin_reactions
                    time_data = np.array(visualizer.time_data, dtype=float)

                    # Интерполяция
                    interpolated_curves = []
                    for reaction in individual_skin_reactions:
                        interpolated = SupportingFunctions.interpolate_data_to_common_timepoints(
                            time_data, reaction, common_timepoints
                        )
                        interpolated_curves.append(interpolated)

                    interpolated_curves = np.array(interpolated_curves)
                    mean_curve = np.nanmean(interpolated_curves, axis=0)

                    # AUC и ошибка
                    auc_mean = SupportingFunctions.calculate_auc(mean_curve, common_timepoints)
                    auc_individual = [SupportingFunctions.calculate_auc(curve, common_timepoints)
                                      for curve in interpolated_curves]
                    auc_sem = np.std(auc_individual, ddof=1) / np.sqrt(len(auc_individual))

                    dose = extract_total_dose(visualizer.experiment_params)
                    # Если доза не найдена (контрольная группа), используем 0
                    if dose is None:
                        dose = 0
                    doses.append(dose)
                    aucs_for_fit.append(auc_mean)
                    errors_for_fit.append(auc_sem)
                    all_individual_aucs.append(auc_individual)  # Сохраняем для теста
                    labels_for_legend.append(format_experiment_params(visualizer.experiment_params))

                except Exception as e:
                    print(f"Ошибка при обработке визуализатора: {e}")
                    continue

            # Построение графика
            plt.figure(figsize=(12, 8))
            bar_width = 2.5 if len(doses) < 10 else 0.8
            bars = plt.bar(doses, aucs_for_fit, yerr=errors_for_fit, width=bar_width,
                           color=colors[:len(doses)], edgecolor="black", zorder=2, capsize=8)
            plt.xlabel("Суммарная доза, Гр", fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.title(title, fontsize=16)
            dose_labels = ["Контроль" if d == 0 else str(d) for d in doses]
            plt.xticks(doses, dose_labels, fontsize=12)

            # Подписи над столбиками
            for i, (bar, auc, err) in enumerate(zip(bars, aucs_for_fit, errors_for_fit)):
                y_text = bar.get_height() - err - 0.03 * max(aucs_for_fit)
                y_text = max(0, y_text)
                plt.text(bar.get_x() + bar.get_width() / 2, y_text, f"{auc:.2f}",
                         ha='center', va='top', fontsize=12, fontweight='bold', color='black')

            # Легенда
            legend_patches = [mpatches.Patch(color=col, label=lab)
                              for col, lab in zip(colors[:len(labels_for_legend)], labels_for_legend)]
            ncol = math.ceil(len(labels_for_legend) / 2) if len(labels_for_legend) > 4 else len(labels_for_legend)
            legend_position = graph_manager.get_current_legend_position()
            if legend_position is not None:
                plt.legend(handles=legend_patches, loc=legend_position,
                           ncol=ncol, fontsize=12, handletextpad=0.5, columnspacing=2.5)

            # Критерий Манна-Уитни
            if perform_stat_test and len(all_individual_aucs) > 1:
                from scipy.stats import mannwhitneyu
                y_max = max(aucs_for_fit) if aucs_for_fit else 0
                y_offset = y_max * 0.05

                # Определяем символы для контролей
                control_symbols = {1: '*', 2: '^', 3: '#'}

                # Если указаны множественные контроли, используем их
                if control_groups_info:
                    for i, (bar, dose) in enumerate(zip(bars, doses)):
                        # Проверяем, не является ли текущий столбец контрольным
                        is_control = any(i in indices for indices in control_groups_info.values())
                        if is_control:
                            continue

                        # Сравниваем с каждой контрольной группой
                        symbols_to_add = []
                        for control_type, control_indices in sorted(control_groups_info.items()):
                            for control_idx in control_indices:
                                if control_idx < len(all_individual_aucs):
                                    control_aucs = all_individual_aucs[control_idx]
                                    exp_aucs = all_individual_aucs[i]
                                    try:
                                        _, p_value = mannwhitneyu(control_aucs, exp_aucs, alternative='two-sided')
                                        if p_value < 0.05:
                                            symbol = control_symbols.get(control_type, '*')
                                            if symbol not in symbols_to_add:
                                                symbols_to_add.append(symbol)
                                    except Exception as e:
                                        print(f"Ошибка при выполнении теста для дозы {dose} с контролем {control_type}: {e}")

                        # Добавляем символы над столбцом
                        if symbols_to_add:
                            y_position = bar.get_height() + errors_for_fit[i] + y_offset
                            symbols_text = ''.join(symbols_to_add)
                            plt.text(bar.get_x() + bar.get_width()/2, y_position,
                                   symbols_text, ha='center', va='bottom',
                                   fontsize=20, color='black', fontweight='bold')

                # Если используется старый формат (один контроль)
                elif control_index < len(all_individual_aucs):
                    control_aucs = all_individual_aucs[control_index]
                    for i, (bar, dose) in enumerate(zip(bars, doses)):
                        if i == control_index:
                            continue

                        if i < len(all_individual_aucs):
                            exp_aucs = all_individual_aucs[i]
                            try:
                                _, p_value = mannwhitneyu(control_aucs, exp_aucs, alternative='two-sided')
                                if p_value < 0.05:
                                    y_position = bar.get_height() + errors_for_fit[i] + y_offset
                                    plt.text(bar.get_x() + bar.get_width()/2, y_position,
                                           '*', ha='center', va='bottom',
                                           fontsize=20, color='black', fontweight='bold')
                            except Exception as e:
                                print(f"Ошибка при выполнении теста для дозы {dose}: {e}")

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

            plt.tight_layout()

    @staticmethod
    def plot_auc_comparison(file_paths: List[str], title="Сравнение AUC кожных реакций", x_label="",
                            y_label="AUC (усл. ед.)", perform_stat_test: bool = False, control_index: int = 0,
                            control_groups_info: dict = None):
        import re
        with sns.axes_style("whitegrid", rc={'font.family': PLOT_FONT_FAMILY}):
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

            common_timepoints = list(range(0, 25))  # как в plot_multiple_experiments
            doses, aucs_for_fit, errors_for_fit, labels_for_legend = [], [], [], []
            all_individual_aucs = []  # Для критерия Манна-Уитни
            colors = sns.color_palette("Set3", n_colors=len(file_paths))

            for file_path, color in zip(file_paths, colors):
                visualizer = SkinReactionsVisualizer(file_path)
                try:
                    individual_skin_reactions = visualizer.skin_reactions
                    time_data = np.array(visualizer.time_data, dtype=float)

                    # Интерполяция
                    interpolated_curves = []
                    for reaction in individual_skin_reactions:
                        interpolated = SupportingFunctions.interpolate_data_to_common_timepoints(
                            time_data, reaction, common_timepoints
                        )
                        interpolated_curves.append(interpolated)

                    interpolated_curves = np.array(interpolated_curves)
                    mean_curve = np.nanmean(interpolated_curves, axis=0)

                    # AUC и ошибка
                    auc_mean = SupportingFunctions.calculate_auc(mean_curve, common_timepoints)
                    auc_individual = [SupportingFunctions.calculate_auc(curve, common_timepoints)
                                      for curve in interpolated_curves]
                    auc_sem = np.std(auc_individual, ddof=1) / np.sqrt(len(auc_individual))

                    dose = extract_total_dose(visualizer.experiment_params)
                    # Если доза не найдена (контрольная группа), используем 0
                    if dose is None:
                        dose = 0
                    doses.append(dose)
                    aucs_for_fit.append(auc_mean)
                    errors_for_fit.append(auc_sem)
                    all_individual_aucs.append(auc_individual)  # Сохраняем для теста
                    labels_for_legend.append(format_experiment_params(visualizer.experiment_params))

                except Exception as e:
                    print(f"Ошибка при обработке {file_path}: {e}")
                    continue

            # Построение графика
            plt.figure(figsize=(12, 8))
            bar_width = 2.5 if len(doses) < 10 else 0.8
            bars = plt.bar(doses, aucs_for_fit, yerr=errors_for_fit, width=bar_width,
                           color=colors[:len(doses)], edgecolor="black", zorder=2, capsize=8)
            plt.xlabel("Суммарная доза, Гр", fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.title(title, fontsize=16)
            dose_labels = ["Контроль" if d == 0 else str(d) for d in doses]
            plt.xticks(doses, dose_labels, fontsize=12)

            # Подписи над столбиками
            for i, (bar, auc, err) in enumerate(zip(bars, aucs_for_fit, errors_for_fit)):
                y_text = bar.get_height() - err - 0.03 * max(aucs_for_fit)
                y_text = max(0, y_text)
                plt.text(bar.get_x() + bar.get_width() / 2, y_text, f"{auc:.2f}",
                         ha='center', va='top', fontsize=12, fontweight='bold', color='black')

            # Легенда
            legend_patches = [mpatches.Patch(color=col, label=lab)
                              for col, lab in zip(colors[:len(labels_for_legend)], labels_for_legend)]
            ncol = math.ceil(len(labels_for_legend) / 2) if len(labels_for_legend) > 4 else len(labels_for_legend)
            legend_position = graph_manager.get_current_legend_position()
            if legend_position is not None:
                plt.legend(handles=legend_patches, loc=legend_position,
                           ncol=ncol, fontsize=12, handletextpad=0.5, columnspacing=2.5)

            # Критерий Манна-Уитни
            if perform_stat_test and len(all_individual_aucs) > 1:
                from scipy.stats import mannwhitneyu
                y_max = max(aucs_for_fit) if aucs_for_fit else 0
                y_offset = y_max * 0.05

                # Определяем символы для контролей
                control_symbols = {1: '*', 2: '^', 3: '#'}

                # Если указаны множественные контроли, используем их
                if control_groups_info:
                    for i, (bar, dose) in enumerate(zip(bars, doses)):
                        # Проверяем, не является ли текущий столбец контрольным
                        is_control = any(i in indices for indices in control_groups_info.values())
                        if is_control:
                            continue

                        # Сравниваем с каждой контрольной группой
                        symbols_to_add = []
                        for control_type, control_indices in sorted(control_groups_info.items()):
                            for control_idx in control_indices:
                                if control_idx < len(all_individual_aucs):
                                    control_aucs = all_individual_aucs[control_idx]
                                    exp_aucs = all_individual_aucs[i]
                                    try:
                                        _, p_value = mannwhitneyu(control_aucs, exp_aucs, alternative='two-sided')
                                        if p_value < 0.05:
                                            symbol = control_symbols.get(control_type, '*')
                                            if symbol not in symbols_to_add:
                                                symbols_to_add.append(symbol)
                                    except Exception as e:
                                        print(f"Ошибка при выполнении теста для дозы {dose} с контролем {control_type}: {e}")

                        # Добавляем символы над столбцом
                        if symbols_to_add:
                            y_position = bar.get_height() + errors_for_fit[i] + y_offset
                            symbols_text = ''.join(symbols_to_add)
                            plt.text(bar.get_x() + bar.get_width()/2, y_position,
                                   symbols_text, ha='center', va='bottom',
                                   fontsize=20, color='black', fontweight='bold')

                # Если используется старый формат (один контроль)
                elif control_index < len(all_individual_aucs):
                    control_aucs = all_individual_aucs[control_index]
                    for i, (bar, dose) in enumerate(zip(bars, doses)):
                        if i == control_index:
                            continue

                        if i < len(all_individual_aucs):
                            exp_aucs = all_individual_aucs[i]
                            try:
                                _, p_value = mannwhitneyu(control_aucs, exp_aucs, alternative='two-sided')
                                if p_value < 0.05:
                                    y_position = bar.get_height() + errors_for_fit[i] + y_offset
                                    plt.text(bar.get_x() + bar.get_width()/2, y_position,
                                           '*', ha='center', va='bottom',
                                           fontsize=20, color='black', fontweight='bold')
                            except Exception as e:
                                print(f"Ошибка при выполнении теста для дозы {dose}: {e}")

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
