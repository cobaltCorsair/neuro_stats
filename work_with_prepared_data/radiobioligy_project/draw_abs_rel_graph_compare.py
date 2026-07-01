# файл draw_abs_rel_graph_compare.py

import math

import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from draw_base_graphs import TumorDataVisualizer
from controls import ControlGroupVisualizer
from utils.plotting_helpers import custom_fill_between, format_experiment_params, MatplotlibConfigurator
from stats_methods.support_stats_methods import SupportingFunctions
from stats_methods.kaplan_meier import log_rank_test
from utils.visualizer import GraphVisualizer
from work_with_prepared_data.radiobioligy_project.gui import graph_manager

# Переопределяем функцию
plt.fill_between = custom_fill_between
configurator = MatplotlibConfigurator()
configurator.apply_custom_styles()
configurator.restore_original_styles()


class TumorDataComparatorAdvanced:
    """
    Класс для сравнения данных произвольного количества экспериментов, представленных экземплярами TumorDataVisualizer.

    Позволяет агрегировать и анализировать данные из различных источников, сравнивая ключевые метрики и визуализируя
    результаты.

    Attributes: visualizers (List[TumorDataVisualizer]): Список визуализаторов, каждый из которых представляет собой
    эксперимент или группу экспериментов.

    Args: *visualizers (TumorDataVisualizer): Произвольное количество объектов TumorDataVisualizer, каждый из которых
    представляет данные одного эксперимента.
    """

    TGI_TIME_GRID_CONTROL_DAYS = "control_days"
    TGI_TIME_GRID_DAILY_INTERPOLATION = "daily_interpolation"

    def __init__(self, *visualizers: TumorDataVisualizer):
        self.visualizers = visualizers
        self._perform_stat_test = False  # Значение по умолчанию
        self._annotation_multiplier = 0
        self._use_ttest = False
        self._use_shapiro = False
        self._use_AUC = False

    def _add_significance_test_legend_if_active(self, drawgraph):
        """
        Отдельная легенда на графике (не подпись под ним), поясняющая, какой критерий
        использован и применена ли поправка Холма — иначе '*'/'(*)' ничего не объясняют.
        Вызывать ПЕРЕД finalize_figure, только если реально запрашивалось сравнение.
        """
        if self.perform_stat_test or self.use_ttest:
            test_name = "Стьюдента" if self.use_ttest else "Манна-Уитни"
            drawgraph.add_legend(
                graph_manager.build_significance_test_legend_label(test_name),
                title="Критерий значимости", loc="lower right", display_marker=False
            )

    @property
    def perform_stat_test(self):
        return self._perform_stat_test

    @perform_stat_test.setter
    def perform_stat_test(self, value: bool):
        self._perform_stat_test = value

    @property
    def annotation_multiplier(self):
        return self._annotation_multiplier

    @annotation_multiplier.setter
    def annotation_multiplier(self, value: int):
        self._annotation_multiplier = value

    @property
    def use_ttest(self):
        return self._use_ttest

    @use_ttest.setter
    def use_ttest(self, value: int):
        self._use_ttest = value

    @property
    def use_shapiro(self):
        return self._use_shapiro

    @use_shapiro.setter
    def use_shapiro(self, value: bool):
        self._use_shapiro = value

    @property
    def use_AUC(self):
        return self._use_AUC

    @use_AUC.setter
    def use_AUC(self, value: int):
        self._use_AUC = value

    def compare_mean_volumes(self):
        """
        Сравнивает средние абсолютные объемы опухолей для всех экспериментов и визуализирует результаты на графике.

        Этот метод агрегирует данные о средних объемах опухолей из всех предоставленных экспериментов, нормализует
        временные данные и строит обобщённый график сравнения, чтобы понять общие тенденции и различия между
        экспериментальными группами.

        Применяется нормализация временных данных для обеспечения корректного сравнения между экспериментами,
        которые могли начинаться в разные моменты времени или иметь различную продолжительность.

        Использует внешнюю функцию `normalize_time_data_min` для нормализации временных данных и
        `prepare_and_add_data_to_graph` для подготовки и добавления данных на график. Результатом является график с
        линиями, каждая из которых представляет средний объем опухоли по времени для каждого эксперимента.

        Args:
            Нет аргументов.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график.

        Использует:
            - `normalize_time_data_min(self.visualizers)` для нормализации временных данных всех экспериментов.
            - `GraphVisualizer` для создания и настройки объекта визуализации графика.
            - `prepare_and_add_data_to_graph` для добавления данных о средних объемах опухолей на график.
            - `finalize_figure` для финализации и отображения графика.
        """
        SupportingFunctions.normalize_time_data_min(self.visualizers)
        drawgraph = GraphVisualizer("Сравнение среднего объема опухолей", "Время, сут.",
                                    "Средний объем опухоли, абс. ед.")
        drawgraph.setup_figure()
        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            lambda visualizer: visualizer.data_processor.get_mean_tumor_volumes(),
            drawgraph,
            "M/V абс.: ",
            self._use_AUC,
            self.perform_stat_test,
            [self.visualizers[0], self.visualizers[1]],
            'up',
            self.annotation_multiplier,
            self.use_ttest,
            self.use_shapiro,
            )

        self._add_significance_test_legend_if_active(drawgraph)
        drawgraph.finalize_figure('', legend_fontsize=18)

    def compare_relative_volumes(self):
        """
        Сравнивает средние относительные объемы опухолей для всех экспериментов и визуализирует результаты на графике.

        Этот метод агрегирует данные о средних относительных объемах опухолей из всех предоставленных экспериментов,
        нормализует временные данные и строит обобщенный график сравнения. Целью является понимание общих тенденций и
        различий между экспериментальными группами в контексте изменения объема опухолей относительно их начального
        размера.

        Для обеспечения корректного сравнения временные данные нормализуются, что позволяет сравнивать эксперименты с
        различной продолжительностью и начальным временем. Относительный объем опухоли выражается как отношение текущего
        объема к начальному, что позволяет оценить динамику роста или уменьшения опухоли.

        Args:
            Нет аргументов.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график.

        Использует:
            - `normalize_time_data_min(self.visualizers)` для нормализации временных данных всех экспериментов.
            - `GraphVisualizer` для создания и настройки объекта визуализации графика.
            - `prepare_and_add_data_to_graph` для добавления данных о средних относительных объемах опухолей на график.
            - `finalize_figure` для финализации и отображения графика.

        Примечание: - В текущей реализации закомментированная строка `drawgraph.add_legend(time_labels, "Интервалы
        между облучениями", "lower right")` предполагает возможность добавления легенды с интервалами между
        облучениями. Эту возможность можно восстановить или модифицировать в соответствии с требованиями к визуализации.
        """
        SupportingFunctions.normalize_time_data_min(self.visualizers)
        drawgraph = GraphVisualizer("Сравнение среднего относительного объема опухолей", "Время, сут.",
                                    "Относительный объем опухоли, отн. ед.")
        drawgraph.setup_figure()

        # Собираем информацию об интервалах
        time_intervals = [visualizer.experiment_params[-1] if visualizer.experiment_params else "—" for visualizer in self.visualizers]

        # Используем лямбда-функцию для извлечения значений
        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            lambda visualizer: visualizer.data_processor.get_mean_relative_tumor_volumes(),  # Лямбда-функция
            drawgraph,
            "",
            self._use_AUC,
            self.perform_stat_test,
            [self.visualizers[0], self.visualizers[1]],
            'down',
            self.annotation_multiplier,
            self.use_ttest,
            self.use_shapiro,
        )
        # Добавляем легенду с интервалами (при необходимости)
        time_labels = [f"Интервал: {interval}" for interval in time_intervals]
        # drawgraph.add_legend(time_labels, "Интервалы между облучениями", "lower right")

        self._add_significance_test_legend_if_active(drawgraph)
        # Увеличиваем размер шрифта легенды для относительных графиков сравнения
        drawgraph.finalize_figure('', legend_fontsize=18)

    def compare_control_and_experiment(self, control_visualizers: List[TumorDataVisualizer]):
        """
        Сравнивает средние относительные объемы опухолей между контрольными и экспериментальными группами.

        Этот метод анализирует и сравнивает динамику изменения объема опухолей в контрольных и экспериментальных группах
        на протяжении всего эксперимента. Относительные объемы опухолей вычисляются как отношение текущего объема к
        начальному, что позволяет оценить эффективность терапии или воздействия в экспериментальной группе по сравнению
        с контролем.

        Метод нормализует временные данные, чтобы обеспечить корректное сравнение экспериментов с различными начальными
        условиями и продолжительностью. Затем строит совместный график, отображающий динамику изменения относительных
        объемов опухолей во времени для обеих групп, позволяя визуально оценить различия между ними.

        Args: control_visualizers (List[TumorDataVisualizer]): Список объектов `TumorDataVisualizer` для контрольных
        групп.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график сравнения.

        Использует: - `normalize_time_data_min` для нормализации временных данных всех экспериментов. -
        `GraphVisualizer` для создания объекта визуализации и настройки параметров графика. -
        `prepare_and_add_data_to_graph` для агрегации и добавления данных о средних относительных объемах на график.
        - `finalize_figure` для финализации графика и его отображения.

        Примечание:
            - Важно правильно выбрать и подготовить контрольные группы, чтобы сравнение было корректным и показательным.
            - Метод позволяет визуализировать влияние экспериментальных условий на динамику роста опухолей.
        """
        SupportingFunctions.normalize_time_data_min(list(self.visualizers) + control_visualizers)
        drawgraph = GraphVisualizer("Сравнение контрольных и экспериментальных групп", "Время, сут.",
                                    "Относительный объем опухоли, отн. ед.")
        drawgraph.setup_figure()

        # Функция для извлечения значений средних относительных объемов из визуализатора
        value_extractor = lambda visualizer: visualizer.data_processor.get_mean_relative_tumor_volumes()

        # Добавляем данные контрольных групп
        GraphVisualizer.prepare_and_add_data_to_graph(
            control_visualizers,
            value_extractor,
            drawgraph,
            "Контроль: без облучения",
            self._use_AUC,  # Указываем, что нужно рассчитать AUC
        )

        # Добавляем данные экспериментальных групп
        visualizers_to_pass = [v for i, v in enumerate(self.visualizers) if i in (0, 1)]

        GraphVisualizer.prepare_and_add_data_to_graph(
            self.visualizers,
            value_extractor,
            drawgraph,
            "Эксперимент: ",
            self._use_AUC,  # Указываем, что нужно рассчитать AUC
            self.perform_stat_test,
            visualizers_to_pass,  # <-- передаём уже проверенный список
            'up',
            self.annotation_multiplier,
            self.use_ttest,
            self.use_shapiro,
        )

        self._add_significance_test_legend_if_active(drawgraph)
        drawgraph.finalize_figure('', legend_fontsize=18)

    def compare_tumor_growth_inhibition_with_multiple_experiments(self, control_visualizer: TumorDataVisualizer,
                                                                  experiment_visualizers: List[TumorDataVisualizer]):
        """
        Сравнивает торможение роста опухоли между одной контрольной и несколькими экспериментальными группами.

        Метод вычисляет и сравнивает процент торможения роста опухоли между контрольной группой и несколькими
        экспериментальными группами на протяжении времени эксперимента. Торможение роста опухоли выражается как
        процентное снижение объема опухоли в экспериментальной группе по сравнению с контрольной группой, что
        позволяет оценить эффективность экспериментального воздействия.

        Args:
            control_visualizer (TumorDataVisualizer): Визуализатор для контрольной группы.
            experiment_visualizers (List[TumorDataVisualizer]): Список визуализаторов для экспериментальных групп.

        Returns:
            None: Функция не возвращает значения, но генерирует и отображает график сравнения.

        Использует:
            - `normalize_time_data_min` для нормализации временных данных всех экспериментов.
            - `GraphVisualizer` для создания объекта визуализации и настройки параметров графика.
            - `calculate_tumor_growth_inhibition` для вычисления процентного торможения роста опухоли.
            - `finalize_figure` для финализации графика и его отображения.

        Примечание:
            - Важно выбрать адекватную контрольную группу для корректного сравнения и интерпретации результатов.
            - Этот метод позволяет исследователям визуально сравнивать эффективность различных экспериментальных
              условий или терапий на основе их способности тормозить рост опухолей.
        """
        drawgraph = GraphVisualizer("Сравнение торможения роста опухоли", "Время, сут.", "Торможение роста опухоли, %")
        drawgraph.setup_figure()

        control_times = SupportingFunctions.to_float_list(control_visualizer.time_data)
        control_mean = SupportingFunctions.to_float_list(control_visualizer.data_processor.get_mean_tumor_volumes())

        x_data_lists = []
        for experiment_visualizer in experiment_visualizers:
            t_exp = SupportingFunctions.to_float_list(experiment_visualizer.time_data)
            exp_mean = SupportingFunctions.to_float_list(experiment_visualizer.data_processor.get_mean_tumor_volumes())

            # контроль → на сетку эксперимента (без обрезания)
            ctrl_on_exp = SupportingFunctions.interpolate_data_to_common_timepoints(
                control_times, control_mean, t_exp
            )

            # TGI имеет ту же длину, что и t_exp
            tgi = SupportingFunctions.calculate_tumor_growth_inhibition(ctrl_on_exp, exp_mean)

            # НИКАКИХ min_length и срезов
            drawgraph.add_plot(
                t_exp,
                tgi,
                experiment_visualizer.experiment_params,
                '',
                self._use_AUC
            )
            x_data_lists.append(t_exp)

        drawgraph.update_axes_limits(x_data_lists)
        # Увеличиваем размер шрифта легенды для графиков торможения роста опухоли
        drawgraph.finalize_figure('', legend_fontsize=18)

    @staticmethod
    def _build_experiment_names(experiment_visualizers: List[TumorDataVisualizer]) -> List[str]:
        experiment_names = []
        duplicate_counters = {}

        for index, experiment_visualizer in enumerate(experiment_visualizers, start=1):
            base_name = format_experiment_params(experiment_visualizer.experiment_params).strip()
            if not base_name:
                base_name = f"Эксперимент {index}"

            duplicate_counters[base_name] = duplicate_counters.get(base_name, 0) + 1
            if duplicate_counters[base_name] > 1:
                experiment_names.append(f"{base_name} ({duplicate_counters[base_name]})")
            else:
                experiment_names.append(base_name)

        return experiment_names

    def _build_tumor_growth_inhibition_series(self, control_visualizer, experiment_visualizers):
        SupportingFunctions.normalize_time_data_min([control_visualizer] + experiment_visualizers)

        control_times = SupportingFunctions.to_float_list(control_visualizer.time_data)
        control_mean = SupportingFunctions.to_float_list(control_visualizer.data_processor.get_mean_tumor_volumes())
        experiment_names = self._build_experiment_names(experiment_visualizers)

        tgi_series_by_experiment = []
        for experiment_name, experiment_visualizer in zip(experiment_names, experiment_visualizers):
            experiment_times = SupportingFunctions.to_float_list(experiment_visualizer.time_data)
            experiment_mean = SupportingFunctions.to_float_list(
                experiment_visualizer.data_processor.get_mean_tumor_volumes()
            )

            control_on_experiment_time = SupportingFunctions.interpolate_data_to_common_timepoints(
                control_times,
                control_mean,
                experiment_times
            )
            tumor_growth_inhibition = SupportingFunctions.calculate_tumor_growth_inhibition(
                control_on_experiment_time,
                experiment_mean
            )

            series = pd.Series(
                tumor_growth_inhibition,
                index=pd.Index(experiment_times, dtype="float64"),
                name=experiment_name,
                dtype="float64"
            )
            series = series[~pd.isna(series.index)]
            series = series.groupby(level=0).mean()
            tgi_series_by_experiment.append((experiment_name, series))

        return tgi_series_by_experiment

    def _build_tgd_table(self, control_visualizer, experiment_visualizers, k=1.5, normalize_time=True):
        """
        Строит таблицу задержки роста опухоли (TGD, сут) по группам относительно контроля.

        normalize_time=False позволяет вызвать этот метод после того, как нормализация времени
        (normalize_time_data_min) уже была выполнена вызывающим кодом для этого же набора
        визуализаторов — повторный вызов на уже сдвинутых данных безопасен лишь случайно
        (минимум после первого сдвига равен 0), поэтому явный флаг лучше, чем полагаться на эту
        случайную идемпотентность.

        Столбец "Лог-ранг p (по животным)" сравнивает не средние кривые, а TGD КАЖДОГО
        животного отдельно (build_tgd_events + log_rank_test) — в отличие от поточечного
        сравнения объёмов (Манна-Уитни/Стьюдента) он не теряет мощность на смешанных
        популяциях (часть животных регрессировала, часть — нет), потому что цензурированных
        животных трактует как цензурированные наблюдения, а не как обычный разброс.
        """
        if normalize_time:
            SupportingFunctions.normalize_time_data_min([control_visualizer] + experiment_visualizers)

        control_times = SupportingFunctions.to_float_list(control_visualizer.time_data)
        control_mean = SupportingFunctions.to_float_list(control_visualizer.data_processor.get_mean_tumor_volumes())
        control_day, control_censored = SupportingFunctions.calculate_tgd_threshold_day(
            control_times, control_mean, k
        )
        control_events = SupportingFunctions.build_tgd_events(
            control_times, control_visualizer.tumor_volumes, k,
            labels=control_visualizer.rat_labels, source_file="control"
        )

        experiment_names = self._build_experiment_names(experiment_visualizers)
        rows = []
        for experiment_name, experiment_visualizer in zip(experiment_names, experiment_visualizers):
            exp_times = SupportingFunctions.to_float_list(experiment_visualizer.time_data)
            exp_mean = SupportingFunctions.to_float_list(
                experiment_visualizer.data_processor.get_mean_tumor_volumes()
            )
            exp_day, exp_censored = SupportingFunctions.calculate_tgd_threshold_day(exp_times, exp_mean, k)
            exp_events = SupportingFunctions.build_tgd_events(
                exp_times, experiment_visualizer.tumor_volumes, k,
                labels=experiment_visualizer.rat_labels, source_file=experiment_name
            )
            _, log_rank_p = log_rank_test(control_events, exp_events)
            rows.append({
                'Группа': experiment_name,
                'TGD, сут': exp_day - control_day,
                'Цензурировано (эксперимент)': exp_censored,
                'Цензурировано (контроль)': control_censored,
                'Лог-ранг p (по животным)': log_rank_p,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _build_control_time_grid(control_visualizer):
        return sorted({
            float(timepoint)
            for timepoint in SupportingFunctions.to_float_list(control_visualizer.time_data)
            if not pd.isna(timepoint)
        })

    @staticmethod
    def _build_daily_experiment_time_grid(tgi_series_by_experiment):
        non_empty_series = [series for _, series in tgi_series_by_experiment if not series.empty]
        if not non_empty_series:
            return []

        start_day = math.floor(min(series.index.min() for series in non_empty_series))
        end_day = math.ceil(max(series.index.max() for series in non_empty_series))
        return [float(day) for day in range(start_day, end_day + 1)]

    @staticmethod
    def _align_tgi_series_to_timepoints(tgi_series_by_experiment, time_grid):
        common_timepoints = sorted({
            float(timepoint)
            for timepoint in time_grid
            if not pd.isna(timepoint)
        })
        if not common_timepoints:
            return [
                (experiment_name, pd.Series(name=experiment_name, dtype="float64"))
                for experiment_name, _ in tgi_series_by_experiment
            ]

        aligned_index = pd.Index(common_timepoints, dtype="float64")
        aligned_series = []

        for experiment_name, series in tgi_series_by_experiment:
            if series.empty:
                aligned_values = [float("nan")] * len(common_timepoints)
            else:
                sorted_series = series.sort_index()
                if len(sorted_series) == 1:
                    aligned_values = [float(sorted_series.iloc[0])] * len(common_timepoints)
                else:
                    aligned_values = SupportingFunctions.interpolate_data_to_common_timepoints(
                        sorted_series.index.tolist(),
                        sorted_series.tolist(),
                        common_timepoints
                    )

            aligned_series.append((
                experiment_name,
                pd.Series(aligned_values, index=aligned_index, name=experiment_name, dtype="float64")
            ))

        return aligned_series

    @staticmethod
    def _build_tgi_table_dataframe(tgi_series_by_experiment):
        if not tgi_series_by_experiment:
            return pd.DataFrame(columns=['Время (сут)'])

        tgi_df = pd.concat(
            [series.rename(experiment_name) for experiment_name, series in tgi_series_by_experiment],
            axis=1
        ).sort_index()
        tgi_df = tgi_df.reset_index().rename(columns={'index': 'Время (сут)'})
        tgi_df['Время (сут)'] = pd.to_numeric(tgi_df['Время (сут)'], errors='coerce')
        return tgi_df

    @staticmethod
    def _create_pairwise_tgi_summary(tgi_series_by_experiment):
        if len(tgi_series_by_experiment) < 2:
            return None

        summary_rows = []
        for left_index, (left_name, left_series) in enumerate(tgi_series_by_experiment[:-1]):
            left_filtered = left_series[left_series.index >= 9]

            for right_name, right_series in tgi_series_by_experiment[left_index + 1:]:
                right_filtered = right_series[right_series.index >= 9]
                paired_values = pd.concat(
                    [left_filtered.rename(left_name), right_filtered.rename(right_name)],
                    axis=1,
                    join='inner'
                ).dropna()

                if paired_values.empty:
                    average_absolute_difference = float('nan')
                    average_relative_difference = float('nan')
                    timepoint_count = 0
                else:
                    absolute_difference = (paired_values[left_name] - paired_values[right_name]).abs()
                    baseline = paired_values[left_name].abs().replace(0, pd.NA)
                    relative_difference = (absolute_difference / baseline) * 100

                    average_absolute_difference = absolute_difference.mean()
                    average_relative_difference = relative_difference.mean()
                    timepoint_count = len(paired_values)

                summary_rows.append({
                    'Пара экспериментов': f"{left_name} vs {right_name}",
                    'Средняя абсолютная разница ТРО, %': average_absolute_difference,
                    'Средняя относительная разница ТРО, %': average_relative_difference,
                    'Число временных точек': timepoint_count,
                })

        return pd.DataFrame(summary_rows)

    def _create_tumor_growth_inhibition_tables_by_mode(self, control_visualizer, experiment_visualizers):
        if not experiment_visualizers:
            empty_df = pd.DataFrame(columns=['Время (сут)'])
            return {
                self.TGI_TIME_GRID_CONTROL_DAYS: (empty_df, None),
                self.TGI_TIME_GRID_DAILY_INTERPOLATION: (empty_df.copy(), None),
            }

        tgi_series_by_experiment = self._build_tumor_growth_inhibition_series(
            control_visualizer,
            experiment_visualizers
        )

        control_days_series = self._align_tgi_series_to_timepoints(
            tgi_series_by_experiment,
            self._build_control_time_grid(control_visualizer)
        )
        daily_interpolated_series = self._align_tgi_series_to_timepoints(
            tgi_series_by_experiment,
            self._build_daily_experiment_time_grid(tgi_series_by_experiment)
        )

        return {
            self.TGI_TIME_GRID_CONTROL_DAYS: (
                self._build_tgi_table_dataframe(control_days_series),
                self._create_pairwise_tgi_summary(control_days_series)
            ),
            self.TGI_TIME_GRID_DAILY_INTERPOLATION: (
                self._build_tgi_table_dataframe(daily_interpolated_series),
                self._create_pairwise_tgi_summary(daily_interpolated_series)
            ),
        }

    def create_tgd_table(self, control_visualizer, experiment_visualizers, k=1.5, normalize_time=True):
        """Возвращает таблицу задержки роста опухоли (TGD, сут) по группам относительно контроля."""
        return self._build_tgd_table(
            control_visualizer, experiment_visualizers, k=k, normalize_time=normalize_time
        )

    def create_tumor_growth_inhibition_tables(self, control_visualizer, experiment_visualizers):
        """
        Возвращает обе таблицы ТРО:
        - по суткам контроля;
        - с ежедневной интерполяцией от начала до конца экспериментов.
        """
        return self._create_tumor_growth_inhibition_tables_by_mode(
            control_visualizer,
            experiment_visualizers
        )

    def _legacy_create_tumor_growth_inhibition_table(self, control_visualizer, experiment_visualizers):
        """
        Возвращает основную таблицу ТРО по времени и, при n >= 2, попарную сводку
        различий между экспериментами после 9-го дня.
        """
        return self.create_tumor_growth_inhibition_table(
            control_visualizer,
            experiment_visualizers,
            time_grid_mode=self.TGI_TIME_GRID_CONTROL_DAYS
        )

        if not experiment_visualizers:
            return pd.DataFrame(columns=['Время (сут)']), None

        tgi_series_by_experiment = self._build_tumor_growth_inhibition_series(
            control_visualizer,
            experiment_visualizers
        )
        tgi_series_by_experiment = self._interpolate_tgi_series_to_common_timepoints(tgi_series_by_experiment)

        tgi_df = pd.concat(
            [series.rename(experiment_name) for experiment_name, series in tgi_series_by_experiment],
            axis=1
        ).sort_index()
        tgi_df = tgi_df.reset_index().rename(columns={'index': 'Время (сут)'})
        tgi_df['Время (сут)'] = pd.to_numeric(tgi_df['Время (сут)'], errors='coerce')

        pairwise_summary_df = self._create_pairwise_tgi_summary(tgi_series_by_experiment)
        return tgi_df, pairwise_summary_df

    def create_tumor_growth_inhibition_table(
            self,
            control_visualizer,
            experiment_visualizers,
            time_grid_mode=TGI_TIME_GRID_CONTROL_DAYS
    ):
        """
        Возвращает основную таблицу ТРО по времени и, при n >= 2, попарную сводку
        различий между экспериментами после 9-го дня.
        """
        tables_by_mode = self._create_tumor_growth_inhibition_tables_by_mode(
            control_visualizer,
            experiment_visualizers
        )
        if time_grid_mode not in tables_by_mode:
            raise ValueError(f"Unsupported TGI time grid mode: {time_grid_mode}")
        return tables_by_mode[time_grid_mode]


if __name__ == "__main__":
    # # Используем с файлом данных
    # file_path1 = './datas/n_7.2_p_25.2_2023_2.xlsx'
    # file_path2 = './datas/p_25.2_n_7.2_2023_2.xlsx'

    # Используем с файлом данных
    # file_path1 = './datas/n_7.2_p_25.2_2023.xlsx'
    # file_path2 = './datas/p_25.2_n_7.2_2023.xlsx'

    # Используем с файлом данных
    # file_path1 = './datas/n_2.56_p_25.6_2019.xlsx'
    # file_path2 = './datas/p_25.6_n_2.56_2019.xlsx'

    # Контроль
    # control_path = './datas/control/16.03.2023_e_36.xlsx'
    # control_path = './datas/control/08.10.2021_p32_прострел.xlsx'

    # Пути к файлам данных для контрольных и экспериментальных групп
    control_paths = [
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\control.xlsx',
    ]
    experiment_paths = [
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\30.03.2022_p_36_прострел.xlsx',
        #r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_12.xlsx',
        r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\02.02.2023_n_18.xlsx',
        #r'C:\dev\neuro_stats\work_with_prepared_data\datas\control\16.03.2023_n_22.xlsx'
    ]

    # Создание объектов визуализатора для контрольных групп
    control_visualizers = [ControlGroupVisualizer(path) for path in control_paths]
    # Создание объекта визуализатора для контрольной группы
    control_visualizer = ControlGroupVisualizer(control_paths[0])

    # Создание объектов визуализатора для экспериментальных групп
    experiment_visualizers = [TumorDataVisualizer(path) for path in experiment_paths]

    # Создание объекта сравнителя
    comparator = TumorDataComparatorAdvanced(*experiment_visualizers)

    comparator.compare_mean_volumes()  # Сравниваем средние абсолютные объемы
    comparator.compare_relative_volumes()  # Сравниваем средние относительные объемы

    # Сравнение контрольных и экспериментальных групп
    #comparator.compare_control_and_experiment(control_visualizers)

    # Сравнение торможения роста опухоли между контрольной и несколькими экспериментальными группами
    comparator.compare_tumor_growth_inhibition_with_multiple_experiments(control_visualizer, experiment_visualizers)
    comparator.create_tumor_growth_inhibition_table(control_visualizer, experiment_visualizers)
