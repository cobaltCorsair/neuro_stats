from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

from work_with_prepared_data.radiobioligy_project.gui import graph_manager
from work_with_prepared_data.radiobioligy_project.stats_methods.support_stats_methods import SupportingFunctions
from work_with_prepared_data.radiobioligy_project.utils.plot_saver import save_plot
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import format_experiment_params


class GraphVisualizer:
    def __init__(self, title, x_label, y_label, figsize=(12, 7)):
        """
        Инициализирует объект GraphVisualizer, предназначенный для упрощения процесса создания и настройки графиков.

        Args:
            title (str): Заголовок графика.
            x_label (str): Подпись оси X.
            y_label (str): Подпись оси Y.
            figsize (Tuple[int, int], optional): Размер фигуры в дюймах. По умолчанию (12, 7).

        Атрибуты:
            figsize (Tuple[int, int]): Размер фигуры.
            title (str): Заголовок графика.
            x_label (str): Подпись оси X.
            y_label (str): Подпись оси Y.
            markers (List[str]): Список маркеров для линий.
            linestyles (List[str]): Список стилей линий.
            linestyle_index (int): Индекс для выбора стиля линии.
            marker_index (int): Индекс для выбора маркера.
            marker_size (int): Размер маркеров.
            lines (List): Список объектов линий (для управления легендами).
            aucs (List[float]): Список значений площади под кривой (AUC).
            max_x (Optional[int]): Максимальное значение по оси X (для настройки масштаба).
            max_y (Optional[int]): Максимальное значение по оси Y.
            legend_info (List[Tuple]): Список данных для создания дополнительных легенд.
        """
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
        self.max_x = None
        self.max_y = None
        self.legend_info = []
        self.legend_position = 'best'
        graph_manager.register_visualizer(self)

    def update_legend_position(self, position):
        self.legend_position = position

    def setup_figure(self):
        """
        Настройка основных параметров фигуры для графика.

        Создает новую фигуру и применяет заданные параметры, такие как размеры фигуры, названия осей и включение сетки.

        Примечания:
        - Заголовок графика задается отдельно и по умолчанию не включен в эту функцию.
        """
        plt.figure(figsize=self.figsize)
        # plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid(True)

    @staticmethod
    def prepare_and_add_data_to_graph(visualizers, value_extractor_func, graph_visualizer, label_prefix,
                                      calculate_auc=False, perform_stat_test=False, experiments_to_compare=None,
                                      annotation_offset_direction='up', annotation_multiplier=0.0, use_ttest=False):
        """
        Подготавливает данные от нескольких экспериментов и добавляет их на график.

        Для каждого объекта визуализатора извлекает необходимые значения, вычисляет статистики и добавляет данные на график.

        Args:
            visualizers (List[VisualizerType]): Список объектов, содержащих данные для визуализации.
            value_extractor_func (Callable): Функция, которая извлекает необходимые данные из объекта визуализатора.
            graph_visualizer (GraphVisualizer): Объект GraphVisualizer, на который нужно добавить данные.
            label_prefix (str): Префикс для легенды, добавляемый к метке каждой серии данных на графике.
            calculate_auc (bool, optional): Флаг, указывающий, нужно ли вычислять площадь под кривой (AUC).
            perform_stat_test (bool, optional): Флаг, необходим ли тест.
            experiments_to_compare (List[VisualizerType], optional): Эксперименты для теста.
            annotation_offset_direction (str): Направление для сдвига аннотаций ('up' или 'down').
            annotation_multiplier (float): Множитель для сдвига аннотаций.
            use_ttest (bool, optional): Если True, выполняется t-тест, иначе используется тест Манна-Уитни.

        Примечание:
            Для расчета стандартного отклонения и погрешности используются функции `calculate_std_dev` и
            `calculate_error_margin` соответственно.
        """
        x_data_lists = []
        upper_bounds_dict = {}

        # Выбор статистического теста
        if perform_stat_test and experiments_to_compare:
            p_values, x_positions_for_annotations = GraphVisualizer.prepare_mann_whitney_test(
                experiments_to_compare, perform_stat_test)
        elif use_ttest and experiments_to_compare:
            p_values, x_positions_for_annotations = GraphVisualizer.prepare_ttest(
                experiments_to_compare, use_ttest)
        else:
            p_values, x_positions_for_annotations = None, None

        # Добавляем графики и вычисляем верхние границы
        for index, visualizer in enumerate(visualizers):
            values = value_extractor_func(visualizer)
            std_dev = [SupportingFunctions.calculate_std_dev(volumes, mean_volume)
                       for volumes, mean_volume in zip(np.transpose(visualizer.tumor_volumes), values)]
            error_margin = [SupportingFunctions.calculate_error_margin(std, len(visualizer.tumor_volumes))
                            for std in std_dev]

            # Добавляем график
            graph_visualizer.add_plot(visualizer.time_data, values, visualizer.experiment_params, f"{label_prefix}",
                                      error_margin, calculate_auc)
            x_data_lists.append(visualizer.time_data)

            # Если визуализатор относится к экспериментам, которые сравниваются, сохраняем его границы
            if experiments_to_compare and visualizer in experiments_to_compare:
                upper_bounds = [val + err + annotation_multiplier for val, err in zip(values, error_margin)]
                upper_bounds_dict[visualizer] = upper_bounds

        # Обновляем оси
        graph_visualizer.update_axes_limits(x_data_lists)

        # Добавление аннотаций значимости
        if (perform_stat_test or use_ttest) and experiments_to_compare:
            GraphVisualizer.add_significance_annotation(
                experiments_to_compare, p_values, x_positions_for_annotations, upper_bounds_dict,
                annotation_visualizer=experiments_to_compare[0]  # Указываем визуализатор для аннотаций
            )

    @staticmethod
    def prepare_mann_whitney_test(experiments_to_compare, perform_stat_test):
        """
        Подготавливает и выполняет статистический тест Манна-Уитни для сравнения экспериментальных групп.

        Для каждой временной точки данных сравнивает группы экспериментов, используя тест Манна-Уитни,
        и собирает значения p для последующей аннотации значимости на графике.

        Args:
            experiments_to_compare (List[VisualizerType]): Список из двух экспериментов для сравнения.
            perform_stat_test (bool): Флаг, определяющий необходимость выполнения статистического теста.

        Returns:
            Tuple[List[float], List[float]]: Возвращает два списка - значения p и позиции по оси X для аннотаций.
        """
        if perform_stat_test and experiments_to_compare:
            # Инициализация списков для результатов тестов и аннотаций
            p_values = []
            x_positions_for_annotations = []

            # Получаем минимальное количество временных точек среди двух групп
            min_time_length = min(len(experiments_to_compare[0].time_data), len(experiments_to_compare[1].time_data))

            # Проходимся по временным точкам, но только до минимальной длины
            for time_index in range(min_time_length):
                # Проверяем, что данные для текущей временной точки есть в обеих группах
                group1_volumes = [
                    vol[time_index] for vol in experiments_to_compare[0].tumor_volumes
                    if len(vol) > time_index  # Проверяем, что объемы существуют для данной точки времени
                ]
                group2_volumes = [
                    vol[time_index] for vol in experiments_to_compare[1].tumor_volumes
                    if len(vol) > time_index  # Проверяем, что объемы существуют для данной точки времени
                ]

                # Если обе группы имеют данные для данной временной точки, выполняем тест Манна-Уитни

                if group1_volumes and group2_volumes:
                    p_value = GraphVisualizer.perform_mann_whitney_test(group1_volumes, group2_volumes)
                    p_values.append(p_value)
                    # Добавляем позиции для аннотаций
                    # индекс тут — это то, для какого эксперимента мы делаем "звёздочки"
                    x_positions_for_annotations.append(experiments_to_compare[0].time_data[time_index])
                else:
                    # Если данных нет для одной из групп, добавляем None, чтобы пропустить аннотацию
                    p_values.append(None)
                    x_positions_for_annotations.append(experiments_to_compare[0].time_data[time_index])

        return p_values, x_positions_for_annotations

    @staticmethod
    def prepare_ttest(experiments_to_compare, use_ttest):
        """
        Подготавливает и выполняет t-тест (критерий Стьюдента) для сравнения экспериментальных групп.

        Для каждой временной точки данных сравнивает группы экспериментов, используя t-тест,
        и собирает значения p для последующей аннотации значимости на графике.

        Args:
            experiments_to_compare (List[VisualizerType]): Список из двух экспериментов для сравнения.
            use_ttest (bool): Флаг, указывающий на выполнение t-теста.

        Returns:
            Tuple[List[float], List[float]]: Возвращает два списка - значения p и позиции по оси X для аннотаций.
        """
        # Инициализируем списки для результатов тестов и аннотаций
        p_values = []
        x_positions_for_annotations = []

        # Проверка входных данных
        if use_ttest and experiments_to_compare and len(experiments_to_compare) == 2:
            # Получаем минимальное количество временных точек среди двух групп
            min_time_length = min(len(experiments_to_compare[0].time_data), len(experiments_to_compare[1].time_data))

            # Проходимся по временным точкам
            for time_index in range(min_time_length):
                # Проверяем, что данные для текущей временной точки есть в обеих группах
                group1_volumes = [
                    vol[time_index] for vol in experiments_to_compare[0].tumor_volumes
                    if len(vol) > time_index  # Проверяем, что объемы существуют для данной точки времени
                ]
                group2_volumes = [
                    vol[time_index] for vol in experiments_to_compare[1].tumor_volumes
                    if len(vol) > time_index  # Проверяем, что объемы существуют для данной точки времени
                ]

                # Если обе группы имеют данные для данной временной точки, выполняем t-тест
                if group1_volumes and group2_volumes:
                    t_stat, p_value = ttest_ind(group1_volumes, group2_volumes,
                                                equal_var=False)  # t-тест с неравными дисперсиями
                    p_values.append(p_value)
                    x_positions_for_annotations.append(experiments_to_compare[0].time_data[time_index])
                else:
                    # Если данных нет для одной из групп, добавляем None
                    p_values.append(None)
                    x_positions_for_annotations.append(experiments_to_compare[0].time_data[time_index])

        else:
            print("Тест не был выполнен, проверьте входные данные.")

        return p_values, x_positions_for_annotations

    @staticmethod
    def perform_mann_whitney_test(group1, group2):
        """
        Выполняет статистический тест Манна-Уитни для двух групп данных.

        Args:
            group1 (List[float]): Первая группа данных для сравнения.
            group2 (List[float]): Вторая группа данных для сравнения.

        Returns:
            float: Значение p, полученное в результате теста Манна-Уитни.
        """
        _, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        return p_value

    @staticmethod
    def add_significance_annotation(experiments_to_compare, p_values, x_positions, upper_bounds_dict,
                                    offset_ratio=0.00, annotation_visualizer=None):
        """
        Добавляет аннотации значимости на график, избегая наложения на доверительные интервалы.

        Args:
            experiments_to_compare (List[VisualizerType]): Список экспериментов для аннотации.
            p_values (List[float]): Список значений p.
            x_positions (List[float]): Список координат по оси X для аннотаций.
            upper_bounds_dict (Dict[VisualizerType, List[float]]): Словарь с верхними границами Y для каждого эксперимента.
            offset_ratio (float): Доля от диапазона Y для смещения аннотаций вверх.
            annotation_visualizer (VisualizerType, optional): Визуализатор, над которым размещаются аннотации.
        """
        # Вычисляем общее смещение на основе диапазона оси Y
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min
        offset = y_range * offset_ratio

        if annotation_visualizer is None:
            annotation_visualizer = experiments_to_compare[0]

        # Получаем time_data и upper_bounds для annotation_visualizer
        time_data = np.array(annotation_visualizer.time_data)
        upper_bounds = upper_bounds_dict.get(annotation_visualizer, None)
        if upper_bounds is None:
            return  # Нет данных для указанного визуализатора

        # Проходим по p-value и добавляем аннотации
        for p_value, x_position in zip(p_values, x_positions):
            if p_value is None:
                continue  # Пропускаем, если p-value не рассчитано
            if p_value < 0.001:
                annotation = '***'  # Сильно значимо
            elif p_value < 0.01:
                annotation = '**'  # Значимо
            elif p_value < 0.05:
                annotation = '*'  # Умеренно значимо
            else:
                annotation = ''  # Не значимо

            if annotation:
                # Найдем индекс ближайшего значения x_position в time_data
                index = np.argmin(np.abs(time_data - x_position))
                # Проверим, что x_position достаточно близок к time_data[index]
                if abs(time_data[index] - x_position) > 1e-6:
                    continue  # x_position не найден в time_data

                # Получаем upper_bound для данного индекса
                upper_bound = upper_bounds[index]

                # Размещаем аннотацию немного выше верхнего значения
                y_position = upper_bound + offset

                # Добавляем аннотацию
                plt.text(x_position, y_position, annotation, ha='center', fontsize=20, color='red')

    def add_individual_plots(self, labels, volumes_data, time_data):
        """
        Добавляет на график индивидуальные линии для каждой единицы данных, например, для каждой крысы в эксперименте.

        Args:
            labels (List[str]): Метки для каждого графика, обычно идентификаторы крыс.
            volumes_data (List[List[float]]): Данные объемов для каждой крысы. Каждый вложенный список содержит значения
            объема для одной крысы.
            time_data (List[float]): Список временных точек, соответствующих измерениям объемов.

        Примечание:
        - Отфильтровывает NaN значения перед добавлением на график.
        - Для каждой крысы рассчитывается среднее значение и стандартное отклонение для определения
        доверительного интервала.
        """
        for label, volumes in zip(labels, volumes_data):
            # Фильтрация NaN значений
            clean_volumes = np.array(volumes)[~np.isnan(volumes)]
            clean_time_data = np.array(time_data)[~np.isnan(volumes)]

            if not list(clean_volumes):
                continue  # Пропуск пустых данных

            # Расчет стандартного отклонения и доверительного интервала
            mean_volume = np.mean(clean_volumes)
            std_dev = SupportingFunctions.calculate_std_dev(clean_volumes, mean_volume)
            error_margin = SupportingFunctions.calculate_error_margin(std_dev, len(clean_volumes))

            # Добавление данных к графику
            self.add_plot(clean_time_data, clean_volumes, {}, label, error_margin)

    def add_plot(self, x_data, y_data, params, label, error_margin=None, calculate_auc=False, fill_alpha=0.2):
        """
        Добавляет на график линию с данными, опционально с доверительными интервалами и расчетом площади под кривой (AUC).

        Args:
            x_data (List[float]): Данные по оси X.
            y_data (List[float]): Данные по оси Y.
            params (list): Параметры эксперимента для включения в подпись графика.
            label (str): Подпись для графика.
            error_margin (List[float], optional): Доверительный интервал или ошибка для каждой точки данных.
            calculate_auc (bool, optional): Если True, будет рассчитана площадь под кривой (AUC).
            fill_alpha (float, optional): Прозрачность заливки для доверительных интервалов.

        Пример использования:
            add_plot([1, 2, 3], [4, 5, 6], {}, "Тест", [0.1, 0.2, 0.1], True, 0.3)
        """
        # Выбор стиля линии и инкремент индекса
        current_linestyle = self.linestyles[self.linestyle_index % len(self.linestyles)]
        self.linestyle_index += 1

        x_data = np.array(x_data, dtype=float)

        # Форматирование подписи с параметрами и датой
        formatted_label = f"{label} {format_experiment_params(params)}"

        # Создаем линейный график
        line, = plt.plot(
            x_data,
            y_data,
            marker=self.markers[self.marker_index % len(self.markers)],
            markersize=self.marker_size,
            linestyle=current_linestyle,
            zorder=2,
            label=formatted_label
        )

        # Расчет и добавление AUC, если необходимо
        if calculate_auc:
            auc_value = SupportingFunctions.calculate_auc(y_data, x_data)
            self.aucs.append(auc_value)

        self.lines.append(line)
        self.marker_index += 1

        # Проверяем, является ли error_margin итерируемым объектом, и если нет, преобразуем его
        if error_margin is not None and not hasattr(error_margin, '__iter__'):
            error_margin = [error_margin] * len(y_data)  # Создаем список с повторяющимся значением error_margin

        # Добавление доверительных интервалов
        if error_margin is not None:
            line_color = line.get_color()
            plt.fill_between(
                x_data,
                [y - e for y, e in zip(y_data, error_margin)],
                [y + e for y, e in zip(y_data, error_margin)],
                color=line_color,
                alpha=fill_alpha
            )

    def add_legend(self, labels, title="", loc="upper left", display_marker=True):
        """
        Добавляет легенду к графику. Позволяет добавлять как одиночные значения, так и списки меток для создания
        комплексных легенд с различными заголовками и расположением.

        Args:
            labels (Union[List[str], str]): Метки для легенды. Может быть как списком строк, так и одиночной строкой.
            title (str, optional): Заголовок легенды. По умолчанию пустая строка.
            loc (str, optional): Расположение легенды на графике. По умолчанию "upper left".
            display_marker (bool, optional): Флаг, указывающий на необходимость отображения маркеров в легенде.
            По умолчанию True.

        Returns:
            None
        """
        if not isinstance(labels, list):  # Если labels не список, преобразуем в список
            labels = [labels]
        self.legend_info.append((labels, title, loc, display_marker))

    def update_axes_limits(self, x_data_lists):
        """
        Обновляет максимальные значения по оси X на основе предоставленных списков данных. Это позволяет гарантировать,
        что все графики имеют одинаковый масштаб оси X для удобства сравнения.

        Args:
            x_data_lists (List[List[float]]): Список списков данных по оси X для всех графиков, которые нужно отобразить
            на одном рисунке.

        Returns:
            None
        """
        # Вычисление максимального значения по оси X из всех предоставленных наборов данных
        self.max_x = max(max(x_data) for x_data in x_data_lists) if x_data_lists else self.max_x

    def finalize_figure(self, file_path, main_legend_title="", ncol=1, legend_fontsize='medium'):
        """
        Финализирует и отображает график, добавляя легенду и соответствующие оформления. Также сохраняет график в файл.

        Args:
            file_path (str): Путь для сохранения графика.
            main_legend_title (str): Основной заголовок легенды.
            ncol (int): Количество колонок в легенде.
            legend_fontsize (Union[str, int]): Размер шрифта для легенды.

        Returns:
            None
        """
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
            first_legend = plt.legend(handles=self.lines, loc=self.legend_position, title=main_legend_title, ncol=ncol,
                                      fontsize=legend_fontsize)
            ax.add_artist(first_legend)  # Важно использовать add_artist для сохранения основной легенды

        # Создаем и добавляем легенду AUC, если есть значения AUC
        if self.aucs:
            auc_labels = [f"AUC: {auc:.2f}" for auc in self.aucs]
            # Создаем объекты легенды AUC. Важно передать 'handles=self.lines', если стили линий важны
            auc_legend = plt.legend(handles=self.lines, labels=auc_labels, title="Площадь под кривой",
                                    loc='upper right', fontsize=legend_fontsize)
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
        #plt.show()
