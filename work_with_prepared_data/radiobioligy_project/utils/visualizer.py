from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

from work_with_prepared_data.radiobioligy_project.gui import graph_manager
from work_with_prepared_data.radiobioligy_project.stats_methods.support_stats_methods import SupportingFunctions
from work_with_prepared_data.radiobioligy_project.utils.plot_saver import save_plot
from work_with_prepared_data.radiobioligy_project.utils.plotting_helpers import format_experiment_params


class GraphVisualizer:
    # Статический словарь для хранения стилей для каждой метки крысы
    label_styles = {}
    
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
            label_styles (dict): Словарь для хранения стилей (маркер, цвет, стиль линии) для каждой метки крысы
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
            p_values, x_positions_for_annotations = GraphVisualizer.prepare_mann_whitney_test_interpolated(
                experiments_to_compare, perform_stat_test)
        elif use_ttest and experiments_to_compare:
            p_values, x_positions_for_annotations = GraphVisualizer.prepare_ttest_interpolated(
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

            # Форматируем параметры эксперимента для текста легенды
            formatted_params = format_experiment_params(visualizer.experiment_params)
            
            # Создаем текст метки для легенды
            legend_label_text = f"{label_prefix} {formatted_params}"

            # Создаем уникальный ключ для стиля, добавляя индекс
            style_key = f"{legend_label_text}_{index}"
            
            # Добавляем график, передавая уникальный ключ стиля и текст для легенды
            graph_visualizer.add_plot(visualizer.time_data, values, {}, legend_label_text,
                                      error_margin, calculate_auc, style_key=style_key)
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
    def prepare_mann_whitney_test_interpolated(experiments_to_compare, perform_stat_test):
        """
        Подготавливает и выполняет статистический тест Манна-Уитни для сравнения экспериментальных групп,
        интерполируя данные для получения значений в общих временных точках, исключая нулевую временную точку.

        Args:
            experiments_to_compare (List[VisualizerType]): Список из двух экспериментов для сравнения.
            perform_stat_test (bool): Флаг, определяющий необходимость выполнения статистического теста.

        Returns:
            Tuple[List[float], List[float]]: Возвращает два списка - значения p и позиции по оси X для аннотаций.
        """
        if not (perform_stat_test and experiments_to_compare and len(experiments_to_compare) == 2):
            print("Тест не был выполнен, проверьте входные данные.")
            return [], []

        exp1, exp2 = experiments_to_compare

        # Выбираем временные точки из первого эксперимента
        time_points = exp1.time_data

        # Интерполируем данные второго эксперимента на временные точки первого
        interpolated_volumes_exp2 = []
        for vol in exp2.tumor_volumes:
            interp_func = interp1d(exp2.time_data, vol, kind='linear', fill_value='extrapolate')
            interpolated_vol = interp_func(time_points)
            interpolated_volumes_exp2.append(interpolated_vol)
        interpolated_volumes_exp2 = np.array(interpolated_volumes_exp2)

        # Преобразуем данные первого эксперимента в массив
        volumes_exp1 = np.array(exp1.tumor_volumes)

        p_values = []
        x_positions_for_annotations = []

        # Проходим по временным точкам и выполняем тесты
        for idx, time_point in enumerate(time_points):
            # Пропускаем нулевую временную точку
            if time_point == 0:
                continue

            # Получаем значения для данной временной точки
            group1_volumes = volumes_exp1[:, idx]
            group2_volumes = interpolated_volumes_exp2[:, idx]

            # Удаляем NaN из данных каждой группы отдельно
            group1_volumes = group1_volumes[~np.isnan(group1_volumes)]
            group2_volumes = group2_volumes[~np.isnan(group2_volumes)]

            if len(group1_volumes) > 0 and len(group2_volumes) > 0:
                # Выполняем тест Манна-Уитни
                p_value = GraphVisualizer.perform_mann_whitney_test(
                    group1_volumes, group2_volumes)
                p_values.append(p_value)
                x_positions_for_annotations.append(time_point)
            else:
                p_values.append(None)
                x_positions_for_annotations.append(time_point)

        return p_values, x_positions_for_annotations

    @staticmethod
    def prepare_ttest_interpolated(experiments_to_compare, use_ttest):
        """
        Подготавливает и выполняет t-тест для сравнения экспериментальных групп,
        интерполируя данные для получения значений в общих временных точках, исключая нулевую временную точку.

        Args:
            experiments_to_compare (List[VisualizerType]): Список из двух экспериментов для сравнения.
            use_ttest (bool): Флаг, указывающий на выполнение t-теста.

        Returns:
            Tuple[List[float], List[float]]: Возвращает два списка - значения p и позиции по оси X для аннотаций.
        """
        if not (use_ttest and experiments_to_compare and len(experiments_to_compare) == 2):
            print("Тест не был выполнен, проверьте входные данные.")
            return [], []

        exp1, exp2 = experiments_to_compare

        # Выбираем временные точки из первого эксперимента
        time_points = exp1.time_data

        # Интерполируем данные второго эксперимента на временные точки первого
        interpolated_volumes_exp2 = []
        for vol in exp2.tumor_volumes:
            interp_func = interp1d(exp2.time_data, vol, kind='linear', fill_value='extrapolate')
            interpolated_vol = interp_func(time_points)
            interpolated_volumes_exp2.append(interpolated_vol)
        interpolated_volumes_exp2 = np.array(interpolated_volumes_exp2)

        volumes_exp1 = np.array(exp1.tumor_volumes)

        p_values = []
        x_positions_for_annotations = []

        for idx, time_point in enumerate(time_points):
            # Пропускаем нулевую временную точку
            if time_point == 0:
                continue

            group1_volumes = volumes_exp1[:, idx]
            group2_volumes = interpolated_volumes_exp2[:, idx]

            # Удаляем NaN из данных каждой группы отдельно
            group1_volumes = group1_volumes[~np.isnan(group1_volumes)]
            group2_volumes = group2_volumes[~np.isnan(group2_volumes)]

            if len(group1_volumes) > 1 and len(group2_volumes) > 1:
                t_stat, p_value = ttest_ind(
                    group1_volumes, group2_volumes, equal_var=False)
                p_values.append(p_value)
                x_positions_for_annotations.append(time_point)
            else:
                p_values.append(None)
                x_positions_for_annotations.append(time_point)

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

        # Создаем словарь для быстрого доступа к upper_bounds по time_point
        upper_bounds_by_time = dict(zip(time_data, upper_bounds))

        # Проходим по p-value и добавляем аннотации
        for p_value, x_position in zip(p_values, x_positions):
            if p_value is None:
                continue  # Пропускаем, если p-value не рассчитано
            # if p_value < 0.001:
            #     annotation = '***'  # Сильно значимо
            # elif p_value < 0.01:
            #     annotation = '**'  # Значимо
            if p_value < 0.05:
                annotation = '*'  # Умеренно значимо
            else:
                annotation = ''  # Не значимо

            if annotation:
                # Получаем upper_bound для текущей временной точки
                upper_bound = upper_bounds_by_time.get(x_position, None)
                if upper_bound is None:
                    # Если нет точного совпадения, ищем ближайшую временную точку
                    closest_time = min(time_data, key=lambda t: abs(t - x_position))
                    upper_bound = upper_bounds_by_time.get(closest_time, None)
                    if upper_bound is None:
                        continue  # Нет верхней границы для данной временной точки

                # Размещаем аннотацию немного выше верхнего значения
                y_position = upper_bound + offset

                # Добавляем аннотацию
                plt.text(x_position, y_position, annotation, ha='center', fontsize=20, color='black')

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

            # Добавление данных к графику, передаем метку крысы как ключ стиля и флаг
            self.add_plot(clean_time_data, clean_volumes, {}, label, error_margin, style_key=label, is_individual_rat=True)

    def add_plot(self, x_data, y_data, params, label, error_margin=None, calculate_auc=False, fill_alpha=0.2, style_key=None, is_individual_rat=False):
        """
        Добавляет на график линию с данными, опционально с доверительными интервалами и расчетом площади под кривой (AUC).

        Args:
            x_data (List[float]): Данные по оси X.
            y_data (List[float]): Данные по оси Y.
            params (list): Параметры эксперимента для включения в подпись графика.
            label (str): Подпись для графика (текст для легенды).
            error_margin (List[float], optional): Доверительный интервал или ошибка для каждой точки данных.
            calculate_auc (bool, optional): Если True, будет рассчитана площадь под кривой (AUC).
            fill_alpha (float, optional): Прозрачность заливки для доверительных интервалов.
            style_key (str, optional): Уникальный ключ для идентификации и сохранения стиля линии. Если None,
                                     используется значение `label`.
            is_individual_rat (bool, optional): Флаг, указывающий, что строится график для отдельной крысы.
                                             Влияет на использование/сохранение стиля в статическом словаре.

        Пример использования:
            add_plot([1, 2, 3], [4, 5, 6], {}, "Тест", [0.1, 0.2, 0.1], True, 0.3, style_key="Тест_1")
        """
        x_data = np.array(x_data, dtype=float)

        # Используем label как ключ стиля по умолчанию, если style_key не предоставлен
        if style_key is None:
            style_key = label

        # Форматирование подписи с параметрами (params здесь могут быть пустыми, как в prepare_and_add_data_to_graph)
        # Основной текст легенды берется из аргумента label
        formatted_legend_label = f"{label} {format_experiment_params(params)}" if params else label
        
        # Проверяем, нужно ли использовать сохраненный стиль (только для индивидуальных крыс)
        should_reuse_style = is_individual_rat and style_key in GraphVisualizer.label_styles

        if should_reuse_style:
            # Используем сохраненный стиль для крысы
            marker, color, linestyle = GraphVisualizer.label_styles[style_key]
            line, = plt.plot(
                x_data,
                y_data,
                marker=marker,
                markersize=self.marker_size,
                linestyle=linestyle,
                color=color,
                zorder=2,
                label=formatted_legend_label # Используем label для легенды
            )
        else:
            # Генерируем новый стиль для этого графика (эксперимент или новая крыса)
            current_linestyle = self.linestyles[self.linestyle_index % len(self.linestyles)]
            current_marker = self.markers[self.marker_index % len(self.markers)]
            
            # Создаем линейный график с новым стилем
            line, = plt.plot(
                x_data,
                y_data,
                marker=current_marker,
                markersize=self.marker_size,
                linestyle=current_linestyle,
                zorder=2,
                label=formatted_legend_label # Используем label для легенды
            )
            new_style = (current_marker, line.get_color(), current_linestyle)

            # Сохраняем стиль в статический словарь ТОЛЬКО если это индивидуальная крыса
            if is_individual_rat:
                GraphVisualizer.label_styles[style_key] = new_style
            
            # Увеличиваем индексы для следующей метки только при генерации нового стиля
            self.linestyle_index += 1
            self.marker_index += 1

        # Расчет и добавление AUC, если необходимо
        if calculate_auc:
            auc_value = SupportingFunctions.calculate_auc(y_data, x_data)
            self.aucs.append(auc_value)

        self.lines.append(line)

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

    def finalize_figure(self, file_path, main_legend_title="", ncol=1, legend_fontsize='medium', legend_title_fontsize=None):
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
        # Сохраняем заголовок основной легенды для использования в отдельном окне
        self.main_legend_title = main_legend_title

        ax = plt.gca()  # Получаем текущий объект Axes

        # Устанавливаем деления оси X
        if not self.max_x and self.lines:
            # Предполагаем, что все линии используют одинаковый набор данных x, поэтому берем максимум из первой
            self.max_x = max(self.lines[0].get_xdata())
            # Устанавливаем деления оси X
        if self.max_x is not None:
            plt.xticks(ticks=range(0, int(self.max_x) + 1, 3), rotation=0)

        # Создаем и добавляем основную легенду с переносом строки перед временем облучения и датой
        # Только если legend_position не равно None
        if self.lines and self.legend_position is not None:
            labels = []
            for line in self.lines:
                original_label = line.get_label()
                if original_label.startswith("Контроль: без облучения"):
                    labels.append("Контроль: без облучения")
                elif ", Интервал:" in original_label:
                    main_part, time_part = original_label.split(", Интервал:", 1)
                    wrapped_label = f"{main_part.strip()}\nИнтервал: {time_part.strip()}"
                    labels.append(wrapped_label)
                else:
                    labels.append(original_label)

            # Создаём основную легенду с новыми метками
            legend_kwargs = dict(handles=self.lines, labels=labels, loc=self.legend_position,
                                 title=main_legend_title, ncol=ncol, fontsize=legend_fontsize)
            if legend_title_fontsize is not None:
                legend_kwargs['title_fontsize'] = legend_title_fontsize
            first_legend = plt.legend(**legend_kwargs)
            first_legend.get_title().set_multialignment('center')
            ax.add_artist(first_legend)  # Важно использовать add_artist для сохранения основной легенды

        # Создаем и добавляем легенду AUC, если есть значения AUC и легенда не скрыта
        if self.aucs and self.legend_position is not None:
            auc_labels = [f"AUC: {auc:.2f}" for auc in self.aucs]
            # Создаем объекты легенды AUC. Важно передать 'handles=self.lines', если стили линий важны
            auc_legend = plt.legend(handles=self.lines, labels=auc_labels, title="Площадь под кривой",
                                    loc='upper right', fontsize=legend_fontsize)
            ax.add_artist(auc_legend)  # Добавляем легенду AUC

        # Обрабатываем дополнительные легенды, только если легенда не скрыта
        if self.legend_position is not None:
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
        # Сохраняем путь к файлу для возможного сохранения легенды
        self.last_save_path = file_path
        #plt.show()

    def save_legend_separately(self, legend_file_path=None):
        """
        Сохраняет легенду в отдельный файл
        
        Args:
            legend_file_path (str, optional): Путь для сохранения легенды. 
                                            Если не указан, используется путь графика с суффиксом '_legend.txt'
        
        Returns:
            str: Путь к сохраненному файлу легенды
        """
        import os
        
        # Формируем путь для файла легенды
        if legend_file_path is None:
            base_path = getattr(self, 'last_save_path', 'legend')
            if base_path.endswith('.png') or base_path.endswith('.jpg') or base_path.endswith('.pdf'):
                legend_file_path = os.path.splitext(base_path)[0] + '_legend.txt'
            else:
                legend_file_path = base_path + '_legend.txt'
        
        # Собираем информацию о легенде
        legend_text = f"Легенда графика: {self.title}\n"
        legend_text += "=" * 60 + "\n\n"
        
        # Основная легенда
        if self.lines:
            legend_text += "Основные элементы графика:\n"
            legend_text += "-" * 30 + "\n"
            for i, line in enumerate(self.lines, 1):
                original_label = line.get_label()
                if original_label.startswith("Контроль: без облучения"):
                    formatted_label = "Контроль: без облучения"
                elif ", Интервал:" in original_label:
                    main_part, time_part = original_label.split(", Интервал:", 1)
                    formatted_label = f"{main_part.strip()}\nИнтервал: {time_part.strip()}"
                else:
                    formatted_label = original_label
                
                legend_text += f"{i}. {formatted_label}\n"
                legend_text += f"   Цвет: {line.get_color()}\n"
                legend_text += f"   Маркер: {line.get_marker()}\n"
                legend_text += f"   Стиль линии: {line.get_linestyle()}\n"
                legend_text += f"   Толщина линии: {line.get_linewidth()}\n\n"
        
        # AUC легенда
        if self.aucs:
            legend_text += "Площади под кривой (AUC):\n"
            legend_text += "-" * 30 + "\n"
            for i, auc in enumerate(self.aucs, 1):
                legend_text += f"{i}. AUC: {auc:.2f}\n"
            legend_text += "\n"
        
        # Дополнительные легенды
        if self.legend_info:
            legend_text += "Дополнительная информация:\n"
            legend_text += "-" * 30 + "\n"
            for i, extra_legend_data in enumerate(self.legend_info, 1):
                labels, title, loc, display_marker = extra_legend_data
                legend_text += f"{i}. {title}:\n"
                for j, label in enumerate(labels):
                    legend_text += f"   {j+1}. {label}\n"
                legend_text += "\n"
        
        # Сохраняем в файл
        try:
            with open(legend_file_path, 'w', encoding='utf-8') as f:
                f.write(legend_text)
            
            # Сохраняем путь для будущего использования
            self.last_save_path = legend_file_path
            
            return legend_file_path
            
        except Exception as e:
            raise Exception(f"Ошибка при сохранении легенды: {str(e)}")

    def save_legend_automatically(self, base_file_path):
        """
        Автоматически сохраняет легенду рядом с основным графиком
        
        Args:
            base_file_path (str): Путь к основному файлу графика
        
        Returns:
            str: Путь к сохраненному файлу легенды
        """
        import os
        
        # Формируем путь для файла легенды
        if base_file_path.endswith('.png') or base_file_path.endswith('.jpg') or base_file_path.endswith('.pdf'):
            legend_file_path = os.path.splitext(base_file_path)[0] + '_legend.txt'
        else:
            legend_file_path = base_file_path + '_legend.txt'
        
        return self.save_legend_separately(legend_file_path)