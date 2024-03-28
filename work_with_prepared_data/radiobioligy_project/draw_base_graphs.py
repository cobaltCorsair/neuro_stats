# файл draw_base_graphs.py

import numpy as np
import matplotlib.pyplot as plt

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

        # Добавление данных на график
        drawgraph.add_plot(self.time_data, mean_volumes, self.experiment_params, "M/V абс.: ", error_margin)
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

        # Добавление данных на график
        drawgraph.add_plot(self.time_data, mean_relative_volumes, self.experiment_params, "M/V отн.: ",
                           error_margin_rel)

        formatted_params = format_experiment_params(self.experiment_params)
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

        # Добавление данных на график
        drawgraph.add_plot(self.time_data, relative_mean_volumes, self.experiment_params, "M/V отн. ср.: ",
                           error_margin_rel_mean)

        formatted_params = format_experiment_params(self.experiment_params)
        drawgraph.finalize_figure(f"{', '.join(self.experiment_params)}_mean_relative_mean_volumes")



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
