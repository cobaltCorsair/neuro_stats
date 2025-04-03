# файл support_stats_methods.py
from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore, t, mannwhitneyu
from sklearn.covariance import EllipticEnvelope
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis


class ExtractOutliers:
    def __init__(self, vis_baseclass):
        """
        Инициализирует экземпляр класса для идентификации и обработки выбросов в данных,
        получаемых из объектов визуализации.

        Данный класс предназначен для работы с экземплярами базового класса визуализаторов,
        позволяя определить и, при необходимости, удалить выбросы из наборов данных перед визуализацией.

        Args:
            vis_baseclass (BaseVisualizerType): Объект базового класса визуализатора, из которого будут извлекаться
                                                данные для анализа на наличие выбросов. 'BaseVisualizerType' здесь
                                                используется как псевдоним для типа базового класса визуализаторов,
                                                который должен предоставлять доступ к данным для анализа.

        """
        self.base_class = vis_baseclass

    def remove_outliers(self, threshold=2):
        """
        Удаляет выбросы из данных об объёмах опухолей, используя Z-score.

        Args:
            threshold (float): Пороговое значение Z-score для определения выбросов.
        """
        # Вычисление Z-score для данных о кожных реакциях
        z_scores = np.abs(zscore(self.base_class.tumor_volumes, nan_policy='omit'))

        # Определение строк (крыс), в которых есть хотя бы одно значение, превышающее пороговое значение
        outlier_rows = np.any(z_scores > threshold, axis=1)

        # Обновление меток и данных о кожных реакциях, исключая выбросы
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      not outlier_rows[idx]]
        self.base_class.tumor_volumes = [reaction for idx, reaction in enumerate(self.base_class.tumor_volumes) if
                                         not outlier_rows[idx]]
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_outliers_iqr(self, k=1.5):
        """
        Удаляет выбросы из данных об объёмах опухолей, используя IQR.

        Args:
            k (float): Множитель для IQR.
        """
        # Преобразование данных в DataFrame для удобства
        skin_reactions_df = pd.DataFrame(self.base_class.tumor_volumes, columns=self.base_class.time_data)

        # Вычисление Q1, Q3 и IQR для каждого временного шага
        Q1 = skin_reactions_df.quantile(0.25)
        Q3 = skin_reactions_df.quantile(0.75)
        IQR = Q3 - Q1

        # Определение выбросов
        outlier_condition = (skin_reactions_df < (Q1 - k * IQR)) | (skin_reactions_df > (Q3 + k * IQR))

        # Удаление строк, содержащих хотя бы один выброс
        clean_skin_reactions_df = skin_reactions_df[~outlier_condition.any(axis=1)]

        # Обновление меток и данных о кожных реакциях
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      idx in clean_skin_reactions_df.index]
        self.base_class.tumor_volumes = clean_skin_reactions_df.values.tolist()
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_outliers_grubbs(self, alpha=0.05):
        """
        Удаляет выбросы из данных об объёмах опухолей, используя тест Граббса.

        Args:
            alpha (float): Уровень значимости для теста Граббса.
        """
        # Преобразование данных в DataFrame для удобства
        skin_reactions_df = pd.DataFrame(self.base_class.tumor_volumes, columns=self.base_class.time_data)

        # Вычисление z-оценок
        z_scores = np.abs(zscore(skin_reactions_df, axis=0))

        # Вычисление критического значения G для каждого временного шага
        N = len(skin_reactions_df)
        t_crit = t.ppf(1 - alpha / (2 * N), N - 2)
        G_crit = (N - 1) * np.sqrt(np.square(t_crit) / (N * (N - 2 + np.square(t_crit))))

        # Определение выбросов
        outlier_condition = z_scores > G_crit

        # Удаление строк, содержащих хотя бы один выброс
        clean_skin_reactions_df = skin_reactions_df[~outlier_condition.any(axis=1)]

        # Обновление меток и данных о кожных реакциях
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      idx in clean_skin_reactions_df.index]
        self.base_class.tumor_volumes = clean_skin_reactions_df.values.tolist()
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_local_outliers(self, window_size=3, threshold=2):
        """
        Удаляет локальные выбросы из данных об объёмах опухолей.

        Args:
            window_size (int): Размер окна для сглаживания.
            threshold (float): Пороговое значение для определения выбросов.
        """
        # Преобразование данных в DataFrame для удобства
        skin_reactions_df = pd.DataFrame(self.base_class.tumor_volumes, columns=self.base_class.time_data)

        # Создание сглаженных данных
        smoothed_df = skin_reactions_df.rolling(window=window_size, min_periods=1, axis=1).mean()

        # Вычисление разницы между оригинальными и сглаженными данными
        diff = np.abs(skin_reactions_df - smoothed_df)

        # Вычисление среднего и стандартного отклонения разницы
        mean_diff = diff.mean().mean()
        std_diff = diff.std().std()

        # Определение выбросов
        outlier_condition = diff > (mean_diff + threshold * std_diff)

        # Удаление строк, содержащих хотя бы один выброс
        clean_skin_reactions_df = skin_reactions_df[~outlier_condition.any(axis=1)]

        # Обновление меток и данных о кожных реакциях
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      idx in clean_skin_reactions_df.index]
        self.base_class.tumor_volumes = clean_skin_reactions_df.values.tolist()
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_outliers_elliptic_envelope(self, contamination=0.1):
        """
        Удаляет выбросы из данных об объёмах опухолей, используя эллиптическую оболочку.

        Args:
            contamination (float): Предполагаемая доля выбросов в данных.
        """
        # Преобразование данных в подходящий формат для EllipticEnvelope
        skin_reactions_array = np.array(self.base_class.tumor_volumes)
        n_samples, n_features = skin_reactions_array.shape

        # Проверка на возможность применения EllipticEnvelope
        if n_samples <= n_features:
            print("Предупреждение: Недостаточно образцов для применения EllipticEnvelope. Пропуск.")
            return

        # Применение EllipticEnvelope
        # Транспонирование для работы с временными рядами
        clf = EllipticEnvelope(contamination=contamination)
        try:
            clf.fit(skin_reactions_array)
            y_pred = clf.predict(skin_reactions_array)
        except Exception as e:
            print(f"Ошибка при применении EllipticEnvelope: {e}. Пропуск.")
            return

        # Идентификация не-выбросов (1) и выбросов (-1)
        non_outlier_mask = y_pred == 1

        # Обновление меток и данных о кожных реакциях
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      non_outlier_mask[idx]]
        self.base_class.tumor_volumes = [reaction for idx, reaction in enumerate(self.base_class.tumor_volumes) if
                                         non_outlier_mask[idx]]
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_outliers_isolation_forest(self, contamination='auto'):
        """
        Удаляет выбросы из данных об объёмах опухолей, используя метод изоляционного леса.

        Args:
            contamination (float or 'auto'): Предполагаемая доля выбросов в данных.
        """
        # Преобразование данных в подходящий формат для IsolationForest
        skin_reactions_array = np.array(self.base_class.tumor_volumes)

        # Применение IsolationForest
        clf = IsolationForest(random_state=42, contamination=contamination)
        try:
            clf.fit(skin_reactions_array)
            y_pred = clf.predict(skin_reactions_array)
        except Exception as e:
            print(f"Ошибка при применении IsolationForest: {e}. Пропуск.")
            return

        # Идентификация не-выбросов (1) и выбросов (-1)
        non_outlier_mask = y_pred == 1

        # Обновление меток и данных о кожных реакциях
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      non_outlier_mask[idx]]
        self.base_class.tumor_volumes = [reaction for idx, reaction in enumerate(self.base_class.tumor_volumes) if
                                         non_outlier_mask[idx]]
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_outliers_mahalanobis(self, alpha=0.01):
        """
        Идентифицирует и удаляет выбросы с использованием квадратичного расстояния Махаланобиса.

        Args:
            alpha (float): Уровень значимости для определения порогового значения расстояния.
        """
        data = pd.DataFrame(self.base_class.tumor_volumes)  # Транспонируем для удобства

        # Вычисляем ковариационную матрицу и её обратную матрицу
        cov_matrix = np.cov(data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mean = np.mean(data, axis=0)

        # Вычисление расстояния Махаланобиса
        mahal_distance = data.apply(lambda x: mahalanobis(x, mean, inv_cov_matrix), axis=1)

        # Пороговое значение на основе распределения chi-square
        threshold = chi2.ppf((1 - alpha), df=data.shape[1])

        # Индексы не выбросов
        non_outlier_indices = np.where(mahal_distance <= threshold)[0]

        # Обновление меток крыс и данных
        self.base_class.rat_labels = [self.base_class.rat_labels[i] for i in non_outlier_indices]
        self.base_class.tumor_volumes = [self.base_class.tumor_volumes[i] for i in non_outlier_indices]
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_outliers_by_euclidean(self, percentile_threshold=90):
        """
        Удаляет выбросы, сравнивая каждую кривую с средней кривой по Евклидовому расстоянию.

        Args:
            percentile_threshold (float): Процентиль для определения выбросов (на основе отклонений).
        """
        # Преобразуем данные в массив numpy для обработки
        data = np.array(self.base_class.tumor_volumes)

        # Рассчитываем среднюю кривую (среднее значение для каждого временного шага)
        mean_curve = np.mean(data, axis=0)

        # Вычисляем Евклидово расстояние каждой кривой до средней кривой
        distances = []
        for curve in data:
            distance = np.linalg.norm(curve - mean_curve)  # Евклидово расстояние
            distances.append(distance)

        # Определяем пороговое значение на основе заданного процентиля
        threshold = np.percentile(distances, percentile_threshold)
        outlier_indices = np.where(np.array(distances) > threshold)[0]

        # Удаляем выбросы
        self.base_class.rat_labels = [label for i, label in enumerate(self.base_class.rat_labels) if
                                      i not in outlier_indices]
        self.base_class.tumor_volumes = [volume for i, volume in enumerate(self.base_class.tumor_volumes) if
                                         i not in outlier_indices]

        print(f'Выбросы удалены. Количество выбросов: {len(outlier_indices)}')
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def remove_outliers_kl_divergence(self, bandwidth=0.5, percentile_threshold=90):
        """
        Удаляет выбросы из данных, используя дивергенцию Кульбака-Лейблера (KL Divergence).

        Args:
            bandwidth (float): Ширина полосы для ядерной оценки плотности.
            percentile_threshold (float): Процентиль для определения выбросов (на основе дивергенции).
        """
        # Преобразуем данные в массив numpy для обработки
        data = np.array(self.base_class.tumor_volumes)

        # Убедимся, что данные имеют двумерную форму (n_samples, 1)
        data_reshaped = data.reshape(-1, 1)

        # Оценка плотности распределения данных с помощью KDE (Ядерная оценка плотности)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data_reshaped)
        log_dens = kde.score_samples(data_reshaped)
        dens = np.exp(log_dens)

        # Рассчитываем среднюю плотность всех элементов
        global_density_mean = np.mean(dens)

        # Вычисляем "аномальность" каждого элемента на основе его отклонения от средней плотности
        kl_divergences = []
        for i in range(len(data)):
            kl_divergence = abs(dens[i] - global_density_mean)  # Простая разница плотности
            kl_divergences.append(kl_divergence)

        # Определяем выбросы на основе заданного процентиля
        threshold = np.percentile(kl_divergences, percentile_threshold)
        outlier_indices = np.where(np.array(kl_divergences) > threshold)[0]

        # Удаляем выбросы на основе "аномальной" плотности
        self.base_class.rat_labels = [label for i, label in enumerate(self.base_class.rat_labels) if
                                      i not in outlier_indices]
        self.base_class.tumor_volumes = [volume for i, volume in enumerate(self.base_class.tumor_volumes) if
                                         i not in outlier_indices]

        print(f'Выбросы удалены. Количество выбросов: {len(outlier_indices)}')
        
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

    def exclude_rats(self, excluded_rats: List[str], data_attribute_name: str):
        """
        Исключает крыс с указанными метками из данных.

        Args:
            excluded_rats (List[str]): Список меток крыс, которые следует исключить.
            data_attribute_name (str): Имя атрибута, который содержит данные для обработки.
        """
        # Получаем индексы крыс, которые необходимо исключить
        print("Исключаемые метки крыс:", excluded_rats)
        exclude_indices = [i for i, label in enumerate(self.base_class.rat_labels) if label in excluded_rats]

        # Исключаем крыс по индексам
        self.base_class.rat_labels = [label for i, label in enumerate(self.base_class.rat_labels) if
                                      i not in exclude_indices]

        # Используем getattr и setattr для работы с динамическими атрибутами
        data_attribute = getattr(self.base_class, data_attribute_name)
        data_attribute = [data for i, data in enumerate(data_attribute) if i not in exclude_indices]
        setattr(self.base_class, data_attribute_name, data_attribute)

        # Обновление данных в data_processor, если он существует и мы работаем с объемами опухоли
        if hasattr(self.base_class, 'data_processor') and data_attribute_name == 'tumor_volumes':
            self.base_class.data_processor.tumor_volumes = self.base_class.tumor_volumes

class SupportingFunctions:

    @staticmethod
    def calculate_std_dev(values, mean_value):
        """
        Расчет стандартного отклонения для заданного набора значений.

        Args:
            values (list): Список значений, для которых вычисляется стандартное отклонение.
            mean_value (float): Среднее значение данных значений.

        Возвращает:
            float: Стандартное отклонение.
        """
        n = len(values)
        sum_squared_deviations = sum((val - mean_value) ** 2 for val in values if not np.isnan(val))
        return np.sqrt(sum_squared_deviations / (n - 1))

    @staticmethod
    def calculate_error_margin(std_dev, n):
        """
        Расчет предела погрешности для заданного стандартного отклонения и размера выборки.

        Args:
            std_dev (float): Стандартное отклонение.
            n (int): Размер выборки.

        Returns:
            float: Предел погрешности.
        """
        return std_dev / np.sqrt(n)

    @staticmethod
    def interpolate_data_to_common_timepoints(time_data, skin_reactions, common_timepoints):
        """
        Интерполирует данные о кожных реакциях на общие временные точки.

        Args:
            time_data (List[int]): Временные точки оригинальных данных.
            skin_reactions (List[float]): Данные о кожных реакциях.
            common_timepoints (List[int]): Общие временные точки для интерполяции.

        Returns:
            List[float]: Интерполированные данные о кожных реакциях.
        """
        return np.interp(common_timepoints, time_data, skin_reactions)

    @staticmethod
    def calculate_auc(y, x):
        """
        Вычисляет площадь под кривой, используя метод трапеций.

        Args:
            x (list): Координаты x точек данных.
            y (list): Координаты y точек данных.

        Returns:
            float: Площадь под кривой.
        """
        return np.trapz(y, x)

    @staticmethod
    def calculate_tumor_growth_inhibition(control_volumes, experiment_volumes):
        """
        Расчет торможения роста опухоли между контрольной и экспериментальными группами.

        Args:
            control_volumes (list): Средние объемы опухоли для контрольной группы.
            experiment_volumes (list): Средние объемы опухоли для экспериментальных групп.

        Returns:
            list: Значения торможения роста опухоли.
        """
        return [(control - experiment) / control * 100 for control, experiment in
                zip(control_volumes, experiment_volumes)]

    @staticmethod
    def trim_data_to_timepoint(time_data, values, last_timepoint):
        """
        Обрезает данные до указанной временной точки.

        Args:
            time_data (List): Список временных точек.
            values (List): Список значений, соответствующих временным точкам.
            last_timepoint (int): Последняя временная точка, до которой следует обрезать данные.

        Returns:
            Tuple[List, List]: Обрезанные списки временных точек и значений.
        """
        trimmed_time_data = []
        trimmed_values = []

        for t, v in zip(time_data, values):
            if int(t) <= last_timepoint:
                trimmed_time_data.append(t)
                trimmed_values.append(v)

        return trimmed_time_data, trimmed_values

    @staticmethod
    def normalize_time_data(visualizers):
        """
        Нормализует временные метки всех переданных визуализаторов, преобразуя их к единообразному числовому формату.

        Приводит все временные метки к относительным значениям, вычитая из каждой временной метки значение первой
        временной метки в соответствующем визуализаторе. Это обеспечивает начало отсчёта времени с 0 для каждого
        эксперимента.

        Args:
            visualizers (List[VisualizerType]): Список объектов визуализаторов, временные метки которых будут нормализованы.
                                                Тип VisualizerType здесь используется как псевдоним для любого класса,
                                                который имеет атрибут `time_data`.

        Returns:
            None
        """
        for visualizer in visualizers:
            visualizer.time_data = [int(time) - int(visualizer.time_data[0]) for time in visualizer.time_data]

    @staticmethod
    def trim_data_to_common_length(visualizers):
        """
        Обрезает временные ряды и соответствующие им значения до минимальной общей длины.

        Args:
            visualizers (list): Список визуализаторов с данными для обработки.

        Returns:
            None: Модифицирует объекты визуализаторов 'in-place', обрезая их временные ряды и данные.
        """
        # Находим минимальную длину временного ряда среди всех визуализаторов
        min_length = min(len(viz.time_data) for viz in visualizers)

        # Обрезаем временные ряды и данные до этой минимальной длины
        for viz in visualizers:
            viz.time_data = viz.time_data[:min_length]
            if hasattr(viz, 'tumor_volumes'):  # Если есть атрибут с объемами опухолей
                viz.tumor_volumes = [vol[:min_length] for vol in viz.tumor_volumes]
            if hasattr(viz, 'mean_tumor_volumes'):  # Если есть атрибут со средними объемами опухолей
                viz.mean_tumor_volumes = viz.mean_tumor_volumes[:min_length]

    @staticmethod
    def normalize_time_data_min(visualizers):
        """
        Выравнивает начальные точки временных рядов всех переданных визуализаторов, используя минимальное
        значение начальной временной метки среди всех экспериментов.

        Этот метод гарантирует, что все временные ряды будут начинаться с одного и того же времени, что обеспечивает
        корректное сравнение временных рядов различных экспериментов на общей временной шкале.

        Args:
            visualizers (List[VisualizerType]): Список объектов визуализаторов, временные метки которых будут
                                                нормализованы. Тип VisualizerType здесь используется как псевдоним
                                                для любого класса, который имеет атрибут `time_data`.

        Returns:
            None
        """
        # Находим минимальную начальную точку времени среди всех экспериментов
        min_start_time = min([int(v.time_data[0]) for v in visualizers])

        # Выравниваем все временные ряды, вычитая минимальную начальную точку
        for visualizer in visualizers:
            visualizer.time_data = [int(time) - min_start_time for time in visualizer.time_data]

    @staticmethod
    def apply_mann_whitney_test(all_reactions, common_timepoints, upper_bounds_by_time, offset_ratio=0.00,
                                annotation_fontsize=18):
        """
        Выполнение статистического теста Манна-Уитни для сравнения реакций кожи между экспериментами.
        Добавляет аннотации на графике для значимых различий с фиксированным отступом над доверительным интервалом.

        Args:
            all_reactions (list): Список данных для каждого эксперимента, включая реакции, средние значения и SEM.
            common_timepoints (list): Общие временные точки, по которым сравниваются реакции.
            upper_bounds_by_time (dict): Словарь верхних границ доверительных интервалов для каждой временной точки.
            offset_ratio (float): Доля от диапазона Y для смещения аннотаций вверх.
            annotation_fontsize (int): Размер шрифта для аннотаций значимости (звёздочек).

        Returns:
            None: Аннотации добавляются непосредственно на график.
        """
        num_experiments = len(all_reactions)

        # Получаем текущие пределы оси Y
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min

        # Задаем фиксированный отступ для аннотаций (например, 2% от диапазона Y)
        fixed_offset = y_range * offset_ratio

        # Собираем все аннотации перед их нанесением
        annotations = []
        for i in range(num_experiments):
            for j in range(i + 1, num_experiments):
                group1 = all_reactions[i]['reactions']
                group2 = all_reactions[j]['reactions']

                for t_idx, time_point in enumerate(common_timepoints):
                    values1 = [reaction[t_idx] for reaction in group1 if not np.isnan(reaction[t_idx])]
                    values2 = [reaction[t_idx] for reaction in group2 if not np.isnan(reaction[t_idx])]

                    if len(values1) > 0 and len(values2) > 0:
                        # Выполняем тест Манна-Уитни
                        _, p_value = mannwhitneyu(values1, values2, alternative='two-sided')

                        # Определяем уровень значимости
                        annotation = '*' if p_value < 0.05 else ''

                        if annotation:
                            # Получаем верхнюю границу для текущей временной точки
                            upper_bound = upper_bounds_by_time[time_point]

                            # Устанавливаем y-координату для аннотации с фиксированным отступом
                            y_annotation = upper_bound + fixed_offset

                            # Сохраняем аннотацию для последующего нанесения
                            annotations.append((time_point, y_annotation, annotation))

        # Если есть аннотации, обновляем пределы оси Y, чтобы вместить все аннотации
        if annotations:
            # Находим максимальную y-координату аннотации
            max_y_annotation = max([ann[1] for ann in annotations])

            # Если максимальная y-координата аннотации превышает текущий y_max, обновляем y_max
            if max_y_annotation > y_max:
                # Добавляем дополнительный отступ (например, 5% от диапазона Y)
                additional_offset = 0.05 * y_range
                new_y_max = max_y_annotation + additional_offset
                plt.ylim(y_min, new_y_max)
                y_range = new_y_max - y_min  # Обновляем диапазон Y после изменения y_max
                fixed_offset = y_range * offset_ratio  # Пересчитываем фиксированный отступ при необходимости

        # Наносим все аннотации на график с использованием заданного размера шрифта
        for ann in annotations:
            time_point, y_annotation, annotation = ann
            plt.text(
                time_point,
                y_annotation,
                annotation,
                ha='center',
                va='bottom',
                fontsize=annotation_fontsize,  # Используем новый параметр для размера шрифта
                color='black'
            )
