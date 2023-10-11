from typing import List

import numpy as np
import pandas as pd
from scipy.stats import zscore, t


class ExtractOutliers:
    def __init__(self, vis_baseclass):
        """
        Удаляем точки, которые могут слишком отклоняться
        """
        self.base_class = vis_baseclass

    def remove_outliers(self, threshold=2):
        """
        Удаляет выбросы из данных о кожных реакциях, используя Z-score.

        Parameters:
            threshold (float): Пороговое значение Z-score для определения выбросов.
        """
        # Вычисление Z-score для данных о кожных реакциях
        z_scores = np.abs(zscore(self.base_class.skin_reactions, nan_policy='omit'))

        # Определение строк (крыс), в которых есть хотя бы одно значение, превышающее пороговое значение
        outlier_rows = np.any(z_scores > threshold, axis=1)

        # Обновление меток и данных о кожных реакциях, исключая выбросы
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      not outlier_rows[idx]]
        self.base_class.skin_reactions = [reaction for idx, reaction in enumerate(self.base_class.skin_reactions) if
                                          not outlier_rows[idx]]

    def remove_outliers_iqr(self, k=1.5):
        """
        Удаляет выбросы из данных о кожных реакциях, используя IQR.

        Parameters:
            k (float): Множитель для IQR.
        """
        # Преобразование данных в DataFrame для удобства
        skin_reactions_df = pd.DataFrame(self.base_class.skin_reactions, columns=self.base_class.time_data)

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
        self.base_class.skin_reactions = clean_skin_reactions_df.values.tolist()

    def remove_outliers_grubbs(self, alpha=0.05):
        """
        Удаляет выбросы из данных о кожных реакциях, используя тест Граббса.

        Parameters:
            alpha (float): Уровень значимости для теста Граббса.
        """
        # Преобразование данных в DataFrame для удобства
        skin_reactions_df = pd.DataFrame(self.base_class.skin_reactions, columns=self.base_class.time_data)

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
        self.base_class.skin_reactions = clean_skin_reactions_df.values.tolist()

    def remove_local_outliers(self, window_size=3, threshold=2):
        """
        Удаляет выбросы, используя локальные средние и стандартные отклонения.

        Parameters:
            window_size (int): Размер окна для вычисления среднего и стандартного отклонения.
            threshold (float): Порог для определения выбросов в единицах стандартного отклонения.
        """
        # Вычисление средних и стандартных отклонений для каждой точки, используя скользящее окно
        means = pd.DataFrame(self.base_class.skin_reactions).rolling(window=window_size, center=True).mean().values
        std_devs = pd.DataFrame(self.base_class.skin_reactions).rolling(window=window_size, center=True).std().values

        # Определение выбросов как точек, которые отклоняются от среднего на threshold*std_dev или более
        outliers = np.abs(np.array(self.base_class.skin_reactions) - means) > threshold * std_devs

        # Удаление строк, содержащих хотя бы один выброс
        non_outlier_indices = ~np.any(outliers, axis=1)

        self.base_class.rat_labels = [label for i, label in enumerate(self.base_class.rat_labels) if
                                      non_outlier_indices[i]]
        self.base_class.skin_reactions = [reactions for i, reactions in enumerate(self.base_class.skin_reactions) if
                                          non_outlier_indices[i]]

    def exclude_rats(self, excluded_rats: List[str], data_attribute_name: str):
        """
        Исключает крыс с указанными метками из данных.

        Parameters:
            excluded_rats (List[str]): Список меток крыс, которые следует исключить.
            data_attribute_name (str): Имя атрибута, который содержит данные для обработки.
        """
        # Получаем индексы крыс, которые необходимо исключить
        exclude_indices = [i for i, label in enumerate(self.base_class.rat_labels) if label in excluded_rats]

        # Исключаем крыс по индексам из данных и меток
        self.base_class.rat_labels = [label for i, label in enumerate(self.base_class.rat_labels) if
                                      i not in exclude_indices]

        # Используем getattr и setattr для работы с динамическими атрибутами
        data_attribute = getattr(self.base_class, data_attribute_name)
        data_attribute = [data for i, data in enumerate(data_attribute) if i not in exclude_indices]
        setattr(self.base_class, data_attribute_name, data_attribute)


class SupportingFunctions:

    @staticmethod
    def calculate_std_dev(values, mean_value):
        """
        Расчет стандартного отклонения для заданного набора значений.

        Параметры:
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

        Параметры:
            std_dev (float): Стандартное отклонение.
            n (int): Размер выборки.

        Возвращает:
            float: Предел погрешности.
        """
        return std_dev / np.sqrt(n)

    @staticmethod
    def interpolate_data_to_common_timepoints(time_data, skin_reactions, common_timepoints):
        """
        Интерполирует данные о кожных реакциях на общие временные точки.

        Parameters:
            time_data (List[int]): Временные точки оригинальных данных.
            skin_reactions (List[float]): Данные о кожных реакциях.
            common_timepoints (List[int]): Общие временные точки для интерполяции.

        Returns:
            List[float]: Интерполированные данные о кожных реакциях.
        """
        return np.interp(common_timepoints, time_data, skin_reactions)

    @staticmethod
    def calculate_auc(x, y):
        """
        Вычисляет площадь под кривой, используя метод трапеций.

        Parameters:
            x (list): Координаты x точек данных.
            y (list): Координаты y точек данных.

        Returns:
            float: Площадь под кривой.
        """
        return np.trapz(y, x) / 1000

    @staticmethod
    def trim_data_to_timepoint(time_data, values, last_timepoint):
        """
        Обрезает данные до указанной временной точки.

        Parameters:
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
    def safe_str(obj):
        """
        Return the string representation of obj.
        Replaces non-string or non-numeric values with an empty string.
        """
        if obj is np.nan:
            return ''
        try:
            return str(obj)
        except Exception:
            return ''

    @staticmethod
    def join_params(params):
        # Преобразование параметров в строку и исключение 'nan'
        valid_params = [str(param) for param in params if str(param).lower() != 'nan']
        return ', '.join(valid_params)
