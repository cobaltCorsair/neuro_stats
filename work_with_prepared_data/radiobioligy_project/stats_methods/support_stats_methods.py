# файл support_stats_methods.py
from typing import List, Optional, Tuple
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
from work_with_prepared_data.radiobioligy_project.gui import graph_manager
from work_with_prepared_data.radiobioligy_project.data_processing.excel_data_processor import (
    RatSurvivalEvent,
    find_rats_that_died_before_experiment_end,
)


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
        # Определяем имя атрибута с данными
        if hasattr(self.base_class, 'tumor_volumes'):
            self.data_attr = 'tumor_volumes'
        elif hasattr(self.base_class, 'skin_reactions'):
            self.data_attr = 'skin_reactions'
        else:
            self.data_attr = 'tumor_volumes'

    def _get_data(self):
        return getattr(self.base_class, self.data_attr)

    def _set_data(self, new_data):
        setattr(self.base_class, self.data_attr, new_data)
        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            if hasattr(self.base_class.data_processor, self.data_attr):
                setattr(self.base_class.data_processor, self.data_attr, new_data)
            elif self.data_attr == 'tumor_volumes' and hasattr(self.base_class.data_processor, 'tumor_volumes'):
                self.base_class.data_processor.tumor_volumes = new_data

    def remove_outliers(self, threshold=2):
        """
        Удаляет выбросы из данных, используя Z-score.

        Args:
            threshold (float): Пороговое значение Z-score для определения выбросов.
        """
        data = self._get_data()
        # Вычисление Z-score
        z_scores = np.abs(zscore(data, nan_policy='omit'))

        # Определение строк (крыс), в которых есть хотя бы одно значение, превышающее пороговое значение
        outlier_rows = np.any(z_scores > threshold, axis=1)

        # Обновление меток и данных, исключая выбросы
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      not outlier_rows[idx]]
        self._set_data([reaction for idx, reaction in enumerate(data) if not outlier_rows[idx]])

    def remove_outliers_iqr(self, k=1.5):
        """
        Удаляет выбросы из данных, используя IQR.

        Args:
            k (float): Множитель для IQR.
        """
        data = self._get_data()
        # Преобразование данных в DataFrame для удобства
        df = pd.DataFrame(data, columns=self.base_class.time_data)

        # Вычисление Q1, Q3 и IQR для каждого временного шага
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        # Определение выбросов
        outlier_condition = (df < (Q1 - k * IQR)) | (df > (Q3 + k * IQR))

        # Удаление строк, содержащих хотя бы один выброс
        clean_df = df[~outlier_condition.any(axis=1)]

        # Обновление меток и данных
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      idx in clean_df.index]
        self._set_data(clean_df.values.tolist())

    def remove_outliers_grubbs(self, alpha=0.05):
        """
        Удаляет выбросы из данных, используя тест Граббса.

        Args:
            alpha (float): Уровень значимости для теста Граббса.
        """
        data = self._get_data()
        # Преобразование данных в DataFrame для удобства
        df = pd.DataFrame(data, columns=self.base_class.time_data)

        # Вычисление z-оценок
        z_scores = np.abs(zscore(df, axis=0))

        # Вычисление критического значения G для каждого временного шага
        N = len(df)
        t_crit = t.ppf(1 - alpha / (2 * N), N - 2)
        G_crit = (N - 1) * np.sqrt(np.square(t_crit) / (N * (N - 2 + np.square(t_crit))))

        # Определение выбросов
        outlier_condition = z_scores > G_crit

        # Удаление строк, содержащих хотя бы один выброс
        clean_df = df[~outlier_condition.any(axis=1)]

        # Обновление меток и данных
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      idx in clean_df.index]
        self._set_data(clean_df.values.tolist())

    def remove_local_outliers(self, window_size=3, threshold=2):
        """
        Удаляет локальные выбросы из данных.

        Args:
            window_size (int): Размер окна для сглаживания.
            threshold (float): Пороговое значение для определения выбросов.
        """
        data = self._get_data()
        # Преобразование данных в DataFrame для удобства
        df = pd.DataFrame(data, columns=self.base_class.time_data)

        # Создание сглаженных данных
        smoothed_df = df.rolling(window=window_size, min_periods=1, axis=1).mean()

        # Вычисление разницы между оригинальными и сглаженными данными
        diff = np.abs(df - smoothed_df)

        # Вычисление среднего и стандартного отклонения разницы
        mean_diff = diff.mean().mean()
        std_diff = diff.std().std()

        # Определение выбросов
        outlier_condition = diff > (mean_diff + threshold * std_diff)

        # Удаление строк, содержащих хотя бы один выброс
        clean_df = df[~outlier_condition.any(axis=1)]

        # Обновление меток и данных
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      idx in clean_df.index]
        self._set_data(clean_df.values.tolist())

    def remove_outliers_elliptic_envelope(self, contamination=0.1):
        """
        Удаляет выбросы из данных, используя эллиптическую оболочку.

        Args:
            contamination (float): Предполагаемая доля выбросов в данных.
        """
        data = self._get_data()
        # Преобразование данных в подходящий формат для EllipticEnvelope
        data_array = np.array(data)
        n_samples, n_features = data_array.shape

        # Проверка на возможность применения EllipticEnvelope
        if n_samples <= n_features:
            print("Предупреждение: Недостаточно образцов для применения EllipticEnvelope. Пропуск.")
            return

        # Применение EllipticEnvelope
        clf = EllipticEnvelope(contamination=contamination)
        try:
            clf.fit(data_array)
            y_pred = clf.predict(data_array)
        except Exception as e:
            print(f"Ошибка при применении EllipticEnvelope: {e}. Пропуск.")
            return

        # Идентификация не-выбросов (1) и выбросов (-1)
        non_outlier_mask = y_pred == 1

        # Обновление меток и данных
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      non_outlier_mask[idx]]
        self._set_data([reaction for idx, reaction in enumerate(data) if non_outlier_mask[idx]])

    def remove_outliers_isolation_forest(self, contamination='auto'):
        """
        Удаляет выбросы из данных, используя метод изоляционного леса.

        Args:
            contamination (float or 'auto'): Предполагаемая доля выбросов в данных.
        """
        data = self._get_data()
        # Преобразование данных в подходящий формат для IsolationForest
        data_array = np.array(data)

        # Применение IsolationForest
        clf = IsolationForest(random_state=42, contamination=contamination)
        try:
            clf.fit(data_array)
            y_pred = clf.predict(data_array)
        except Exception as e:
            print(f"Ошибка при применении IsolationForest: {e}. Пропуск.")
            return

        # Идентификация не-выбросов (1) и выбросов (-1)
        non_outlier_mask = y_pred == 1

        # Обновление меток и данных
        self.base_class.rat_labels = [label for idx, label in enumerate(self.base_class.rat_labels) if
                                      non_outlier_mask[idx]]
        self._set_data([reaction for idx, reaction in enumerate(data) if non_outlier_mask[idx]])

    def remove_outliers_mahalanobis(self, alpha=0.05):
        """
        Идентифицирует и удаляет выбросы с использованием квадратичного расстояния Махаланобиса.

        Args:
            alpha (float): Уровень значимости для определения порогового значения расстояния.
                Выброс: D_M^2 > chi2_{k, 1-alpha}. По умолчанию 0.05.
        """
        data_list = self._get_data()
        data = pd.DataFrame(data_list)

        # Проверка на достаточное количество наблюдений
        n_samples, n_features = data.shape
        if n_samples < n_features:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Недостаточно наблюдений ({n_samples}) для количества признаков ({n_features})")
            print("Метод Махаланобиса не будет применён. Попробуйте другой метод исключения выбросов.")
            return

        # Вычисляем ковариационную матрицу
        cov_matrix = np.cov(data, rowvar=False)
        mean = np.mean(data, axis=0)

        # Проверяем, является ли матрица сингулярной
        try:
            det = np.linalg.det(cov_matrix)
            if abs(det) < 1e-10:  # Матрица почти сингулярная
                print("ПРЕДУПРЕЖДЕНИЕ: Ковариационная матрица сингулярна. Используем псевдообратную матрицу.")
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
            else:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = np.linalg.pinv(cov_matrix)

        # Вычисление квадрата расстояния Махаланобиса: D_M^2 = (x-mu)^T Sigma^{-1} (x-mu)
        try:
            mahal_distance_sq = data.apply(
                lambda x: mahalanobis(x, mean, inv_cov_matrix) ** 2, axis=1
            )
        except Exception as e:
            print(f"ОШИБКА при вычислении расстояния Махаланобиса: {e}")
            return

        # Пороговое значение chi2_{k, 1-alpha}: выброс при D_M^2 > threshold
        threshold = chi2.ppf((1 - alpha), df=data.shape[1])

        # Индексы не выбросов
        non_outlier_indices = np.where(mahal_distance_sq <= threshold)[0]

        if len(non_outlier_indices) == 0:
            print("ПРЕДУПРЕЖДЕНИЕ: Все наблюдения классифицированы как выбросы. Метод не применён.")
            return

        # Обновление меток крыс и данных
        self.base_class.rat_labels = [self.base_class.rat_labels[i] for i in non_outlier_indices]
        self._set_data([data_list[i] for i in non_outlier_indices])

    def remove_outliers_by_euclidean(self, percentile_threshold=90):
        """
        Удаляет выбросы, сравнивая каждую кривую с средней кривой по Евклидовому расстоянию.

        Args:
            percentile_threshold (float): Процентиль для определения выбросов (на основе отклонений).
        """
        data_list = self._get_data()
        # Преобразуем данные в массив numpy для обработки
        data = np.array(data_list)

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
        self._set_data([volume for i, volume in enumerate(data_list) if i not in outlier_indices])

        print(f'Выбросы удалены. Количество выбросов: {len(outlier_indices)}')

    def remove_outliers_kl_divergence(self, bandwidth=0.5, percentile_threshold=90):
        """
        Удаляет выбросы из данных, используя дивергенцию Кульбака-Лейблера (KL Divergence).

        Args:
            bandwidth (float): Ширина полосы для ядерной оценки плотности.
            percentile_threshold (float): Процентиль для определения выбросов (на основе дивергенции).
        """
        data_list = self._get_data()
        # Преобразуем данные в массив numpy для обработки
        data = np.array(data_list)

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
        self._set_data([volume for i, volume in enumerate(data_list) if i not in outlier_indices])

        print(f'Выбросы удалены. Количество выбросов: {len(outlier_indices)}')

    def exclude_rats(self, excluded_rats: List[str], data_attribute_name: str = None):
        """
        Исключает крыс с указанными метками из данных.

        Args:
            excluded_rats (List[str]): Список меток крыс, которые следует исключить.
            data_attribute_name (str): Имя атрибута, который содержит данные для обработки.
                                       Если None, используется автоматически определенный атрибут.
        """
        if data_attribute_name is None:
            data_attribute_name = self.data_attr

        # Используем точное соответствие меток для исключения
        # Сначала находим, какие крысы из excluded_rats действительно есть в этом эксперименте
        exclude_indices = []
        for i, label in enumerate(self.base_class.rat_labels):
            if label in excluded_rats:
                exclude_indices.append(i)

        # Если нет крыс для исключения в этом конкретном эксперименте, ничего не делаем
        if not exclude_indices:
            return

        # Проверка: если после исключения не останется крыс, отменяем операцию
        if len(exclude_indices) >= len(self.base_class.rat_labels):
            return

        # Исключаем крыс по индексам
        new_rat_labels = [label for i, label in enumerate(self.base_class.rat_labels) if i not in exclude_indices]

        # Используем getattr и setattr для работы с динамическими атрибутами
        data_attribute = getattr(self.base_class, data_attribute_name)
        new_data_attribute = [data for i, data in enumerate(data_attribute) if i not in exclude_indices]

        # Обновление данных в базовом классе
        self.base_class.rat_labels = new_rat_labels
        setattr(self.base_class, data_attribute_name, new_data_attribute)

        # Обновление данных в data_processor, если он существует
        if hasattr(self.base_class, 'data_processor'):
            if hasattr(self.base_class.data_processor, data_attribute_name):
                updated_data = getattr(self.base_class, data_attribute_name)
                # Убедимся, что данные в правильном формате (список списков для skin_reactions)
                if data_attribute_name == 'skin_reactions':
                    if not isinstance(updated_data, np.ndarray):
                        updated_data = np.array(updated_data, dtype=float)
                setattr(self.base_class.data_processor, data_attribute_name, updated_data)
            elif data_attribute_name == 'tumor_volumes' and hasattr(self.base_class.data_processor, 'tumor_volumes'):
                self.base_class.data_processor.tumor_volumes = getattr(self.base_class, data_attribute_name)

    def exclude_dead_rats(self):
        """
        Автоматически исключает животных, подтверждённо умерших до конца наблюдения в этом
        же файле (см. find_rats_that_died_before_experiment_end) — в отличие от остальных
        методов этого класса, не статистический выброс, а прямое чтение маркеров смерти
        (тех же, что уже используются для Каплана-Майера) из исходного файла эксперимента.
        """
        dead_labels = find_rats_that_died_before_experiment_end(self.base_class.file_path)
        self.exclude_rats(dead_labels)

class SupportingFunctions:

    @staticmethod
    def calculate_std_dev(values, mean_value):
        """
        Расчет стандартного отклонения для заданного набора значений.

        n берётся как число НЕ-NaN значений (а не общая длина values) — иначе при выбывании
        животных (смерть/цензурирование, NaN на хвосте) дисперсия систематически
        недооценивается: сумма квадратов считается по меньшему числу слагаемых, а делится на
        исходное (большее) n-1.

        Args:
            values (list): Список значений, для которых вычисляется стандартное отклонение.
            mean_value (float): Среднее значение данных значений.

        Возвращает:
            float: Стандартное отклонение. NaN, если валидных значений меньше двух.
        """
        valid = [val for val in values if not np.isnan(val)]
        n = len(valid)
        if n < 2:
            return float('nan')
        sum_squared_deviations = sum((val - mean_value) ** 2 for val in valid)
        return np.sqrt(sum_squared_deviations / (n - 1))

    @staticmethod
    def calculate_error_margin(std_dev, n):
        """
        Расчет предела погрешности (SEM) для заданного стандартного отклонения и размера выборки.

        Args:
            std_dev (float): Стандартное отклонение.
            n (int): Размер выборки — число животных, ДЕЙСТВИТЕЛЬНО давших измерение в этой
                точке (не общее число животных в эксперименте; иначе SEM занижается на
                временных точках, где часть животных выбыла).

        Returns:
            float: Предел погрешности. NaN, если n <= 0.
        """
        if n <= 0:
            return float('nan')
        return std_dev / np.sqrt(n)

    @staticmethod
    def count_at_risk(values) -> int:
        """Число НЕ-NaN значений в столбце — корректный n для SEM в этой временной точке."""
        return int(np.sum(~np.isnan(np.asarray(values, dtype=float))))

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
    def holm_correction(p_values: List[Optional[float]], alpha: float = 0.05) -> List[bool]:
        """
        Пошаговая (step-down) поправка Холма-Бонферрони для контроля FWER при множественных
        сравнениях.

        None/NaN элементы не входят в число сравнений m и в результате помечаются False —
        иначе тесты, пропущенные из-за нехватки данных в конкретной точке, занижали бы m и
        тем самым ослабляли поправку для остальных.

        Args:
            p_values (List[Optional[float]]): Список p-значений (может содержать None/NaN).
            alpha (float): Уровень семейственной ошибки первого рода.

        Returns:
            List[bool]: Список той же длины и в том же порядке, что и p_values — True,
                если сравнение значимо после поправки Холма.
        """
        result = [False] * len(p_values)
        indexed = [
            (i, p) for i, p in enumerate(p_values)
            if p is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        m = len(indexed)
        if m == 0:
            return result

        indexed.sort(key=lambda ip: ip[1])
        still_rejecting = True
        for rank, (orig_idx, p) in enumerate(indexed, start=1):
            threshold = alpha / (m - rank + 1)
            if still_rejecting and p <= threshold:
                result[orig_idx] = True
            else:
                still_rejecting = False
                result[orig_idx] = False
        return result

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
    def _skin_reactions_to_rtog_matrix(skin_reactions: List[List[float]]) -> np.ndarray:
        """
        Переводит сырые баллы кожной реакции КАЖДОГО животного в степень RTOG (0-4) по уже
        проверенным на реальных данных порогам map_to_rtog. Общий шаг для aggregate_rtog_grades
        и calculate_rtog_iqr — оба должны агрегировать ПОСЛЕ перевода в RTOG, а не до.

        Returns:
            np.ndarray: Матрица степеней RTOG той же формы (животные x дни), NaN сохраняются.
        """
        # Локальный импорт: from_our_scale_to_rtog.py при загрузке модуля применяет глобальные
        # стили seaborn/matplotlib (sns.set_theme, plt.rcParams.update) — нежелательный побочный
        # эффект для любого, кто просто импортирует SupportingFunctions.
        from stats_methods.from_our_scale_to_rtog import SkinReactionAnalyzer

        arr = np.array(skin_reactions, dtype=float)
        if arr.size == 0:
            return arr

        rtog_arr = np.full_like(arr, np.nan)
        valid_mask = ~np.isnan(arr)
        rtog_arr[valid_mask] = [SkinReactionAnalyzer.map_to_rtog(v) for v in arr[valid_mask]]
        return rtog_arr

    @staticmethod
    def aggregate_rtog_grades(skin_reactions: List[List[float]]) -> List[float]:
        """
        Переводит сырой балл кожной реакции каждого животного в степень RTOG, затем агрегирует
        МЕДИАНОЙ по животным на каждый день. RTOG - порядковая шкала, поэтому медиана корректнее
        среднего: перевод уже усреднённого по группе сырого балла может дать степень, которой не
        было ни у одного животного (например, при поровну расколотой группе вокруг порога).

        Args:
            skin_reactions (List[List[float]]): Сырые баллы по животным (строки) и дням
                (столбцы), как self.skin_reactions в SkinReactionsVisualizer. Может содержать NaN.

        Returns:
            List[float]: Медианная степень RTOG по дням (NaN, если все животные NaN в этот день).
        """
        rtog_arr = SupportingFunctions._skin_reactions_to_rtog_matrix(skin_reactions)
        if rtog_arr.size == 0:
            return []

        with np.errstate(invalid='ignore'):
            return np.nanmedian(rtog_arr, axis=0).tolist()

    @staticmethod
    def calculate_rtog_iqr(skin_reactions: List[List[float]]) -> Tuple[List[float], List[float]]:
        """
        Межквартильный интервал [Q1, Q3] степени RTOG по дням — показатель разброса,
        согласованный с медианной агрегацией aggregate_rtog_grades (в отличие от SEM,
        который предполагает нормальное распределение непрерывной величины, а не порядковую
        шкалу 0-4).

        Returns:
            Tuple[List[float], List[float]]: (Q1 по дням, Q3 по дням).
        """
        rtog_arr = SupportingFunctions._skin_reactions_to_rtog_matrix(skin_reactions)
        if rtog_arr.size == 0:
            return [], []

        with np.errstate(invalid='ignore'):
            q1 = np.nanpercentile(rtog_arr, 25, axis=0)
            q3 = np.nanpercentile(rtog_arr, 75, axis=0)
        return q1.tolist(), q3.tolist()

    @staticmethod
    def calculate_skin_reaction_peak(time_data: List[float], values: List[float]) -> Tuple[float, float]:
        """
        Пиковый балл кожной реакции и день его достижения (на уже агрегированной по группе
        кривой, например усреднённой). При плато на максимуме возвращается ПЕРВЫЙ день.

        Returns:
            Tuple[float, float]: (peak_value, peak_day).

        Raises:
            ValueError: если нет ни одного валидного измерения.
        """
        pairs = [
            (t, v) for t, v in zip(time_data, values)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not pairs:
            raise ValueError("Нет валидных измерений для расчёта пика кожной реакции")

        peak_value = max(v for _, v in pairs)
        peak_day = next(t for t, v in pairs if v == peak_value)
        return peak_value, peak_day

    @staticmethod
    def calculate_skin_reaction_duration_above_threshold(
            time_data: List[float], values: List[float], threshold: float) -> Tuple[float, bool]:
        """
        Суммарная продолжительность (сут), в течение которой кривая >= threshold. Учитывает ВСЕ
        эпизоды превышения порога за период наблюдения (не только первый), с линейной
        интерполяцией моментов входа/выхода между соседними измерениями.

        Returns:
            Tuple[float, bool]: (duration_days, censored). censored=True, если на последней
                точке наблюдения кривая всё ещё >= threshold (эпизод не завершён).
        """
        pairs = [
            (t, v) for t, v in zip(time_data, values)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not pairs:
            return 0.0, False
        pairs.sort(key=lambda tv: tv[0])

        total_duration = 0.0
        in_episode = pairs[0][1] >= threshold
        episode_start = pairs[0][0] if in_episode else None

        for i in range(1, len(pairs)):
            t_prev, v_prev = pairs[i - 1]
            t_curr, v_curr = pairs[i]

            if v_prev < threshold <= v_curr:
                entry_t = t_prev + (threshold - v_prev) / (v_curr - v_prev) * (t_curr - t_prev)
                episode_start = entry_t
                in_episode = True
            elif v_prev >= threshold > v_curr:
                exit_t = t_prev + (threshold - v_prev) / (v_curr - v_prev) * (t_curr - t_prev)
                total_duration += exit_t - episode_start
                in_episode = False
                episode_start = None

        censored = in_episode
        if in_episode:
            total_duration += pairs[-1][0] - episode_start

        return total_duration, censored

    @staticmethod
    def calculate_skin_reaction_time_to_normalization(
            time_data: List[float], values: List[float], normalization_grade: float = 1.0) -> Tuple[float, bool]:
        """
        День устойчивой нормализации кожной реакции (возврат к normalization_grade и ниже,
        БЕЗ последующего повторного превышения до конца наблюдения).

        В отличие от calculate_tgd_threshold_day (первое пересечение — необратимая потеря
        контроля над опухолью), здесь ищется ПОСЛЕДНЕЕ пересечение сверху вниз: кратковременный
        провал балла в середине пика не должен ошибочно засчитываться как нормализация, пока
        реакция фактически продолжается.

        Returns:
            Tuple[float, bool]: (day, censored). censored=True, если кривая так и не опустилась
                устойчиво до normalization_grade к концу наблюдения; day в этом случае — последний
                наблюдённый день (нижняя граница).

        Raises:
            ValueError: если нет ни одного валидного измерения.
        """
        pairs = [
            (t, v) for t, v in zip(time_data, values)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not pairs:
            raise ValueError("Нет валидных измерений для расчёта времени нормализации")
        pairs.sort(key=lambda tv: tv[0])

        if pairs[-1][1] > normalization_grade:
            return pairs[-1][0], True

        for i in range(len(pairs) - 1, 0, -1):
            t_prev, v_prev = pairs[i - 1]
            t_curr, v_curr = pairs[i]
            if v_prev > normalization_grade >= v_curr:
                t_cross = t_prev + (normalization_grade - v_prev) / (v_curr - v_prev) * (t_curr - t_prev)
                return t_cross, False

        # Кривая ни разу не поднималась выше порога — реакция отсутствовала с самого начала.
        return pairs[0][0], False

    @staticmethod
    def calculate_tgd_threshold_day(time_data: List[float], volumes: List[float], k: float = 1.5) -> Tuple[float, bool]:
        """
        Задержка роста опухоли (TGD): находит первый момент времени, когда объём впервые
        достигает k * V(0) (линейная интерполяция между соседними точками наблюдения).

        Сообщается ПЕРВОЕ пересечение порога, даже если кривая немонотонна (опухоль
        регрессировала, затем повторно достигла порога) — это стандартное определение TGD
        в радиобиологии: момент потери контроля над опухолью, а не последующая динамика.

        Args:
            time_data (List[float]): Временные точки наблюдения.
            volumes (List[float]): Объёмы опухоли в эти моменты (может содержать NaN).
            k (float): Кратность исходного объёма, определяющая порог. По умолчанию 1.5.

        Returns:
            Tuple[float, bool]: (день_пересечения, censored). censored=True означает, что
                порог не достигнут в пределах наблюдения — возвращённый день является
                нижней границей (последний наблюдённый день).

        Raises:
            ValueError: если нет ни одного валидного (не-NaN) измерения объёма.
        """
        pairs = [
            (t, v) for t, v in zip(time_data, volumes)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not pairs:
            raise ValueError("Нет валидных измерений объёма для расчёта TGD")
        pairs.sort(key=lambda tv: tv[0])

        t0, v0 = pairs[0]
        threshold = k * v0

        if v0 >= threshold:
            # k > 1, поэтому это достижимо лишь при v0 <= 0 (патологические данные) —
            # возвращаем без деления на ноль вместо падения.
            return t0, False

        for i in range(1, len(pairs)):
            t_prev, v_prev = pairs[i - 1]
            t_curr, v_curr = pairs[i]
            if v_curr >= threshold:
                t_cross = t_prev + (threshold - v_prev) / (v_curr - v_prev) * (t_curr - t_prev)
                return t_cross, False

        return pairs[-1][0], True

    @staticmethod
    def build_tgd_events(time_data: List[float], volumes_per_animal: List[List[float]], k: float = 1.5,
                          labels: Optional[List[str]] = None, source_file: str = "") -> List[RatSurvivalEvent]:
        """
        Строит по одному событию TGD (день пересечения k*V0, либо цензурирование) НА КАЖДОЕ
        животное — в отличие от calculate_tgd_threshold_day, которая обычно применяется к одной
        (чаще средней по группе) кривой.

        Нужно для лог-рангового теста (stats_methods.kaplan_meier.log_rank_test): в отличие от
        поточечного сравнения объёмов (Манна-Уитни/Стьюдента) он трактует животных, не достигших
        порога, как цензурированные наблюдения, а не как обычный разброс — поэтому не теряет
        мощность на смешанных популяциях (часть животных регрессировала, часть — нет).

        Args:
            time_data: Временные точки наблюдения, общие для всех животных группы.
            volumes_per_animal: Список кривых объёма опухоли, по одной на животное.
            k: Кратность исходного объёма, определяющая порог. По умолчанию 1.5.
            labels: Метки животных; если не заданы, используются "#1", "#2", ...
            source_file: Значение поля source_file для всех созданных событий.

        Returns:
            List[RatSurvivalEvent]: По одному событию на животное с хотя бы одним валидным
                измерением объёма (животные без единого валидного измерения пропускаются).
        """
        events = []
        for index, volumes in enumerate(volumes_per_animal):
            label = labels[index] if labels else f"#{index + 1}"
            try:
                day, censored = SupportingFunctions.calculate_tgd_threshold_day(time_data, volumes, k)
            except ValueError:
                continue
            events.append(RatSurvivalEvent(
                label=label, day=day, event_observed=not censored,
                reason="" if censored else f"TGD (k={k})", source_file=source_file,
            ))
        return events

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
    def build_common_timepoints_from_data(experiments_time_lists, step=None):
        """
        Собирает общую сетку времени как объединение реальных дней всех экспериментов.
        step: если задано (например, 2 или 3), прореживает узлы примерно с таким шагом (в сутках).
        """

        def _to_float(x):
            s = str(x).strip()
            if s == "" or s.lower() in ("nan", "none"):
                return None
            try:
                return float(s.replace(',', '.'))
            except Exception:
                return None

        all_pts = []
        for tl in experiments_time_lists:
            for x in tl:
                v = _to_float(x)
                if v is not None:
                    all_pts.append(v)

        pts = sorted(set(all_pts))
        if not step or step <= 1:
            return pts

        picked, last = [], None
        for t in pts:
            if last is None or (t - last) >= step - 1e-9:
                picked.append(t)
                last = t
        return picked

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

        # Сначала собираем p-значения ВСЕХ попарных поточечных сравнений этого вызова
        # (одно "семейство" тестов для поправки Холма), и только потом решаем, что значимо.
        collected = []
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
                        collected.append((time_point, p_value))

        # Поправка Холма включается/выключается чекбоксом "Поправка Холма" в интерфейсе.
        if graph_manager.is_holm_correction_enabled():
            holm_significant = SupportingFunctions.holm_correction([p for _, p in collected])
        else:
            holm_significant = [False] * len(collected)

        # Собираем все аннотации перед их нанесением
        annotations = []
        for (time_point, p_value), is_holm_significant in zip(collected, holm_significant):
            if is_holm_significant:
                annotation = '*'  # Значимо после поправки Холма
            elif p_value < 0.05:
                annotation = '(*)' if graph_manager.is_holm_correction_enabled() else '*'
            else:
                annotation = ''

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
    @staticmethod
    def to_float_list(seq):
        out = []
        for x in seq:
            s = str(x).strip()
            if s == "" or s.lower() == "nan" or s.lower() == "none":
                out.append(float('nan'))
            else:
                # На случай десятичной запятой из Excel
                try:
                    out.append(float(s.replace(',', '.')))
                except Exception:
                    out.append(float('nan'))
        return out

    @staticmethod
    def calculate_pairwise_divergence(
            relative_volumes: List[List[float]],
            rat_labels: List[str]
    ) -> List[Tuple[str, str, List[float]]]:
        """
        Вычисляет попарное расхождение между нормированными кривыми роста крыс.

        Модернизированная формула (Кизилова, 2026):
            d(t) = 2 * |V₁(t) − V₂(t)| / (V₁(t) + V₂(t))

        Функционально эквивалентна точечному коэффициенту вариации для двух наблюдений.
        Позволяет оценить межособевую вариабельность в контрольной (необлучённой) группе.

        Args:
            relative_volumes (List[List[float]]): Нормированные объёмы V/V₀ для каждой крысы.
                Форма: (n_крыс, n_временных_точек).
            rat_labels (List[str]): Метки крыс (длина = n_крыс).

        Returns:
            List[Tuple[str, str, List[float]]]: Список кортежей (label_i, label_j, [d(t0), d(t1), ...])
                для каждой пары (i < j). NaN-значения в точке t заменяются на NaN.
        """
        from itertools import combinations
        result = []
        n_timepoints = len(relative_volumes[0]) if len(relative_volumes) > 0 else 0

        for (i, label_i), (j, label_j) in combinations(enumerate(rat_labels), 2):
            v1 = np.array(relative_volumes[i], dtype=float)
            v2 = np.array(relative_volumes[j], dtype=float)
            denom = v1 + v2
            # Избегаем деления на ноль: если сумма равна 0 или оба NaN → NaN
            with np.errstate(invalid='ignore', divide='ignore'):
                d = np.where(denom == 0, np.nan, 2.0 * np.abs(v1 - v2) / denom)
            result.append((label_i, label_j, d.tolist()))

        return result

    @staticmethod
    def calculate_cv(relative_volumes: List[List[float]]) -> Tuple[List[float], List[float]]:
        """
        Вычисляет коэффициент вариации CV(t) по группе крыс в каждой временной точке.

        Стандартная формула:
            CV(t) = σ(t) / μ(t) × 100%

        где σ(t) — стандартное отклонение, μ(t) — среднее нормированных объёмов группы в точке t.

        Args:
            relative_volumes (List[List[float]]): Нормированные объёмы V/V₀ для каждой крысы.
                Форма: (n_крыс, n_временных_точек).

        Returns:
            Tuple[List[float], List[float]]:
                - cv_values: CV(t) в % для каждой временной точки.
                - mean_values: среднее V/V₀ по группе в каждой точке (для справки).
        """
        data = np.array(relative_volumes, dtype=float)  # (n_крыс, n_точек)
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            cv = np.where(mean_vals == 0, np.nan, std_vals / mean_vals * 100.0)
        return cv.tolist(), mean_vals.tolist()
