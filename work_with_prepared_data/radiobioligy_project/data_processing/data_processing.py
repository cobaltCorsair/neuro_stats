import numpy as np


class TumorDataProcessor:
    def __init__(self, tumor_volumes=None):
        self.tumor_volumes = tumor_volumes

    def get_mean_tumor_volumes(self, volumes=None) -> np.ndarray:
        """
        Вычисляет средний объем опухоли для всех крыс на каждом временном интервале.
        Поддерживает внешние данные о объемах опухоли.

        Параметры:
            volumes (np.ndarray, optional): Внешние данные объемов опухоли. Если не указан, используется self.tumor_volumes.

        Возвращает:
            np.ndarray: Массив средних объемов опухоли.
        """
        if volumes is None:
            volumes = self.tumor_volumes
        return np.nanmean(volumes, axis=0)

    def get_relative_tumor_volumes(self) -> np.ndarray:
        """
        Вычисляет относительные объемы опухолей для каждой крысы.

        Возвращает:
            np.ndarray: Массив относительных объемов опухолей.
        """
        return np.array([[vol / volumes[0] for vol in volumes] for volumes in self.tumor_volumes])

    def get_mean_relative_tumor_volumes(self) -> np.ndarray:
        """
        Вычисляет средний относительный усреднённый объем опухоли для всех крыс.

        Возвращает:
            np.ndarray: Массив средних относительных объемов опухоли.
        """
        # Получение средних объемов опухоли
        mean_volumes = self.get_mean_tumor_volumes()

        # Вычисление среднего относительного объема опухоли
        mean_rel_volumes = mean_volumes / mean_volumes[0]

        return mean_rel_volumes
