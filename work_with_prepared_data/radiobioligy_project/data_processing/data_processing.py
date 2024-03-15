import numpy as np

from work_with_prepared_data.radiobioligy_project.stats_methods.support_stats_methods import SupportingFunctions


class TumorDataProcessor:
    """
       Обработчик данных об объемах опухолей, предоставляющий методы для расчета средних и относительных объемов опухолей.
       """
    def __init__(self, tumor_volumes=None):
        """
        Инициализация обработчика данных об объемах опухолей.

        Args:
            tumor_volumes (Optional[np.ndarray], optional): Данные об объемах опухолей. Defaults to None.
        """
        self.tumor_volumes = tumor_volumes

    def get_mean_tumor_volumes(self, volumes=None) -> np.ndarray:
        """
         Вычисляет средний объем опухоли для всех крыс на каждом временном интервале.

         Args:
             volumes (Optional[np.ndarray], optional): Массив объемов опухолей для использования вместо self.tumor_volumes.
             Defaults to None.

         Returns:
             np.ndarray: Массив средних объемов опухоли на каждом временном интервале.
         """
        if volumes is None:
            volumes = self.tumor_volumes
        return np.nanmean(volumes, axis=0)

    def get_relative_tumor_volumes(self) -> np.ndarray:
        """
        Вычисляет средний относительный объем опухолей для всех крыс.

        Returns:
            np.ndarray: Массив средних относительных объемов опухолей на каждом временном интервале.
        """
        return np.array([[vol / volumes[0] for vol in volumes] for volumes in self.tumor_volumes])

    def get_mean_relative_tumor_volumes(self) -> np.ndarray:
        """
        Вычисляет средний относительный усреднённый объем опухоли для всех крыс.

        Returns:
            np.ndarray: Массив средних относительных объемов опухоли.
        """
        # Получение средних объемов опухоли
        mean_volumes = self.get_mean_tumor_volumes()

        # Вычисление среднего относительного объема опухоли
        mean_rel_volumes = mean_volumes / mean_volumes[0]

        return mean_rel_volumes


class SkinReactionsDataProcessor:
    """
    Обработчик данных о кожных реакциях, предоставляющий методы для расчета средних реакций и их статистических показателей.
    """
    def __init__(self, skin_reactions=None):
        """
        Инициализация обработчика данных о кожных реакциях.

        Args:
            skin_reactions (Optional[np.ndarray], optional): Данные о кожных реакциях. Defaults to None.
        """
        self.skin_reactions = skin_reactions

    def get_mean_skin_reactions(self):
        """
        Вычисляет средние значения кожных реакций и их статистические характеристики.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Кортеж, содержащий средние значения реакций,
            стандартные отклонения и доверительные интервалы.
        """
        mean_reactions = np.nanmean(self.skin_reactions, axis=0)
        std_dev = [SupportingFunctions.calculate_std_dev(values, mean_value) for values, mean_value in
                   zip(np.transpose(self.skin_reactions), mean_reactions)]
        error_margin = [SupportingFunctions.calculate_error_margin(std, len(self.skin_reactions)) for std in std_dev]
        return mean_reactions, np.array(std_dev), np.array(error_margin)
