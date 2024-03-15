# файл controls.py

from typing import List, Tuple
from draw_base_graphs import TumorDataVisualizer
from data_processing.excel_data_processor import process_tumor_data_excel


class ControlGroupVisualizer(TumorDataVisualizer):
    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
        """
        Обрабатывает данные эксперимента с контрольной группой из файла Excel.

        Этот метод расширяет базовый класс `TumorDataVisualizer`, предоставляя специализированную обработку
        для данных контрольной группы. Он извлекает временные метки, метки крыс, объемы опухолей и другие
        параметры эксперимента из файла Excel. Обработка файла осуществляется с помощью функции
        `process_tumor_data_excel`, которая должна быть определена вне этого класса.

        Returns:
            Tuple[List[str], List[str], List[str], List[List[float]]]: Кортеж, содержащий списки параметров
            эксперимента, временных меток, меток крыс и объемов опухолей соответственно.

        Пример использования:
            visualizer = ControlGroupVisualizer("путь/к/файлу.xlsx")
            experiment_params, time_data, rat_labels, tumor_volumes = visualizer.process_excel()
        """
        # Используем функцию process_tumor_data_excel, передавая путь к файлу
        return process_tumor_data_excel(self.file_path)
