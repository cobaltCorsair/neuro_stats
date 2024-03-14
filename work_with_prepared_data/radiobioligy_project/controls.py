# файл controls.py

from typing import List, Tuple
from draw_base_graphs import TumorDataVisualizer
from data_processing.excel_data_processor import process_tumor_data_excel


class ControlGroupVisualizer(TumorDataVisualizer):
    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
        """
        Обрабатывает данные из файла Excel и извлекает необходимые данные,
        используя функцию process_tumor_data_excel из другого файла.
        """
        # Используем функцию process_tumor_data_excel, передавая путь к файлу
        return process_tumor_data_excel(self.file_path)
