import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from work_with_prepared_data.draw_base_grapfs import TumorDataVisualizer


class ControlGroupVisualizer(TumorDataVisualizer):
    def process_excel(self) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
        """
        Обрабатывает данные из файла Excel и извлекает необходимые данные.
        Особенно подходит для контрольных групп, где только один параметр в первой строке данных.

        Возвращает:
            tuple: Кортеж, содержащий:
                - experiment_params (List[str]): Параметры эксперимента.
                - time_data (List[str]): Метки времени для каждого измерения.
                - rat_labels (List[str]): Метки крыс.
                - tumor_volumes (List[List[float]]): Объемы опухолей для каждой крысы на каждом временном интервале.
        """
        data = pd.read_excel(self.file_path, header=None)
        # Учитываем, что у нас всего один параметр в первой строке
        experiment_params = [data.iloc[0, 0]]
        tumor_data = data.iloc[2:, :].copy()
        time_data = [str(int(item.split(' ')[0].replace('V', '0'))) for item in data.iloc[1, 1:]]

        tumor_data = tumor_data.applymap(
            lambda x: str(x).strip().replace(',', '.').replace(' -', '-') if pd.notna(x) else "NA")
        rat_labels = tumor_data.iloc[:, 0].tolist()

        tumor_volumes = []
        for _, row in tumor_data.iterrows():
            rat_volumes = []
            for item in row[1:]:
                if "-" in item:
                    a, b, c = map(float, item.split("-"))
                    volume = (np.pi * a * b * c) / 6
                elif item.replace(".", "").isdigit():
                    volume = float(item)
                else:
                    volume = np.nan
                rat_volumes.append(volume)
            tumor_volumes.append(rat_volumes)

        return experiment_params, time_data, rat_labels, tumor_volumes
