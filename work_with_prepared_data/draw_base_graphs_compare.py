import os
import matplotlib.pyplot as plt
from draw_base_grapfs import TumorDataVisualizer


class TumorDataComparator:
    def __init__(self, *visualizers):
        """
        Инициализатор класса сравнителя данных о опухолях.

        Parameters:
            visualizers (list of TumorDataVisualizer): Список визуализаторов данных экспериментов.
        """
        self.visualizers = visualizers

    def save_plot(self, comparison_type: str):
        """
        Сохраняет текущий график в файл PNG.

        Parameters:
            comparison_type (str): Тип сравнения, используется в имени файла.
        """
        file_names = [os.path.splitext(os.path.basename(visualizer.file_path))[0] for visualizer in self.visualizers]
        file_name = f"{comparison_type}_{'_vs_'.join(file_names)}.png"
        plt.savefig(file_name, format='png', dpi=300)
        print(f"Plot saved as {file_name}")

    def normalize_time_data(self):
        """
        Нормализует временные метки всех экспериментов, приводя их к числовому формату и вычитая начальное время.
        """
        for visualizer in self.visualizers:
            visualizer.time_data = [int(time) - int(visualizer.time_data[0]) for time in visualizer.time_data]

    def format_experiment_params(self, params):
        """
        Форматирует параметры эксперимента для отображения в легенде.

        Parameters:
            params (list): Список параметров эксперимента.

        Returns:
            str: Отформатированная строка параметров эксперимента.
        """
        # Удаляем пустые строки и значения 'nan'
        cleaned_params = [str(param).replace('nan', '').strip() for param in params if str(param).strip()]
        # Преобразуем список в строку, исключая квадратные скобки
        return ', '.join(cleaned_params)

    def compare_tumor_volumes(self):
        """
        Сравнивает абсолютные объемы опухолей между экспериментами и строит соответствующий график.
        """
        # Нормализовать временные метки
        self.normalize_time_data()

        plt.figure(figsize=(15, 8))
        linestyles = ['-', '--', '-.', ':']
        for visualizer, linestyle in zip(self.visualizers, linestyles[:len(self.visualizers)]):
            formatted_params = self.format_experiment_params(visualizer.experiment_params)
            for label, volumes in zip(visualizer.rat_labels, visualizer.tumor_volumes):
                plt.plot(visualizer.time_data, volumes, marker='o', linestyle=linestyle, label=f"{formatted_params}: {label}")

        plt.title("Сравнение экспериментов")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()
        self.save_plot("compare_tumor_volumes")
        plt.show()

    def compare_relative_tumor_volumes(self):
        """
        Сравнивает относительные объемы опухолей между экспериментами и строит соответствующий график.
        """
        # Нормализовать временные метки
        self.normalize_time_data()

        plt.figure(figsize=(15, 8))
        linestyles = ['-', '--', '-.', ':']
        for visualizer, linestyle in zip(self.visualizers, linestyles[:len(self.visualizers)]):
            formatted_params = self.format_experiment_params(visualizer.experiment_params)
            for label, volumes in zip(visualizer.rat_labels, visualizer.get_relative_tumor_volumes()):
                plt.plot(visualizer.time_data, volumes, marker='o', linestyle=linestyle, label=f"{formatted_params}: {label}")


        plt.title("Сравнение относительных объемов опухолей")
        plt.xticks(rotation=45)
        plt.xlabel("Время (дни)")
        plt.ylabel("Относительный объем опухоли")
        plt.grid(True)
        plt.legend(title="Метка крысы")
        plt.tight_layout()

        # Сохранение графика
        self.save_plot("compare_relative_tumor_volumes")
        plt.show()

# Используем с файлом данных
# file_path1 = './datas/n_7.2_p_25.2_2023_2.xlsx'
# file_path2 = './datas/p_25.2_n_7.2_2023_2.xlsx'

# Используем с файлом данных
# file_path1 = './datas/n_7.2_p_25.2_2023.xlsx'
# file_path2 = './datas/p_25.2_n_7.2_2023.xlsx'

# Используем с файлом данных
# file_path1 = './datas/n_2.56_p_25.6_2019.xlsx'
# file_path2 = './datas/p_25.6_n_2.56_2019.xlsx'

# Используем с файлом данных
file_path1 = './datas/control/02.02.2023_n_12.xlsx'
file_path2 = './datas/control/02.02.2023_n_18.xlsx'
file_path3 = './datas/control/16.03.2023_n_22.xlsx'

# Создаем объекты визуализатора для каждого файла данных
visualizers = [TumorDataVisualizer(file_path) for file_path in [file_path1, file_path2, file_path3]]
# visualizers = [TumorDataVisualizer(file_path) for file_path in [file_path1, file_path2]]
# Создаем объект сравнителя и сравниваем данные из всех экспериментов
comparator = TumorDataComparator(*visualizers)
comparator.compare_tumor_volumes()  # Сравниваем абсолютные объемы
comparator.compare_relative_tumor_volumes()  # Сравниваем относительные объемы
