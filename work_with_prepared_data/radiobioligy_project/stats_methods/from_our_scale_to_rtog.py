# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
import numpy as np
import os

# Установка современного стиля seaborn
palette = 'Spectral'
sns.set_theme(style="ticks", palette=palette)

# Настройка параметров matplotlib для более современного вида
plt.rcParams.update({
    'figure.figsize': (16, 8),
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
})

# Данные для коррекции по дням
day_corrections = {
    2: 95, 3: 93, 4: 90, 5: 87, 6: 85, 7: 82, 8: 80, 9: 78,
    10: 75, 11: 73, 12: 70, 13: 68, 14: 65, 15: 63, 16: 60,
    17: 54, 18: 55, 19: 53, 20: 50, 21: 48, 22: 45, 23: 43,
    24: 40, 25: 37, 26: 35
}


# Функция для извлечения номера дня из названия столбца
def extract_day_number(day_label: str) -> int:
    """Извлекает номер дня из названия столбца."""
    # Предполагается, что день является первым числом в названии столбца
    # Например, '2 сут. - 21.06' -> 2
    try:
        return int(day_label.split()[0])
    except (ValueError, IndexError):
        # Если не удается преобразовать, вернуть 0 или другое значение по умолчанию
        return 0


# Класс для анализа кожных реакций
class SkinReactionAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.apply(pd.to_numeric, errors='coerce')
        self.days = self.data.columns
        self.num_days = range(len(self.days))
        self.day_labels = [day for day in self.days]  # Используем исходные названия столбцов

    @staticmethod
    def map_to_rtog(score: Union[int, float]) -> int:
        """Перевод значения из старой шкалы в RTOG."""
        if score <= 0:
            return 0  # No reaction
        elif 0 < score <= 100:
            return 1  # Mild reaction (RTOG Grade 1)
        elif 101 <= score <= 200:
            return 1  # Mild to moderate reaction (still fits RTOG Grade 1)
        elif 201 <= score <= 400:
            return 2  # Moderate reaction (fits RTOG Grade 2)
        elif 401 <= score <= 600:
            return 3  # Severe reaction (fits RTOG Grade 3)
        elif score > 600:
            return 4  # Necrosis or Ulceration (fits RTOG Grade 4)
        return 0

    def map_data_to_rtog(self) -> pd.DataFrame:
        """Переводит все данные на шкалу RTOG."""
        return self.data.applymap(self.map_to_rtog)

    def _plot(self, title: str, ylabel: str, legend: List[str],
              values: Union[pd.DataFrame, pd.Series],
              palette: str = palette,
              fill_between: bool = False,
              fill_lower: pd.Series = None,
              fill_upper: pd.Series = None):
        """Вспомогательный метод для построения графиков."""
        plt.figure(figsize=(16, 8))
        if isinstance(values, pd.DataFrame):
            colors = sns.color_palette(palette, n_colors=len(legend))
            for label, color in zip(legend, colors):
                plt.plot(self.num_days, values.loc[label], marker='o', linestyle='-', label=label, color=color)
                if fill_between and fill_lower is not None and fill_upper is not None:
                    plt.fill_between(self.num_days, fill_lower.loc[label], fill_upper.loc[label],
                                     color=color, alpha=0.2)
        elif isinstance(values, pd.Series):
            color = sns.color_palette(palette, 1)[0]
            plt.plot(self.num_days, values, marker='o', linestyle='-', label=legend[0], color=color)
            if fill_between and fill_lower is not None and fill_upper is not None:
                plt.fill_between(self.num_days, fill_lower, fill_upper, color=color, alpha=0.2)
        else:
            raise ValueError("Unsupported type for values in _plot")

        # Настройка оси X с метками из названий столбцов
        plt.xticks(self.num_days, self.day_labels, rotation=45)

        # Современные настройки графика
        plt.title(title, fontsize=20, weight='bold', pad=20)
        plt.xlabel("Дни наблюдения", fontsize=16, labelpad=15)
        plt.ylabel(ylabel, fontsize=16, labelpad=15)
        plt.legend(title="Группа", frameon=False, fontsize=14, title_fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.show()


class GroupAnalyzer(SkinReactionAnalyzer):
    def calculate_group_mean(self) -> pd.Series:
        """Вычисляет среднее значение по группе."""
        return self.data.mean()

    def calculate_group_std(self) -> pd.Series:
        """Вычисляет стандартное отклонение по группе."""
        return self.data.std()

    def analyze_group(self):
        """Выполняет анализ группы по старой шкале и по шкале RTOG."""
        # Расчет среднего значения и стандартного отклонения для каждого дня
        group_mean = self.calculate_group_mean()
        group_std = self.calculate_group_std()

        # Перевод среднего значения в шкалу RTOG
        group_mean_rtog = group_mean.apply(self.map_to_rtog)
        group_std_rtog = group_std.apply(self.map_to_rtog)

        # Определение границ для заливки (например, mean +/- std)
        group_lower = group_mean - group_std
        group_upper = group_mean + group_std

        group_lower_rtog = group_mean_rtog - group_std_rtog
        group_upper_rtog = group_mean_rtog + group_std_rtog

        # Построение графика по старой шкале с заливкой
        self._plot_with_fill_between(
            title="Динамика кожных реакций по группе (старая шкала)",
            ylabel="Средний балл по шкале повреждений",
            legend=["Старая шкала"],
            mean_values=group_mean,
            fill_lower=group_lower,
            fill_upper=group_upper,
            color=sns.color_palette(palette)[0]
        )

        # Построение графика по шкале RTOG с заливкой
        self._plot_with_fill_between(
            title="Динамика кожных реакций по группе (шкала RTOG)",
            ylabel="Средняя степень кожной реакции (RTOG)",
            legend=["Шкала RTOG"],
            mean_values=group_mean_rtog,
            fill_lower=group_lower_rtog,
            fill_upper=group_upper_rtog,
            color=sns.color_palette(palette)[1]
        )

    def _plot_with_fill_between(self, title: str, ylabel: str, legend: List[str],
                                mean_values: pd.Series, fill_lower: pd.Series,
                                fill_upper: pd.Series, color: str):
        """Метод для построения графиков с заливкой между нижней и верхней границей."""
        plt.figure(figsize=(16, 8))
        plt.plot(self.num_days, mean_values, marker='o', linestyle='-', label=legend[0], color=color)
        plt.fill_between(self.num_days, fill_lower, fill_upper, color=color, alpha=0.2)

        # Настройка оси X с метками из названий столбцов
        plt.xticks(self.num_days, self.day_labels, rotation=45)

        # Современные настройки графика
        plt.title(title, fontsize=20, weight='bold', pad=20)
        plt.xlabel("Дни наблюдения", fontsize=16, labelpad=15)
        plt.ylabel(ylabel, fontsize=16, labelpad=15)
        plt.legend(title="Шкала", frameon=False, fontsize=14, title_fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.show()


class IndividualAnalyzer(SkinReactionAnalyzer):
    def analyze_individuals(self):
        """Выполняет анализ по каждому животному отдельно в старой шкале и шкале RTOG."""
        # Графики на основе старой шкалы для каждого животного
        self._plot(
            title="Динамика кожных реакций для каждого животного (старая шкала)",
            ylabel="Баллы по шкале повреждений",
            legend=self.data.index.tolist(),
            values=self.data,
            palette=palette,
            fill_between=False  # Заливка не требуется для индивидуальных данных
        )

        # Перевод данных для каждого животного на шкалу RTOG
        data_rtog = self.map_data_to_rtog()
        self._plot(
            title="Динамика кожных реакций для каждого животного (шкала RTOG)",
            ylabel="Степень кожной реакции (RTOG)",
            legend=data_rtog.index.tolist(),
            values=data_rtog,
            palette=palette,
            fill_between=False  # Заливка не требуется для индивидуальных данных
        )


# Новый класс для анализа откорректированных данных
class CorrectedSkinReactionAnalyzer(SkinReactionAnalyzer):
    def __init__(self, data: pd.DataFrame):
        self.original_data = data.apply(pd.to_numeric, errors='coerce')
        self.days = self.original_data.columns
        self.num_days = range(len(self.days))
        self.day_labels = [day for day in self.days]  # Используем исходные названия столбцов
        self.data = self.apply_day_correction()

    def apply_day_correction(self) -> pd.DataFrame:
        """Применяет коррекцию значений по дням на основе таблицы коррекций."""
        corrected_data = self.original_data.copy()
        for day in self.days:
            day_int = extract_day_number(day)
            if day_int in day_corrections:
                correction_value = day_corrections[day_int]
                corrected_data[day] = self.original_data[day] - correction_value
        return corrected_data


# Класс для группового анализа откорректированных данных
class CorrectedGroupAnalyzer(GroupAnalyzer):
    def __init__(self, data: pd.DataFrame):
        corrected_analyzer = CorrectedSkinReactionAnalyzer(data)
        self.data = corrected_analyzer.data
        self.days = corrected_analyzer.days
        self.num_days = corrected_analyzer.num_days
        self.day_labels = corrected_analyzer.day_labels

    def analyze_group(self):
        """Анализирует группу по откорректированным данным."""
        # Расчёт среднего значения и стандартного отклонения для каждого дня
        group_mean = self.calculate_group_mean()
        group_std = self.calculate_group_std()

        # Перевод среднего значения в шкалу RTOG
        group_mean_rtog = group_mean.apply(self.map_to_rtog)
        group_std_rtog = group_std.apply(self.map_to_rtog)

        # Определение границ для заливки (например, mean +/- std)
        group_lower = group_mean - group_std
        group_upper = group_mean + group_std

        group_lower_rtog = group_mean_rtog - group_std_rtog
        group_upper_rtog = group_mean_rtog + group_std_rtog

        # Построение графика по старой шкале с заливкой
        self._plot_with_fill_between(
            title="Динамика кожных реакций по группе (откорректированные данные, старая шкала)",
            ylabel="Средний балл по шкале повреждений",
            legend=["Старая шкала (откорректированные)"],
            mean_values=group_mean,
            fill_lower=group_lower,
            fill_upper=group_upper,
            color=sns.color_palette(palette)[0]
        )

        # Построение графика по шкале RTOG с заливкой
        self._plot_with_fill_between(
            title="Динамика кожных реакций по группе (откорректированные данные, шкала RTOG)",
            ylabel="Средняя степень кожной реакции (RTOG)",
            legend=["Шкала RTOG (откорректированные)"],
            mean_values=group_mean_rtog,
            fill_lower=group_lower_rtog,
            fill_upper=group_upper_rtog,
            color=sns.color_palette(palette)[1]
        )


# Класс для индивидуального анализа откорректированных данных
class CorrectedIndividualAnalyzer(IndividualAnalyzer):
    def __init__(self, data: pd.DataFrame):
        corrected_analyzer = CorrectedSkinReactionAnalyzer(data)
        self.data = corrected_analyzer.data
        self.days = corrected_analyzer.days
        self.num_days = corrected_analyzer.num_days
        self.day_labels = corrected_analyzer.day_labels

    def analyze_individuals(self):
        """Анализирует каждого животного по откорректированным данным."""
        # Графики на основе старой шкалы для каждого животного
        self._plot(
            title="Динамика кожных реакций для каждого животного (откорректированные данные, старая шкала)",
            ylabel="Баллы по шкале повреждений",
            legend=self.data.index.tolist(),
            values=self.data,
            palette=palette,
            fill_between=False  # Заливка не требуется для индивидуальных данных
        )

        # Перевод данных для каждого животного на шкалу RTOG
        data_rtog = self.map_data_to_rtog()
        self._plot(
            title="Динамика кожных реакций для каждого животного (откорректированные данные, шкала RTOG)",
            ylabel="Степень кожной реакции (RTOG)",
            legend=data_rtog.index.tolist(),
            values=data_rtog,
            palette=palette,
            fill_between=False  # Заливка не требуется для индивидуальных данных
        )


# Класс для комбинированного анализа
class CombinedAnalyzer(SkinReactionAnalyzer):
    def __init__(self, raw_data: pd.DataFrame, corrected_data: pd.DataFrame):
        self.raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
        self.corrected_data = corrected_data.apply(pd.to_numeric, errors='coerce')
        self.days = self.raw_data.columns
        self.num_days = range(len(self.days))
        self.day_labels = [day for day in self.days]  # Используем исходные названия столбцов

    def plot_combined_scale(self):
        """Строит график с двумя осями Y для старой шкалы без коррекции и RTOG с коррекцией, включая доверительные интервалы."""
        # Расчет среднего значения и стандартного отклонения для необработанных данных (старая шкала)
        group_mean_raw = self.raw_data.mean()
        group_std_raw = self.raw_data.std()
        n_raw = self.raw_data.count()
        se_raw = group_std_raw / np.sqrt(n_raw)
        ci_raw = 1.96 * se_raw  # 95% доверительный интервал

        # Расчет среднего значения и стандартного отклонения для откорректированных данных
        group_mean_corrected = self.corrected_data.mean()
        group_std_corrected = self.corrected_data.std()
        n_corrected = self.corrected_data.count()
        se_corrected = group_std_corrected / np.sqrt(n_corrected)
        ci_corrected = 1.96 * se_corrected  # 95% доверительный интервал

        # Перевод среднего значения откорректированных данных в шкалу RTOG
        group_mean_rtog = group_mean_corrected.apply(self.map_to_rtog)
        # Примечание: Доверительные интервалы для RTOG рассчитаны на основе откорректированных данных,
        # что может быть не совсем корректно из-за преобразования шкалы. Это приближение.
        ci_rtog = ci_corrected.apply(self.map_to_rtog)

        # Создание фигуры и осей
        fig, ax1 = plt.subplots(figsize=(16, 8))

        palette = sns.color_palette("tab10")  # Убедитесь, что палитра определена
        color1 = palette[0]  # Цвет для старой шкалы
        color2 = palette[1]  # Цвет для RTOG

        # Плот для старой шкалы без коррекции
        ax1.set_xlabel('Дни наблюдения', fontsize=16, labelpad=15)
        ax1.set_ylabel('Средний балл по шкале повреждений (старая шкала)', color=color1, fontsize=16, labelpad=15)
        ax1.plot(self.num_days, group_mean_raw, marker='o', linestyle='-', color=color1, label='Старая шкала без коррекции')
        ax1.fill_between(self.num_days, group_mean_raw - ci_raw, group_mean_raw + ci_raw, color=color1, alpha=0.2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(self.num_days)
        ax1.set_xticklabels(self.day_labels, rotation=45)
        ax1.set_ylim(0, 650)  # Установка лимитов для левой оси

        # Создание второго Y-axes для RTOG
        ax2 = ax1.twinx()

        # Плот для RTOG с коррекцией
        ax2.set_ylabel('Средняя степень кожной реакции (RTOG)', color=color2, fontsize=16, labelpad=15)
        ax2.plot(self.num_days, group_mean_rtog, marker='o', linestyle='-', color=color2, label='RTOG с коррекцией')
        ax2.fill_between(self.num_days, group_mean_rtog - ci_rtog, group_mean_rtog + ci_rtog, color=color2, alpha=0.2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 4)  # Установка лимитов для правой оси

        # Заголовок графика
        plt.title('Сравнение шкал: старая шкала без коррекции и RTOG с коррекцией', fontsize=20, weight='bold', pad=20)

        # Объединение легенд для обоих осей
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=14)

        # Настройка сетки и оформления
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.show()


def main(file_path: str, save_plots: bool = False, save_dir: str = "plots"):
    # Загрузка данных
    data = pd.read_excel(file_path)
    data.columns = data.iloc[0]  # Установка имен колонок из первой строки
    data = data.drop(0).reset_index(drop=True).set_index("Метка")

    # Создание экземпляров анализаторов
    group_analyzer = GroupAnalyzer(data)
    individual_analyzer = IndividualAnalyzer(data)

    # Анализ необработанных данных
    group_analyzer.analyze_group()
    individual_analyzer.analyze_individuals()

    # Создание экземпляров анализаторов для откорректированных данных
    corrected_analyzer = CorrectedSkinReactionAnalyzer(data)
    corrected_group_analyzer = CorrectedGroupAnalyzer(data)
    corrected_individual_analyzer = CorrectedIndividualAnalyzer(data)

    # Анализ откорректированных данных
    corrected_group_analyzer.analyze_group()
    corrected_individual_analyzer.analyze_individuals()

    # Создание экземпляра комбинированного анализатора и построение комбинированного графика
    combined_analyzer = CombinedAnalyzer(data, corrected_analyzer.data)
    combined_analyzer.plot_combined_scale()


if __name__ == "__main__":
    # Указание пути к файлу данных
    #file_path = r"V:\Kizilova\Крысы сканы\Нейтроны + протоны (2024 год)\skin_reactions_p_36_in_peak_19.06.2024.xlsx"
    file_path = r"V:\Kizilova\Крысы сканы\Нейтроны + протоны (2024 год)\consolidate\skin_reactions_n_3.6_n_3.6_p.25.2_all_consolidate.xlsx"

    main(file_path, save_plots=False)
