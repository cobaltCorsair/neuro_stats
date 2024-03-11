# файл plot_saver.py
import os
from matplotlib import pyplot as plt


def save_plot(file_path, plot_title, comparison_type=None, file_suffix=None, set_limits=True):
    """
    Сохраняет текущий график в файл PNG с учетом различных параметров.

    Parameters:
        file_path (str): Путь к файлу данных, используемый для создания части имени файла.
        plot_title (str): Основа для имени файла, обычно название графика.
        comparison_type (str, optional): Тип сравнения, если применимо, для дополнения имени файла.
        file_suffix (str, optional): Суффикс для имени файла для уточнения типа графика.
        set_limits (bool, optional): Флаг для установки минимальных значений осей X и Y равными 0.
    """
    # Базовое имя файла из пути к файлу данных
    file_name_base = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = r'C:\dev\neuro_stats\work_with_prepared_data\saved_graphics'

    # Составление имени файла из предоставленных компонентов
    components = [component for component in [file_name_base, plot_title, comparison_type, file_suffix] if component]
    file_name = f"{'_'.join(components).replace(' ', '_')}.png"

    if set_limits:
        plt.xlim(left=0)  # Установка минимального значения для оси X равным 0
        plt.ylim(bottom=0)  # Установка минимального значения для оси Y равным 0

    # Создаём путь для сохранения, если он не существует
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    full_file_path = os.path.join(save_dir, file_name)  # Полный путь к файлу
    plt.savefig(full_file_path, format='png', dpi=300)  # Сохранение файла по полному пути
    print(f"Plot saved as {full_file_path}")
