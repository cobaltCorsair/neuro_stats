# файл plot_saver.py
import os
from matplotlib import pyplot as plt


def save_plot(file_path, plot_title, comparison_type=None, file_suffix=None, set_limits=True):
    """
    Сохраняет текущий график в файл PNG.

    Args:
        file_path (str): Путь к файлу данных, используемый для создания части имени файла.
        plot_title (str): Заголовок графика, используемый в качестве части имени сохраняемого файла.
        comparison_type (Optional[str]): Дополнительное описание сравнения для включения в имя файла.
        file_suffix (Optional[str]): Суффикс имени файла для дополнительного уточнения.
        set_limits (bool): Если True, устанавливает минимальные значения осей X и Y в 0.

    Пример:
        save_plot("data.xlsx", "График роста опухоли", "контроль_vs_эксперимент", "график_1")
    """
    # Извлечение базового имени файла из пути к файлу для использования в имени сохраняемого файла
    file_name_base = os.path.splitext(os.path.basename(file_path))[0]
    # Определение директории для сохранения графиков
    save_dir = 'C:\\dev\\neuro_stats\\work_with_prepared_data\\saved_graphics'

    # Комбинирование компонентов для формирования имени файла
    components = [file_name_base, plot_title, comparison_type, file_suffix]
    file_name = "_".join(filter(None, components)).replace(" ", "_") + ".png"

    # Установка ограничений для осей, если требуется
    if set_limits:
        plt.xlim(left=0)  # Установка минимального значения оси X
        plt.ylim(bottom=0)  # Установка минимального значения оси Y

    # Проверка наличия директории для сохранения и её создание при необходимости
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Формирование полного пути к файлу и сохранение графика
    #full_file_path = os.path.join(save_dir, file_name)
    #plt.savefig(full_file_path, format="png", dpi=300)
    #print(f"График сохранён как {full_file_path}")