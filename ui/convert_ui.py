import subprocess


def convert_ui_to_py(ui_file):
    # Путь к интерпретатору Python, используйте абсолютный путь к нужной версии Python
    python_interpreter = "C:\\Program Files\\Python310\\python.exe"

    # Определение имени выходного файла, замена расширения .ui на .py
    py_file = ui_file.replace('.ui', '.py')

    # Вызов pyuic6 для конвертации, используя указанный интерпретатор Python
    try:
        subprocess.run([python_interpreter, "-m", "PyQt6.uic.pyuic", "-x", ui_file, "-o", py_file], check=True)
        print(f"Файл {ui_file} успешно конвертирован в {py_file}")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при конвертации файла: {e}")


# Пример использования
convert_ui_to_py(r'gui.ui')
