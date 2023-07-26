import os
from vision_recognition import image_to_text

# Путь к директории с изображениями
directory_path = '../neuro_stats/output'

# Итерация по файлам в директории
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
        file_path = os.path.join(directory_path, filename)
        results = image_to_text(file_path)
        print(results)
