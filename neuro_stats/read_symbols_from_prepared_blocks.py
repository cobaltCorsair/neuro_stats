import cv2
import numpy as np
import os


# Функция для вырезки и изменения размера символов до 28x28
def extract_and_resize_to_28x28(file_path):
    # Читаем изображение в оттенках серого
    img = cv2.imread(file_path, 0)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Бинаризация изображения с использованием адаптивного порога
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Находим контуры
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Перебираем все контуры
    for i, cnt in enumerate(contours):
        # Получаем прямоугольник, ограничивающий контур
        x, y, w, h = cv2.boundingRect(cnt)

        # Если контур достаточно большой, чтобы быть символом
        if w * h > 50:
            # Рисуем контур на исходном изображении
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Вырезаем символ из исходного изображения
            digit = img[y:y + h, x:x + w]

            digit = cv2.bitwise_not(digit)

            # Если изображение символа больше 28x28, уменьшаем его
            if max(digit.shape) > 28:
                digit = cv2.resize(digit, (28, 28))

            # Увеличение резкости
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            digit = cv2.filter2D(digit, -1, kernel)

            # Создаем новое черное изображение размером 28x28
            new_img = np.zeros((28, 28), dtype=np.uint8)

            # Вычисляем координаты для центрирования изображения символа
            x_offset = (28 - digit.shape[1]) // 2
            y_offset = (28 - digit.shape[0]) // 2

            # Копируем изображение символа в центр нового изображения
            new_img[y_offset:y_offset + digit.shape[0], x_offset:x_offset + digit.shape[1]] = digit
            # Сохраняем символ в файл
            output_file = f"symbol_{i}.png"
            # cv2.imwrite(output_file, new_img)

    # Сохраняем исходное изображение с контурами
    cv2.imwrite(f"contours/contours_{os.path.basename(file_path)}", img_color)


# Путь к директории с изображениями
directory_path = 'output'

# Итерация по файлам в директории
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
        file_path = os.path.join(directory_path, filename)
        extract_and_resize_to_28x28(file_path)
