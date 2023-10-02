import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

# Открытие PDF файла
doc = fitz.open("../../test.pdf")

# Итерация по страницам
for page_number in range(len(doc)):
    # Взятие страницы
    page = doc.load_page(page_number)

    # Преобразование страницы в изображение
    image_matrix = page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72))

    # Получение данных изображения в формате PIL, затем преобразование в numpy array
    image = Image.frombytes("RGB", [image_matrix.width, image_matrix.height], image_matrix.samples)
    img = np.array(image)

    # Поворот изображения на 90 градусов влево
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Преобразование изображения в оттенки серого
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения
    _, binary_img = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

    # # Создание структурирующего элемента для детектирования горизонтальных и вертикальных линий
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    #
    # # Применение морфологической операции для выделения горизонтальных и вертикальных линий
    # detected_horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # detected_vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    #
    # # Удаление линий из изображения
    # img_without_lines = cv2.subtract(cv2.subtract(binary_img, detected_horizontal_lines), detected_vertical_lines)

    # Находим области с текстом
    dilated_img = cv2.dilate(binary_img, None, iterations=2)

    # Края
    edges = cv2.Canny(dilated_img, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Рисование контуров
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Отсечение слишком маленьких и слишком больших контуров
        if w * h > 10 and w * h < 2000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение изображения
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
