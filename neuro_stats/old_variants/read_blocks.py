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
    image_matrix = page.get_pixmap()

    # Получение данных изображения в формате PIL, затем преобразование в numpy array
    image = Image.frombytes("RGB", [image_matrix.width, image_matrix.height], image_matrix.samples)
    img = np.array(image)

    # Поворот изображения на 90 градусов влево
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Преобразование изображения в оттенки серого
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения
    _, binary_img = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Применение преобразования Хафа
    lines = cv2.HoughLines(binary_img, 1, np.pi / 180, 220)

    # Рисование линий
    tolerance = np.pi / 36  # 5 градусов в радианах
    rho_tolerance = 15  # настраиваемое значение
    drawn_lines = []
    for rho, theta in lines[:, 0]:
        # проверяем, есть ли линия, похожая на те, которые уже нарисованы
        if any(abs(rho - r) <= rho_tolerance and abs(theta - t) <= tolerance for r, t in drawn_lines):
            continue

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # добавляем линию в список нарисованных линий
        drawn_lines.append((rho, theta))

    # Отображение изображения
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
