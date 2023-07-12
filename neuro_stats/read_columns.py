import cv2
import numpy as np

# Загрузка изображения
img = cv2.imread('../test.png', 0)

# Бинаризация изображения
_, binary_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

# Создание структурирующего элемента для детектирования горизонтальных линий
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))

# Применение морфологической операции для выделения горизонтальных линий
detected_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Удаление линий из изображения
img_without_lines = cv2.subtract(binary_img, detected_lines)

# Находим области с текстом
dilated_img = cv2.dilate(img_without_lines, None, iterations=2)
# Края
edges = cv2.Canny(dilated_img, 50, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Рисование контуров
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Отсечение слишком маленьких и слишком больших контуров
    if w * h > 100 and w * h < 1200: # измените числа в соответствии с вашими потребностями
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Отображение изображения
cv2.imshow('image', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
