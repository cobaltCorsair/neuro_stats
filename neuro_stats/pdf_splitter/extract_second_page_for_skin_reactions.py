import fitz
from PIL import Image


def save_page_as_image(pdf_path, page_number, image_path, zoom_x, zoom_y):
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # Установка матрицы преобразования для увеличения DPI изображения
    # zoom_x и zoom_y это коэффициенты масштабирования для осей x и y соответственно.
    mat = fitz.Matrix(zoom_x, zoom_y)

    # Получение изображения страницы с учетом масштабирования
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Поворот изображения на 90 градусов против часовой стрелки
    img_rotated = img.rotate(90, expand=True)

    img_rotated.save(image_path)


# Пример использования функции:
pdf_path = "rats_data/25.10.2023_n7,2_p25,2.pdf"
image_path = "n_p_only_images/" + f'{pdf_path.split("/")[0]}' + "_skin_image.jpg"
zoom_x = 2  # Увеличение DPI в 2 раза по оси X
zoom_y = 2  # Увеличение DPI в 2 раза по оси Y

try:
    save_page_as_image(pdf_path, 1, image_path, zoom_x, zoom_y)  # Страницы отсчитываются с 0
    print(f"Image saved at {image_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
