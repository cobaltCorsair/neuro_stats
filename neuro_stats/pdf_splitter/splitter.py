import os
import fitz
import numpy as np
from PIL import Image
from PyPDF2 import PdfWriter, PdfReader
import io


def pdf_to_image(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(alpha=False)
    data = pix.samples
    image = Image.frombytes("RGB", [pix.width, pix.height], data)
    return image


def is_white_page(image):
    grayscale = image.convert('L')
    numpy_array = np.array(grayscale)
    std_dev = numpy_array.std()
    return std_dev < 10  # Adjust this threshold as per definition of "white page"


def split_pdf_on_white_pages(file_path, output_folder):
    inputpdf = PdfReader(file_path)
    total_pages = len(inputpdf.pages)

    writer = PdfWriter()
    doc_number = 1  # Initialize document number

    for i in range(total_pages):
        image = pdf_to_image(file_path, i)
        if is_white_page(image) or i == total_pages - 1:
            # Save the previous document and start a new one
            if writer.pages:
                output_file_path = os.path.join(output_folder, f'output_{doc_number}.pdf')
                with open(output_file_path, 'wb') as output_pdf:
                    writer.write(output_pdf)
                writer = PdfWriter()
                doc_number += 1  # Increase document number
        else:
            # Add the page to the current document
            writer.add_page(inputpdf.pages[i])


split_pdf_on_white_pages('V:\Kizilova\Крысы сканы\Статистика_крысы_3.pdf', 'V:\Kizilova\Крысы сканы')
