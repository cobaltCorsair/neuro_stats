import os
import cv2
import fitz  # PyMuPDF
import numpy as np
import json


def save_cell_info(cell_info, filename):
    with open(filename, 'w') as f:
        json.dump(cell_info, f)


def count_rows_cols(boundingBoxes, width_threshold=50, height_threshold=50):
    # Initialize row and column counts
    row_count = 1
    col_count = 1

    # Initialize current position
    current_x = boundingBoxes[0][0]
    current_y = boundingBoxes[0][1]

    for bbox in boundingBoxes[1:]:
        x, y, _, _ = bbox

        # If x coordinate changes significantly, increment column count
        if abs(x - current_x) > width_threshold:
            col_count += 1
            current_x = x

        # If y coordinate changes significantly, increment row count
        if abs(y - current_y) > height_threshold:
            row_count += 1
            current_y = y

    return row_count, col_count


def is_white_page(np_img, threshold=0):
    # Adjust this threshold as per definition of "white page"
    return np_img.std() < threshold


def sort_contours(cnts, method="left-to-right"):
    # Initialize variables
    reverse = False
    i = 0

    # Check if reverse is needed
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # Check if sort is based on y-coordinate rather than x-coordinate
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # Create a bounding box for each contour
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # Sort contours based on the bounding boxes
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)


def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    # Read the image
    img = cv2.imread(img_for_box_extraction_path, 0)
    # Threshold the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    # Define a kernel length
    kernel_length = np.array(img).shape[1] // 40

    # Define vertical and horizontal kernels
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Use erosion and dilation to eliminate noise
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha

    # Add two images to get a complete image with both horizontal and vertical lines
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    # Initialize cell info list
    cell_info_list = []

    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # If the box is a rectangle and not a square and has a certain minimum size, then save it
        if (w > 80 and h > 20) and w > 2 * h:
            idx += 1
            new_img = img[y:y + h, x:x + w]
            if not is_white_page(new_img):
                output_file = os.path.join(cropped_dir_path, f'{idx}.png')
                cv2.imwrite(output_file, new_img)

                # Save cell info
                cell_info = {'x': x, 'y': y, 'w': w, 'h': h, 'img_file': output_file.split('\\')[1]}
                cell_info_list.append(cell_info)

    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imwrite("output/img_contour.jpg", img)

    # Calculate row and column counts
    row_count, col_count = count_rows_cols(boundingBoxes)

    # Save cell info to file
    cell_info_file = os.path.join(cropped_dir_path, 'cell_info.json')
    save_cell_info({'row_count': row_count, 'col_count': col_count, 'cells': cell_info_list}, cell_info_file)


# Open the PDF document
doc = fitz.open("../test.pdf")

# For each page in the document
for page_number in range(len(doc)):
    # Load the page and render it as a pixmap
    page = doc.load_page(page_number)
    image_matrix = page.get_pixmap(matrix=fitz.Matrix(100 / 72, 100 / 72))
    # Convert the pixmap into an image object
    img = np.frombuffer(image_matrix.samples, np.uint8).reshape(image_matrix.height, image_matrix.width, 3)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    temp_file_name = "temp_image.png"
    cv2.imwrite(temp_file_name, img)
    box_extraction(temp_file_name, "output")

# Clean up the temporary image file
os.remove(temp_file_name)
