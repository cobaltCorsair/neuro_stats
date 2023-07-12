from imutils.contours import sort_contours
import numpy as np
import imutils
import cv2
import torch


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread("d:/dev/pythonProjects/neuro_stats/test.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

cnts = cv2.findContours(edges.copy(), cv.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

print(cnts)
#cnts = sort_contours(cnts, method="left-to-right")[0]

#import sys
#sys.exit()


# # load the input image from disk, convert it to grayscale, and blur
# # it to reduce noise
# image = cv2.imread("d:/dev/pythonProjects/neuro_stats/test.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# # perform edge detection, find contours in the edge map, and sort the
# # resulting contours from left-to-right
# edged = cv2.Canny(gray, 100, 200)
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sort_contours(cnts, method="left-to-right")[0]
# # initialize the list of contour bounding boxes and associated
# # characters that we'll be OCR'ing
# chars = []
#
# # loop over the contours
# for c in cnts:
# 	# compute the bounding box of the contour
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	# filter out bounding boxes, ensuring they are neither too small
# 	# nor too large
# 	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
# 		# extract the character and threshold it to make the character
# 		# appear as *white* (foreground) on a *black* background, then
# 		# grab the width and height of the thresholded image
# 		roi = gray[y:y + h, x:x + w]
# 		thresh = cv2.threshold(roi, 0, 255,
# 			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 		(tH, tW) = thresh.shape
# 		# if the width is greater than the height, resize along the
# 		# width dimension
# 		if tW > tH:
# 			thresh = imutils.resize(thresh, width=32)
# 		# otherwise, resize along the height
# 		else:
# 			thresh = imutils.resize(thresh, height=32)
# 		# re-grab the image dimensions (now that its been resized)
# 		# and then determine how much we need to pad the width and
# 		# height such that our image will be 32x32
# 		(tH, tW) = thresh.shape
# 		dX = int(max(0, 32 - tW) / 2.0)
# 		dY = int(max(0, 32 - tH) / 2.0)
# 		# pad the image and force 32x32 dimensions
# 		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
# 			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
# 			value=(0, 0, 0))
# 		padded = cv2.resize(padded, (32, 32))
# 		# prepare the padded image for classification via our
# 		# handwriting OCR model
# 		padded = padded.astype("float32") / 255.0
# 		padded = np.expand_dims(padded, axis=-1)
# 		# update our list of characters that will be OCR'd
# 		chars.append((padded, (x, y, w, h)))
#
# # extract the bounding box locations and padded characters
# boxes = [b[1] for b in chars]
# chars = np.array([c[0] for c in chars], dtype="float32")
#
# print(boxes, chars)

for cnt in cnts:
	x, y, w, h = cv2.boundingRect(cnt)

	# Отсечение слишком маленьких контуров, которые могут быть шумом
	#if w * h > 100:
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Отображение изображения
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
