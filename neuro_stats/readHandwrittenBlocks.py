import numpy as np
import imutils
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

img = cv2.imread("/neuro_stats/output/20.png", cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv2.Canny(img,150,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

print(cnts)

model = torch.jit.load("d:/dev/pythonProjects/neuro_stats/neuro_stats/LeNet5_full_28.pt")
model.eval()
#model = torch.jit.load("d:/dev/pythonProjects/neuro_stats/neuro_stats/modelf28.pt")
#model.eval()
#img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Преобразование изображения в оттенки серого
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = img

# Бинаризация изображения
_, binary_img = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

print(binary_img.shape)

plt.imshow(binary_img)
plt.show()

ress = []
for i in range(0, binary_img.shape[1]-15, 15):
	imgSlice = binary_img[:, i:i+15]
	#imgSlice = np.rot90(imgSlice)
	imgSlice2 = imutils.resize(imgSlice, width=32)
	padded = cv2.copyMakeBorder(imgSlice2, top=2, bottom=2,
				left=2, right=2, borderType=cv2.BORDER_CONSTANT,
				value=(0, 0, 0))
	padded = cv2.resize(padded, (28, 28))
	padded = padded.astype("float32") / 255.0
	#padded = imgSlice2.astype("float32") / 255.0
	plt.subplot(121),plt.imshow(imgSlice)
	plt.subplot(122),plt.imshow(padded)
	chars3 = padded.reshape(-1, 784)
	chars4 = chars3.reshape((28, 28))
	chars4s = torch.from_numpy(chars4)
	chars5 = torch.zeros((1, 1, 28, 28))
	chars5[0, 0, :, :] = chars4s
	a = model(chars5)
	print(a, torch.argmax(a, 1))
	#a = model(torch.from_numpy(chars3))
	#ress.append(a)
	ress.append(torch.argmax(a, 1))
	plt.show()
import pprint
pprint.pprint(ress)
import sys
sys.exit()

#for i in range(0, )

# Находим области с текстом
#dilated_img = cv2.dilate(binary_img, None, iterations=2)

# Края
#edges = cv2.Canny(dilated_img, 50, 200)
edges = cv2.Canny(binary_img, 50, 200)
#contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Рисование контуров
for cnt in contours:
	x, y, w, h = cv2.boundingRect(cnt)

	# Отсечение слишком маленьких и слишком больших контуров
	char = torch.zeros((28, 28))
	if w * h > 10 and w * h < 2000:
	#if True:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


		print(w, h)
		#char[0:0+w, 0:0+h] = torch.from_numpy(img[x:x+w, y:y+h])
		print(x, y, w, h)
		print(img, img.shape)
		img2 = img[y:y+h, x:x+w]
		plt.imshow(img2)
		plt.show()
		#cv2.imshow("Image2", img2)
		#cv2.waitKey(0)

#cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# chars = []
# ress = []
# #
# # # loop over the contours
# for c in cnts:
# # 	# compute the bounding box of the contour
# 	(x, y, w, h) = cv2.boundingRect(c)
# # 	# filter out bounding boxes, ensuring they are neither too small
# # 	# nor too large
# 	#if (w >= 5 and w <= 30) and (h >= 15 and h <= 30):
# 	if True:
# # 		# extract the character and threshold it to make the character
# # 		# appear as *white* (foreground) on a *black* background, then
# # 		# grab the width and height of the thresholded image
# 		roi = img[y:y + h, x:x + w]
# 		thresh = cv2.threshold(roi, 0, 255,
# 			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 		(tH, tW) = thresh.shape
# # 		# if the width is greater than the height, resize along the
# # 		# width dimension
# 		if tW > tH:
# 			thresh = imutils.resize(thresh, width=32)
# # 		# otherwise, resize along the height
# 		else:
# 			thresh = imutils.resize(thresh, height=32)
# # 		# re-grab the image dimensions (now that its been resized)
# # 		# and then determine how much we need to pad the width and
# # 		# height such that our image will be 32x32
# 		(tH, tW) = thresh.shape
# 		dX = int(max(0, 32 - tW) / 2.0)
# 		dY = int(max(0, 32 - tH) / 2.0)
# # 		# pad the image and force 32x32 dimensions
# 		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
# 			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
# 			value=(0, 0, 0))
# 		padded = cv2.resize(padded, (28, 28))
# # 		# prepare the padded image for classification via our
# # 		# handwriting OCR model
# 		padded = padded.astype("float32") / 255.0
# 		padded = np.expand_dims(padded, axis=-1)
# # 		# update our list of characters that will be OCR'd
# 		chars.append((padded, (x, y, w, h)))
# 		#a = model(torch.from_numpy(padded))
# 		cv2.imshow("Image", padded)
# 		cv2.waitKey(0)
# 		chars3 = padded.reshape(-1, 784)
# 		chars4 = chars3.reshape((28, 28))
# 		chars4s = torch.from_numpy(chars4)
# 		chars5 = torch.zeros((1, 1, 28, 28))
# 		chars5[0, 0, :, :] = chars4s
# 		a = model(chars5)
# 		#a = model(torch.from_numpy(chars3))
# 		#ress.append(a)
# 		ress.append(torch.argmax(a, 1))
# #
# # # extract the bounding box locations and padded characters
# boxes = [b[1] for b in chars]
# chars = np.array([c[0] for c in chars], dtype="float32")
#
# print(boxes)
# print(ress)
# import pprint
# pprint.pprint(ress)
#
# #model = torch.load("d:/dev/pythonProjects/neuro_stats/neuro_stats/LeNet5_full_28.pth")
# #model.eval()
#
#
#
# chars2 = torch.from_numpy(chars)
#
# #chars2 = torch.utils.data.DataLoader(chars, shuffle=True, batch_size=100)
#
# #print(chars2, chars2.shape, "chars")
#
# #preds = model(chars2)
# #for i in chars2:
# #	pred = model(i)
import sys
sys.exit()

for (pred, (x, y, w, h)) in zip(preds, boxes):
	# find the index of the label with the largest corresponding
	# probability, then extract the probability and label
	#i = np.argmax(pred)
	print(pred, torch.argmax(pred))
	i = torch.argmax(pred, 0).float()

	label = str(i)
	prob = 1.
	#prob = pred[i]
	#label = labelNames[i]
	# draw the prediction on the image
	print("[INFO] {} - {:.2f}%".format(label, prob * 100))
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(img, label, (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
	# show the image
	cv2.imshow("Image", img)
	cv2.waitKey(0)