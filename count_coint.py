import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import dilate
import contours
import canny
import gaussianblur
import cvt_color as cvt
from keras.models import load_model
from cnn_model import generate_cnn_model
import os
if len(sys.argv) < 2:
    print("usage: python count_coins.py target.jpg")
    exit()

image = cv2.imread(sys.argv[1])
#image = cv2.imread(os.getcwd()+"\\tests\\test_test.png")
gray = cvt.rgb2gray(image)
plt.imshow(gray, cmap='gray')

# 1. make it blur
blur = gaussianblur.gaussian_blur(gray, 31, 1)
plt.imshow(blur, cmap='gray')
plt.show()

# 2. apply edge detection
canny = canny.canny_edge_detection(blur)
plt.imshow(canny, cmap='gray')

# make edges clear
dilated = dilate.dilate_image(canny, (3, 3), iterations = 2)
plt.imshow(dilated, cmap='gray')
plt.show()

# 3. Contour the coins
(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (255, 0,0), 2)



# put model here
cnn_model = load_model("cnn_model/coin_detection_model.h5")

# Array to store pixel values of each circle
circle_pixels = []
# List to store cropped coin images
coin_images = []
# Iterate over each contour and crop the coins
result = 0
for contour in cnt:
    (x, y, w, h) = cv2.boundingRect(contour)
    coin = image[y:y+h, x:x+w]
    coin_images.append(coin)

for coin in coin_images:
  predicted_label, amount = generate_cnn_model.predict(coin, cnn_model)
  result += amount
  plt.imshow(coin)
  plt.title(predicted_label)
  plt.show()

print("Total amount: " + str(result))
