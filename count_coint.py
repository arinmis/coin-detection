import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import dilate
import contours
import canny
import gaussianblur
import cvt_color as cvt

if len(sys.argv) < 2:
    print("usage: python count_coins.py target.jpg")
    exit()

# return numpy array
image = cv2.imread(sys.argv[1])
gray = cvt.rgb2gray(image)
plt.imshow(gray, cmap='gray');

# 1. make it blur
blur = gaussianblur.gaussian_blur(gray, 31, 1)
plt.imshow(blur, cmap='gray')
plt.show()

# 2. apply edge detection
canny = canny.canny_edge_detection(blur)
plt.imshow(canny, cmap='gray')

# make edges clear
dilated = dilate.dilate_image(canny, (4, 4), iterations = 5)
plt.imshow(dilated, cmap='gray')
plt.show()

# 3. Contour the coins
(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (255, 0,0), 2) #  

# cnt= contours.find_contours(dilated.copy())
# cnt= cv2.find_contours(dilated.copy())
# rgb = cvt.bgr2rgb(image)
# contours.draw_contours(rgb, cnt,(255, 0,0))  

# put model here
# https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
print('Coins in the image: ', len(cnt))
plt.imshow(rgb)
plt.show()
