import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import dilate
import contours

if len(sys.argv) < 2:
    print("usage: python count_coins.py target.jpg")
    exit()

# return numpy array
image = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray');

# 1. make it blur
blur = cv2.GaussianBlur(gray, (31,31), 1)
plt.imshow(blur, cmap='gray')
# plt.show()

# 2. apply edge detection
canny = cv2.Canny(blur, 10, 200)
plt.imshow(canny, cmap='gray')

# make edges clear
dilated = dilate.dilate_image(canny, (4, 4), iterations = 5)
plt.imshow(dilated, cmap='gray')
plt.show()

# 3. Contour the coins
(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
contours.draw_contours(rgb, cnt,(255, 0,0))  

# put model here
# https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
print('Coins in the image: ', len(cnt))
plt.imshow(rgb)
plt.show()
