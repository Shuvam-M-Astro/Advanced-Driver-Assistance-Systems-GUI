import cv2
import numpy as np
import cvzone

img2 = cv2.imread('stop.png',cv2.IMREAD_UNCHANGED)
img1 = cv2.imread('GUI_def.jpg',cv2.IMREAD_UNCHANGED)
#dst = cv2.addWeighted(img1, 1, img2, 0.7, 0)

#img_arr = np.hstack((img1, img2))

# Resize smaller image
scale_percent = 15 # percent of original size
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)


#Resize bigger Image
scale_percent = 60 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)



imgResult = cvzone.overlayPNG(img1,resized,[700,200])
#cv2.imshow('Input Images',img_arr)
cv2.imshow('Blended Image',imgResult)

cv2.waitKey(0)
cv2.destroyAllWindows()
