import cv2
import numpy as np

def detect_lane(img):
    stencil = np.zeros_like(img[:,:,0])
    polygon = np.array([[50,270], [220,160], [360,160], [480,270]])
    cv2.fillConvexPoly(stencil, polygon, 1)
    masked = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)
    _, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return img