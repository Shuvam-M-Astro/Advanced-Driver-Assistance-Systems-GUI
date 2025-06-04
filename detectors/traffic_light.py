import cv2
import numpy as np

def detect_traffic_light(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0,100,100]), np.array([10,255,255])
    lower_red2, upper_red2 = np.array([160,100,100]), np.array([180,255,255])
    lower_green, upper_green = np.array([40,50,50]), np.array([90,255,255])
    lower_yellow, upper_yellow = np.array([15,150,150]), np.array([35,255,255])

    masks = {
        'red': cv2.add(cv2.inRange(hsv, lower_red1, upper_red1),
                      cv2.inRange(hsv, lower_red2, upper_red2)),
        'green': cv2.inRange(hsv, lower_green, upper_green),
        'yellow': cv2.inRange(hsv, lower_yellow, upper_yellow)
    }

    for color, mask in masks.items():
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 60,
                                   param1=50, param2=10, minRadius=0, maxRadius=30)
        if circles is not None:
            return color
    return 'green'
