import cv2
import numpy as np
import matplotlib
import json
import argparse
import time
import glob

with open('values.json') as json_file:
    data = json.load(json_file)

lower_color = np.array(data['lower-color-list'])
upper_color = np.array([data['upper-color-list'][0], 255, 255])
count = 6
while (1):
    img = cv2.imread("C:/Users/Nightmaze/Documents/Github/FRC/GreenVision/Test_Images/opencv_image_{}in.jpg".format(count))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    print("{} in shot".format(count))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    for contour in contours:
        if cv2.contourArea(contour) > 75:
            ncontours.append(contour)
            contourarea = cv2.contourArea(contour)
            print("Contour Area: {}".format(contourarea))
    print("Number of contours: ", len(ncontours))
    count +=3


