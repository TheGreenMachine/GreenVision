import cv2
import numpy as np
from statistics import mean
import json
import os
import csv

with open('values.json') as json_file:
    data = json.load(json_file)

lower_color = np.array(data['lower-color-list'])
upper_color = np.array([data['upper-color-list'][0], 255, 255])
count = 6

xs = np.array([], dtype=np.float64)
ys = np.array([], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)

    return m, b


def capture(count):
    path = '/home/pi/Desktop/GreenVision/Test_Images'
    img_name = os.path.join(path, 'opencv_image_{}in.jpg'.format(count))
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])
while count < 31:
    print('Line up camera at {}, and press C to capture'.format(count))
    ret, frame = cap.read()
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        capture(count)
        count += 1
count = 6
while count < 31:
    img = cv2.imread("/home/pi/Desktop/GreenVision/Test_Images/opencv_image_{}in.jpg".format(count))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    print("{} in shot".format(count))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    for contour in contours:
        if cv2.contourArea(contour) > 75:
            ncontours.append(contour)
    if len(ncontours) <= 2 and len(ncontours) != 0 and len(ncontours) != 1 and len(ncontours) != 3:
        contourarea = cv2.contourArea(ncontours[0])
        ys = np.append(ys, contourarea)
        xs = np.append(xs, count)
        print(ys)
        print(xs)
    count += 1


with open('distance_calibrate_dump', 'w') as output:
    pass
m, b = best_fit_slope_and_intercept(xs, ys)

print("M: {} B: {}".format(m, b))
