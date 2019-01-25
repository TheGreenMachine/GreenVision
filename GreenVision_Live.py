import numpy as np
import cv2
import sys
import networktables as nt
from imutils.video import WebcamVideoStream
import json

vision_flag = '-v' in sys.argv
debug_flag = '-d' in sys.argv
threshold_flag = '-t' in sys.argv
multithread_flag = '-mt' in sys.argv

with open('values.json') as json_file:
    data = json.load(json_file)

cap = WebcamVideoStream(src=0).start() if multithread_flag else cv2.VideoCapture(0)

nt.NetworkTables.initialize(server=data['server-ip'])
table = nt.NetworkTables.getTable("SmartDashboard")
if table:
    print("table OK")
table.putNumber("visionX", -1)
table.putNumber("visionY", -1)

if debug_flag:
    print('Vision flag: {v}\nDebug flag: {d}\nThreshold Flag: {t}\nMultithread Flag: {mt}'.format(v=vision_flag,
                                                                                                  d=debug_flag,
                                                                                                  t=threshold_flag,
                                                                                                  mt=multithread_flag))

lower_color = np.array(data["lower-color-list-thresh"]) if threshold_flag else np.array(data["lower-color-list"])
upper_color = np.array(data["upper-color-list-thresh"]) if threshold_flag else np.array(data["upper-color-list"])


def draw_points(center1_x, center1_y, center2_x, center2_y, avg_center_x, avg_center_y):
    cv2.line(frame, (center1_x, center1_y), (center1_x, center1_y), (255, 0, 0), 8)
    cv2.line(frame, (center2_x, center2_y), (center2_x, center2_y), (255, 0, 0), 8)
    cv2.line(frame, (avg_center_x, avg_center_y), (avg_center_x, avg_center_y), (255, 0, 0), 8)


def def_rec(rectangle):
    top_left_x = rectangle[0]
    top_left_y = rectangle[1]
    width = rectangle[2]
    height = rectangle[3]
    bottom_right_x = top_left_x + width
    bottom_right_y = top_left_y + height
    center_x = int((top_left_x + bottom_right_x) / 2)
    center_y = int((top_left_y + bottom_right_y) / 2)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y


def get_avg_points(center1_x, center2_x, center1_y, center2_y):
    avg_center_x = int((center1_x + center2_x) / 2)
    avg_center_y = int((center1_y + center2_y) / 2)

    return avg_center_x, avg_center_y


def is_pair(topLeftX, topLeftX1, bottomRightX, bottomRightX1):
    topDiff = abs(topLeftX - topLeftX1)
    bottomDiff = abs(bottomRightX - bottomRightX1)
    print('Top diff: {td}\nBottom diff: {bd}'.format(td=topDiff, bd=bottomDiff))
    return bottomDiff > topDiff


def update_net_table(n, center1x=-1, center1y=-1, center2x=-1, center2y=-1, averagedCenterX=-1, averagedCenterY=-1):
    table.putNumber("center{n}X".format(n=n), center1x)
    table.putNumber("center{n}Y".format(n=n), center1y)
    table.putNumber("center{n}X".format(n=n + 1), center2x)
    table.putNumber("center{n}Y".format(n=n + 1), center2y)
    table.putNumber("averagedCenterX", averagedCenterX)
    table.putNumber("averagedCenterY", averagedCenterY)
    if debug_flag:
        print("center{n}X".format(n=n), center1x)
        print("center{n}Y".format(n=n), center1y)
        print("center{n}X".format(n=n + 1), center2x)
        print("center{n}Y".format(n=n + 1), center2y)
        print("averagedCenterX", averagedCenterX)
        print("averagedCenterY", averagedCenterY)


while True:
    if multithread_flag:
        frame = cap.read()
    else:
        _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    for contour in contours:
        if cv2.contourArea(contour) > 75:
            print('Contour area:', cv2.contourArea(contour))
            ncontours.append(contour)
    print("Number of contours: ", len(ncontours))
    rectangles = []
    for c in ncontours:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rectangles.append(cv2.boundingRect(c))
        if len(rectangles) > 1:
            topLeft1X, topLeft1Y, bottomRight1X, bottomRight1Y, center1X, center1Y = def_rec(rectangles[0])
            topLeft2X, topLeft2Y, bottomRight2X, bottomRight2Y, center2X, center2Y = def_rec(rectangles[1])
            averagedCenterX, averagedCenterY = get_avg_points(center1X, center2X, center1Y, center2Y)
            if True:
                # if is_pair(topLeft1X, topLeft2X, bottomRight1X, bottomRight2X):
                update_net_table(1, center1X, center1Y, center2X, center2Y, averagedCenterX, averagedCenterY)
                draw_points(center1X, center1Y, center2X, center2Y, averagedCenterX, averagedCenterY)

            if len(rectangles) > 3:
                topLeft3X, topLeft3Y, bottomRight3X, bottomRight3Y, center3X, center3Y = def_rec(rectangles[2])
                topLeft4X, topLeft4Y, bottomRight4X, bottomRight4Y, center4X, center4Y = def_rec(rectangles[3])
                averagedCenter1X, averagedCenter1Y = get_avg_points(center3X, center4X, center3Y, center4Y)
                if is_pair(topLeft3X, topLeft4X, bottomRight3X, bottomRight4X):
                    update_net_table(2, center3X, center3Y, center4X, center4Y, averagedCenter1X, averagedCenter1Y)
                    draw_points(center3X, center3Y, center4X, center4Y, averagedCenter1X, averagedCenter1Y)

                if len(rectangles) > 5:
                    topLeft5X, topLeft5Y, bottomRight5X, bottomRight5Y, center5X, center5Y = def_rec(rectangles[4])
                    topLeft6X, topLeft6Y, bottomRight6X, bottomRight6Y, center6X, center6Y = def_rec(rectangles[5])
                    averagedCenter2X, averagedCenter2Y = get_avg_points(center5X, center6X, center5Y, center6Y)
                    if is_pair(topLeft5X, topLeft6X, bottomRight5X, bottomRight6X):
                        update_net_table(3, center5X, center5Y, center6X, center6Y, averagedCenter2X, averagedCenter2Y)
                        draw_points(center4X, center4Y, center5X, center5Y, averagedCenter2X, averagedCenter2Y)

    if vision_flag:
        cv2.imshow('Contour Window', frame)
        cv2.imshow('Mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
