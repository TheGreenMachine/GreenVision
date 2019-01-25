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


def draw_points(rec_a, rec_b, avgcx, avgcy):
    # center1_x, center1_y, center2_x, center2_y, avg_center_x, avg_center_y
    # cv2.line(frame, (center1_x, center1_y), (center1_x, center1_y), (255, 0, 0), 8)
    # cv2.line(frame, (center2_x, center2_y), (center2_x, center2_y), (255, 0, 0), 8)
    # cv2.line(frame, (avg_center_x, avg_center_y), (avg_center_x, avg_center_y), (255, 0, 0), 8)
    cv2.line(frame, (rec_a['c_x'], rec_a['c_y']), (rec_a['c_x'], rec_a['c_y']), (255, 0, 0), 8)
    cv2.line(frame, (rec_b['c_x'], rec_b['c_y']), (rec_b['c_x'], rec_b['c_y']), (255, 0, 0), 8)
    cv2.line(frame, (avgcx, avgcy), (avgcx, avgcy), (255, 0, 0), 8)


def def_rec(rectangle):
    top_left_x = rectangle[0]
    top_left_y = rectangle[1]
    width = rectangle[2]
    height = rectangle[3]
    bottom_right_x = top_left_x + width
    bottom_right_y = top_left_y + height
    center_x = int((top_left_x + bottom_right_x) / 2)
    center_y = int((top_left_y + bottom_right_y) / 2)

    return {'tl_x': top_left_x, 'tl_y': top_left_y, 'br_x': bottom_right_x, 'br_y': bottom_right_y, 'c_x': center_x,
            'c_y': center_y}


def get_avg_points(rec_a, rec_b):
    # (center1_x, center2_x, center1_y, center2_y)
    # avg_center_x = int((center1_x + center2_x) / 2)
    # avg_center_y = int((center1_y + center2_y) / 2)
    avg_center_x = int((rec_a['c_x'] + rec_b['c_x']) / 2)
    avg_center_y = int((rec_a['c_y'] + rec_b['c_y']) / 2)
    if debug_flag:
        print('Average Center (x , y): ({acx} , {acy})'.format(acx=avg_center_x, acy=avg_center_y))
    return avg_center_x, avg_center_y


def is_pair(tl1_x, tl2_x, br1_x, br2_x):
    top_diff = abs(tl1_x - tl2_x)
    bottom_diff = abs(br1_x - br2_x)
    if debug_flag:
        print('Top diff: {td}\nBottom diff: {bd}'.format(td=top_diff, bd=bottom_diff))
    return bottom_diff > top_diff


def update_net_table(n, c1_x=-1, c1_y=-1, c2_x=-1, c2_y=-1, avgc_x=-1, avgc_y=-1):
    table.putNumber("center{n}X".format(n=n), c1_x)
    table.putNumber("center{n}Y".format(n=n), c1_y)
    table.putNumber("center{n}X".format(n=n + 1), c2_x)
    table.putNumber("center{n}Y".format(n=n + 1), c2_y)
    table.putNumber("averagedCenterX", avgc_x)
    table.putNumber("averagedCenterY", avgc_y)
    if debug_flag:
        print("center{n}X".format(n=n), c1_x)
        print("center{n}Y".format(n=n), c1_y)
        print("center{n}X".format(n=n + 1), c2_x)
        print("center{n}Y".format(n=n + 1), c2_y)
        print("averagedCenterX", avgc_x)
        print("averagedCenterY", avgc_y)


while True:
    print('=========================================================')
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
    rec_list = []
    for c in ncontours:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rec_list.append(cv2.boundingRect(c))
        if len(rec_list) > 1:
            # topLeft1X, topLeft1Y, bottomRight1X, bottomRight1Y, center1X, center1Y = def_rec(rec_list[0])
            # topLeft2X, topLeft2Y, bottomRight2X, bottomRight2Y, center2X, center2Y = def_rec(rec_list[1])
            # averagedCenterX, averagedCenterY = get_avg_points(center1X, center2X, center1Y, center2Y)

            rec1 = def_rec(rec_list[0])
            rec2 = def_rec(rec_list[1])
            avg_c1_x, avg_c1_y = get_avg_points(rec1, rec2)
            if True:
                # if is_pair(topLeft1X, topLeft2X, bottomRight1X, bottomRight2X):
                # update_net_table(1, center1X, center1Y, center2X, center2Y, averagedCenterX, averagedCenterY)
                # draw_points(center1X, center1Y, center2X, center2Y, averagedCenterX, averagedCenterY)
                update_net_table(1, rec1['c_x'], rec1['c_y'], rec2['c_x'], rec2['c_y'], avg_c1_x, avg_c1_y)
                draw_points(rec1, rec2, avg_c1_x, avg_c1_y)

            if len(rec_list) > 3:
                # topLeft3X, topLeft3Y, bottomRight3X, bottomRight3Y, center3X, center3Y = def_rec(rec_list[2])
                # topLeft4X, topLeft4Y, bottomRight4X, bottomRight4Y, center4X, center4Y = def_rec(rec_list[3])
                # if is_pair(topLeft3X, topLeft4X, bottomRight3X, bottomRight4X):
                rec3 = def_rec(rec_list[2])
                rec4 = def_rec(rec_list[3])
                avg_c2_x, avg_c2_y = get_avg_points(rec3, rec4)
                if True:
                    update_net_table(2, rec3['c_x'], rec3['c_y'], rec4['c_x'], rec4['c_y'], avg_c2_x, avg_c2_y)
                    draw_points(rec3, rec4, avg_c2_x, avg_c2_y)

                if len(rec_list) > 5:
                    # topLeft5X, topLeft5Y, bottomRight5X, bottomRight5Y, center5X, center5Y = def_rec(rec_list[4])
                    # topLeft6X, topLeft6Y, bottomRight6X, bottomRight6Y, center6X, center6Y = def_rec(rec_list[5])
                    rec5 = def_rec(rec_list[4])
                    rec6 = def_rec(rec_list[5])
                    avg_c3_x, avg_c3_y = get_avg_points(rec5, rec6)
                    # if is_pair(topLeft5X, topLeft6X, bottomRight5X, bottomRight6X):
                    if True:
                        update_net_table(3, rec5['c_x'], rec5['c_y'], rec6['c_x'], rec6['c_y'], avg_c3_x, avg_c3_y)
                        draw_points(rec5, rec6, avg_c3_x, avg_c3_y)

    if vision_flag:
        cv2.imshow('Contour Window', frame)
        cv2.imshow('Mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
