import numpy as np
import cv2
import networktables as nt
from imutils.video import WebcamVideoStream
import json
import math
import time
import argparse

with open('values.json') as json_file:
    data = json.load(json_file)

ap = argparse.ArgumentParser(description='Team 1816 Vision Processing for the 2019 Deep Space Season')
ap.add_argument('src', help='Choose source for processing: integer for camera, file path for image/video')
ap.add_argument('-v', help='Toggle contour and mask window', action='store_true')
ap.add_argument('-db', help='Toggle debug print statements', action='store_true')
ap.add_argument('-th', help='Adjust thresholds for lower_color and upper_color by 50 or less', default=0, type=int, )
ap.add_argument('-mt', help='Toggle multi-threading', action='store_true')
ap.add_argument('-nt', help='Toggle network tables', action='store_true')

args = vars(ap.parse_args())
source_input = args['src']
vision_flag = args['v']
debug_flag = args['db']
threshold = args['th'] if 0 < args['th'] <= 50 else 0
multi_thread_flag = args['mt']
nt_flag = args['nt']

# cap = WebcamVideoStream(src=0).start() if multi_thread_flag else cv2.VideoCapture(0)

if multi_thread_flag:
    cap = WebcamVideoStream(0)
    cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
    cap.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])
    cap.start()

else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])

if nt_flag:
    nt.NetworkTables.initialize(server=data['server-ip'])
    table = nt.NetworkTables.getTable("SmartDashboard")
    if table:
        print("table OK")
    table.putNumber("visionX", -1)
    table.putNumber("visionY", -1)

if debug_flag:
    print('----------------------------------------------------------------')
    print('Current Source: {}'.format(source_input))
    print('Vision Flag: {}'.format(vision_flag))
    print('Debug Flag: {}'.format(debug_flag))
    print('Threshold Value: {}'.format(threshold))
    print('Multi-Thread Flag: {}'.format(multi_thread_flag))
    print('Network Tables Flag: {}'.format(nt_flag))
    print('----------------------------------------------------------------')

horizontal_aspect = data['horizontal-aspect']
vertical_aspect = data['vertical-aspect']

horizontal_view = data['fish-eye-cam-HFOV']
vertical_view = data['fish-eye-cam-VFOV']

H_FOCAL_LENGTH = data['image-width'] / (2 * math.tan((horizontal_view / 2)))
V_FOCAL_LENGTH = data['image-height'] / (2 * math.tan((vertical_view / 2)))

lower_color = np.array(data['lower-color-list']) - threshold
upper_color = np.array([data['upper-color-list'][0] + threshold, 255, 255])


def calc_distance(area):
    equation = (-3.07571647 * (10 ** -15))(area ** 2) + (1.00000000 * 10 ** -2)(area) + 1
    return equation


def calc_pitch(pixel_y, center_y, v_foc_len):
    p = math.degrees(math.atan((pixel_y - center_y) / v_foc_len)) * -1
    return round(p)


def calc_yaw(pixel_x, center_x, h_foc_len):
    ya = math.degrees(math.atan((pixel_x - center_x) / h_foc_len))
    return round(ya)


def draw_points(rec_a, rec_b, avgcx, avgcy):
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
    starttime = time.time()
    if multi_thread_flag:
        frame = cap.read()
    else:
        _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    screen_c_x = (data['image-width'] / 2) - 0.5
    screen_c_y = (data['image-height'] / 2) - 0.5
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    for contour in contours:
        if cv2.contourArea(contour) > 75:
            print('Contour area:', cv2.contourArea(contour))
            contourarea = cv2.contourArea(contour)
            ncontours.append(contour)
    print("Number of contours: ", len(ncontours))
    rec_list = []
    for c in ncontours:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rec_list.append(cv2.boundingRect(c))
        if len(rec_list) > 1:
            rec1 = def_rec(rec_list[0])
            rec2 = def_rec(rec_list[1])
            avg_c1_x, avg_c1_y = get_avg_points(rec1, rec2)
            if True:
                if nt_flag:
                    update_net_table(1, rec1['c_x'], rec1['c_y'], rec2['c_x'], rec2['c_y'], avg_c1_x, avg_c1_y)
                draw_points(rec1, rec2, avg_c1_x, avg_c1_y)
                pitch = calc_pitch(avg_c1_y, screen_c_y, V_FOCAL_LENGTH)
                distance = calc_distance(contourarea)
                yaw = calc_yaw(avg_c1_x, screen_c_x, H_FOCAL_LENGTH)
                print('Pitch = {} \t Distance = {} \t Yaw = {}'.format(pitch, distance, yaw))

            if len(rec_list) > 3:
                rec3 = def_rec(rec_list[2])
                rec4 = def_rec(rec_list[3])
                avg_c2_x, avg_c2_y = get_avg_points(rec3, rec4)
                if True:
                    if nt_flag:
                        update_net_table(2, rec3['c_x'], rec3['c_y'], rec4['c_x'], rec4['c_y'], avg_c2_x, avg_c2_y)
                    draw_points(rec3, rec4, avg_c2_x, avg_c2_y)

                if len(rec_list) > 5:
                    rec5 = def_rec(rec_list[4])
                    rec6 = def_rec(rec_list[5])
                    avg_c3_x, avg_c3_y = get_avg_points(rec5, rec6)
                    if True:
                        if nt_flag:
                            update_net_table(3, rec5['c_x'], rec5['c_y'], rec6['c_x'], rec6['c_y'], avg_c3_x, avg_c3_y)
                        draw_points(rec5, rec6, avg_c3_x, avg_c3_y)
    print("Elasped Time: {}".format(time.time() - starttime))
    if vision_flag:
        cv2.imshow('Contour Window', frame)
        cv2.imshow('Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break