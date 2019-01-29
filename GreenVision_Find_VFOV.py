import numpy as np
import cv2
import json
import math
import time


def calc_distance(p):
    height_diff = data['height-of-target'] - data['height-of-camera']
    d = math.fabs(height_diff / math.tan(math.radians(p)))
    return d


def calc_pitch(pixel_y, center_y, v_foc_len):
    p = math.degrees(math.atan((pixel_y - center_y) / v_foc_len))
    p *= -1
    return round(p)


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

    return avg_center_x, avg_center_y


with open('values.json') as json_file:
    data = json.load(json_file)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])

horizontal_aspect = data['horizontal-aspect']
vertical_aspect = data['vertical-aspect']

lower_color = np.array(data["lower-color-list"])
upper_color = np.array(data["upper-color-list"])

for v_fov in range(60, 66):
    V_FOCAL_LENGTH = data['image-height'] / (2 * math.tan((v_fov / 2)))
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    screen_c_x = (data['image-width'] / 2) - 0.5
    screen_c_y = (data['image-height'] / 2) - 0.5
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    for contour in contours:
        if cv2.contourArea(contour) > 75:
            ncontours.append(contour)
    rec_list = []
    for c in ncontours:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rec_list.append(cv2.boundingRect(c))
        if len(rec_list) > 1:
            rec1 = def_rec(rec_list[0])
            rec2 = def_rec(rec_list[1])
            avg_c1_x, avg_c1_y = get_avg_points(rec1, rec2)
            if True:
                draw_points(rec1, rec2, avg_c1_x, avg_c1_y)
                pitch = calc_pitch(avg_c1_y, screen_c_y, V_FOCAL_LENGTH)
                if pitch != 0:
                    distance = calc_distance(pitch)
                else:
                    continue
                print('V_FOV: {}, pitch: {}, distance: {}'.format(v_fov, pitch, distance))
                print('=========================================================')
                time.sleep(5)
    cv2.imshow('Contour Window', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
