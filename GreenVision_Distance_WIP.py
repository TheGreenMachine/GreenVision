import numpy as np
import cv2
import sys
import math
import json

with open('values.json') as json_file:
    data = json.load(json_file)


def calc_distance(pitch):
    height_diff = data['height-of-target'] - data['height-of-camera']
    distance = math.fabs(height_diff / math.tan(math.radians(pitch)))
    return distance


def calc_pitch(py, cy, v_foc_len):
    p = math.degrees(math.atan((py - cy) / v_foc_len)) * -1
    return p


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


def draw_points(rec_a, rec_b, avgcx, avgcy):
    cv2.line(frame, (rec_a['c_x'], rec_a['c_y']), (rec_a['c_x'], rec_a['c_y']), (255, 0, 0), 8)
    cv2.line(frame, (rec_b['c_x'], rec_b['c_y']), (rec_b['c_x'], rec_b['c_y']), (255, 0, 0), 8)
    cv2.line(frame, (avgcx, avgcy), (avgcx, avgcy), (255, 0, 0), 8)


def get_avg_points(rec_a, rec_b):
    avg_center_x = int((rec_a['c_x'] + rec_b['c_x']) / 2)
    avg_center_y = int((rec_a['c_y'] + rec_b['c_y']) / 2)
    print('Average Center (x , y): ({acx} , {acy})'.format(acx=avg_center_x, acy=avg_center_y))
    return avg_center_x, avg_center_y


cam_fov = data['fish-eye-cam-FOV']
diagonal_view = math.radians(cam_fov)

horizontal_aspect = 16
vertical_aspect = 9

diagonal_aspect = math.hypot(horizontal_aspect, vertical_aspect)
horizontal_view = math.atan(math.tan(diagonal_view / 2) * (horizontal_aspect / diagonal_aspect)) * 2
vertical_view = math.atan(math.tan(diagonal_view / 2) * (vertical_aspect / diagonal_aspect)) * 2

H_FOCAL_LENGTH = data['image-width'] / (2 * math.tan(horizontal_view / 2))
V_FOCAL_LENGTH = data['image-height'] / (2 * math.tan(vertical_view / 2))

lower_color = np.array(data['lower-color-list'])
upper_color = np.array(data['upper-color-list'])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)

while True:
    print('=========================================================')
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
        M = cv2.moments(c)
        if M["m00"] != 0:
            # cy = int(M["m01"] / M["m00"])
            cy = M["m01"] / M["m00"]
        else:
            cy = 0, 0
        print('cy: {}'.format(cy))
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rec_list.append(cv2.boundingRect(c))
        if len(rec_list) > 1:
            rec1 = def_rec(rec_list[0])
            rec2 = def_rec(rec_list[1])
            avg_c1_x, avg_c1_y = get_avg_points(rec1, rec2)
            if True:
                draw_points(rec1, rec2, avg_c1_x, avg_c1_y)
                pitch = calc_pitch(cy, avg_c1_y, V_FOCAL_LENGTH)
                distance = calc_distance(pitch) if pitch != 0 else 0
                print('Pitch = {} \t Distance = {}'.format(pitch, distance))
    cv2.imshow('Contour Window', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
