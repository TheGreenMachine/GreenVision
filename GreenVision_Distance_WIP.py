import numpy as np
import cv2
import sys
import math
import json

with open('values.json') as json_file:
    data = json.load(json_file)
cap = cv2.VideoCapture(0)

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


def calc_distance(pitch):
    height_diff = data['height-of-target'] - data['height-of-camera']
    distance = math.fabs(height_diff / math.tan(math.radians(pitch)))
    return distance


def calc_pitch(py, cy, v_foc_len):
    pitch = math.degrees(math.atan((py - cy) / v_foc_len)) * -1
    return round(pitch)
