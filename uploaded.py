import numpy as np
import cv2
import networktables as nt
from imutils.video import WebcamVideoStream
import json
import math

with open('values.json') as json_file:
    data = json.load(json_file)

src = int(args['source']) if args['source'].isdigit() else args['source']
model = args['model']
view = args['view']
debug = args['debug']
threshold = args['threshold'] if 0 < args['threshold'] < 50 else 0
multi = args['multithread']
net_table = args['networktables']

if multi:
    cap = WebcamVideoStream(src)
    cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
    cap.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])
    cap.start()

else:
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])

if net_table:
    nt.NetworkTables.initialize(server=data['server-ip'])
    table = nt.NetworkTables.getTable('SmartDashboard')
    if table:
        print('table OK')
    table.putNumber('visionX', -1)
    table.putNumber('visionY', -1)
    table.putNumber('width', data['width'])
    table.putNumber('height', data['height'])

if debug:
    print('----------------------------------------------------------------')
    print('Current Source: {}'.format(src))
    print('Vision Flag: {}'.format(view))
    print('Debug Flag: {}'.format(debug))
    print('Threshold Value: {}'.format(threshold))
    print('Function Model Value: {}'.format(model))
    print('Multi-Thread Flag: {}'.format(multi))
    print('Network Tables Flag: {}'.format(net_table))
    print('----------------------------------------------------------------')

horizontal_aspect = data['horizontal-aspect']
horizontal_fov = data['fish-eye-cam-HFOV']
h_focal_length = data['image-width'] / (2 * math.tan((horizontal_fov / 2)))

lower_color = np.array(data['lower-color-list']) - threshold
upper_color = np.array([data['upper-color-list'][0] + threshold, 255, 255])


class Rect:
    def __init__(self, rectangle, theta, area):
        self.tlx = rectangle[0]
        self.tly = rectangle[1]
        self.width = rectangle[2]
        self.height = rectangle[3]
        self.brx = self.tlx + self.width
        self.bry = self.tly + self.height
        self.cx = int((self.tlx + self.brx) / 2)
        self.cy = int((self.tly + self.bry) / 2)
        self.angle = theta
        self.cont_area = area


def get_rec(rec_l, theta_l, contour_l):
    big_cont = max(contour_l)
    big_index = contour_l.index(big_cont)
    rec = Rect(rec_l[big_index], theta_l[big_index], contour_l[big_index])

    rec_l.pop(big_index)
    theta_l.pop(big_index)
    contour_l.pop(big_index)
    return rec


def is_pair(ca, cb):
    if ca.angle < 0:
        print('is_pair() math: {}'.format(ca.angle + cb.angle))
        return 0 < abs(ca.angle + cb.angle) < 10
    else:
        return False


def calc_angle(con):
    _, _, theta = cv2.minAreaRect(con)
    if theta < -50:
        angle = abs(theta + 90)
    else:
        angle = round(theta)

    return angle


def calc_distance(ca, cb):
    avg_contour = (ca + cb) / 2
    if debug:
        print('Coeff A: {}, Coeff B: {}, Avg Contour Area: {}'.format(data['A'], data['B'], avg_contour))
    if model == 'power':
        return data['A'] * avg_contour ** data['B']
    elif model == 'exponential':
        return data['A'] * data['B'] ** avg_contour


def calc_yaw(pixel_x, center_x, h_foc_len):
    ya = math.degrees(math.atan((pixel_x - center_x) / h_foc_len))
    return round(ya)


def draw_points(rec_a, rec_b, acx, acy, color):
    cv2.line(frame, (rec_a.cx, rec_a.cy), (rec_a.cx, rec_a.cy), color, 8)
    cv2.line(frame, (rec_b.cx, rec_b.cy), (rec_b.cx, rec_b.cy), color, 8)
    cv2.line(frame, (acx, acy), (acx, acy), (255, 0, 0), 8)


def get_avg_points(rec_a, rec_b):
    avgcx = int((rec_a.cx + rec_b.cx) / 2)
    avgcy = int((rec_a.cy + rec_b.cy) / 2)
    if debug:
        print('Average Center (x , y): ({} , {})'.format(avgcx, avgcy))
    return avgcx, avgcy


def update_net_table(n, c1_x=-1, c1_y=-1, c2_x=-1, c2_y=-1, avgc_x=-1, avgc_y=-1, dis=-1):
    table.putNumber("center{}_x".format(n), c1_x)
    table.putNumber("center{}_y".format(n), c1_y)
    table.putNumber("center{}_x".format(n + 1), c2_x)
    table.putNumber("center{}_y".format(n + 1), c2_y)
    table.putNumber("center_x", avgc_x)
    table.putNumber("center_y", avgc_y)
    table.putNumber('distance_esti', dis)
    if debug:
        print("center{}_x".format(n), c1_x)
        print("center{}_y".format(n), c1_y)
        print("center{}_x".format(n + 1), c2_x)
        print("center{}_y".format(n + 1), c2_y)
        print("center_x", avgc_x)
        print("center_y", avgc_y)
        print('distance_esti', dis)


while True:
    print('=========================================================')
    if multi:
        frame = cap.read()
    else:
        _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    screen_c_x = data['image-width'] / 2 - 0.5
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    contour_area_arr = []
    theta_list = []
    rec_list = []
    for contour in contours:
        if cv2.contourArea(contour) > 75:
            print('Contour area:'.format(cv2.contourArea(contour)))
            theta_list.append(calc_angle(contour))
            contour_area_arr.append(cv2.contourArea(contour))
            ncontours.append(contour)
            rec_list.append(cv2.boundingRect(contour))
    print("Number of contours: ", len(ncontours))
    for contour in ncontours:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)
        if len(rec_list) > 1:
            print('Angles: {}'.format(theta_list))

            rec1 = get_rec(rec_list, theta_list, contour_area_arr)
            rec2 = get_rec(rec_list, theta_list, contour_area_arr)
            print('Is pair: {}'.format(is_pair(rec1, rec2)))
            avg_c1_x, avg_c1_y = get_avg_points(rec1, rec2)
            if is_pair(rec1, rec2):
                draw_points(rec1, rec2, avg_c1_x, avg_c1_y, (255, 0, 0))  # (255, 0, 0) == blue (bgr)
                distance = calc_distance(rec1.cont_area, rec2.cont_area)
                yaw = calc_yaw(avg_c1_x, screen_c_x, h_focal_length)
                if debug:
                    print('Distance = {} \t Yaw = {} \t is_pair() = {}'.format(distance, yaw, is_pair(rec1, rec2)))
                if net_table:
                    update_net_table(1, rec1.cx, rec1.cy, rec2.cx, rec2.cy, avg_c1_x, avg_c1_y, distance)

    if view:
        cv2.imshow('Contour Window', frame)
        cv2.imshow('Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
