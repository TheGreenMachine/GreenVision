import argparse
import json
import math
import os
import time
import glob
import cv2
import imutils
import networktables as nt
import numpy as np
import csv
import logging
import datetime
import getpass
import sys

cwd = os.getcwd()
file_path = os.path.join(cwd, 'values.json')
with open(file_path) as json_file:
    data = json.load(json_file)


def program_description():
    return 'Team 1816 Vision Processing Utility for the 2019 Deep Space Season'


def program_help():
    print(parser.description)
    print("""
Usage: GreenVision.py [program] [-optional arguments]
     
Available parameters:
WIP
""")


def init_parser_vision():
    parser.add_argument('-src', '--source',
                        required=True,
                        type=str,
                        help='set source for processing: [int] for camera, [path] for file')
    parser.add_argument('-r', '--rotate',
                        action='store_true',
                        default=False,
                        help='rotate 90 degrees')
    parser.add_argument('-f', '--flip',
                        action='store_true',
                        default=False,
                        help='flip camera image')
    parser.add_argument('-v', '--view',
                        action='store_true',
                        help='enable contour and mask window')
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='enable debug output to console')
    parser.add_argument('-th', '--threshold',
                        default=0,
                        type=int,
                        help='increases color thresholds by 50.0 or less')
    parser.add_argument('-nt', '--networktables',
                        action='store_true',
                        help='enable network tables')
    parser.add_argument('--pi',
                        action='store_true',
                        default=False,
                        help='must enable for the script to work the pi -- GVLogging USB must be plugged in')
    parser.add_argument('--crash',
                        action='store_true',
                        default=False,
                        help='enable to simulate a crash during vision loop')
    parser.add_argument('-l', '--log',
                        action='store_true',
                        default=False,
                        help='enable logging')


def vision():
    class Rect:
        def __init__(self, contur):
            center, wid_hei, theta = cv2.minAreaRect(contur)
            self.center = center
            self.width = wid_hei[0]
            self.height = wid_hei[1]
            self.theta = theta
            self.box = cv2.boxPoints(cv2.minAreaRect(contur))
            self.area = cv2.contourArea(contur)
            self.draw = np.int0(self.box)

    def sort_contours(cnts):
        bounding_boxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][0], reverse=False))
        return cnts, bounding_boxes

    def calc_distance(coord, screen_y, v_foc_len):

        # d = distance
        # h = height between camera and target
        # a = angle = pitch
        # tan a = h/d (opposite over adjacent)
        # d = h / tan a
        #                      .
        #                     /|
        #                    / |
        #                   /  |h
        #                  /a  |
        #           camera -----
        #                    d
        target_height = data['target-height']
        cam_height = data['camera-height']
        h = abs(target_height - cam_height)
        temp = math.tan(math.radians(calc_angles(coord)[1]))
        dist = math.fabs(h / temp) if temp != 0 else -1
        return dist

    def calc_angles(coords):
        yaw = math.degrees(math.atan((coords[0] - screen_c_x) / h_focal_length))
        pitch = math.degrees(math.atan((coords[1] - screen_c_y) / v_focal_length))
        return yaw, pitch

    def draw_rect(rect, color):
        cv2.line(frame, (rect.box[0][0], rect.box[0][1]), (rect.box[1][0], rect.box[1][1]), color, 2)
        cv2.line(frame, (rect.box[1][0], rect.box[1][1]), (rect.box[2][0], rect.box[2][1]), color, 2)
        cv2.line(frame, (rect.box[2][0], rect.box[2][1]), (rect.box[3][0], rect.box[3][1]), color, 2)
        cv2.line(frame, (rect.box[3][0], rect.box[3][1]), (rect.box[0][0], rect.box[0][1]), color, 2)

    def draw_center_dot(cord, color):
        cv2.line(frame, cord, cord, color, 5)

    def update_net_table(avgc_x=-1, avgc_y=-1, yaaw=-1, dis=-1, conts=-1, targets=-1, pitch=-1):
        table.putNumber('center_x', avgc_x)
        table.putNumber('center_y', avgc_y)
        table.putNumber('yaw', yaaw)
        table.putNumber('distance_esti', dis)
        table.putNumber('contours', conts)
        table.putNumber('targets', targets)
        table.putNumber('pitch', pitch)

    def get_system_stats():
        def cpu_temp():
            res = os.popen('vcgencmd measure_temp').readline()
            return res.replace("temp=", "").replace("'C\n", "")

        def cpu_usage():
            return str(os.popen("top -n1 | awk '/Cpu\(s\):/ {print $2}'").readline().strip())

        def ram_usage():
            """
            :return: [total RAM, used RAM, free RAM]
            """
            p = os.popen('free')
            i = 0
            while 1:
                i = i + 1
                line = p.readline()
                if i == 2:
                    return line.split()[1:4]

        if is_pi:
            return cpu_temp(), cpu_usage(), ram_usage()
        else:
            return -1, -1, -1

    def capture_frame(name):
        images_fp = os.path.join(log_fp, 'images')
        if not os.path.exists(images_fp):
            os.makedirs(images_fp)
        biggest_num = max([file[:-3] for file in os.listdir(images_fp)])
        fname = '{}{}.jpg'.format(name, biggest_num + 1)
        cv2.imwrite(images_fp, fname)
        if debug:
            print('Frame Captured!')

    def log_data():
        with open(os.path.join(log_fp, 'vision_log.csv'), mode='a+') as vl_file:
            vl_writer = csv.writer(vl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            vl_writer.writerow(
                [datetime.datetime.now(),
                 [cv2.contourArea(c) for c in all_contours if cv2.contourArea(contour) > 25],
                 biggest_contour_area,
                 [cv2.contourArea(c) for c in filtered_contours],
                 [cv2.contourArea(c) for c in sorted_contours],
                 len(rectangle_list),  # num rectangles
                 len(sorted_contours),  # num contours
                 len(average_coord_list),  # num targets
                 average_coord_list,
                 index,
                 best_center_average_coords,
                 abs(data['image-width'] / 2 - best_center_average_coords[0]),
                 get_system_stats(),
                 end - start,
                 image_written])

    def value_changed(table, key, value, isNew):
        global values
        values[key] = value
        if key == 'vision_active' and value:
            capture_frame('vision')

    def debug_output():
        return """
=========================================================
Unfiltered Contour Area: {}
Filtered Contour Area: {}
Sorted Contour Area: {}
Biggest Contour Area: {}
Rectangle List: {}
Contours: {}
Targets: {}
Avg_center_list: {}
Best Center Coords: {}
Index: {}
Distance: {}
Pitch: {}
Yaw: {}
Temp, Cpu, Ram: {}
Execute Time: {}\r"""

    src = int(args['source']) if args['source'].isdigit() else args['source']
    flip = args['flip']
    rotate = args['rotate']
    view = args['view']
    debug = args['debug']
    threshold = args['threshold'] if 0 < args['threshold'] < 50 else 0
    net_table = args['networktables']
    is_pi = args['pi']
    crash = args['crash']
    log = args['log']
    can_log = False
    sequence = False

    # sudo mount /dev/sda1 media/pi/GVLOGGING

    # logging_fp = '/media/{}/GVLOGGING/'.format(getpass.getuser()) if is_pi else os.path.join(os.getcwd(), 'Logs')
    # /media/pi/GVLOGGING or /home/[user]/Documents/GreenVision/Logs

    log_fp = os.path.join(os.getcwd(), 'Logs')
    if is_pi:
        log_fp = '/media/pi/GVLOGGING/'
        if not os.path.exists(log_fp):
            os.makedirs(log_fp)
        logging.basicConfig(level=logging.DEBUG,
                            filename=os.path.join(log_fp, 'crash.log'),
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        can_log = True
    elif not is_pi and getpass.getuser() != 'pi':
        if not os.path.exists(log_fp):
            os.makedirs(log_fp)
        logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_fp, 'crash.log'))
        can_log = True

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])
    cap.set(cv2.CAP_PROP_FPS, data['fps'])

    if net_table:
        nt.NetworkTables.initialize(server=data['server-ip'])
        table = nt.NetworkTables.getTable('SmartDashboard')
        if table:
            print('table OK')
        table.putNumber('center_x', -1)
        table.putNumber('center_y', -1)
        table.putNumber('contours', -1)
        table.putNumber('targets', -1)
        table.putNumber('width', data['image-width'])
        table.putNumber('height', data['image-height'])
        # values = {'vision_active': False}
        # table.addEntryListener(value_changed, key='vision_active')

    if debug:
        print('----------------------------------------------------------------')
        print('Current Source: {}'.format(src))
        print('Vision Flag: {}'.format(view))
        print('Debug Flag: {}'.format(debug))
        print('Threshold Value: {}'.format(threshold))
        print('Network Tables Flag: {}'.format(net_table))
        print('----------------------------------------------------------------')

    v_focal_length = data['camera_matrix'][1][1]
    h_focal_length = data['camera_matrix'][0][0]
    lower_color = np.array([
        data['lower-color-list'][0] - threshold,
        data['lower-color-list'][1],
        data['lower-color-list'][2]])  # HSV to test: 0, 220, 25
    upper_color = np.array([
        data['upper-color-list'][0] + threshold,
        data['upper-color-list'][1],
        data['upper-color-list'][2]])
    center_coords = (int(data['image-width'] / 2), int(data['image-height'] / 2))
    screen_c_x = data['image-width'] / 2 - 0.5
    screen_c_y = data['image-height'] / 2 - 0.5

    first_read = True
    try:
        filtered_contours = []
        rectangle_list = []
        sorted_contours = []
        average_coord_list = []
        can_log = os.path.exists(log_fp)
        while True:
            if crash:
                raise Exception('boop')
            start = time.time()
            best_center_average_coords = (-1, -1)
            pitch = -1
            yaw = -1
            distance = -1
            image_written = False
            if not first_read:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                if sequence and key != ord(' '):
                    continue

            first_read = False

            _, frame = cap.read()

            if frame is None:
                continue

            if flip:
                frame = cv2.flip(frame, -1)
            if rotate:
                frame = imutils.rotate_bound(frame, 90)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            all_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index = 0
            biggest_contour_area = cv2.contourArea(
                max(all_contours, key=lambda x: cv2.contourArea(x) if 50 < cv2.contourArea(x) < 2000 else 0)) if len(
                all_contours) != 0 else 0
            for contour in all_contours:
                if 50 < cv2.contourArea(contour) < 2000 and cv2.contourArea(contour) > 0.75 * biggest_contour_area:
                    filtered_contours.append(contour)
            if len(filtered_contours) > 1:
                sorted_contours, _ = sort_contours(filtered_contours)
                sorted_contours = list(sorted_contours)
            if len(sorted_contours) > 1:
                if len(sorted_contours) > 6:
                    capture_frame('toomany')
                    image_written = True
                for contour in sorted_contours:
                    rectangle_list.append(Rect(contour))
                for index, rect in enumerate(rectangle_list):
                    # positive angle means it's the left tape of a pair
                    if -82 < rect.theta < -74 and index != len(rectangle_list) - 1:
                        if view:
                            draw_rect(rect, (0, 255, 255))
                        # only add rect if the second rect is the correct angle
                        if -12 > rectangle_list[index + 1].theta > -20:
                            if view:
                                draw_rect(rectangle_list[index + 1], (0, 0, 255))

                            cx = int((rect.center[0] + rectangle_list[index + 1].center[0]) / 2)
                            cy = int((rect.center[1] + rectangle_list[index + 1].center[1]) / 2)
                            average_coord_list.append((cx, cy))

            if len(average_coord_list) == 1:
                best_center_average_coords = average_coord_list[0]
                yaw, pitch = calc_angles(best_center_average_coords)
                end = time.time()
                if log:
                    log_data()
                if debug:
                    sys.stdout.write(debug_output().format(
                        [cv2.contourArea(contour) for contour in all_contours if cv2.contourArea(contour) > 50],
                        [cv2.contourArea(contour) for contour in filtered_contours],
                        [cv2.contourArea(contour) for contour in sorted_contours],
                        biggest_contour_area,
                        len(rectangle_list),
                        len(sorted_contours),
                        len(average_coord_list),
                        average_coord_list,
                        best_center_average_coords,
                        0,
                        distance,
                        pitch,
                        yaw,
                        get_system_stats(),
                        end - start))

                if view:
                    cv2.line(frame, best_center_average_coords, center_coords, (0, 255, 0), 2)
                    for coord in average_coord_list:
                        draw_center_dot(coord, (255, 0, 0))
            elif len(average_coord_list) > 1:
                # finds c_x that is closest to the center of the center
                best_center_average_x = min(average_coord_list,
                                            key=lambda xy: abs(xy[0] - data['image-width'] / 2))[0]
                index = [coord[0] for coord in average_coord_list].index(best_center_average_x)
                best_center_average_y = average_coord_list[index][1]
                best_center_average_coords = (best_center_average_x, best_center_average_y)
                yaw, pitch = calc_angles(best_center_average_coords)
                end = time.time()
                if log:
                    log_data()
                if debug:
                    sys.stdout.write(debug_output().format(
                        [cv2.contourArea(contour) for contour in all_contours if cv2.contourArea(contour) > 50],
                        [cv2.contourArea(contour) for contour in filtered_contours],
                        [cv2.contourArea(contour) for contour in sorted_contours],
                        biggest_contour_area,
                        len(rectangle_list),
                        len(sorted_contours),
                        len(average_coord_list),
                        average_coord_list,
                        best_center_average_coords,
                        index,
                        distance,
                        pitch,
                        yaw,
                        get_system_stats(),
                        end - start))
                if view:
                    cv2.line(frame, best_center_average_coords, center_coords, (0, 255, 0), 2)
                    for coord in average_coord_list:
                        draw_center_dot(coord, (255, 0, 0))

            if net_table:
                update_net_table(best_center_average_coords[0], best_center_average_coords[1], yaw, distance,
                                 len(sorted_contours), len(average_coord_list), pitch)
            filtered_contours.clear()
            sorted_contours.clear()
            rectangle_list.clear()
            average_coord_list.clear()

            if view:
                cv2.imshow('Mask', mask)
                cv2.imshow('Contour Window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as err:
        if net_table:
            table.putNumber('contours', -99)
            table.putNumber('targets', -99)
        print('Vision has crashed! The error has been logged to {}. The error will now be displayed: \n{}'.format(
            log_fp, err))
        if can_log:
            logging.exception('Vision Machine Broke')

    cap.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description=program_description(), add_help=False)
parser.add_argument('-h', '--help', action='store_true')
init_parser_vision()
args = vars(parser.parse_args())
if args['help']:
    program_help()
else:
    del args['help']
    vision()
