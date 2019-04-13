import argparse
import json
import math
import os
import time
import cv2
import imutils
from imutils.video import FPS
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
    sequence = False

    # sudo mount /dev/sda1 media/pi/GVLOGGING

    # logging_fp = '/media/{}/GVLOGGING/'.format(getpass.getuser()) if is_pi else os.path.join(os.getcwd(), 'Logs')
    # /media/pi/GVLOGGING or /home/[user]/Documents/GreenVision/Logs

    log_fp = os.path.join(os.getcwd(), 'Logs')
    if is_pi:
        print('Log running in PI mode...')
        log_fp = '/media/pi/GVLOGGING'
        logging.basicConfig(level=logging.DEBUG,
                            filename=os.path.join(log_fp, 'crash.log'),
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    elif not is_pi and getpass.getuser() != 'pi':
        print('Log running in laptop mode...')
        if not os.path.exists(log_fp):
            os.makedirs(log_fp)
        logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_fp, 'crash.log'))

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
        print('View Flag: {}'.format(view))
        print('Debug Flag: {}'.format(debug))
        print('Threshold Value: {}'.format(threshold))
        print('Network Tables Flag: {}'.format(net_table))
        print('----------------------------------------------------------------\n')

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
        all_contours_area = []
        rectangle_list = []
        sorted_contours = []
        average_coord_list = []
        append = average_coord_list.append
        can_log = os.path.exists(log_fp)
        if log and can_log:
            vl_file = open(os.path.join(log_fp, 'vision_log.csv'), mode='a+')
            vl_writer = csv.writer(vl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        while True:
            fps = FPS().start()
            if crash:
                raise Exception('Get bamboozled...')
            start_time = time.time()
            biggest_contour_area = -1
            best_center_average_coords = (-1, -1)
            index = -1
            distance = -1
            pitch = -999
            yaw = -999
            image_written = False

            if view:
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
            # find contours from mask
            all_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # remove super small or super big contours that exist due to light noise/objects
            all_contours_area = [c for c in all_contours if 50 < cv2.contourArea(c)]
            filtered_contours = [c for c in all_contours if 50 < cv2.contourArea(c) < 15000]

            filtered_contours_area = [cv2.contourArea(c) for c in all_contours if 50 < cv2.contourArea(c)]

            # find the contour with the biggest area so we can further remove contours created from light noise
            if len(all_contours) > 0:
                biggest_contour_area = max([cv2.contourArea(c) for c in all_contours])
            # create a contour list that removes contours smaller than the biggest * some constant

            filtered_contours = [c for c in filtered_contours if cv2.contourArea(c) > 0.5 * biggest_contour_area]

            # sort contours by left to right, top to bottom
            if len(filtered_contours) > 1:
                bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
                sorted_contours, _ = zip(
                    *sorted(zip(filtered_contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
                sorted_contours = list(sorted_contours)

            if len(sorted_contours) > 1:
                # gets ((cx, cy), (width, height), angle of rot) for each contour
                rectangle_list = [cv2.minAreaRect(c) for c in sorted_contours]
                for pos, rect in enumerate(rectangle_list):
                    angle_constant = 15
                    if biggest_contour_area < 10000:
                        if -77 - angle_constant < rect[2] < -75 + angle_constant and pos != len(rectangle_list) - 1:

                            if view:
                                color = (0, 255, 255)
                                box = np.int0(cv2.boxPoints(rect))
                                cv2.drawContours(frame, [box], 0, color, 2)
                            # only add rect if the second rect is the correct angle
                            if -13 - angle_constant < rectangle_list[pos + 1][2] < -15 + angle_constant:
                                if view:
                                    color = (0, 0, 255)
                                    rect2 = rectangle_list[pos + 1]
                                    box2 = np.int0(cv2.boxPoints(rect2))
                                    cv2.drawContours(frame, [box2], 0, color, 2)
                                cx = int((rect[0][0] + rectangle_list[pos + 1][0][0]) / 2)
                                cy = int((rect[0][1] + rectangle_list[pos + 1][0][1]) / 2)
                                append((cx, cy))
                    else:
                        if pos != len(rectangle_list) - 1:
                            if view:
                                color = (0, 255, 255)
                                box = np.int0(cv2.boxPoints(rect))
                                cv2.drawContours(frame, [box], 0, color, 2)
                                rect2 = rectangle_list[pos + 1]
                                box2 = np.int0(cv2.boxPoints(rect2))
                                color = (255, 255, 0)
                                cv2.drawContours(frame, [box2], 0, color, 2)
                            cx = int((rect[0][0] + rectangle_list[pos + 1][0][0]) / 2)
                            cy = int((rect[0][1] + rectangle_list[pos + 1][0][1]) / 2)
                            append((cx, cy))

            if len(average_coord_list) == 1:
                best_center_average_coords = average_coord_list[index]
                index = 0
                yaw = math.degrees(math.atan((best_center_average_coords[0] - screen_c_x) / h_focal_length))
                pitch = math.degrees(math.atan((best_center_average_coords[1] - screen_c_y) / v_focal_length))
                if view:
                    cv2.line(frame, best_center_average_coords, center_coords, (0, 255, 0), 2)
                    cv2.line(frame, best_center_average_coords, best_center_average_coords, (255, 0, 0), 5)

            elif len(average_coord_list) > 1:
                # finds c_x that is closest to the center of the center
                t1 = time.time()
                best_center_average_coords = np.amin(average_coord_list, axis=0)
                t2 = time.time()
                print('Method 1: {} in {} seconds'.format(best_center_average_coords, ((t2 - t1)*10**5)))
                t1 = time.time()
                best_center_average_x = min(average_coord_list, key=lambda xy: abs(xy[0] - data['image-width'] / 2))[0]
                t2 = time.time()
                print('Method 2: {} in {} seconds'.format(best_center_average_coords, ((t2 - t1)*10**5)))
                index = [coord[0] for coord in average_coord_list].index(best_center_average_x)
                best_center_average_y = average_coord_list[index][1]
                best_center_average_coords = (best_center_average_x, best_center_average_y)
                yaw = math.degrees(math.atan((best_center_average_coords[0] - screen_c_x) / h_focal_length))
                pitch = math.degrees(math.atan((best_center_average_coords[1] - screen_c_y) / v_focal_length))
                if view:
                    cv2.line(frame, best_center_average_coords, center_coords, (0, 255, 0), 2)
                    for coord in average_coord_list:
                        cv2.line(frame, coord, coord, (255, 0, 0), 5)

            if net_table:
                table.putNumber('center_x', best_center_average_coords[0])
                table.putNumber('center_y', best_center_average_coords[1])
                table.putNumber('yaw', yaw)
                table.putNumber('distance_esti', distance)
                table.putNumber('contours', len(sorted_contours))
                table.putNumber('targets', len(average_coord_list))
                table.putNumber('pitch', pitch)

            if view:
                cv2.imshow('Mask', mask)
                cv2.imshow('Contour Window', frame)
                cv2.moveWindow('Mask', 300, 250)
                cv2.moveWindow('Contour Window', 1100, 250)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end_time = time.time()

            fps.update()
            fps.stop()
            if debug:
                sys.stdout.write("""
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
FPS: {}
Execute time: {}\r""".format(filtered_contours_area,
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
                             fps.fps(),
                             end_time - start_time))
                if log and can_log:
                    vl_writer.writerow(
                        [datetime.datetime.now(),
                         filtered_contours_area,
                         biggest_contour_area,
                         len(sorted_contours),  # num contours
                         len(rectangle_list),  # num rectangles
                         len(average_coord_list),  # num targets
                         average_coord_list,
                         index,
                         best_center_average_coords,
                         abs(data['image-width'] / 2 - best_center_average_coords[0]),
                         fps.fps(),
                         end_time - start_time])
                    vl_file.flush()

            filtered_contours.clear()
            sorted_contours.clear()
            rectangle_list.clear()
            average_coord_list.clear()

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
