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

def vision():
    src = 0
    flip = False
    rotate = False
    view = True
    debug = True
    threshold = 30
    #angle_threshold = args['athreshold'] if 0 < args['athreshold'] < 30 else 0
    #filter_threshold = .5
    net_table = True
    is_pi = False
    crash = False
    log = False
    window_moved = False
    sequence = False

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])

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
    rectangle_list = []
    sorted_contours = []
    average_coord_list = []
    append = average_coord_list.append
    # can_log = os.path.exists(log_fp)
    # if log and can_log:
        # vl_file = open(os.path.join(log_fp, 'vision_log.csv'), mode='a+')
        # vl_writer = csv.writer(vl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    while True:
        fps = FPS().start()
        if crash:
            raise Exception('Get bamboozled...')
        start_time = time.time()
        biggest_contour_area = -1
        best_center_average_coords = (-1, -1)
        index = -1
        pitch = -999
        yaw = -999

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
        filtered_contours = [c for c in all_contours if 10000 < cv2.contourArea(c) < 100000]

        filtered_contours_area = [cv2.contourArea(c) for c in all_contours if 50 < cv2.contourArea(c)]

        # find the contour with the biggest area so we can further remove contours created from light noise
        if len(all_contours) > 0:
            biggest_contour_area = max([cv2.contourArea(c) for c in all_contours])
        # create a contour list that removes contours smaller than the biggest * some constant

        filtered_contours = [c for c in filtered_contours if
                                cv2.contourArea(c) == biggest_contour_area]
        for c in filtered_contours:
          print(cv2.contourArea(c))
        # sort contours by left to right, top to bottom
        # if len(filtered_contours) > 0:
        #     bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
        #     sorted_contours, _ = zip(
        #         *sorted(zip(filtered_contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
        #     sorted_contours = list(sorted_contours)

        if len(filtered_contours) > 0:
            # gets ((cx, cy), (width, height), angle of rot) for each contour
            rectangle_list = [cv2.minAreaRect(c) for c in filtered_contours]
            for pos, rect in enumerate(rectangle_list):
                if biggest_contour_area < 100000:
                    if True:

                        if view:
                            color = (0, 255, 255)
                            box = np.int0(cv2.boxPoints(rect))
                            cv2.drawContours(frame, [box], 0, color, 2)

                            cx = int((rect[0][0]))
                            cy = int((rect[0][1]))
                            append((cx, cy))

        if len(average_coord_list) == 1:
            best_center_average_coords = average_coord_list[index]
            index = 0
            yaw = math.degrees(math.atan((best_center_average_coords[0] - screen_c_x) / h_focal_length))
            pitch = math.degrees(math.atan((best_center_average_coords[1] - screen_c_y) / v_focal_length))
            if view:
                cv2.line(frame, best_center_average_coords, center_coords, (0, 255, 0), 2)
                cv2.line(frame, best_center_average_coords, best_center_average_coords, (255, 0, 0), 5)

        if view:
            cv2.imshow('Mask', mask)
            cv2.imshow('Contour Window', frame)
            if not window_moved:
                cv2.moveWindow('Mask', 300, 250)
                cv2.moveWindow('Contour Window', 1100, 250)
                window_moved = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()

        fps.update()
        fps.stop()
        curr_fps = fps.fps()
        if debug:
            sys.stdout.write("""
=========================================================
Filtered Contour Area: {}
Sorted Contour Area: {}
Biggest Contour Area: {}
Rectangle List: {}
Contours: {}
Targets: {}
Avg_center_list: {}
Best Center Coords: {}
Index: {}
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
                            pitch,
                            yaw,
                            curr_fps,
                            end_time - start_time))

        if net_table:
            table.putNumber('center_x', best_center_average_coords[0])
            table.putNumber('center_y', best_center_average_coords[1])
            table.putNumber('yaw', yaw)
            table.putNumber('contours', len(sorted_contours))
            table.putNumber('targets', len(average_coord_list))
            table.putNumber('pitch', pitch)
            table.putNumber('fps', curr_fps)
        filtered_contours.clear()
        sorted_contours.clear()
        rectangle_list.clear()
        average_coord_list.clear()

    # except Exception as err:
    #     if net_table:
    #         table.putNumber('contours', -99)
    #         table.putNumber('targets', -99)
#        print('Vision has crashed! The error has been logged to {}. The error will now be displayed: \n{}'.format(
#            log_fp, err))
#        if can_log:
#            logging.exception('Vision Machine Broke')

    cap.release()
    cv2.destroyAllWindows()
vision()
