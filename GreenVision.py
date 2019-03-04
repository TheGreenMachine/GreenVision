import numpy as np
import cv2
import networktables as nt
from imutils.video import WebcamVideoStream
import json
import math
import time
import argparse
import os
import pandas as pd
import imutils
import bisect

with open('./values.json') as json_file:
    data = json.load(json_file)


def program_description():
    return 'Team 1816 Vision Processing Utility for the 2019 Deep Space Season'


def program_help():
    print(parser.description)
    print("""
Usage: GreenVision.py [program] [-optional arguments]
     
Available Programs:
vision              start vision processing
image_capture       save frame from camera
video_capture       save video from camera
distance_table      generate CSV containing contour area and distance
""")


def program_usage():
    return 'Greenvision.py [vision] or [image_capture] or [video_capture] or [distance_table]'


def init_parser_vision():
    parser_vision = subparsers.add_parser('vision')
    parser_vision.add_argument('-src', '--source',
                               required=True,
                               type=str,
                               help='set source for processing: [int] for camera, [path] for file')
    parser_vision.add_argument('-r', '--rotate',
                               action='store_true',
                               default=False,
                               help='rotate 90 degrees')
    parser_vision.add_argument('-f', '--flip',
                               action='store_true',
                               default=False,
                               help='flip camera image')
    parser_vision.add_argument('-v', '--view',
                               action='store_true',
                               help='toggle contour and mask window')
    parser_vision.add_argument('-m', '--model',
                               type=str,
                               default='power',
                               help='choose function model for distance calculating: [power], [exponential]')
    parser_vision.add_argument('-d', '--debug',
                               action='store_true',
                               help='toggle debug output to console')
    parser_vision.add_argument('-th', '--threshold',
                               default=0,
                               type=int,
                               help='adjust color thresholds by 50.0 or less')
    parser_vision.add_argument('-mt', '--multithread',
                               action='store_true',
                               help='toggle multi-threading')
    parser_vision.add_argument('-nt', '--networktables',
                               action='store_true',
                               help='toggle network tables')


def init_parser_image():
    parser_image = subparsers.add_parser('image_capture')
    parser_image.add_argument('-s', '--src', '--source',
                              required=True,
                              type=int,
                              help='set source for processing: [int] for camera')
    parser_image.add_argument('-cw', '--width',
                              type=int,
                              default=data['image-width'],
                              help='set width of the camera resolution')
    parser_image.add_argument('-ch', '--height',
                              type=int,
                              default=data['image-height'],
                              help='set height of the camera resolution')
    parser_image.add_argument('-n', '--name',
                              type=str,
                              required=True,
                              default='opencv_image',
                              help='choose a different name for the file')


def init_parser_video():
    parser_video = subparsers.add_parser('video_capture')
    parser_video.add_argument_group('Video Capture Arguments')
    parser_video.add_argument('-s', '--src', '--source',
                              required=True,
                              type=int,
                              help='set source for processing: [int] for camera')
    parser_video.add_argument('-f', '--fps',
                              type=float,
                              default=30.0,
                              help='set fps of the video')
    parser_video.add_argument('-cw', '--width',
                              type=int,
                              default=data['image-width'],
                              help='set width of the camera resolution')
    parser_video.add_argument('-ch', '--height',
                              type=int,
                              default=data['image-height'],
                              help='set height of the camera resolution')
    parser_video.add_argument('-n', '--name',
                              type=str,
                              default='opencv_video',
                              required=True,
                              help='choose a different name for the file')


def init_parser_distance_table():
    parser_distance_table = subparsers.add_parser('distance_table')
    parser_distance_table.add_argument_group('Distance table Arguments')
    parser_distance_table.add_argument('-s', '--src', '--source',
                                       type=int,
                                       default=0,
                                       required=True,
                                       help='set source for processing: [int] for camera')
    parser_distance_table.add_argument('-c', '--capture',
                                       action='store_true',
                                       default=False,
                                       help='toggle capture of new images')
    parser_distance_table.add_argument('-th', '--threshold',
                                       default=0,
                                       type=float,
                                       help='adjust color thresholds by 50.0 or less')
    parser_distance_table.add_argument('-cw', '--width',
                                       type=int,
                                       default=data['image-width'],
                                       help='set width of the camera resolution')
    parser_distance_table.add_argument('-ch', '--height',
                                       type=int,
                                       default=data['image-height'],
                                       help='set height of the camera resolution')
    parser_distance_table.add_argument('-o', '--output',
                                       type=str,
                                       default='distance_table',
                                       help='choose name for csv file')


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

    def draw_points(rect, color):
        cv2.line(frame, (rect.box[0][0], rect.box[0][1]), (rect.box[1][0], rect.box[1][1]), color, 2)
        cv2.line(frame, (rect.box[1][0], rect.box[1][1]), (rect.box[2][0], rect.box[2][1]), color, 2)
        cv2.line(frame, (rect.box[2][0], rect.box[2][1]), (rect.box[3][0], rect.box[3][1]), color, 2)
        cv2.line(frame, (rect.box[3][0], rect.box[3][1]), (rect.box[0][0], rect.box[0][1]), color, 2)

    def update_net_table(avgc_x=-1, dis=-1):
        table.putNumber("center_x", avgc_x)
        # table.putNumber('distance_esti', dis)
        if debug:
            print("center_x", avgc_x)
            # print('distance_esti', dis)

    src = int(args['source']) if args['source'].isdigit() else args['source']
    flip = args['flip']
    rotate = args['rotate']
    model = args['model']
    view = args['view']
    debug = args['debug']
    threshold = args['threshold'] if 0 < args['threshold'] < 50 else 0
    multi = args['multithread']
    net_table = args['networktables']
    sequence = False
    if multi:
        cap = WebcamVideoStream(src)
        cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
        cap.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])
        if cap.stream.get(cv2.CAP_PROP_FRAME_COUNT) < 50:
            sequence = True
        cap.start()

    else:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, data['image-width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, data['image-height'])
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) < 50:
            sequence = True

    if net_table:
        nt.NetworkTables.initialize(server=data['server-ip'])
        table = nt.NetworkTables.getTable('SmartDashboard')
        if table:
            print('table OK')
        table.putNumber('visionX', -1)
        table.putNumber('visionY', -1)
        table.putNumber('width', data['image-width'])
        table.putNumber('height', data['image-height'])

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

    first_read = True

    while True:
        start = time.time()
        if not first_read:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if sequence and key != ord(' '):
                continue

        first_read = False

        if multi:
            frame = cap.read()
        else:
            _, frame = cap.read()

        if frame is None:
            continue

        print('=========================================================')

        if flip:
            frame = cv2.flip(frame, -1)
        if rotate:
            frame = imutils.rotate_bound(frame, 90)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        all_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        rectangle_list = []
        average_cx_list = []
        average_cy_list = []
        sorted_contours = []
        for contour in all_contours:
            if 50 < cv2.contourArea(contour) < 10000:
                filtered_contours.append(contour)
        if len(filtered_contours) != 0:
            sorted_contours, _ = sort_contours(filtered_contours)
        print("Number of contours: ", len(filtered_contours))
        if len(sorted_contours) > 1:
            for contour in sorted_contours:
                rectangle_list.append(Rect(contour))
            if len(rectangle_list) > 1:
                for index, rect in enumerate(rectangle_list):
                    # positive angle means it's the left tape of a pair
                    if abs(rect.theta) > 40 and index != len(rectangle_list) - 1:
                        draw_points(rect, (0, 255, 255))
                        # only add rect if the second rect is the correct pair
                        if abs(rectangle_list[index + 1].theta) < 40:
                            draw_points(rectangle_list[index + 1], (0, 0, 255))
                            average_cx_list.append(int((rect.center[0] + rectangle_list[index + 1].center[0]) / 2) + 1)
                            average_cy_list.append(int((rect.center[1] + rectangle_list[index + 1].center[1]) / 2) + 1)
        if len(average_cx_list) > 0:
            # finds c_x that is closest to the center of the center
            best_center_average_x = min(average_cx_list, key=lambda x: abs(x - 320))
            best_center_average_y = min(average_cy_list, key=lambda y: abs(y - 240))
            best_center_average = (best_center_average_x, best_center_average_y)
            # could use bisect algorithm to find best center if you want O(ln n) instead of O(n)
            # best_center_average = bisect.bisect((average_cx_list, 320))
        # if len(average_center_list) > 0:
        #     best_center_average = min(average_center_list[0], key=lambda x: abs(x - 320))
        #     print(type(best_center_average))
            # cv2.line(frame, (best_center_average, 0), (best_center_average, data['image-height']), (0, 255, 0), 2)
            cv2.line(frame, best_center_average, (int(data['image-width'] / 2), int(data['image-height'] / 2)), (0, 255, 0), 2)
            if net_table:
                update_net_table(best_center_average[0])

        if view:
            cv2.imshow('Contour Window', frame)
            cv2.imshow('Mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()
        print('Execute time: {}'.format(end - start))

    cap.release()
    cv2.destroyAllWindows()


def image_capture():
    cap = cv2.VideoCapture(args['src'])
    width = args['width']
    height = args['height']
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cv2.namedWindow('Image Capture')

    while True:
        print('Hold C to capture, Hold Q to quit')
        ret, frame = cap.read()
        cv2.imshow("Image Capture", frame)
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('c'):
            file_name = args['name']
            cwd = os.path.join(os.getcwd(), 'Image_Capture')  # /home/pi/Desktop/GreenVision/Image_Capture
            if not os.path.exists(cwd):
                os.makedirs(cwd)
            path = os.path.join(cwd, file_name)
            cv2.imwrite(path, frame)
            print("{} saved!".format(file_name))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def video_capture():
    src = args['src']
    file_name = args['name']
    cwd = os.path.join(os.getcwd(), 'Video_Capture')  # /home/pi/Desktop/GreenVision/Video_Capture
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    path = os.path.join(cwd, (file_name + '.avi'))
    fps = args['fps']
    res = (args['width'], args['height'])

    cap = cv2.VideoCapture(src)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path, fourcc, fps, res)
    print('Hold Q to stop recording and quit')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Video Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def distance_table():
    def capture():
        length = input('Hold camera still and enter distance in inches from target: ')
        img_name = os.path.join(cwd, 'distance_{}in.jpg'.format(length))
        cv2.imwrite(img_name, frame)
        print('{} saved!'.format(img_name))

    src = args['src']
    cwd = os.path.join(os.getcwd(), 'Distance_Table')  # /home/pi/Desktop/GreenVision/Distance_Table
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    threshold = args['threshold']
    lower_color = np.array(data['lower-color-list']) - threshold
    upper_color = np.array([data['upper-color-list'][0] + threshold, 255, 255])
    distance_arr = np.array([], dtype=np.float64)
    contour_area_arr = np.array([], dtype=np.float64)

    if args['capture']:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args['height'])
        while True:
            print('Press [C] to capture frame | Press [Q] to exit capture mode')
            ret, frame = cap.read()
            cv2.imshow('Image Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                capture()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    files = []
    for _, _, filenames in os.walk(cwd, topdown=False):
        files = filenames.copy()
    for file in files:
        if file.endswith('in.jpg'):
            inches = int(file[9:-6])
            img = cv2.imread(os.path.join(cwd, file))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            n_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > 75:
                    n_contours.append(contour)
            if len(n_contours) == 2:
                print('{} is a valid image!'.format(file))
                contour_area = cv2.contourArea(n_contours[0])
                contour_area1 = cv2.contourArea(n_contours[1])
                contour_area_average = (contour_area + contour_area1) / 2
                contour_area_arr = np.append(contour_area_arr, contour_area_average)
                distance_arr = np.append(distance_arr, inches)
            else:
                print('{} is not a valid image! Please retake the image from that distance.'.format(file))

    df = pd.DataFrame({'x': contour_area_arr, 'y': distance_arr})
    df.to_csv(os.path.join(cwd, 'Distance_Table.csv'), index=False)


parser = argparse.ArgumentParser(description=program_description(), add_help=False)
parser.add_argument('-h', '--help', action='store_true')
subparsers = parser.add_subparsers(help='commands', dest='program')
init_parser_vision()
init_parser_image()
init_parser_video()
init_parser_distance_table()

args = vars(parser.parse_args())
prog = args['program']
if args['help']:
    program_help()
if prog is None and not args['help']:
    print('No command selected, please rerun the script with the "-h" flag to see available commands.')
    print('To see the flags of each command, include a "-h" after choosing a command')
elif prog == 'vision':
    del args['program']
    del args['help']
    vision()
elif prog == 'image_capture':
    del args['program']
    del args['help']
    image_capture()
elif prog == 'video_capture':
    del args['program']
    del args['help']
    video_capture()
elif prog == 'distance_table':
    del args['program']
    del args['help']
    distance_table()
