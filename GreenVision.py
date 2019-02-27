import numpy as np
import cv2
import networktables as nt
from imutils.video import WebcamVideoStream
import json
import math
import argparse
import os
import pandas as pd
import imutils
import threading

with open('/home/pi/GreenVision/values.json') as json_file:
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
                               type=str,
                               default='none',
                               help='rotate image to the [left], [right], or [flip]')
    # parser_vision.add_argument('-f', '--flip',
    #                            action='store_true',
    #                            default=False,
    #                            help='flip camera image')
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
    src = int(args['source']) if args['source'].isdigit() else args['source']
    flip = args['flip']
    rotate = args['rotate']
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
        cond = threading.Condition()
        notified = [False]

        def connection_listener(connected, info):
            print('{}; Connected={}'.format(info, connected))
            with cond:
                notified[0] = True
                cond.notify()

        nt.NetworkTables.initialize(server=data['server-ip'])
        nt.NetworkTables.addConnectionListener(connection_listener, immediateNotify=True)
        with cond:
            print('Waiting...')
            if not notified[0]:
                cond.wait()

        print('Connected!')
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
        if (ca.angle < 0 and cb.angle > 0) or (ca.angle > 0 and cb.angle < 0):
            print('is_pair() math: {}'.format(ca.angle + cb.angle))
            are_opposite = np.sign(ca.angle) != np.sign(cb.angle)
            return are_opposite
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
        yaw, yaw1, yaw2 = -99, -99, -99
        avg_c1_x, avg_c1_y, avg_c2_x, avg_c2_y, avg_c3_x, avg_c3_y = 0, 0, 0, 0, 0, 0
        rec1, rec2, rec3, rec4, rec5, rec6 = 0, 0, 0, 0, 0, 0

        print('=========================================================')
        if multi:
            frame = cap.read()
        else:
            _, frame = cap.read()

        # if flip:
        #     frame = cv2.flip(frame, -1)
        if not rotate == 'none':
            if rotate == 'left':
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotate == 'right':
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 'flip':
                frame = cv2.flip(frame, -1)
            else:
                raise ValueError('Incorrect rotate keyword!')
            # frame = imutils.rotate_bound(frame, 90)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        screen_c_x = data['image-width'] / 2 - 0.5
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ncontours = []

        contour_area_arr = []

        theta_list = []
        rec_list = []
        for contour in contours:
            if 50 < cv2.contourArea(contour) < 6000:
                theta_list.append(calc_angle(contour))
                contour_area_arr.append(cv2.contourArea(contour))
                cont_input = contour_area_arr.copy()
                ncontours.append(contour)
                rec_list.append(cv2.boundingRect(contour))
        print("Number of contours: ", len(ncontours))
        if len(rec_list) == 0:
            update_net_table(1)
            continue
        for contour in ncontours:
            # cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)
            rec_list1 = rec_list.copy()
            if len(rec_list) > 1:

                print('Angles: {}'.format(theta_list))

                rec1 = get_rec(rec_list, theta_list, cont_input)
                rec2 = get_rec(rec_list, theta_list, cont_input)
                print(contour_area_arr)
                print('Is pair: {}'.format(is_pair(rec1, rec2)))
                avg_c1_x, avg_c1_y = get_avg_points(rec1, rec2)
                if is_pair(rec1, rec2):
                    # draw_points(rec1, rec2, avg_c1_x, avg_c1_y, (255, 0, 0))  # (255, 0, 0) == blue (bgr)
                    distance = calc_distance(rec1.cont_area, rec2.cont_area)
                    yaw = calc_yaw(avg_c1_x, screen_c_x, h_focal_length)
                    if debug:
                        print('Distance = {} \t Yaw = {} \t is_pair() = {}'.format(distance, yaw, is_pair(rec1, rec2)))
                    if net_table:
                        update_net_table(1, rec1.cx, rec1.cy, rec2.cx, rec2.cy, avg_c1_x, avg_c1_y, distance)
                if len(rec_list) > 3:
                    rec3 = get_rec(rec_list, theta_list, cont_input)
                    rec4 = get_rec(rec_list, theta_list, cont_input)
                    avg_c2_x, avg_c2_y = get_avg_points(rec3, rec4)
                    yaw1 = calc_yaw(avg_c2_x, screen_c_x, h_focal_length)
                    if is_pair(rec3, rec4):
                        # draw_points(rec3, rec4, avg_c2_x, avg_c2_y, (0, 204, 204))  # (0, 204, 204) == yellow (bgr)
                        if net_table:
                            update_net_table(2, rec3.cx, rec3.cy, rec4.cx, rec4.cy, avg_c2_x, avg_c2_y)

                    if len(rec_list) > 5:
                        rec5 = get_rec(rec_list, theta_list, cont_input)
                        rec6 = get_rec(rec_list, theta_list, cont_input)
                        avg_c3_x, avg_c3_y = get_avg_points(rec5, rec6)
                        yaw2 = calc_yaw(avg_c3_x, screen_c_x, h_focal_length)
                        if is_pair(rec5, rec6):
                            if net_table:
                                update_net_table(3, rec5.cx, rec5.cy, rec6.cx, rec6.cy, avg_c3_x,
                                                 avg_c3_y)
                            # draw_points(rec5, rec6, avg_c3_x, avg_c3_y, (255, 0, 255))  # (255, 0, 255) == fuschia
                yaw_list = [math.fabs(yaw), math.fabs(yaw1), math.fabs(yaw2)]
                if yaw_list[0] != yaw_list[1] or yaw_list[1] != yaw_list[2]:
                    yaw_selection = min(yaw_list)
                else:
                    continue
                if yaw_selection == yaw_list[0] and is_pair(rec1, rec2) and len(rec_list1) > 1:
                    draw_points(rec1, rec2, avg_c1_x, avg_c1_y, (255, 0, 0))  # (255, 0, 0) == blue (bgr)
                elif yaw_selection == yaw_list[1] and is_pair(rec3, rec4) and len(rec_list1) > 3:
                    draw_points(rec3, rec4, avg_c2_x, avg_c2_y, (0, 204, 204))  # (0, 204, 204) == yellow (bgr)
                elif yaw_selection == yaw_list[2] and is_pair(rec5, rec6) and len(rec_list1) > 5:
                    draw_points(rec5, rec6, avg_c3_x, avg_c3_y, (255, 0, 255))  # (255, 0, 255) == fuschia

        if view:
            cv2.putText(frame, 'Yaws: {} {} {}'.format(yaw, yaw1, yaw2), (10, 500), cv2.FONT_HERSHEY_COMPLEX, .5,
                        (255, 255, 255), 2)
            cv2.putText(frame, 'Contour areas: {}'.format(contour_area_arr), (10, 600), cv2.FONT_HERSHEY_COMPLEX, .5,
                        (255, 255, 255), 2)
            cv2.imshow('Contour Window', frame)
            cv2.imshow('Mask', mask)
            cv2.waitKey(30)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


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
