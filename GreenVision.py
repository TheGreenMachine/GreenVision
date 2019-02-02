import numpy as np
import cv2
import networktables as nt
from imutils.video import WebcamVideoStream
import json
import math
import time
import argparse
import os

with open('values.json') as json_file:
    data = json.load(json_file)


def init_parser_vision(sp):
    parser_vision = sp.add_parser('vision')
    parser_vision.add_argument('-s', '--src', '--source',
                               required=True,
                               type=str,
                               help='set source for processing: [int] for camera, [file path] for image/video')
    parser_vision.add_argument('-v', '--view',
                               action='store_true',
                               help='toggle contour and mask window')
    parser_vision.add_argument('-d', '--debug',
                               action='store_true',
                               help='toggle debug output to console')
    parser_vision.add_argument('-th', '--threshold',
                               default=0,
                               type=int,
                               help='adjust thresholds for lower_color and upper_color by 50 or less')
    parser_vision.add_argument('-mt', '--multithread',
                               action='store_true',
                               help='toggle multi-threading')
    parser_vision.add_argument('-nt', '--networktables',
                               action='store_true',
                               help='toggle network tables')


def init_parser_image(sp):
    parser_image = sp.add_parser('image_capture')
    parser_image.add_argument('-s', '--src', '--source',
                              required=True,
                              type=int,
                              help='set source for processing: [int] for camera')
    parser_image.add_argument('-d', '--distance',
                              type=int,
                              required=True,
                              help='set distance of target in inches')
    parser_image.add_argument('-p', '--path',
                              type=str,
                              default='home/pi/Desktop/GreenVision/Test_Images',
                              help='choose a different path to store file')
    parser_image.add_argument('-n', '--name',
                              type=str,
                              default='opencv_image_{}',
                              help='choose a different name for the file')


def init_parser_video(sp):
    parser_video = sp.add_parser('video_capture')
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
    parser_video.add_argument('-p', '--path',
                              type=str,
                              default='home/pi/Desktop/GreenVision/Test_Images',
                              help='choose a different path to store file')
    parser_video.add_argument('-n', '--name',
                              type=str,
                              default='opencv_video',
                              required=True,
                              help='choose a different name for the file')


def init_parser_cali_distance(sp):
    parser_cali_distance = sp.add_parser('calibrate_distance')
    parser_cali_distance.add_argument_group('Calibrate Distance Arguments')
    parser_cali_distance.add_argument('-s', '--src', '--source',
                                      type=int,
                                      default=0,
                                      required=True,
                                      help='set source for processing: [int] for camera')
    parser_cali_distance.add_argument('-c', '--capture',
                                      action='store_true',
                                      help='toggle capture of new images')
    parser_cali_distance.add_argument('-n', '--number',
                                      type=int,
                                      help='set number of images to take', )
    parser_cali_distance.add_argument('-o', '--output',
                                      type=str,
                                      default='distance_calibration_dump',
                                      help='choose name for output file')


def vision():
    src = args['src']
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
        table = nt.NetworkTables.getTable("SmartDashboard")
        if table:
            print("table OK")
        table.putNumber("visionX", -1)
        table.putNumber("visionY", -1)

    if debug:
        print('----------------------------------------------------------------')
        print('Current Source: {}'.format(src))
        print('Vision Flag: {}'.format(view))
        print('Debug Flag: {}'.format(debug))
        print('Threshold Value: {}'.format(threshold))
        print('Multi-Thread Flag: {}'.format(multi))
        print('Network Tables Flag: {}'.format(net_table))
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
        top = area - data['b']
        full = top / data['m']
        return full

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
        if debug:
            print('Average Center (x , y): ({acx} , {acy})'.format(acx=avg_center_x, acy=avg_center_y))
        return avg_center_x, avg_center_y

    def is_pair(tl1_x, tl2_x, br1_x, br2_x):
        top_diff = abs(tl1_x - tl2_x)
        bottom_diff = abs(br1_x - br2_x)
        if debug:
            print('Top diff: {td}\nBottom diff: {bd}'.format(td=top_diff, bd=bottom_diff))
        return bottom_diff > top_diff

    def update_net_table(n, c1_x=-1, c1_y=-1, c2_x=-1, c2_y=-1, avgc_x=-1, avgc_y=-1):
        table.putNumber("center{n}X".format(n=n), c1_x)
        table.putNumber("center{n}Y".format(n=n), c1_y)
        table.putNumber("center{n}X".format(n=n + 1), c2_x)
        table.putNumber("center{n}Y".format(n=n + 1), c2_y)
        table.putNumber("averagedCenterX", avgc_x)
        table.putNumber("averagedCenterY", avgc_y)
        if debug:
            print("center{n}X".format(n=n), c1_x)
            print("center{n}Y".format(n=n), c1_y)
            print("center{n}X".format(n=n + 1), c2_x)
            print("center{n}Y".format(n=n + 1), c2_y)
            print("averagedCenterX", avgc_x)
            print("averagedCenterY", avgc_y)

    while True:
        print('=========================================================')
        starttime = time.time()
        if multi:
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
                    if net_table:
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
                        if net_table:
                            update_net_table(2, rec3['c_x'], rec3['c_y'], rec4['c_x'], rec4['c_y'], avg_c2_x, avg_c2_y)
                        draw_points(rec3, rec4, avg_c2_x, avg_c2_y)

                    if len(rec_list) > 5:
                        rec5 = def_rec(rec_list[4])
                        rec6 = def_rec(rec_list[5])
                        avg_c3_x, avg_c3_y = get_avg_points(rec5, rec6)
                        if True:
                            if net_table:
                                update_net_table(3, rec5['c_x'], rec5['c_y'], rec6['c_x'], rec6['c_y'], avg_c3_x,
                                                 avg_c3_y)
                            draw_points(rec5, rec6, avg_c3_x, avg_c3_y)
        print("Elasped Time: {}".format(time.time() - starttime))
        if view:
            cv2.imshow('Contour Window', frame)
            cv2.imshow('Mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def image_capture():
    cap = cv2.VideoCapture(args['src'])
    cv2.namedWindow('Image Capture')

    while True:
        print('Hold C to capture, Hold Q to quit')
        ret, frame = cap.read()
        cv2.imshow("test", frame)
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('c'):
            distance = args['distance']
            path = args['path']
            img_name = os.path.join(path, (args['name'] + 'in.jpg').format(distance))
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def video_capture():
    cap = cv2.VideoCapture(args['src'])
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_name = args['name']
    path = args['path']
    fps = args['fps']
    res = (args['width'], args['height'])
    out = cv2.VideoWriter(os.path.join(path, (output_name + '.avi')), fourcc, fps, res)
    print('Hold Q to stop recording and quit')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def calibrate_distance():
    pass


parser = argparse.ArgumentParser(description='Team 1816 Vision Processing Utility for the 2019 Deep Space Season',
                                 usage='Greenvision.py [vision] or [image_capture] or [video_capture] or [calibrate_distance]',
                                 add_help=False)

subparsers = parser.add_subparsers(help='commands', dest='program')
init_parser_vision(subparsers)
init_parser_image(subparsers)
init_parser_video(subparsers)
init_parser_cali_distance(subparsers)

args = vars(parser.parse_args())

if len(args) == 0:
    print('No command selected, please rerun the script with the "-h" flag to see available commands.')
    print('To see the flags of each command, include a "-h" choosing a command')
elif 'vision' in args.values():
    del args['program']
    vision()
elif 'image_capture' in args.values():
    del args['program']
    image_capture()
elif 'video_capture' in args.values():
    del args['program']
    video_capture()
elif 'calibrate_distance' in args.values():
    del args['program']
    calibrate_distance()
