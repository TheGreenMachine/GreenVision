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

logging.basicConfig(level=logging.DEBUG, filename='crash.log')
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
     
Available Programs:
vision              start vision processing
image       save frame from camera
video       save video from camera
calibration           generate json containing camera matrix and distortion values
""")


def program_usage():
    return 'Greenvision.py [vision] or [image] or [video] or [calibrate]'


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
    parser_vision.add_argument('-nt', '--networktables',
                               action='store_true',
                               help='toggle network tables')


def init_parser_image():
    parser_image = subparsers.add_parser('image')
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


def init_parser_video():
    parser_video = subparsers.add_parser('video')
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


def init_camera_calibration():
    parser_calibration = subparsers.add_parser('calibrate')
    parser_calibration.add_argument_group('Camera Calibration Arguments')
    parser_calibration.add_argument('--length', '-l',
                                    type=int,
                                    default=9,
                                    help='length of checkerboard (number of corners)')
    parser_calibration.add_argument('--width', '-w',
                                    type=int,
                                    default=6,
                                    help='width of checkerboard (number of corners)')
    parser_calibration.add_argument('--size', '-s',
                                    type=float,
                                    default=1.0,
                                    help='size of square')


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

    def calc_distance(c_y, screen_y, v_foc_len):

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
        temp = math.tan(math.radians(calc_pitch(c_y, screen_y, v_foc_len)))
        dist = math.fabs(h / temp) if temp != 0 else -1
        return dist

    def calc_pitch(c_y, screen_y, v_foc_len):
        pitch = math.degrees(math.atan((c_y - screen_y) / v_foc_len))
        return pitch

    def calc_yaw(c_x, screen_x, h_foc_len):
        ya = math.degrees(math.atan((c_x - screen_x) / h_foc_len))
        return round(ya)

    def draw_rect(rect, color):
        cv2.line(frame, (rect.box[0][0], rect.box[0][1]), (rect.box[1][0], rect.box[1][1]), color, 2)
        cv2.line(frame, (rect.box[1][0], rect.box[1][1]), (rect.box[2][0], rect.box[2][1]), color, 2)
        cv2.line(frame, (rect.box[2][0], rect.box[2][1]), (rect.box[3][0], rect.box[3][1]), color, 2)
        cv2.line(frame, (rect.box[3][0], rect.box[3][1]), (rect.box[0][0], rect.box[0][1]), color, 2)

    def undistort_frame(frame):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undst = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undst

    def solve_pnp(rect1, rect2, cy):

        model_points = [
            # Left target
            (-5.938, 2.938, 0.0),  # top left
            (-4.063, 2.375, 0.0),  # top right
            (-5.438, -2.938, 0.0),  # bottom left
            (-7.375, -2.500, 0.0),  # bottom right

            # Right target
            (3.938, 2.375, 0.0),  # top left
            (5.875, 2.875, 0.0),  # top right
            (7.313, -2.500, 0.0),  # bottom left
            (5.375, -2.938, 0.0),  # bottom right
        ]

        mp = np.array(model_points)

        image_points = np.concatenate((rect1.box, rect2.box))
        image_points[:, 0] -= data['image-width'] / 2
        image_points[:, 1] -= cy
        image_points[:, 1] *= -1

        ret, rvec, tvec = cv2.solvePnP(mp, image_points, K, D)

        x = tvec[0][0]
        y = tvec[1][0]
        z = tvec[2][0]

        distance = math.sqrt(x ** 2 + z ** 2)
        angle1 = math.atan2(x, z)
        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()
        pzero_world = np.matmul(rot_inv, -tvec)
        angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])
        if debug:
            print('Distance: {}, Angle1: {}, X: {}, Y: {}, Z: {}, cy: {}'.format(distance, angle1, angle2, x, y, z, cy))
        if net_table:
            pass
            # TODO: Implement table update depending on the data that Andrew wants
        return distance, angle1, angle2

    def draw_center_dot(cord, color):
        cv2.line(frame, cord, cord, color, 3)

    def update_net_table(avgc_x=-1, avgc_y=-1, yaaw=-1, dis=-1, conts=-1, targets=-1, pitch=-1):
        table.putNumber('center_x', avgc_x)
        table.putNumber('center_y', avgc_y)
        table.putNumber('yaw', yaaw)
        table.putNumber('distance_esti', dis)
        table.putNumber('contours', conts)
        table.putNumber('targets', targets)
        table.putNumber('pitch', pitch)

    src = int(args['source']) if args['source'].isdigit() else args['source']
    flip = args['flip']
    rotate = args['rotate']
    model = args['model']
    view = args['view']
    debug = args['debug']
    threshold = args['threshold'] if 0 < args['threshold'] < 50 else 0
    net_table = args['networktables']
    sequence = False

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
        table.putNumber('width', data['image-width'])
        table.putNumber('height', data['image-height'])

    if debug:
        print('----------------------------------------------------------------')
        print('Current Source: {}'.format(src))
        print('Vision Flag: {}'.format(view))
        print('Debug Flag: {}'.format(debug))
        print('Threshold Value: {}'.format(threshold))
        print('Function Model Value: {}'.format(model))
        print('Network Tables Flag: {}'.format(net_table))
        print('----------------------------------------------------------------')

    v_focal_length = data['camera_matrix'][1][1]
    h_focal_length = data['camera_matrix'][0][0]
    lower_color = np.array(data['lower-color-list']) - threshold
    upper_color = np.array([data['upper-color-list'][0] + threshold, 255, 255])
    center_coords = (int(data['image-width'] / 2), int(data['image-height'] / 2))
    screen_c_x = data['image-width'] / 2 - 0.5
    screen_c_y = data['image-height'] / 2 - 0.5

    DIM = tuple(data['dim'])
    K = np.array(data['camera_matrix'])
    D = np.array(data['distortion'])
    first_read = True
    try:
        while True:
            start = time.time()
            best_center_average_coords = (-1, -1)
            pitch = -1
            yaw = -1
            distance = -1
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

            print('=========================================================')

            if flip:
                frame = cv2.flip(frame, -1)
            if rotate:
                frame = imutils.rotate_bound(frame, 90)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            all_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index = 0
            filtered_contours = []
            rectangle_list = []
            average_cx_list = []
            average_cy_list = []
            sorted_contours = []
            biggest_contour_area = cv2.contourArea(
                max(all_contours, key=lambda x: cv2.contourArea(x) if cv2.contourArea(x) < 1750 else 0)) if len(
                all_contours) != 0 else 0
            print('Biggest Area: {}'.format(biggest_contour_area))
            for contour in all_contours:
                if cv2.contourArea(contour) > 50:
                    print('Unfiltered Contour Area: {}'.format(cv2.contourArea(contour)))
                if 50 < cv2.contourArea(contour) < 1750 and cv2.contourArea(contour) > 0.75 * biggest_contour_area:
                    filtered_contours.append(contour)
            if len(filtered_contours) != 0:
                print('Filtered Contour Area: {}'.format([cv2.contourArea(contour) for contour in filtered_contours]))
                sorted_contours, _ = sort_contours(filtered_contours)
                print('Sorted Contours Area: {}'.format([cv2.contourArea(contour) for contour in sorted_contours]))
            if len(sorted_contours) > 1:
                for contour in sorted_contours:
                    rectangle_list.append(Rect(contour))
                print('Rectangle List: {}'.format(len(rectangle_list)))
                for index, rect in enumerate(rectangle_list):
                    # positive angle means it's the left tape of a pair
                    print('Rect angle {}'.format(rect.theta))
                    if -82 < rect.theta < -74 and index != len(rectangle_list) - 1:
                        if view:
                            draw_rect(rect, (0, 255, 255))
                        # only add rect if the second rect is the correct angle
                        if -12 > rectangle_list[index + 1].theta > -20:
                            if view:
                                draw_rect(rectangle_list[index + 1], (0, 0, 255))
                            average_cx_list.append(int((rect.center[0] + rectangle_list[index + 1].center[0]) / 2))
                            average_cy_list.append(int((rect.center[1] + rectangle_list[index + 1].center[1]) / 2))
            if len(average_cx_list) == 1:
                best_center_average_coords = (average_cx_list[0], average_cy_list[0])
                pitch = calc_pitch(average_cy_list[0], screen_c_y, v_focal_length)
                yaw = calc_yaw(average_cx_list[0], screen_c_x, h_focal_length)
                if debug:
                    print('Contours: {}'.format(len(sorted_contours)))
                    print('Targets: {}'.format(len(average_cx_list)))
                    print('Avg_cx_list: {}'.format(average_cx_list))
                    print('Avg_cy_list: {}'.format(average_cy_list))
                    print('Best Center Coords: {}'.format(best_center_average_coords))
                    print('Index: {}'.format(0))
                    print('Distance: {}'.format(distance))
                    print('Pitch: {}'.format(pitch))
                    print('Yaw: {}'.format(yaw))

                if view:
                    cv2.line(frame, best_center_average_coords, center_coords, (0, 255, 0), 2)
                    for index, x in enumerate(average_cx_list):
                        draw_center_dot((x, average_cy_list[index]), (255, 0, 0))
            elif len(average_cx_list) > 1:
                # finds c_x that is closest to the center of the center
                best_center_average_x = min(average_cx_list, key=lambda x: abs(x - 320))
                index = average_cx_list.index(best_center_average_x)
                best_center_average_y = average_cy_list[index]

                best_center_average_coords = (best_center_average_x, best_center_average_y)
                pitch = calc_pitch(best_center_average_y, screen_c_y, v_focal_length)
                yaw = calc_yaw(best_center_average_x, screen_c_x, h_focal_length)
                if debug:
                    print('Contours: {}'.format(len(sorted_contours)))
                    print('Targets: {}'.format(len(average_cx_list)))
                    print('Avg_cx_list: {}'.format(average_cx_list))
                    print('Avg_cy_list: {}'.format(average_cy_list))
                    print('Best Center Coords: {}'.format(best_center_average_coords))
                    print('Index: {}'.format(index))
                    print('Distance: {}'.format(distance))
                    print('Pitch: {}'.format(pitch))
                    print('Yaw: {}'.format(yaw))
                if view:
                    cv2.line(frame, best_center_average_coords, center_coords, (0, 255, 0), 2)
                    for index, x in enumerate(average_cx_list):
                        draw_center_dot((x, average_cy_list[index]), (255, 0, 0))

            if net_table:
                update_net_table(best_center_average_coords[0], best_center_average_coords[1], yaw, distance,
                                 len(sorted_contours), len(average_cx_list), pitch)
            if view:
                cv2.imshow('Contour Window', frame)
                cv2.imshow('Mask', mask)
            with open('vision_log.csv', mode='a+') as vl_file:
                vl_writer = csv.writer(vl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                # vl_writer.writerow([str([cv2.contourArea(contour) for contour in all_contours]),
                #                    biggest_contour_area,
                #                    str([cv2.contourArea(contour) for contour in filtered_contours]),
                #                    str([cv2.contourArea(contour) for contour in sorted_contours]),
                #                    len(rectangle_list),  # num rectangles
                #                    len(sorted_contours),  # num contours
                #                    len(average_cx_list),  # num targets
                #                    str(average_cx_list),
                #                    str(average_cy_list),
                #                    index,
                #                    str(best_center_average_coords),
                #                    abs(320 - best_center_average_coords[0])])
                vl_writer.writerow(
                    [[cv2.contourArea(contour) for contour in all_contours if cv2.contourArea(contour) > 25],
                     biggest_contour_area,
                     [cv2.contourArea(contour) for contour in filtered_contours],
                     [cv2.contourArea(contour) for contour in sorted_contours],
                     len(rectangle_list),  # num rectangles
                     len(sorted_contours),  # num contours
                     len(average_cx_list),  # num targets
                     average_cx_list,
                     average_cy_list,
                     index,
                     best_center_average_coords,
                     abs(320 - best_center_average_coords[0])])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end = time.time()
            if debug:
                print('Execute time: {}'.format(end - start))
    except Exception as err:
        print('Crashed!: {}'.format(err))
        logging.exception('Heck: ')

    cap.release()
    cv2.destroyAllWindows()


def image_capture():
    cap = cv2.VideoCapture(args['src'])
    width = args['width']
    height = args['height']
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cv2.namedWindow('Image Capture')

    n = 0

    while True:
        print('Hold C to capture, Hold Q to quit')
        ret, frame = cap.read()
        cv2.imshow("Image Capture", frame)
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('c'):
            # file_name = input('Enter file name: ') + '.jpg'
            file_name = input('Enter file name: ') + '.jpg'
            cwd = os.path.join(os.getcwd(), 'Image_Capture/')  # /home/pi/Desktop/GreenVision/Image_Capture
            if not os.path.exists(cwd):
                os.makedirs(cwd)
            path = os.path.join(cwd, file_name)
            cv2.imwrite(path, frame)
            print("{} saved!".format(file_name))
            n += 1
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def video_capture():
    src = args['src']
    file_name = input('Enter file name: ')
    cwd = '/home/pi/GreenVison/Video_Capture'
    # cwd = os.path.join(os.getcwd(), 'Video_Capture')
    # /home/pi/Desktop/GreenVision/Video_Capture
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


def camera_calibration():
    CHECKERBOARD = (6, 9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    #
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    f_path = os.path.join(os.getcwd(), 'Camera_Calibration')
    images = glob.glob(f_path + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape is None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints,
                                                         imgpoints,
                                                         gray.shape[::-1],
                                                         K,
                                                         D,
                                                         rvecs,
                                                         tvecs,
                                                         calibration_flags,
                                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    print('Found {} valid images for calibration'.format(N_OK))
    print('Reprojection error = {}'.format(ret))
    print('Image center = ({:.2f}, {:.2f})'.format(mtx[0][2], mtx[1][2]))
    fov_x = math.degrees(2.0 * math.atan(data['image-width'] / 2.0 / mtx[0][0]))
    fov_y = math.degrees(2.0 * math.atan(data['image-height'] / 2.0 / mtx[1][1]))
    print('FOV = ({:.2f}, {:.2f}) degrees'.format(fov_x, fov_y))
    print('Mtx = {}\n'.format(mtx))
    print('Dist = {}\n'.format(dist))

    # Writing JSON data
    with open('calibration_output.json', 'w') as f:
        json.dump({"camera_matrix": mtx.tolist(),
                   "distortion": dist.tolist(),
                   "xfov": fov_x,
                   "yfov": fov_y,
                   "dim": gray.shape[::-1]}, f)


parser = argparse.ArgumentParser(description=program_description(), add_help=False)
parser.add_argument('-h', '--help', action='store_true')
subparsers = parser.add_subparsers(help='commands', dest='program')
init_parser_vision()
init_parser_image()
init_parser_video()
init_camera_calibration()

args = vars(parser.parse_args())
prog = args['program']
if args['help']: program_help()
if prog is None and not args['help']:
    print('No command selected, please rerun the script with the "-h" flag to see available commands.')
    print('To see the flags of each command, include a "-h" after choosing a command')
elif prog == 'vision':
    del args['program']
    del args['help']
    vision()
elif prog == 'image':
    del args['program']
    del args['help']
    image_capture()
elif prog == 'video':
    del args['program']
    del args['help']
    video_capture()
elif prog == 'calibration':
    del args['program']
    del args['help']
    camera_calibration()
