import networktables as nt
import yaml
import cv2
import numpy as np
import math

class GreenVision:
    def __init__(self, settings):
        with open(settings) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            print(data)

        camera_types = ['ZED', 'BASIC']
        if data['camera']['type'] not in camera_types:
            raise ValueError('Invalid camera type. Expected one of: %s' % camera_types)
        self.flags = data['flags']
        self.view = False
        if self.flags['view'] == 'true':
            self.view = True
        self.camera = data['camera']
        self.nt = data['nt']
        self.colors = data['colors']
        self.contours = data['contours']
        self.angles = data['angles']
        self.cap = cv2.VideoCapture(self.camera['source'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera['height'])
        self.screen_c_x = self.camera['width'] / 2 - 0.5
        self.screen_c_y = self.camera['height'] / 2 - 0.5
        self.frame = 0
        self.biggest_contour_area = 0
        self.sorted_contours = 0
        self.cords_list = []
        self.append = self.cords_list.append


    def initNetworkTables(self):
        nt.NetworkTables.initialize(server=self.nt['server-ip'])
        table = nt.NetworkTables.getTable(self.nt['table-name'])
        if table:
            print('table OK')
        for var in self.nt['variables']:
            table.putNumber(var, -1)
    def capturePhotoAndFilter(self):
        lower_bound = np.array([self.colors['lower'][0], self.colors['lower'][1], self.colors['lower'][2]])
        upper_bound = np.array([self.colors['upper'][0], self.colors['upper'][1], self.colors['upper'][2]])
        threshold = self.colors['threshold']
        _, self.frame = self.cap.read()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.frame = cv2.inRange(self.frame, lower_bound, upper_bound)
    def findContours(self):
        all_contours, _ = cv2.findContours(self.frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in all_contours if self.contours['min-size'] < cv2.contourArea(c) < self.contours['max-size']]
        # find the contour with the biggest area so we can further remove contours created from light noise
        if len(all_contours) > 0:
            self.biggest_contour_area = max([cv2.contourArea(c) for c in all_contours])
        # create a contour list that removes contours smaller than the biggest * some constant
        self.sorted_contours = [c for c in filtered_contours if
                             cv2.contourArea(c) > self.contours['filter-threshold'] * self.biggest_contour_area]
        # sort contours by left to right, top to bottom
        if len(filtered_contours) > 1:
            bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
            sorted_contours, _ = zip(
                *sorted(zip(filtered_contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
            self.sorted_contours = list(sorted_contours)
    def genRectangleList(self):
        if len(self.sorted_contours) > 1:
            # gets ((cx, cy), (width, height), angle of rot) for each contour
            rectangle_list = [cv2.minAreaRect(c) for c in self.sorted_contours]
            for pos, rect in enumerate(rectangle_list):
                if self.biggest_contour_area < 10000:
                    if -78 - self.angles['angle_threshold'] < rect[2] < -74 + self.angles['angle_threshold'] and pos != len(rectangle_list) - 1:

                        if self.view:
                            color = (0, 255, 255)
                            box = np.int0(cv2.boxPoints(rect))
                            cv2.drawContours(self.frame, [box], 0, color, 2)
                        # only add rect if the second rect is the correct angle
                        if -16 - self.angles['angle_threshold'] < rectangle_list[pos + 1][2] < -12 + self.angles['angle_threshold']:
                            if self.view:
                                color = (0, 0, 255)
                                rect2 = rectangle_list[pos + 1]
                                box2 = np.int0(cv2.boxPoints(rect2))
                                cv2.drawContours(self.frame, [box2], 0, color, 2)
                            cx = int((rect[0][0] + rectangle_list[pos + 1][0][0]) / 2)
                            cy = int((rect[0][1] + rectangle_list[pos + 1][0][1]) / 2)
                            self.append((cx, cy))
                else:
                    if pos != len(rectangle_list) - 1:
                        if self.view:
                            color = (0, 255, 255)
                            box = np.int0(cv2.boxPoints(rect))
                            cv2.drawContours(self.frame, [box], 0, color, 2)
                            rect2 = rectangle_list[pos + 1]
                            box2 = np.int0(cv2.boxPoints(rect2))
                            color = (255, 255, 0)
                            cv2.drawContours(self.frame, [box2], 0, color, 2)
                        cx = int((rect[0][0] + rectangle_list[pos + 1][0][0]) / 2)
                        cy = int((rect[0][1] + rectangle_list[pos + 1][0][1]) / 2)
                        self.append((cx, cy))
    def checkCenterCords(self):
        if len(self.cords_list) == 1:
            best_center_average_coords = self.cords_list[0]
            yaw = math.degrees(math.atan((best_center_average_coords[0] - self.screen_c_x) / self.camera['horizontal-focal-length']))
            pitch = math.degrees(math.atan((best_center_average_coords[1] - self.screen_c_y) / self.camera['vertical-focal-length']))
            if self.view:
                cv2.line(self.frame, best_center_average_coords, self.center_coords, (0, 255, 0), 2)
                cv2.line(self.frame, best_center_average_coords, best_center_average_coords, (255, 0, 0), 5)

        elif len(self.cords_list) > 1:
            # finds c_x that is closest to the center of the center
            best_center_average_x = min(self.cords_list, key=lambda xy: abs(xy[0] - self.camera['width'] / 2))[0]
            index = [coord[0] for coord in self.cords_list].index(best_center_average_x)
            best_center_average_y = self.cords_list[index][1]
            best_center_average_coords = (best_center_average_x, best_center_average_y)
            yaw = math.degrees(math.atan((best_center_average_coords[0] - self.screen_c_x) / self.camera['horizontal-focal-length']))
            pitch = math.degrees(math.atan((best_center_average_coords[1] - self.screen_c_y) /self.camera['vertical-focal-length']))
            if self.view:
                cv2.line(self.frame, best_center_average_coords, self.center_coords, (0, 255, 0), 2)
                for coord in self.cords_list:
                    cv2.line(self.frame, coord, coord, (255, 0, 0), 5)
    def showWindow(self):
        # if view:
        cv2.imshow('Contour Window', self.frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
