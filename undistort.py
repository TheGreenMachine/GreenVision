import numpy as np
import sys
import cv2
import json
import os

fpath = os.path.join(os.getcwd(), 'calibration_output.json')
with open(fpath) as json_file:
    vals = json.load(json_file)

DIM = tuple(vals['dim'])
K = np.array(vals['camera_matrix'])
D = np.array(vals['distortion'])


def undistort(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow('distorted', img)
    print('distorted dim: {}'.format(img.shape[::-1]))
    cv2.imshow('undistorted', undistorted_img)
    print('undistorted dim: {}'.format(undistorted_img.shape[::-1]))
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
