## Tasks

- ~~Take lots of pictures at the camera's default values~~
- ~~Get camera calibration values/matrices from images~~
- ~~Store constants and camera matrix in values.json~~
- ~~Undistort frame using camera calibration values~~
- ~~Implement frame undistortion into main processing code~~
- ~~Implement solvePnP() code~~


## Random Notes

```python
model_points =
[
    # Left target
    (-5.938, 2.938, 0.0), # top left
    (-4.063, 2.375, 0.0), # top right
    (-5.438, -2.938, 0.0), # bottom left
    (-7.375, -2.500, 0.0), # bottom right

    # Right target
    (3.938, 2.375, 0.0), # top left
    (5.875, 2.875, 0.0), # top right
    (7.313, -2.500, 0.0), # bottom left
    (5.375, -2.938, 0.0), # bottom right
]

(ret, rvec, tvec) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
```
```
camera matrix format:
    c_ = optical center of axis
    f_ = focal length along axis

    f_x  0    c_x
    0    f_y  c_y
    0    0    1
```

## Relevant links and documentation

* Liger Bots White Paper
* https://www.chiefdelphi.com/t/finding-camera-location-with-solvepnp/159685/6
* https://www.chiefdelphi.com/t/how-to-know-your-robot-angle-compared-to-vision-taget/340707/10
* **https://www.chiefdelphi.com/t/vision-following-while-incoming-at-angle/345034/2**
* https://www.chiefdelphi.com/t/how-does-opencv-camera-calibration-work/346585/2
* https://www.chiefdelphi.com/t/camera-pose-estimation-help/137045/2
* https://www.chiefdelphi.com/t/using-pixy/343335
* https://www.chiefdelphi.com/t/solvepnp-angle-quesiton/348029
* https://www.chiefdelphi.com/t/calculating-a-distance-between-bot-and-point-on-image-from-camera-and-finding-an-angle-to-that-point/348580/2
* https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html
* https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp
* https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
* https://docs.opencv.org/trunk/db/d58/group__calib3d__fisheye.html#gga287a8008e05b4cc9a4dfbb1883a4e30da0899eaa2f96d6eed9927c4b4f4464e05
* https://gist.github.com/andrewda/f2461ebffdaabd7b37f3b4af15182acd
* https://github.com/ligerbots/VisionServer/blob/master/utils/camera_calibration.py
* https://github.com/team3997/ChickenVision/blob/master/ChickenVision.py
* https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
* **https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0**
* https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
* https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298

