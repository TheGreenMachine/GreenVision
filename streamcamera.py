from flask_opencv_streamer.streamer import Streamer
import cv2
import pyzed.sl as sl
import numpy as np
from threading import Thread
from networktables import  NetworkTables
NetworkTables.initialize("10.18.16.2")
camera_table = NetworkTables.getTable("CameraPublisher")
table = camera_table.getSubTable("Camera")
table.getEntry("streams").setStringArray(["mjpg:http://10.18.16.142:3030/video_feed"])
port1 = 3030
port2 = 3031
require_login = False

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_VGA
init_params.camera_fps = 100

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

image_zed = sl.Mat(zed.get_resolution().width, zed.get_resolution().height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
image_ocv = image_zed.get_data()

runtime_parameters = sl.RuntimeParameters()

streamer = Streamer(port1, require_login)


while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT)
        # Use get_data() to get the numpy array
        image_ocv = image_zed.get_data()
        # Display the left image from the numpy array

    cv2.rectangle(image_ocv, (0, 0), (200, 200), (255, 0, 0), 2)
    streamer.update_frame(image_ocv)

    if not streamer.is_streaming:
        streamer.start_streaming()

    cv2.waitKey(30)