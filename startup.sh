#!/bin/bash


source /home/pi/prof
workon cv
v4l2-ctl --set-ctrl exposure_auto=1
v4l2-ctl --set-ctrl exposure_absolute=4
v4l2-ctl --set-ctrl brightness=1

python3 /home/pi/GreenVision/GreenVision.py vision -d -mt -nt -f -th 30 -s 0 -f
