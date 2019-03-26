#!/bin/bash

source /home/pi/prof
workon cv
#v4l2-ctl --set-ctrl exposure_auto=1
v4l2-ctl --set-ctrl exposure_absolute=10
#v4l2-ctl --set-ctrl brightness=1
python3 /home/pi/seecam.py
sleep 3
cd GreenVision
#python3 /home/pi/GreenVision/GreenVision.py vision -d -mt -nt -f -th 30 -s 0
python3 GreenVision.py vision -d -nt -th 15 -s 0
