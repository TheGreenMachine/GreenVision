#!/bin/bassh

source ~/.profile
workon cv
v4l2-ctl --set-ctrl exposure_auto=1
v4l2-ctl --set-ctrl exposure_absolute=10
python3 /home/pi/GreenVision/GreenVision.py vision -d -mt -nt -r -th 30 -s 0
