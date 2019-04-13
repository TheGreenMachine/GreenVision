#!/bin/bash
source /home/pi/prof
workon cv
sleep 3
cd GreenVision
python3 seecam_comp.py
v4l2-ctl --set-fmt-video=pixelformat=MJPG
v4l2-ctl --set-parm=120
if [ -e /dev/sda1 ]
then sudo mount -o rw,users,umask=000 /dev/sda1 /media/pi/GVLOGGING
	echo "Mounting logging USB..."
	sleep 1
	python3 GreenVision_NSR.py -s 0 -d -nt --pi -l
else
	echo "Failed to mount USB!"
	python3 GreenVision_NSR.py -s 0 -d -nt --pi
fi
