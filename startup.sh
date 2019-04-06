#!/bin/bash
source /home/pi/prof
workon cv
sleep 3
cd GreenVision
python3 seecam_comp.py
if [ -e /dev/sda1 ];
then mount /dev/sda1 /media/pi/GVLOGGING;
python3 GreeVision.py -s 0 -f -nt --pi -l;
else
python3 GreenVision.py -s 0 -d -nt --pi;
fi