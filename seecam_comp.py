import pyv4l2

from pyv4l2.frame import Frame
from pyv4l2.control import Control

control = Control("/dev/video0")
control.get_controls()
control.get_control_value(9963776)

# get all default settings
settings = control.get_controls()

for setting in settings:
    print("Value {} = {}".format(setting['name'], setting['value']))

# set desired values
print()
print("------ setting values ----")
control.set_control_value(9963776, 0)  # brightness
control.set_control_value(9963777, 32)  # contrast
control.set_control_value(10094849, 1)  # exposure_auto
control.set_control_value(10094850, 3)  # exposure_absolute
control.set_control_value(9963778, 64)  # saturation
control.set_control_value(9963779, 0)  # hue
control.set_control_value(9963792, 200)  # gamma
control.set_control_value(9963802, 3250)  # white_balance_temperature
control.close()
print("------    DONE    ----")

# COMP CAM (as of 3/19/19)
#                     brightness (int)    : min=-64 max=64 step=1 default=-8193 value=-64
#                       contrast (int)    : min=0 max=64 step=1 default=57343 value=32
#                     saturation (int)    : min=0 max=128 step=1 default=57343 value=64
#                            hue (int)    : min=-40 max=40 step=1 default=-8193 value=0
# white_balance_temperature_auto (bool)   : default=1 value=1
#                          gamma (int)    : min=72 max=500 step=1 default=57343 value=200
#                           gain (int)    : min=0 max=100 step=1 default=57343 value=0
#           power_line_frequency (menu)   : min=0 max=2 default=1 value=2
#                0: Disabled
#                1: 50 Hz
#                2: 60 Hz
#      white_balance_temperature (int)    : min=2800 max=6500 step=1 default=57343 value=3250 flags=inactive
#                      sharpness (int)    : min=0 max=6 step=1 default=57343 value=2
#         backlight_compensation (int)    : min=0 max=2 step=1 default=57343 value=1
#                  exposure_auto (menu)   : min=0 max=3 default=0 value=1
#                1: Manual Mode
#                3: Aperture Priority Mode
#              exposure_absolute (int)    : min=1 max=5000 step=1 default=157 value=20
#         exposure_auto_priority (bool)   : default=0 value=1

# ----------------------------------------------------------------------------------------------------------------- #
#                     brightness (int)    : min=0 max=15 step=1 default=-8193 value=1
#                       contrast (int)    : min=0 max=15 step=1 default=57343 value=15
#                     saturation (int)    : min=0 max=15 step=1 default=57343 value=15
#                            hue (int)    : min=-10 max=10 step=1 default=-8193 value=-10
# white_balance_temperature_auto (bool)   : default=1 value=1
#                          gamma (int)    : min=1 max=10 step=1 default=57343 value=10
#                           gain (int)    : min=0 max=0 step=0 default=20478 value=0
#           power_line_frequency (menu)   : min=0 max=2 default=2 value=2
#                0: Disabled
#                1: 50 Hz
#                2: 60 Hz
#      white_balance_temperature (int)    : min=2800 max=6500 step=1 default=57343 value=2800 flags=inactive
#                      sharpness (int)    : min=0 max=15 step=1 default=57343 value=15
#         backlight_compensation (int)    : min=0 max=1 step=1 default=57343 value=1
#                  exposure_auto (menu)   : min=0 max=3 default=0 value=1
#                1: Manual Mode
#                3: Aperture Priority Mode
#              exposure_absolute (int)    : min=4 max=5000 step=1 default=625 value=4
#                 focus_absolute (int)    : min=0 max=21 step=1 default=57343 value=16 flags=inactive
#                     focus_auto (bool)   : default=1 value=1
#
