==============
Distance Table
==============

The Distance Table module reads files in from the Test Images folder following the default naming scheme of *opencv_image_{}*, and then
outputs a csv file with contour area on x and distance on y.

The Distance Table module can be used through the following command::

    python3 GreenVision.py distance_table [arguments]

The supported OPTIONAL arguments are::

    -c or --capture: Toggles capture of new images. Default is False.

    -cw [int] or --width [int]: Sets width of camera resolution. Default is defined as image-width in values.json.

    -ch [int] or --height [int]: Sets height of camera resolution. Default is defined as image-height in values.json.

    -o [string] or --output [string]: Sets a name for output csv file. Default is distance_table.

    -th [double] or --threshold [double]: Adjusts color thresholds by 50.0 or less. Use if recognition is not working.

    -h: Shows command help.

The supported REQUIRED arguments are::

    -s [input] or -src [input] or --source [input]: Sets source for image processing - [int] for camera.
