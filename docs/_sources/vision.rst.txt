=================
Vision Processing
=================

The Vision Processing module is used on the actual pi during matches to detect retro-reflective tape and send data back to Network Tables.

The Vision Processing module can be used through the following command::

    python3 GreenVision.py vision [arguments]

The supported OPTIONAL arguments are::

    -v or --view: Toggle contour and mask window. This allows you to see what the pi is seeing.

    -d or --debug: Toggles debug output to console. This shows coordinates of all the contours, as well as additional data.

    -mt or --multithread: Toggles multi-threading. Increases processing speed.

    -th [double] or --threshold [double]: Adjusts color thresholds by 50.0 or less. Use if recognition is not working.

    -nt or --networktables: Toggles sending data to Network Tables. Default is false.

    -h: Shows command help.

The supported REQUIRED arguments are::

    -src [input] or --source [input]: Sets source for image processing - [int] for camera, [path] for file.
