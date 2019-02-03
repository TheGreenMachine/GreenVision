=============
Video Capture
=============

The Video Capture module is used to record video form the camera with specific settings.

The Video Capture module can be used through the following command::

    python3 GreenVision.py video_capture [arguments]

The supported OPTIONAL arguments are::

    -f [int] or --fps [int]: Sets FPS for video. Default is 30.

    -cw [int] or --width [int]: Sets width of camera resolution. Default is defined as image-width in values.json.

    -ch [int] or --height [int]: Sets height of camera resolution. Default is defined as image-height in values.json.

    -n [string] or --name [string]: Sets a name for output file.

    -h: Shows command help.

The supported REQUIRED arguments are::

    -s [input] or -src [input] or --source [input]: Sets source for image processing - [int] for camera.
