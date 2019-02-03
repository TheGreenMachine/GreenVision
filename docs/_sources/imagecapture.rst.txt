=============
Image Capture
=============

The Image Capture module is used to capture a picture form the camera with specific settings.

The Image Capture module can be used through the following command::

    python3 GreenVision.py image_capture [arguments]

The supported OPTIONAL arguments are::

    -cw [int] or --width [int]: Sets width of camera resolution. Default is defined as image-width in values.json.

    -ch [int] or --height [int]: Sets height of camera resolution. Default is defined as image-height in values.json.

    -n [string] or --name [string]: Sets a name for output file. Can use {} to input distance.

    -h: Shows command help.

The supported REQUIRED arguments are::

    -s [input] or -src [input] or --source [input]: Sets source for image processing - [int] for camera.

    -d [int] or --distance [int]: Sets distance from target in inches. Required for naming.
