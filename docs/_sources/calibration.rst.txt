===========
Calibration
===========

Now that you've cloned and installed the requirements for GreenVision, you now need to change the values in **values.json** to match your current setup.

First of all, the server IP. This is the ip of the Robo-Rio, so that the program can connect to network tables and push values. Change this to match your IP.

.. code-block:: JSON

    {
      "server-ip": "10.18.16.2",
    }

Second, color range. You really shouldn't need to change these, but if contour recognition is not working with your green light, you may need to.

.. code-block:: JSON

    {
      "lower-color-list": [
        50.0,
        55.03597122302158,
        174.28057553956833
      ],
      "upper-color-list": [
        90.60606060606061,
        255,
        255
      ],
    }

Third, camera horizontal field of view. This should be what your camera says its FOV is in its tech specs.

.. code-block:: JSON

    {
      "fish-eye-cam-HFOV": 90,
    }

Finally, image resolution. The lower this is set, the less extra contours your camera will pick up. However, if you go too low, it won't pick up anything.

.. code-block:: JSON

    {
      "image-width": 640,
      "image-height": 480,
    }

Once you've tuned all of these, move on to the next section, usage.