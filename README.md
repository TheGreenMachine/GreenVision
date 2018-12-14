# GreenVision
Hello, and welcome to GreenVision, team 1816's Open Source C++ Vision Code. 

## Setup
In order to get setup with GreenVision, install one of the following:

* [Visual Studio Community](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15)
* [Qt Creator](https://www.qt.io/download-qt-installer?hsCtaTracking=9f6a2170-a938-42df-a8e2-a9f0b1d6cdce%7C6cb0de4f-9bb5-4778-ab02-bfb62735f3e5)

Then, follow each IDE's own guide below for setting up OpenCV.
## Qt Creator - Linux
In order to use OpenCV with Qt Creator, you must build OpenCV from source. The steps to do so are below:
1. First, make sure you have the required dependencies. Run the following(assuming debian based distro):

`sudo apt-get install build-essential`

`sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev`

2. Next, we are going to download OpenCV via git. Run the following:

`mkdir opencv`

`cd opencv`

`git clone https://github.com/opencv/opencv.git`

3. Now, we are going to start to build. Change into the OpenCV directory and run:

`mkdir build`

`cd build`

`cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..`

4. This will take a while. Once it's done, run:

`make -j10`

5. Once this is finished, run:

`sudo make install`

6. That's it! Now open the GreenVision.pro file in Qt Creator and it should work! Remember to only open one main cpp at a time.
