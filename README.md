# GreenVision
Hello, and welcome to GreenVision, Team 1816's Open Source C++ Vision Code. 

## Setup
In order to get setup with GreenVision, install one of the following:

* [Visual Studio Community](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15)
* [Qt Creator](https://www.qt.io/download-qt-installer?hsCtaTracking=9f6a2170-a938-42df-a8e2-a9f0b1d6cdce%7C6cb0de4f-9bb5-4778-ab02-bfb62735f3e5)
* [Clion](https://www.jetbrains.com/clion/)

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

## Visual Studio - Windows

1. Download OpenCV - We will not be building from source for windows to head to [OpenCV](https://opencv.org/releases.html) and download the latest release. Click on the self extracting archive, and move the files to where you want your libraries to be stored. I suggest the root of the C drive, so C:\opencv.

2. Next, you are going to want to add the bin directory to your system path. On Windows, go to Advanced System Settings > Enviroment Variables > System Variables > Path. Click "Edit", and then click "New". In the following prompt copy and paste the path to the bin folder in OpenCV.  For example, if you installed to C:\opencv, it would be C:\opencv\build\x64\vc14\bin

3. Open the project in Visual Studio. You should get some errors, that's normal.

4. Include OpenCV in Visual Studio. 
* First, make sure the solution platforn reads x64, not x86. 
* Next, open the project properties. Go to C/C++ > General, and copy and paste the path to the OpenCV include folder into the "Additional Library Directories" box, and click apply. For me, the path looks like C:\opencv\build\include.
* Now go to Linker > General in the properties pane. Copy and paste the path to the OpenCV lib folder in the "Additional Library Directories" box and hit apply.
* Finally, go to Input. Open your OpenCV directory and navigate to the lib folder. Inside there should be a .lib file that is similar to opencv_world341d.lib. Make sure to pick the one that ends in "d".
5. That's it! If you did everything correctly and rebuild, GreenVision should run!

## Clion - Windows
Clion with windows uses cmake to manage it's dependancies, for this you will need to build the WPI dependancies with gradle and import them into the project

1. [Clone or download allwpilib](https://github.com/wpilibsuite/allwpilib)

2. open your terminal of choice in the allwpilib root then run:

`./gradlew build` or `gradle build` depending on if you have gradle installed on your system. THE BUILD WILL TAKE TIME. Unfortinately there doesn't seem to be a way to individually build the wpiutil and ntcore, which are the two things you need.

3. After the build go into the `wpiutil/build/libs/wpiutil/static/x86-x64/release` folder and copy `wpiutil.lib` into `GreenVision/libs/wpiutil`

4. Do the same for ntcore: `ntcore/build/libs/ntcore/static/x86-x64/release` copy `ntcore.lib` into `GreenVision/libs/networktables`

5. Sync the project and build
