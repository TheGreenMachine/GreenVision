cmake_minimum_required(VERSION 3.12)
project(GreenVision)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR D:\\Robotics\\OpenCV-MinGW-Build\\x64\\mingw\\lib)
set(wpiutil_DIR ${PROJECT_SOURCE_DIR}/libs/wpiutil)
set(ntcore_DIR ${PROJECT_SOURCE_DIR}/libs/networktables)

find_package(OpenCV REQUIRED)

include_directories(${wpiutil_DIR}/include ${ntcore_DIR}/include)

add_executable(GreenVision GreenVision_Offline.cpp RectCoordinates.cpp RectCoordinates.h)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(GreenVision ${OpenCV_LIBS} ${wpiutil_DIR}/wpiutil.lib ${ntcore_DIR}/ntcore.lib)