cmake_minimum_required(VERSION 3.12)
project(GreenVision)

set(CMAKE_CXX_STANDARD 17)

set(wpiutil_DIR ${PROJECT_SOURCE_DIR}/libs/wpiutil)
set(ntcore_DIR ${PROJECT_SOURCE_DIR}/libs/networktables)
set(json_DIR ${PROJECT_SOURCE_DIR}/libs/json)

find_package(OpenCV REQUIRED)

include_directories(${wpiutil_DIR}/include ${ntcore_DIR}/include ${json_DIR})
find_library(ntcore ${ntcore_DIR}/ntcore.lib)
find_library(wpiutil ${wpiutil_DIR}/wpituil.lib)

#add_executable(GreenVision GreenVision_Offline.cpp RectCoordinates.cpp RectCoordinates.h)
#add_executable(GreenVision GreenVision_Live.cpp RectCoordinates.cpp RectCoordinates.h)
add_executable(GreenVision GreenVision.cpp RectCoordinates.cpp RectCoordinates.h)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(GreenVision ${wpiutil_DIR}/wpiutil.lib  ${ntcore_DIR}/ntcore.lib ${OpenCV_LIBS})