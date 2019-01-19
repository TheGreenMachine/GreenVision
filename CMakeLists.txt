cmake_minimum_required(VERSION 3.12)
project(GreenVision)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR D:\\Robotics\\OpenCV-MinGW-Build\\x64\\mingw\\lib)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/libs/wpiutil/include ${PROJECT_SOURCE_DIR}/libs/networktables/include)

add_executable(GreenVision GreenVision_Offline.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(GreenVision ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/libs/wpiutil/lib/wpiutil.lib ${PROJECT_SOURCE_DIR}/libs/networktables/ntcore.lib)