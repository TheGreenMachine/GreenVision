######################################################################
# Automatically generated by qmake (3.1) Tue Dec 4 17:48:22 2018
######################################################################

TEMPLATE = app
TARGET = GreenVision
INCLUDEPATH += .

# The following define makes your compiler warn you if you use any
# feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

# Input
HEADERS += GreenVision/pch.h
SOURCES += GreenVision/pch.cpp \
    GreenVision/GreenVision_Live.cpp
INCLUDEPATH += /usr/local/include/opencv4
LIBS += -L/usr/local/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio

DISTFILES += \
    GreenVision/light.jpg \
    GreenVision/nolight.jpg
