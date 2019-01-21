#include <iostream>
#include <opencv2/opencv.hpp>
#include "RectCoordinates.h"

using namespace cv;

std::string help();

void parseArguments(int argc, char **argv);

RectCoordinates defineRec(Rect rectangle);

bool debug = false;
bool vision = false;

int main(int argc, char **argv) {
    parseArguments(argc, argv);

    VideoCapture capture;
    double lower[3] = {50.0, 55.03597122302158, 174.28057553956833};
    double upper[3] = {90.60606060606061, 255, 255};

    if (!capture.open(0)) {
        std::cout << "Failed to open video stream";
        return 1;
    }

    if (vision) {
        const std::string source_window = "Filter";
        namedWindow(source_window);
    }

    while (capture.isOpened()) {
        Mat frame;
        capture >> frame;

        if (frame.empty()) {
            std::cout << "Frame is empty, breaking";
            break;
        }

        cvtColor(frame, frame, COLOR_BGR2HSV);
        inRange(frame, (InputArray) lower, (InputArray) upper, frame);

        if (vision)
            imshow("Mask", frame);

        std::vector<std::vector<Point>> contours;
        std::vector<std::vector<Point>> fContours;
        findContours(frame, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        for (const auto &contour : contours) {
            if (contourArea(contour) > 75)
                fContours.push_back(contour);
        }

        if (debug)
            std::cout << "number of contours: " << fContours.size();

        std::vector<Rect> rectangles;
        for (const auto &contour : fContours) {
            Scalar scalar = Scalar(0, 0, 255);

            drawContours(frame, contour, -1, scalar, 2);
            rectangles.push_back(boundingRect(contour));
        }

        switch (rectangles.size()) {
            case 2:
                auto firstCoordinates = defineRec(rectangles.at(0));
                auto secondCoordinates = defineRec(rectangles.at(1));

                break;
            case 4:
                break;
            case 6:
                break;
        }
    }
};

String help() {
    return "Flag options: -d for debugging information, -v to show the windows";
}

void parseArguments(int argc, char **argv) {
    if (argc > 2) {
        std::cout << help();
    } else if (argc == 1 && argv[0][1] == 'h')
        help();
    else if (argc == 1) {
        debug = argv[0][1] == 'd';
        vision = argv[0][1] == 'v';
    } else if (argc == 2) {
        debug = argv[0][1] == 'd' || argv[1][1] == 'd';
        vision = argv[0][1] == 'v' || argv[1][1] == 'v';
    }
}

RectCoordinates defineRec(Rect rectangle) {
    float bottomRightX = rectangle.x + rectangle.width;
    float bottomRightY = rectangle.y + rectangle.width;
    int centerX = int((rectangle.x + bottomRightX) / 2);
    int centerY = int((rectangle.y + bottomRightY) / 2);

    return RectCoordinates(rectangle.x, rectangle.y, rectangle.width, rectangle.height, bottomRightX, bottomRightY,
                           centerX, centerY);
}