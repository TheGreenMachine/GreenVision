#include <iostream>
#include <opencv2/opencv.hpp>
#include "RectCoordinates.h"

using namespace cv;

std::string help();

void parseArguments(int argc, char **argv);

RectCoordinates defineRec(Rect rectangle);

int getAverageX(int center1, int center2);

int getAverageY(int center1, int center2);

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

        Scalar center = Scalar(255, 0, 0);
        switch (rectangles.size()) {
            case 0:
                if (debug)
                    std::cout << "No contours detected";
                break;
            case 2: {
                auto firstCoordinates = defineRec(rectangles.at(0));
                auto secondCoordinates = defineRec(rectangles.at(1));

                int averageCenterX = getAverageX(firstCoordinates.getCenterX(), secondCoordinates.getCenterY());
                int averageCenterY = getAverageY(firstCoordinates.getCenterY(), secondCoordinates.getCenterY());

                line(frame, Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()),
                     Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()), center, 8);
                line(frame, Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()),
                     Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()), center, 8);
                line(frame, Point(averageCenterX, averageCenterY), Point(averageCenterX, averageCenterY), center, 8);
                break;
            }
            case 4: {
                auto firstCoordinates = defineRec(rectangles.at(0));
                auto secondCoordinates = defineRec(rectangles.at(1));
                auto thirdCoordinates = defineRec(rectangles.at(2));
                auto fourthCoordinates = defineRec(rectangles.at(3));

                int averageCenterX = getAverageX(firstCoordinates.getCenterX(), secondCoordinates.getCenterY());
                int averageCenterY = getAverageY(firstCoordinates.getCenterY(), secondCoordinates.getCenterY());
                int averageCenterX1 = getAverageX(thirdCoordinates.getCenterX(), fourthCoordinates.getCenterY());
                int averageCenterY1 = getAverageY(thirdCoordinates.getCenterY(), fourthCoordinates.getCenterY());

                line(frame, Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()),
                     Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()), center, 8);
                line(frame, Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()),
                     Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()), center, 8);
                line(frame, Point(thirdCoordinates.getCenterX(), thirdCoordinates.getCenterY()),
                     Point(thirdCoordinates.getCenterX(), thirdCoordinates.getCenterY()), center, 8);
                line(frame, Point(fourthCoordinates.getCenterX(), fourthCoordinates.getCenterY()),
                     Point(fourthCoordinates.getCenterX(), fourthCoordinates.getCenterY()), center, 8);
                line(frame, Point(averageCenterX, averageCenterY), Point(averageCenterX, averageCenterY), center, 8);
                line(frame, Point(averageCenterX1, averageCenterY1), Point(averageCenterX1, averageCenterY1), center, 8);
                break;
            }
            case 6: {
                auto firstCoordinates = defineRec(rectangles.at(0));
                auto secondCoordinates = defineRec(rectangles.at(1));
                auto thirdCoordinates = defineRec(rectangles.at(2));
                auto fourthCoordinates = defineRec(rectangles.at(3));
                auto fifthCoordinates = defineRec(rectangles.at(4));
                auto sixthCoordinates = defineRec(rectangles.at(4));

                int averageCenterX = getAverageX(firstCoordinates.getCenterX(), secondCoordinates.getCenterY());
                int averageCenterY = getAverageY(firstCoordinates.getCenterY(), secondCoordinates.getCenterY());
                int averageCenterX1 = getAverageX(thirdCoordinates.getCenterX(), fourthCoordinates.getCenterY());
                int averageCenterY1 = getAverageY(thirdCoordinates.getCenterY(), fourthCoordinates.getCenterY());
                int averageCenterX2 = getAverageX(fifthCoordinates.getCenterX(), sixthCoordinates.getCenterY());
                int averageCenterY2 = getAverageY(fifthCoordinates.getCenterY(), sixthCoordinates.getCenterY());

                line(frame, Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()),
                     Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()), center, 8);
                line(frame, Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()),
                     Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()), center, 8);
                line(frame, Point(thirdCoordinates.getCenterX(), thirdCoordinates.getCenterY()),
                     Point(thirdCoordinates.getCenterX(), thirdCoordinates.getCenterY()), center, 8);
                line(frame, Point(fourthCoordinates.getCenterX(), fourthCoordinates.getCenterY()),
                     Point(fourthCoordinates.getCenterX(), fourthCoordinates.getCenterY()), center, 8);
                line(frame, Point(fifthCoordinates.getCenterX(), fifthCoordinates.getCenterY()),
                     Point(fifthCoordinates.getCenterX(), fifthCoordinates.getCenterY()), center, 8);
                line(frame, Point(sixthCoordinates.getCenterX(), sixthCoordinates.getCenterY()),
                     Point(sixthCoordinates.getCenterX(), sixthCoordinates.getCenterY()), center, 8);
                line(frame, Point(averageCenterX, averageCenterY), Point(averageCenterX, averageCenterY), center, 8);
                line(frame, Point(averageCenterX1, averageCenterY1), Point(averageCenterX1, averageCenterY1), center, 8);
                line(frame, Point(averageCenterX2, averageCenterY2), Point(averageCenterX2, averageCenterY2), center, 8);
                break;
            }
            default:
                if (debug)
                    std::cout << "Number of rectangles: " << rectangles.size();
        }

        if (vision)
            imshow("Contour Window", frame);

        char c = (char) waitKey(10);
        if (c == 27)
            break;
    }

    if (capture.isOpened())
        capture.release();

    return 0;
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

int getAverageX(int center1, int center2) {
    return (center1 + center2) / 2;
}

int getAverageY(int center1, int center2) {
    return (center1 + center2) / 2;
}