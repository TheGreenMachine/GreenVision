#include <iostream>
#include <opencv2/opencv.hpp>
#include <networktables/NetworkTableInstance.h>
#include <json.hpp>
#include "RectCoordinates.h"

using namespace cv;

std::string help();

void parseArguments(int argc, char **argv);

void updateNetTables(int n, std::shared_ptr<nt::NetworkTable> &table, int center1X = -1, int center1Y = -1,
                     int center2X = -1, int center2Y = -1, int averageCenterX = -1, int averageCenterY = -1);

RectCoordinates defineRec(Rect rectangle);

int getAverageX(int center1, int center2);

int getAverageY(int center1, int center2);

bool debug = false;
bool vision = false;

int main(int argc, char **argv) {
    parseArguments(argc, argv);
    VideoCapture capture;

    const std::vector<double> lower = {50.0, 55.03597122302158, 174.28057553956833};
    const std::vector<double> upper = {90.60606060606061, 255, 255};

    auto inst = nt::NetworkTableInstance::GetDefault();
    auto table = inst.GetTable("SmartDashboard");

    if (inst.IsConnected()) {
        if (debug)
            std::cout << "Successfully connected to network tables server: " << inst.GetConnections().at(0).remote_ip;

        table->PutNumber("visionX", -1);
        table->PutNumber("visionY", -1);
    }

    if (!capture.open(0)) {
        std::cout << "Failed to open video stream";
        return 1;
    }

    while (capture.isOpened()) {
        Mat frame;
        capture >> frame;

        if (vision)
            imshow("Raw", frame);
        if (frame.empty()) {
            std::cout << "Frame is empty, continuing";
            continue;
        }

        Mat convert;
        cvtColor(frame, convert, COLOR_BGR2HSV);
        inRange(convert, lower, upper, convert);

        std::vector<std::vector<Point>> contours;
        std::vector<std::vector<Point>> fContours;
        findContours(convert, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        for (const auto &contour : contours) {
            if (contourArea(contour) > 75)
                fContours.push_back(contour);
        }

        if (debug)
            std::cout << "number of contours: " << fContours.size();

        std::vector<Rect> rectangles;

        for (const auto &contour : fContours) {
            Scalar scalar = Scalar(0, 0, 255);

            drawContours(convert, contours, -1, scalar, 2);
            rectangles.push_back(boundingRect(contour));
        }

        Scalar center = Scalar(255, 0, 0);
        switch (rectangles.size()) {
            case 0:
                if (debug)
                    std::cout << "No contours detected";
                break;
            case 6: {
                auto fifthCoordinates = defineRec(rectangles.at(4));
                auto sixthCoordinates = defineRec(rectangles.at(4));

                int averageCenterX2 = getAverageX(fifthCoordinates.getCenterX(), sixthCoordinates.getCenterY());
                int averageCenterY2 = getAverageY(fifthCoordinates.getCenterY(), sixthCoordinates.getCenterY());

                line(convert, Point(fifthCoordinates.getCenterX(), fifthCoordinates.getCenterY()),
                     Point(fifthCoordinates.getCenterX(), fifthCoordinates.getCenterY()), center, 8);
                line(convert, Point(sixthCoordinates.getCenterX(), sixthCoordinates.getCenterY()),
                     Point(sixthCoordinates.getCenterX(), sixthCoordinates.getCenterY()), center, 8);
                line(convert, Point(averageCenterX2, averageCenterY2), Point(averageCenterX2, averageCenterY2),
                     center,
                     8);

                updateNetTables(1, table, fifthCoordinates.getCenterX(), fifthCoordinates.getCenterY(),
                                sixthCoordinates.getCenterX(), sixthCoordinates.getCenterY(), averageCenterX2,
                                averageCenterY2);
            }
            case 4: {
                auto thirdCoordinates = defineRec(rectangles.at(2));
                auto fourthCoordinates = defineRec(rectangles.at(3));

                int averageCenterX1 = getAverageX(thirdCoordinates.getCenterX(), fourthCoordinates.getCenterY());
                int averageCenterY1 = getAverageY(thirdCoordinates.getCenterY(), fourthCoordinates.getCenterY());

                line(convert, Point(thirdCoordinates.getCenterX(), thirdCoordinates.getCenterY()),
                     Point(thirdCoordinates.getCenterX(), thirdCoordinates.getCenterY()), center, 8);
                line(convert, Point(fourthCoordinates.getCenterX(), fourthCoordinates.getCenterY()),
                     Point(fourthCoordinates.getCenterX(), fourthCoordinates.getCenterY()), center, 8);
                line(convert, Point(averageCenterX1, averageCenterY1), Point(averageCenterX1, averageCenterY1),
                     center,
                     8);

                updateNetTables(2, table, thirdCoordinates.getCenterX(), thirdCoordinates.getCenterY(),
                                fourthCoordinates.getCenterX(), fourthCoordinates.getCenterY(), averageCenterX1,
                                averageCenterY1);
            }
            case 2: {
                auto firstCoordinates = defineRec(rectangles.at(0));
                auto secondCoordinates = defineRec(rectangles.at(1));

                int averageCenterX = getAverageX(firstCoordinates.getCenterX(), secondCoordinates.getCenterY());
                int averageCenterY = getAverageY(firstCoordinates.getCenterY(), secondCoordinates.getCenterY());

                line(convert, Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()),
                     Point(firstCoordinates.getCenterX(), firstCoordinates.getCenterY()), center, 8);
                line(convert, Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()),
                     Point(secondCoordinates.getCenterX(), secondCoordinates.getCenterY()), center, 8);
                line(convert, Point(averageCenterX, averageCenterY), Point(averageCenterX, averageCenterY), center,
                     8);

                updateNetTables(1, table, firstCoordinates.getCenterX(), firstCoordinates.getCenterY(),
                                secondCoordinates.getCenterX(), secondCoordinates.getCenterY(), averageCenterX,
                                averageCenterY);
                break;
            }
            default:
                if (debug)
                    std::cout << "Number of rectangles: " << rectangles.size();
        }

        if (vision)
            imshow("Contour Window", convert);

        char c = (char) waitKey(10);
        if (c == 113)
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

void
updateNetTables(int n, std::shared_ptr<nt::NetworkTable> &table, int center1X, int center1Y, int center2X, int center2Y,
                int averageCenterX, int averageCenterY) {
    table->PutNumber(format("center%dX", n), center1X);
    table->PutNumber(format("center%dY", n), center1Y);
    table->PutNumber(format("center%dX", n), center2X);
    table->PutNumber(format("center%dY", n), center2Y);
    table->PutNumber("averageCenterX", averageCenterX);
    table->PutNumber("averageCenterY", averageCenterY);

    if (debug) {
        std::cout << format("center%dX: ", n) << center1X;
        std::cout << format("center%dY: ", n) << center1Y;
        std::cout << format("center%dX: ", n) << center2X;
        std::cout << format("center%dY: ", n) << center2Y;
        std::cout << "averageCenterX: " << averageCenterX;
        std::cout << "averageCenterY: " << averageCenterY;
    }
}