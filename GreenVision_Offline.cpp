// OpenCV.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ntcore.h"

using namespace std;
using namespace cv;

int main() {
    Mat image = imread("test_images/light.png");
    Mat mask;
    vector<vector<Point>> Contours;
    vector<Vec4i> hier;
    Mat contourimg = image.clone();
    Rect large;
    Rect secondLarge;
    int largestIndex = 0;
    int largestContour = 0;
    int secondLargestIndex = 0;
    int secondLargestContour = 0;
    cvtColor(image, image, COLOR_BGR2HSV);
    GaussianBlur(image, image, Size(5, 5), 0);
    namedWindow("absolute", WINDOW_AUTOSIZE);
    imshow("absolute", image);
    //threshold(finalimage, mask, 40, 255, THRESH_BINARY);
    inRange(image, Scalar(45, 20, 205), Scalar(255, 255, 255), mask);
    imshow("Imaage1", mask);
    erode(mask, mask, 1);
    findContours(mask, Contours, hier, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    //Might need to loop through here to check for size
    for (int i = 0; i < Contours.size(); i++) {
        if (Contours[i].size() > largestContour) {
            secondLargestContour = largestContour;
            secondLargestIndex = largestIndex;
            largestContour = Contours[i].size();
            largestIndex = i;
        } else if (Contours[i].size() > secondLargestContour) {
            secondLargestContour = Contours[i].size();
            secondLargestIndex = i;
        }
    }
    large = boundingRect(Contours[largestIndex]);
    secondLarge = boundingRect(Contours[secondLargestIndex]);
    drawContours(contourimg, Contours, largestIndex, Scalar(232, 12, 122), 3, 8, hier);
    drawContours(contourimg, Contours, secondLargestIndex, Scalar(232, 12, 122), 3, 8, hier);

    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", contourimg);
    waitKey(0);
};

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

