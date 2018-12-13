// OpenCV.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap(0);
    while (true) {
    Mat image1;
    Mat image2;
    cap >> image1;
    cap >> image2;
    Mat finalimage;
    Mat mask;
    cvtColor(image1,image1, COLOR_BGR2GRAY);
    cvtColor(image2, image2, COLOR_BGR2GRAY);
    GaussianBlur(image1, image1, Size(5, 5), 0);
    GaussianBlur(image2, image2, Size(5, 5), 0);
    absdiff(image1, image2, finalimage);
    //threshold(finalimage, mask, 40, 255, THRESH_BINARY);
    inRange(finalimage, Scalar(50.0, 55.03597122302158, 174.28057553956833), Scalar(90.60606060606061, 255, 255), mask);
    erode(mask, mask, 1);
	namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", mask);
    waitKey(15);
   }
};

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

