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
    Mat image1 = imread("/home/ianmcvann/Documents/Github/FRC/GreenVision/GreenVision/light.jpg");
    Mat image2 = imread("/home/ianmcvann/Documents/Github/FRC/GreenVision/GreenVision/nolight.jpg");
	Mat finalimage;
	Mat mask;
    vector<vector<Point>> Contours;
    vector<Vec4i> hier;
    Mat contourimg = image1.clone();
    double maxArea = 0.0;
    int savedContour = -1;
	cvtColor(image1,image1, COLOR_BGR2GRAY);
	cvtColor(image2, image2, COLOR_BGR2GRAY);
	GaussianBlur(image1, image1, Size(5, 5), 0);
	GaussianBlur(image2, image2, Size(5, 5), 0);
	absdiff(image1, image2, finalimage);
    //threshold(finalimage, mask, 40, 255, THRESH_BINARY);
    inRange(finalimage, Scalar(50.0, 55.03597122302158, 174.28057553956833), Scalar(90.60606060606061, 255, 255), mask);
    erode(mask, mask, 1);
    findContours(mask, Contours, hier, RETR_TREE, CHAIN_APPROX_SIMPLE);
    //Might need to loop through here to check for size
    for (int i = 0; i < Contours.size(); i++) {
        double area = contourArea(Contours[i]);
        if (area > maxArea) {
            maxArea = area;
            savedContour = i;
        }
    }
    drawContours(contourimg, Contours, savedContour, Scalar(70, 237, 54), 3, 8, hier);

	namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", contourimg);
	waitKey(0);
};

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

