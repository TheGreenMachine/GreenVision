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
    Mat image1 = imread("/home/ianmcvann/Documents/Data/OpenCVTest/OpenCV/OpenCV/light.jpg");
    Mat image2 = imread("/home/ianmcvann/Documents/Data/OpenCVTest/OpenCV/OpenCV/nolight.jpg");
	Mat finalimage;
	Mat mask;
	cvtColor(image1,image1, COLOR_BGR2GRAY);
	cvtColor(image2, image2, COLOR_BGR2GRAY);
	GaussianBlur(image1, image1, Size(5, 5), 0);
	GaussianBlur(image2, image2, Size(5, 5), 0);
	absdiff(image1, image2, finalimage);
	threshold(finalimage, mask, 40, 255, THRESH_BINARY);
	erode(mask, mask, 1);
	namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", mask);
	waitKey(0);
};

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

