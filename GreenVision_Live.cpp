// OpenCV.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
using namespace std;
using namespace cv;
int main()
{
	VideoCapture cap(0);
	Mat image;
	Mat mask;
	Mat contourimg;
	vector<Vec4i> hier;
	Rect large;
	Rect secondLarge;
	int largestIndex = 0;
	int largestContour = 0;
	int secondLargestIndex = 0;
	int secondLargestContour = 0;
	vector<vector<Point>> Contours;
	while (true) {
		cap >> image;
		contourimg = image.clone();
		cvtColor(image, image, COLOR_BGR2GRAY);
		GaussianBlur(image, image, Size(5, 5), 0);
		inRange(image, Scalar(45, 20, 205), Scalar(255, 255, 255), mask);
		erode(mask, mask, 1);
		findContours(mask, Contours, hier, RETR_TREE, CHAIN_APPROX_SIMPLE);
		//Might need to loop through here to check for size
		for (int i = 0; i < Contours.size(); i++)
		{
			if (Contours[i].size() > largestContour) {
				secondLargestContour = largestContour;
				secondLargestIndex = largestIndex;
				largestContour = Contours[i].size();
				largestIndex = i;
			}
			else if (Contours[i].size() > secondLargestContour) {
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
		waitKey(15);
	}
};

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

