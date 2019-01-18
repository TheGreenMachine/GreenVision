import cv2
import networktables
import numpy as np

cap = cv2.VideoCapture(1)
lowerbound = np.array([45, 20, 205])
upperbound = np.array([255, 255, 255])
erodesize = np.ones((5,5),np.uint8)
areaArray = []
largestarea = 0
secondlargestarea = 0
while (True):
    retr, image = cap.read()
    maskCopy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.GaussianBlur(image, (5,5), 0)
    mask = cv2.inRange(image, lowerbound, upperbound)
    mask = cv2.erode(mask, erodesize,iterations = 1)
    secondlargestcontouridx = 0
    largestcontouridx = 0
    areaArray = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    try:
        if (sorteddata[1][1].any() > 0):
            print(len(sorteddata[1][1]))
            secondlargestcontouridx = sorteddata[1][1]
        if (sorteddata[0][1].any() > 0):
            largestcontouridx = sorteddata[0][1]
        x, y, w, h = cv2.boundingRect(largestcontouridx)
        x1, y1, w1, h1 = cv2.boundingRect(secondlargestcontouridx)
    except IndexError:
        print("You died")
        continue



    #Find cordinates of first rectangle top left bottom corner

    topLeft1x = x
    topLeft1y = y
    topLeft2x = x1
    topLeft2y = y1

    #Find cordinates of first rectangle bottom corner

    bottomRight1x = x + w
    bottomRight1y = y + h
    bottomRight2x = x1 + w1
    bottomRight2y = y1 + h1

    #Calculate Centers

    center1x = int((topLeft1x + bottomRight1x) / 2)
    center1y = int((topLeft1y + bottomRight1y) / 2)
    center2x = int((topLeft2x + bottomRight2x) / 2)
    center2y = int((topLeft2y + bottomRight2y) / 2)
    averagedCenterx = int((center1x + center2x) / 2)
    averagedCentery = int((center1y + center2y) / 2)

    #Draw a mask
    cv2.line(maskCopy, (center1x, center1y), (center1x, center1y), (255, 0, 0), 8)
    cv2.line(maskCopy, (center2x, center2y), (center2x, center2y), (255, 0, 0), 8)
    cv2.line(maskCopy, (averagedCenterx, averagedCentery), (averagedCenterx, averagedCentery), (255, 0, 0), 8)

    cv2.drawContours(maskCopy, [largestcontouridx], -2, (255, 0, 0), 3)
    cv2.drawContours(maskCopy, [secondlargestcontouridx], -2, (0, 255, 0), 3)
    cv2.namedWindow("yo", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("yo", maskCopy)
    cv2.waitKey(15)