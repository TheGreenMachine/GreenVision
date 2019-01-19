import cv2
import networktables
import numpy as np

cap = cv2.VideoCapture(0)
def nothing(x):
    pass
networktables.NetworkTables.initialize(server='10.18.16.2')
table = networktables.NetworkTables.getTable("SmartDashboard")
if table:
    print("Table OK")
table.putNumber("center1", 2)
table.putNumber("center2", 2)
table.putNumber("averagedCenter", 1)
threshold = 40
cv2.namedWindow("yo", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Threshold", "yo", 0, 255, nothing)
lowercolor = [0, 0, 231]
uppercolor = [180, 35, 255]

erodesize = np.ones((5,5),np.uint8)
size = 0
maxsize = 400
while (True):
    retr, image = cap.read()
    threshold = cv2.getTrackbarPos("Threshold", "yo")
    lowerbound = (lowercolor[0] - threshold, lowercolor[1] - threshold, lowercolor[2] - threshold)
    upperbound = (uppercolor[0] + threshold, uppercolor[1] + threshold, uppercolor[2] + threshold)
    print(threshold)
    maskCopy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.GaussianBlur(image, (5,5), 0)
    mask = cv2.inRange(image, lowerbound, upperbound)
    mask = cv2.erode(mask, erodesize,iterations = 1)
    secondlargestcontouridx = 0
    largestcontouridx = 0
    largestarea = 0
    secondlargestarea = 0
    isLargestContour = False
    isSecondLargestContour = False
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if (area > largestarea):
            largestarea = area
            largestcontouridx = c
            if (secondlargestarea > size and maxsize > largestarea):
                isLargestContour = True

        elif (area > secondlargestarea):
            secondlargestarea = area
            secondlargestcontouridx = c
            if (largestarea > size and maxsize > secondlargestarea):
                isSecondLargestContour = True
    if isLargestContour:
        x, y, w, h = cv2.boundingRect(largestcontouridx)
        if isSecondLargestContour:
            x1, y1, w1, h1 = cv2.boundingRect(secondlargestcontouridx)


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
            print("center1X", center1x)
            print("center1Y", center1y)
            print("center2X", center2x)
            print("center2Y", center2y)
            print("averagedCenterX", averagedCenterx)
            print("averagedCenterY", averagedCentery)
            table.putNumber("center1X", center1x)
            table.putNumber("center1Y", center1y)
            table.putNumber("center2X", center2x)
            table.putNumber("center2Y", center2y)
            table.putNumber("averagedCenterX", averagedCenterx)
            table.putNumber("averagedCenterY", averagedCentery)
            #Draw a mask
            cv2.line(maskCopy, (center1x, center1y), (center1x, center1y), (255, 0, 0), 8)
            cv2.line(maskCopy, (center2x, center2y), (center2x, center2y), (255, 0, 0), 8)
            cv2.line(maskCopy, (averagedCenterx, averagedCentery), (averagedCenterx, averagedCentery), (255, 0, 0), 8)

            cv2.drawContours(maskCopy, [largestcontouridx], -1, (255, 0, 0), 3)
            cv2.drawContours(maskCopy, [secondlargestcontouridx], -1, (0, 255, 0), 3)
    cv2.namedWindow("yo", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("yo", maskCopy)
    cv2.waitKey(15)