import numpy as np
import cv2
import sys
import networktables as nt

cap = cv2.VideoCapture('test2.mp4')
nt.NetworkTables.initialize(server='10.18.16.2')

table = nt.NetworkTables.getTable("SmartDashboard")
if table:
    print("table OK")
table.putNumber("visionX", -1)
table.putNumber("visionY", -1)

visionFlag = True
debugFlag = True
if len(sys.argv) > 1:
    visionFlag = sys.argv[1] == "-v" or sys.argv[2] == "-v"
    debugFlag = sys.argv[1] == "-d" or sys.argv[2] == "-d"
    print("Vision flag is: {};  debug flag: {}".format(visionFlag, debugFlag))

lower_color = np.array([50.0, 55.03597122302158, 174.28057553956833])
upper_color = np.array([90.60606060606061, 255, 255])

while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    if visionFlag:
        cv2.imshow('Mask', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    for contour in contours:
        if cv2.contourArea(contour) > 400:
            ncontours.append(contour)
    print("Number of contours: ", len(ncontours))
    rectangles = []
    for c in ncontours:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rectangles.append(cv2.boundingRect(c))
    if (len(rectangles)) > 0:
        x = rectangles[0][0]
        y = rectangles[0][1]
        w = rectangles[0][2]
        h = rectangles[0][3]
        if (len(rectangles) > 1):
            x1 = rectangles[1][0]
            y1 = rectangles[1][1]
            w1 = rectangles[1][2]
            h1 = rectangles[1][3]
            topLeft1x = x
            topLeft1y = y
            topLeft2x = x1
            topLeft2y = y1

            # Find cordinates of first rectangle bottom corner

            bottomRight1x = x + w
            bottomRight1y = y + h
            bottomRight2x = x1 + w1
            bottomRight2y = y1 + h1

            # Calculate Centers

            center1x = int((topLeft1x + bottomRight1x) / 2)
            center1y = int((topLeft1y + bottomRight1y) / 2)
            center2x = int((topLeft2x + bottomRight2x) / 2)
            center2y = int((topLeft2y + bottomRight2y) / 2)
            averagedCenterX = int((center1x + center2x) / 2)
            averagedCenterY = int((center1y + center2y) / 2)
            cv2.line(frame, (center1x, center1y), (center1x, center1y), (255, 0, 0), 8)
            cv2.line(frame, (center2x, center2y), (center2x, center2y), (255, 0, 0), 8)
            cv2.line(frame, (averagedCenterX, averagedCenterY), (averagedCenterX, averagedCenterY), (255, 0, 0), 8)
    if len(ncontours) == 0:
        table.putNumber("rec1X", -1)
        table.putNumber("rec1Y", -1)
        table.putNumber("rec2X", -1)
        table.putNumber("rec2Y", -1)
        table.putNumber("averageX", -1)
        table.putNumber("averageY", -1)
        if debugFlag:
            print("TARGET NOT FOUND")
    else:
        table.putNumber("center1X", center1x)
        table.putNumber("center1Y", center1y)
        table.putNumber("center2X", center2x)
        table.putNumber("center2Y", center2y)
        table.putNumber("averagedCenterX", averagedCenterX)
        table.putNumber("averagedCenterY", averagedCenterY)
        if debugFlag:
            print("center1X", center1x)
            print("center1Y", center1y)
            print("center2X", center2x)
            print("center2Y", center2y)
            print("averagedCenterX", averagedCenterX)
            print("averagedCenterY", averagedCenterY)
    if visionFlag:
        cv2.imshow('Contour Window', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

