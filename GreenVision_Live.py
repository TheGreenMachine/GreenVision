import numpy as np
import cv2
import sys
import networktables as nt

cap = cv2.VideoCapture(0)
nt.NetworkTables.initialize(server='10.18.16.2')

table = nt.NetworkTables.getTable("SmartDashboard")
if table:
    print("table OK")
table.putNumber("visionX", -1)
table.putNumber("visionY", -1)

visionFlag = False
debugFlag = False

if len(sys.argv) == 2:
    visionFlag = sys.argv[1] == "-v"
    debugFlag = sys.argv[1] == "-d"
elif len(sys.argv) == 3:
    visionFlag = sys.argv[1] == "-v" or sys.argv[2] == "-v"
    debugFlag = sys.argv[1] == "-d" or sys.argv[2] == "-d"
if debugFlag:
    print("Vision flag is:", visionFlag, "debug flag:", debugFlag)

lower_color = np.array([50.0, 55.03597122302158, 174.28057553956833])
upper_color = np.array([90.60606060606061, 255, 255])


def defineRec(rectangle):
    topleftX = rectangle[0]
    topleftY = rectangle[1]
    width = rectangle[2]
    height = rectangle[3]
    bottomrightX = topleftX + width
    bottomrightY = topleftY + height
    centerX = int((topleftX + bottomrightX) / 2)
    centerY = int((topleftY + bottomrightY) / 2)

    return topleftX, topleftY, bottomrightX, bottomrightY, centerX, centerY


def getAverage(center1x, center2x, center1y, center2y):
    averagedCenterX = int((center1x + center2x) / 2)
    averagedCenterY = int((center1y + center2y) / 2)

    return averagedCenterX, averagedCenterY

def updateNetTable(n, center1x = -1, center1y = -1, center2x = -1, center2y = -1, averagedCenterX = -1, averagedCenterY = -1, debugFlag = False):
        table.putNumber("center{n}X".format(n=n), center1x)
        table.putNumber("center{n}Y".format(n=n), center1y)
        table.putNumber("center{n}X".format(n=n+1), center2x)
        table.putNumber("center{n}Y".format(n=n+1), center2y)
        table.putNumber("averagedCenterX", averagedCenterX)
        table.putNumber("averagedCenterY", averagedCenterY)
        if debugFlag:
            print("center{n}X".format(n=n), center1x)
            print("center{n}Y".format(n=n), center1y)
            print("center{n}X".format(n=n + 1), center2x)
            print("center{n}Y".format(n=n + 1), center2y)
            print("averagedCenterX", averagedCenterX)
            print("averagedCenterY", averagedCenterY)


while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    if visionFlag:
        cv2.imshow('Mask', mask)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ncontours = []
    for contour in contours:
        if cv2.contourArea(contour) > 75:
            print('Contour area:', cv2.contourArea(contour))
            ncontours.append(contour)
    print("Number of contours: ", len(ncontours))
    rectangles = []
    for c in ncontours:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rectangles.append(cv2.boundingRect(c))
    if len(rectangles) > 0:

        topLeft1x, topLeft1y, bottomRight1x, bottomRight1y, center1x, center1y = defineRec(rectangles[0])
        if len(rectangles) > 1:

            topLeft2x, topLeft2y, bottomRight2x, bottomRight2y, center2x, center2y = defineRec(rectangles[1])
            averagedCenterX, averagedCenterY = getAverage(center1x, center2x, center1y, center2y)
            updateNetTable(1, center1x, center1y, center2x, center2y, averagedCenterX, averagedCenterY,debugFlag)
            cv2.line(frame, (center1x, center1y), (center1x, center1y), (255, 0, 0), 8)
            cv2.line(frame, (center2x, center2y), (center2x, center2y), (255, 0, 0), 8)
            cv2.line(frame, (averagedCenterX, averagedCenterY), (averagedCenterX, averagedCenterY), (255, 0, 0), 8)


            if len(rectangles) > 2:
                topLeft3x, topLeft3y, bottomRight3x, bottomRight3y, center3x, center3y = defineRec(rectangles[2])
                if len(rectangles) > 3:
                    topLeft4x, topLeft4y, bottomRight4x, bottomRight4y, center4x, center4y = defineRec(rectangles[3])
                    averagedCenter1X, averagedCenter1Y = getAverage(center3x, center4x, center3y, center4y)
                    updateNetTable(2, center3x, center3y, center4x, center4y, averagedCenter1X, averagedCenter1Y,debugFlag)
                    cv2.line(frame, (center3x, center3y), (center3x, center3y), (255, 0, 0), 8)
                    cv2.line(frame, (center4x, center4y), (center4x, center4y), (255, 0, 0), 8)
                    cv2.line(frame, (averagedCenter1X, averagedCenter1Y), (averagedCenter1X, averagedCenter1Y), (255, 0, 0), 8)

                    if len(rectangles) > 4:
                        topLeft5x, topLeft5y, bottomRight5x, bottomRight5y, center5x, center5y = defineRec(rectangles[4])
                        if len(rectangles) > 5:
                            topLeft6x, topLeft6y, bottomRight6x, bottomRight6y, center6x, center6y = defineRec(rectangles[5])
                            averagedCenter2X, averagedCenter2Y = getAverage(center5x, center6x, center5y, center6y)
                            updateNetTable(3, center5x, center5y, center6x, center6y, averagedCenter2X, averagedCenter2Y, debugFlag)
                            cv2.line(frame, (center4x, center4y), (center4x, center4y), (255, 0, 0), 8)
                            cv2.line(frame, (center5x, center5y), (center5x, center5y), (255, 0, 0), 8)
                            cv2.line(frame, (averagedCenter2X, averagedCenter2Y), (averagedCenter2X, averagedCenter2Y), (255, 0, 0), 8)



    if visionFlag:
        cv2.imshow('Contour Window', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
