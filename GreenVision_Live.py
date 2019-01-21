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
    print("Vision flag is:",visionFlag,"debug flag:",debugFlag)

lower_color = np.array([50.0, 55.03597122302158, 174.28057553956833])
upper_color = np.array([90.60606060606061, 255, 255])
def defineRecs(rectangle):
    topleftX = rectangle[0]
    topleftY = rectangle[1]
    width = rectangle[2]
    height= rectangle[3]
    bottomrightX = topleftX + width
    bottomrightY = topleftY + height
    centerX = int((topleftX + bottomrightX) / 2)
    centerY = int((topleftY + bottomrightY) / 2)
    
    return topleftX, topleftY, bottomrightX, bottomrightY, centerX, centerY
def getAverage(center1x, center2x, center1y, center2y):
    averagedCenterX = int((center1x + center2x) / 2)
    averagedCenterY = int((center1y + center2y) / 2)

    return averagedCenterX, averagedCenterY
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
            print('Contour area:',cv2.contourArea(contour))
            ncontours.append(contour)
    print("Number of contours: ", len(ncontours))
    rectangles = []
    for c in ncontours:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 3)
        rectangles.append(cv2.boundingRect(c))
    if (len(rectangles)) > 0:
        
        topLeft1x, topLeft1y, bottomRight1x, bottomRight1y, center1x, center1y = defineRecs(rectangles[0])
        if (len(rectangles)) > 1:
            
            topLeft2x, topLeft2y, bottomRight2x, bottomRight2y, center2x, center2y = defineRecs(rectangles[1])
            averagedCenterX, averagedCenterY = getAverage(center1x, center2x, center1y, center2y)
            
            cv2.line(frame, (center1x, center1y), (center1x, center1y), (255, 0, 0), 8)
            cv2.line(frame, (center2x, center2y), (center2x, center2y), (255, 0, 0), 8)
            cv2.line(frame, (averagedCenterX, averagedCenterY), (averagedCenterX, averagedCenterY), (255, 0, 0), 8)
            if len(ncontours) == 0:
                table.putNumber("center1X", -1)
                table.putNumber("center1Y", -1)
                table.putNumber("center2X", -1)
                table.putNumber("center2Y", -1)
                table.putNumber("averagedCenterX", -1)
                table.putNumber("averagedCenterY", -1)
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

