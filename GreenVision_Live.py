import numpy as np
import cv2
import sys
import networktables as nt
from imutils.video import WebcamVideoStream
import json


visionFlag = '-v' in sys.argv
debugFlag = '-d' in sys.argv
thresholdFlag = '-t' in sys.argv
multithreadFlag = '-mt' in sys.argv

with open('values.json') as json_file:
    data = json.load(json_file)

cap = WebcamVideoStream(src=0).start() if multithreadFlag else cv2.VideoCapture(0)

nt.NetworkTables.initialize(server=data['server-ip'])
table = nt.NetworkTables.getTable("SmartDashboard")
if table:
    print("table OK")
table.putNumber("visionX", -1)
table.putNumber("visionY", -1)

if debugFlag:
    print('Vision flag: {v}\nDebug flag: {d}\nThreshold Flag: {t}\nMultithread Flag: {mt}'.format(v=visionFlag, d=debugFlag, t=thresholdFlag, mt=multithreadFlag))

lower_color = np.array(data["lower-color-list-thresh"]) if thresholdFlag else np.array(data["lower-color-list"])
upper_color = np.array(data["upper-color-list-thresh"]) if thresholdFlag else np.array(data["upper-color-list"])

def drawPoints(frame, center1x, center1y, center2x, center2y, averagedCenterX, averagedCenterY):
    cv2.line(frame, (center1x, center1y), (center1x, center1y), (255, 0, 0), 8)
    cv2.line(frame, (center2x, center2y), (center2x, center2y), (255, 0, 0), 8)
    cv2.line(frame, (averagedCenterX, averagedCenterY), (averagedCenterX, averagedCenterY), (255, 0, 0), 8)


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


def isPair(topLeftX, topLeftX1, bottomRightX, bottomRightX1):
    topDiff = abs(topLeftX - topLeftX1)
    bottomDiff = abs(bottomRightX - bottomRightX1)
    print('Top diff: {td}\nBottom diff: {bd}'.format(td=topDiff,bd=bottomDiff))
    return bottomDiff > topDiff
        
    
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
    if multithreadFlag:
        frame = cap.read()
    else:
        _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

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
        if len(rectangles) > 1:
            topLeft1X, topLeft1Y, bottomRight1X, bottomRight1Y, center1X, center1Y = defineRec(rectangles[0])
            topLeft2X, topLeft2Y, bottomRight2X, bottomRight2Y, center2X, center2Y = defineRec(rectangles[1])
            averagedCenterX, averagedCenterY = getAverage(center1X, center2X, center1Y, center2Y)
            if True:
            #if isPair(topLeft1X, topLeft2X, bottomRight1X, bottomRight2X):
                updateNetTable(1, center1X, center1Y, center2X, center2Y, averagedCenterX, averagedCenterY, debugFlag)
                drawPoints(frame, center1X, center1Y, center2X, center2Y, averagedCenterX, averagedCenterY)

            if len(rectangles) > 3:
                topLeft3X, topLeft3Y, bottomRight3X, bottomRight3Y, center3X, center3Y = defineRec(rectangles[2])
                topLeft4X, topLeft4Y, bottomRight4X, bottomRight4Y, center4X, center4Y = defineRec(rectangles[3])
                averagedCenter1X, averagedCenter1Y = getAverage(center3X, center4X, center3Y, center4Y)
                if isPair(topLeft3X, topLeft4X, bottomRight3X, bottomRight4X):
                    updateNetTable(2, center3X, center3Y, center4X, center4Y, averagedCenter1X, averagedCenter1Y, debugFlag)
                    drawPoints(frame, center3X, center3Y, center4X, center4Y, averagedCenter1X, averagedCenter1Y)

                if len(rectangles) > 5:
                    topLeft5X, topLeft5Y, bottomRight5X, bottomRight5Y, center5X, center5Y = defineRec(rectangles[4])
                    topLeft6X, topLeft6Y, bottomRight6X, bottomRight6Y, center6X, center6Y = defineRec(rectangles[5])
                    averagedCenter2X, averagedCenter2Y = getAverage(center5X, center6X, center5Y, center6Y)
                    if isPair(topLeft5X, topLeft6X, bottomRight5X, bottomRight6X):
                        updateNetTable(3, center5X, center5Y, center6X, center6Y, averagedCenter2X, averagedCenter2Y, debugFlag)
                        drawPoints(frame, center4X, center4Y, center5X, center5Y, averagedCenter2X, averagedCenter2Y)

    if visionFlag:
        cv2.imshow('Contour Window', frame)
        cv2.imshow('Mask', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
