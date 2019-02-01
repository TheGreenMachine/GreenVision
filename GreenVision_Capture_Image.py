import cv2
import json
import os

with open('values.json') as json_file:
    data = json.load(json_file)

cap = cv2.VideoCapture(0)
cv2.namedWindow('image cap')

while True:
    ret, frame = cap.read()
    cv2.imshow("test", frame)
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        distance = int(input('Enter distance in inches: '))
        path = '/home/pi/Desktop/GreenVision/Test_Images'
        img_name = os.path.join(path, 'opencv_image_{}in.jpg'.format(distance))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
