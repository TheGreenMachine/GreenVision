import cv2
import json
import os

with open('values.json') as json_file:
    data = json.load(json_file)

cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_name = input('Enter name for file: ')
path = '/home/pi/Desktop/GreenVision/Test_Images'
out = cv2.VideoWriter(os.path.join(path, (output_name + '.avi')), fourcc, 30.0,
                      (data['image-width'], data['image-height']))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
