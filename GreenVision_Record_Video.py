import numpy as np
import cv2
import json

with open('values.json') as json_file:
    data = json.load(json_file)

cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_name = input('Enter name for file')
out = cv2.VideoWriter(output_name, fourcc, 30.0, (data['image-width'], data['image-height']))
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
