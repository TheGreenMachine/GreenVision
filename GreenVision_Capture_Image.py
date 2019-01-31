import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow('image cap')

print('ESC to exit, SPACE to capture')

while True:
    distance = input('Enter distance in inches: ')
    ret, frame = cap.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1) & 0xFF

    # if k % 256 == 27:
    if k == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    # elif k % 256 == 32:
    elif k == 32:
        # SPACE pressed
        img_name = 'opencv_image_{}in_'.format(distance)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
cap.release()
cv2.destroyAllWindows()
