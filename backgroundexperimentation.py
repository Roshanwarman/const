import cv2
import numpy
import imutils


cap = cv2.VideoCapture(0)
fconvolve = cv2.createBackgroundSubtractorMOG()

while(cap.isOpened()):



    t, frame = cap.read()



    result = fconvolve.apply(frame)

    # result = cv2.dilate(result, None, iterations = 2)
    countours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    countours = imutils.grab_contours(countours)

    for c in countours:
        if cv2.contourArea(c) < 10000:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h) , (0,255,0), 1)

    cv2.imshow('hi', frame)

    if cv2.waitKey(30) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
