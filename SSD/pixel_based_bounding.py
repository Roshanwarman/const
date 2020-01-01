from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2
import time

cap = cv2.VideoCapture(0)

first = None

while(True):

    t, frame = cap.read()

    if t:

        frame = imutils.resize(frame, width=500)
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray1, (21, 21), 0)

        if first is None:
            # first_time = time.time()
            first = gray
            continue

        # if time.time() - first_time > 1:
        #     first_time = time.time()
        #     first = gray



        frame_difference = cv2.absdiff(first, gray)
        threshold = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)[1]

        threshold = cv2.dilate(threshold, None, iterations = 2)
        countours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        countours = imutils.grab_contours(countours)

        for c in countours:
            if cv2.contourArea(c) < 10000:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h) , (0,255,0), 1)




        cv2.imshow("constructionbox", frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy
# import matplotlib.pyplot as plt
#
# cap = cv2.VideoCapture('movie.mp4')
# t = True
# while(t):
#     t, frame = cap.read()
#     if t:
#
#         cv2.imshow('video', frame)
#
#         if cv2.waitKey(1) & 0xFF  == ord('q'):
#             break
#
# cap.release()
# cv2.destroyAllWindows()
# # img = cv2.imread('image.JPG', cv2.IMREAD_COLOR)
# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # from urllib import request
# #
# # import re
# #
# # new_url = re.sub("tube", "pak", url_initial)
