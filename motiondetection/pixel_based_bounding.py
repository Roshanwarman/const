# import cv2
#
#
# img1 = cv2.imread('puc.jpg')
#
# e1 = cv2.getTickCount()
# for i in range(1,49,2):
#     img1 = cv2.medianBlur(img1,i)
#     cv2.imshow("hie", img1)
#     if cv2.waitKey(0):
#         continue
# e2 = cv2.getTickCount()
# t = (e2 - e1)/cv2.getTickFrequency()
#
#
# print(t)

import matplotlib.pyplot as plt
from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#
# class Block:
#
#     block_index = None
#
#     def __init__(self):
#
#     def set_block_index(self, horizontal, veritcal):
#
#         return $$
#
#     def adjacent_block_diff(self, block1, block2):
#
#         return difference
#




#
#
# def compute_SAD(block, image1, image2, image):
#
#     for t in width_array:
#         frame = cv2.line(frame, (0, t), (frame_width, t), (255,0, 255), 1)
#     for f in height_array:
#         frame = cv2.line(frame, (f, 0), (f, frame_height), (255,0,255), 1)
#
#
#     First_image = cv2.imread(image1)
    # Second_image = cv2.imread(image2)
#
#
#     np.abs(np.difference(First_image, Se


def create_blocks(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # width_array = np.array([x for x in range(0, width, np.floor(width/16))])
    # height_array  = np.array([y for y in range(0, height, np.floor(height/16))])
    height_array  = np.array([y for y in range(0, height, 330)])
    width_array = np.array([x for x in range(0, width, 330)])


    blocks = []
    for i in range(len(width_array) -1 ):
        for j in range(len(height_array) - 1):
            blocks.append(image[height_array[j] : height_array[j+1], width_array[i] : width_array[i+1]])

    return blocks


def SAD(im1_block, im2_block):
    SAD = []

    for i in range(len(im1_block)):
        SAD.append(np.sum(np.absolute(np.subtract(im1_block[i], im2_block[i]))))

    return SAD




if __name__ == "__main__":

    image = cv2.imread("puc.JPG")
    # print(image)
    blocks = np.array(create_blocks(image))
    first_block  = blocks[0]
    second_block = blocks[1]
    # plt.subplot(212)
    # plt.imshow(first_block)
    # plt.subplot(222)
    # plt.imshow(second_block)
    # plt.show()
    print(np.sum(np.abs(first_block - first_block)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
# first = None
# while(True):
#
#     t, frame = cap.read()
#
#     if t:
#
#         frame = imutils.resize(frame, width=500)
#         gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         width, height = gray1.shape
#
#         width_array = np.array([x for x in range(0, width, round(width/16))])
#         height_array  = np.array([y for y in range(0, height, round(height/16))])
#
#
#         gridx, gridy= np.meshgrid(height_array, width_array)
#
#         # print(np.amax(height_array))
#         # print(np.amax(width_array))
#
#         frame_height, frame_width = gray1.shape
#         # print(gray1.shape)
#
#         # border for frame; consider removing later if SAD doesn't work
#
#         frame = cv2.line(frame, (0, 0), (0, frame_width), (0,0,0), 1)
#         frame = cv2.line(frame, (0, 0), (frame_height, 0), (0,0,0), 1)
#         frame = cv2.line(frame, (frame_height, frame_width), (0, frame_width), (0,255,0), 1)
#         frame = cv2.line(frame, (frame_height, 0), (frame_height, frame_width),  (0,255,0), 1)
#
#
#
#         for t in width_array:
#             frame = cv2.line(frame, (0, t), (frame_width, t), (255,0, 255), 1)
#         for f in height_array:
#             frame = cv2.line(frame, (f, 0), (f, frame_height), (255,0,255), 1)
#
#         gray = cv2.GaussianBlur(gray1, (21, 21), 0)
#
#         if first is None:
#             # first_time = time.time()
#             first = gray
#             continue
#
#         # if time.time() - first_time > 1:
#         #     first_time = time.time()
#         #     first = gray
#
#
#
#         frame_difference = cv2.absdiff(first, gray)
#         threshold = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)[1]
#
#         # threshold = cv2.dilate(threshold, None, iterations = 2)
#         countours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#         countours = imutils.grab_contours(countours)
#
#         for c in countours:
#             if cv2.contourArea(c) < 10000:
#                 continue
#
#             (x, y, w, h) = cv2.boundingRect(c)
#             frame = cv2.rectangle(frame, (x,y), (x+w, y+h) , (0,255,0), 1)
#
#
#
#         cv2.imshow("constructionbox", frame)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
#
# # # import cv2
# # # import numpy
# # # import matplotlib.pyplot as plt
# # #
# # # cap = cv2.VideoCapture('movie.mp4')
# # # t = True
# # # while(t):
# # #     t, frame = cap.read()
# # #     if t:
# # #
# # #         cv2.imshow('video', frame)
# # #
# # #         if cv2.waitKey(1) & 0xFF  == ord('q'):
# # #             break
# # #
# # # cap.release()
# # # cv2.destroyAllWindows()
# # # # img = cv2.imread('image.JPG', cv2.IMREAD_COLOR)
# # # # cv2.imshow('image', img)
# # # # cv2.waitKey(0)
# # # # cv2.destroyAllWindows()
# # #
# # # # from urllib import request
# # # #
# # # # import re
# # # #
# # # # new_url = re.sub("tube", "pak", url_initial)
