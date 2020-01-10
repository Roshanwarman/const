import matplotlib.pyplot as plt
from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2
import numpy as np


class Block:


    def __init__(self, value):
        self.value = value
        # self.index = index


def create_blocks(image):

    # gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image)
    # return image
    height, width = gray.shape
    # width_array = np.array([x for x in range(0, width, np.floor(width/16))])
    # height_array  = np.array([y for y in range(0, height, np.floor(height/16))])
    height_array  = np.array([y for y in range(0, height, 30)])
    width_array = np.array([x for x in range(0, width, 30)])


    blocks = []
    wi = 0
    he = 0
    for i in range(len(width_array) -1 ):
        he = 0
        for j in range(len(height_array) - 1):
            if i == len(width_array) - 2 and j == len(height_array) - 2:
                blocks.append(Block(gray[height_array[j] : height -1, width_array[i] : width -1]))
            elif i == len(width_array) - 1:
                blocks.append(Block(gray[height_array[j] : height_array[j+1], width_array[i] : width-1 ]))
            elif j == len(height_array) -1:
                blocks.append(Block(gray[height_array[j] : height -1, width_array[i] : width_array[i+1]]))
            else:
                blocks.append(Block(gray[height_array[j] : height_array[j+1], width_array[i] : width_array[i+1]]))
            he = he+1

        wi = wi+1
            #add edge blocks for image; this only goes to len(array) -1, so capture the end array separately
    return blocks, wi, he


def reconstruct_background(blocks, wi, he):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # height, width = gray.shape
    #
    # height_array  = np.array([y for y in range(0, height, 30)])
    # width_array = np.array([x for x in range(0, width, 30)])


    # h, w = len(height_array) -1 , len(width_array) -1

    length = len(blocks)
    re_back = []
    for i in range(he):
        row = np.array([])
        for j in range(wi):
            index = i*wi + j
            row.append(blocks[index].value)

        re_back.append(row)
    hi = np.array(re_back)
    return hi

if __name__ == "__main__":

    img = cv2.imread('test.JPG')

    blockedimg, width, height = create_blocks(img)

    print(height, width)
    cv2.imshow("hello", reconstruct_background(blockedimg, width, height))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
