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
    for i in range(len(width_array)):
        he = 0
        for j in range(len(height_array)):
            if i == len(width_array) - 1 and j == len(height_array) - 1:
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


    # for i in range(len(blocks)):
    #     print(blocks[i].value.shape)
    # return blocks[len(blocks)-1].value

    # blocks_new = np.asarray(blocks).reshape((he, wi))
    # a = blocks_new.value
    # print(blocks_new.shape)
    # a = np.empty((blocks_new.shape))
    # for i in range(len(a)):
    #     for j in range(len(a[0])):
    #         a[i][j] = blocks_new[i][j].value

    a = np.empty((blocks[0].value.shape), int)

    for j in range(he):
        a = np.vstack((a, blocks[j].value))

    column = a

    for i in range(1, wi):
        a = np.empty((blocks[i*he].value.shape), int)
        for j in range(he):
            index = i * he + j
            a = np.vstack((a, blocks[index].value))
        print(a.shape)
        column = np.hstack((column, a))

    return column.astype(np.uint8)



    # print(a.shape)
    # print(blocks[0].value.shape)
    # print(a.shape)
    # a = np.vstack((a, blocks[1].value))
    # a = np.vstack((a, blocks[2].value))
    # a = np.vstack((a, blocks[3].value))
    #
    # a = np.vstack((a, blocks[4].value))
    # a = np.vstack((a, blocks[5].value))
    #
    #
    # print(a.shape)
    # a = np.hstack((a, blocks[5].value))
    # return a.astype(np.uint8)
    # length = len(blocks)
    # re_back = []
    # for i in range(he):
    #     row = np.array([])
    #     row = np.empty((0,3), int)
    #     for j in range(wi):
    #         index = i*wi + j
    #         row.append(blocks[index].value)
    #
    #     re_back.append(row)
    # hi = np.array(re_back)
    # return hi

if __name__ == "__main__":

    img = cv2.imread('screenshot.JPG')

    blockedimg, width, height = create_blocks(img)

    # print(height, width)
    cv2.imshow("hi", reconstruct_background(blockedimg, width, height))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
