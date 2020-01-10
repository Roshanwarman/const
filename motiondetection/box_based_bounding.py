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


def SAD(image1, image2):
    im1_block = create_blocks(image1)
    im2_block = create_blocks(image2)
    SAD = []
    for i in range(len(im1_block)):
        SAD.append(np.sum(np.absolute(im1_block[i].value - im2_block[i].value)))

    return SAD

def minimum_indexed_block(images):
    # two = images[0].shape
    SAD_tensor = SAD(images[0], images[1])
    gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    (h, w) = gray.shape
    height_array  = np.array([y for y in range(0, h, 330)])
    width_array = np.array([x for x in range(0, w, 330)])
    height, width = len(height_array), len(width_array)

    for i in range(1, len(images)-1):
        vadd = SAD(images[i], images[i+1])
        SAD_tensor = np.vstack((SAD_tensor, vadd))


    flipped = SAD_tensor.transpose()
    minimum_blocks = np.array([])

    for i in flipped:
        minimum_blocks = np.append(minimum_blocks, np.argmin(i))

    minimum_blocks = minimum_blocks.astype(np.uint8)

    memo = {} #memoizer for DP

    Images = np.array([])

    for i in minimum_blocks:
        Images.append((images[minimum_blocks[i]], i))

    for j in Images:

        if j[1] in memo:
            #memoization here


    # minimum_blocks returns array [b1, b2, ... , bk] where b1 is the image that contains the arg min block in position (i,j) -> reshape

    # h1, w1 = images[0].shape
    # reconstructed_background = np.array(create_blocks(images[minimum_blocks[0]])[0].value)


    # for i in range(1, len(minimum_blocks)):
    #     addendum = np.array(create_blocks(images[minimum_blocks[i]])[i].value)
    #     reconstructed_background = np.append(reconstructed_background, addendum)

    #reconstructed_background is 1D array of Block objects that are minimium SAD for each block location
    # print(reconstructed_background.shape)
    # background_initialization = reconstructed_background.reshape((h, w))
    #
    background = np.zeros((height, width))
    print(background.shape)

    #reshape and make background matrix which converts the matrix of Block objects to the values of the Block objects
    for i in range(len(background)):
        for j in range(len(background[i])):
            index = i*len(background[i]) + j
            valueij = create_blocks(images[minimum_blocks[index]])
            print(valueij[0].value)
            break
            background[i][j] = valueij


    return background


def reconstruct_background(blocks, wi, he):
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


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # first = None
    image_sequence = []
    initial_time = time.time()
    while(True):
        t, frame = cap.read()
        # print(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_sequence.append(frame)
        if t:
            cv2.imshow("current frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

        if time.time() - initial_time > 1:
            break

    cap.release()

    cv2.imshow('hi', minimum_indexed_block(image_sequence))


    # print(create_blocks(image_sequence[0]))
    # print(image_sequence[0].shape)

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        # if t:
        #
        #     frame = imutils.resize(frame, width=500)
        #     gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        #     width, height = gray1.shape
        #
        #     width_array = np.array([x for x in range(0, width, round(width/16))])
        #     height_array  = np.array([y for y in range(0, height, round(height/16))])
        #
        #
        #     gridx, gridy= np.meshgrid(height_array, width_array)

            # print(np.amax(height_array))
            # print(np.amax(width_array))

            # frame_height, frame_width = gray1.shape
            # print(gray1.shape)

            # border for frame; consider removing later if SAD doesn't work

            # frame = cv2.line(frame, (0, 0), (0, frame_width), (0,0,0), 1)
            # frame = cv2.line(frame, (0, 0), (frame_height, 0), (0,0,0), 1)
            # frame = cv2.line(frame, (frame_height, frame_width), (0, frame_width), (0,255,0), 1)
            # frame = cv2.line(frame, (frame_height, 0), (frame_height, frame_width),  (0,255,0), 1)


            #
            # for t in width_array:
            #     frame = cv2.line(frame, (0, t), (frame_width, t), (255,0, 255), 1)
            # for f in height_array:
            #     frame = cv2.line(frame, (f, 0), (f, frame_height), (255,0,255), 1)
            #
            # gray = cv2.GaussianBlur(gray1, (21, 21), 0)
            #
            # if first is None:
                # first_time = time.time()
                # first = gray
                # continue

            # if time.time() - first_time > 1:
            #     first_time = time.time()
            #     first = gray


            #
            # frame_difference = cv2.absdiff(first, gray)
            # threshold = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)[1]

            # threshold = cv2.dilate(threshold, None, iterations = 2)
            # countours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
