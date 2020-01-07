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



class Block:


    def __init__(self, value):
        self.block_matrix = value
        # self.index = index



def create_blocks(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY )

    height, width = gray.shape

    # width_array = np.array([x for x in range(0, width, np.floor(width/16))])
    # height_array  = np.array([y for y in range(0, height, np.floor(height/16))])
    height_array  = np.array([y for y in range(0, height, 330)])
    width_array = np.array([x for x in range(0, width, 330)])


    blocks = []
    for i in range(len(width_array) -1 ):
        for j in range(len(height_array) - 1):
            if i == len(width_array) - 2 and j == len(height_array) - 2:
                blocks.append(Block(image[height_array[j] : height -1, width_array[i] : width -1]))
            elif i == len(width_array) - 1:
                blocks.append(Block(image[height_array[j] : height_array[j+1], width_array[i] : width-1 ]))
            elif j == len(height_array) -1:
                blocks.append(Block(image[height_array[j] : height -1, width_array[i] : width_array[i+1]]))
            else:
                blocks.append(Block(image[height_array[j] : height_array[j+1], width_array[i] : width_array[i+1]]))

            #add edge blocks for image; this only goes to len(array) -1, so capture the end array separately
    return blocks


def SAD(image1, image2):
    im1_blocks = create_blocks(image1)
    im2_blocks = create_blocks(image2)
    SAD = []
    for i in range(len(im1_block)):
        SAD.append(np.sum(np.absolute(im1_block[i].value - im2_block[i].value)))

    return SAD



def minimum_indexed_block(images):

    SAD_tensor = SAD(images[0], images[1])


    for i in range(1, len(images)-1):
        SAD_tensor = np.vstack(SAD_tensor, SAD(images[i], images[i+1]))

    flipped = SAD_tensor.transpose()
    minimum_blocks = np.array([])

    for i in flipped:
        minimum_blocks = np.append(minmum_blocks, np.where(i == i.min())[0][0])
    #minimum_blocks returns array [b1, b2, ... , bk] where b1 is the image that contains the arg min block in position (i,j) -> reshape
    reconstructed_background = np.array([])
    for i in range(len(minimum_blocks)):
        reconstructed_background = np.append(reconstructed_background, create_blocks[images[minimum_blocks[i]]][i])

    #reconstructed_background is 1D array of Block objects that are minimium SAD for each block location

    background_initialization = reconstructed_background.reshape((image[0].shape))
    background = np.zeros((background_initialization.shape))

    #reshape and make background matrix which converts the matrix of Block objects to the values of the Block objects
    for i in range(len(background)):
        for j in range(len(background[i])):
            background[i][j] = background_initialization[i][j].value


    return background



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # first = None
    image_sequence = np.array([])
    initial_time = time.time()
    while(True):
        t, frame = cap.read()
        print(frame)
        image_sequence = np.append(image_sequence, frame)
        if t:
            cv2.imshow("current frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

        if time.time() - initial_time > 10:
            break

    cap.release()

    cv2.imshow("helo", minimum_indexed_block(image_sequence))


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
