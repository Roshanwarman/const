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

    def change_value(self, new):
        self.value = new

    def pdistribution(self):

        h, w = self.value.shape
        size = h*w
        flat = self.value.flatten()

        sorted = np.sort(flat, kind = 'heapsort')

        distribution = []
        initial = sorted[0]
        counter = 1

        for i in range(1, len(sorted)):
            if sorted[i] == sorted[i-1]:
                counter = counter +1
            elif i == len(sorted) - 1:
                counter = counter + 1
                distribution.append((sorted[i], counter/size))

            else:
                distribution.append((sorted[i-1], counter/size))
                counter = 1

        return np.asarray(distribution)


    def compute_entropy(self):

        probabilities = [i[1] for i in self.pdistribution()]
        entropy_array = [i*np.log(2*i) for i in probabilities]
        entropy = -1 * np.sum(entropy_array)
        return entropy




def create_blocks(image):

    # gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image)
    # return image
    height, width = gray.shape
    # width_array = np.array([x for x in range(0, width, np.floor(width/16))])
    # height_array  = np.array([y for y in range(0, height, np.floor(height/16))])
    height_array  = np.array([y for y in range(0, height, 100)])
    width_array = np.array([x for x in range(0, width, 100)])


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

    im1_block, w, h = create_blocks(image1)
    im2_block, w1, h1 = create_blocks(image2)
    SAD = []
    for i in range(len(im1_block)):
        diff = np.absolute(im1_block[i].value - im2_block[i].value)
        mu = np.mean(im1_block[i].value)
        standard = diff / mu
        SAD.append(np.sum(np.absolute(standard)))

    return np.asarray(SAD)

def minimum_indexed_block(images):

    SAD_tensor = SAD(images[0], images[1])

    for i in range(1, len(images)-1):
        vadd = SAD(images[i], images[i+1])
        if vadd.astype(np.uint8)[0] == 0:
            continue
        SAD_tensor = np.vstack((SAD_tensor, vadd))

    print(SAD_tensor.shape)

    flipped = SAD_tensor.T
    print(flipped.shape)
    minimum_blocks = []

    for i in range(len(flipped)):
        minimum_blocks.append(np.argmin(flipped[i]))
        # print("hi",np.min(flipped[i]), flipped[i][np.argmin(flipped[i])])
    for i in minimum_blocks:
        print(i)
    minimum_blocks = np.asarray(minimum_blocks).astype(np.uint8)
    for i in minimum_blocks:
        print(i)

    memo = {" " : None}

    Images = []

    for i in range(len(minimum_blocks)):
        Images.append((images[minimum_blocks[i]], minimum_blocks[i], i))

    final_blocks = []
    for j in Images:

        if j[1] in memo:
            final_blocks.append(memo[j[1]][j[2]])

        else:
            blocks, w, h = create_blocks(j[0])
            final_blocks.append(blocks[j[2]])
            memo.update({j[1] : blocks})

    return final_blocks, w, h

    # minimum_blocks returns array [b1, b2, ... , bk] where b1 is the image that contains the arg min block in position (i,j) -> reshape
    #
    # h1, w1 = images[0].shape
    # reconstructed_background = np.array(create_blocks(images[minimum_blocks[0]])[0].value)
    #
    #
    # for i in range(1, len(minimum_blocks)):
    #     addendum = np.array(create_blocks(images[minimum_blocks[i]])[i].value)
    #     reconstructed_background = np.append(reconstructed_background, addendum)
    #
    # reconstructed_background is 1D array of Block objects that are minimium SAD for each block location
    # print(reconstructed_background.shape)
    # background_initialization = reconstructed_background.reshape((h, w))
    #
    # background = np.zeros((height, width))
    # print(background.shape)
    #
    # #reshape and make background matrix which converts the matrix of Block objects to the values of the Block objects
    # for i in range(len(background)):
    #     for j in range(len(background[i])):
    #         index = i*len(background[i]) + j
    #         valueij = create_blocks(images[minimum_blocks[index]])
    #         print(valueij[0].value)
    #         break
    #         background[i][j] = valueij
    #
    #
    # return background


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


def background_update(current_blocks, previous_blocks):

    h,w = current_blocks[0].value.shape

    for i in range(len(current_blocks)):
        A = None

        EBt = previous_blocks[i].compute_entropy()
        EIt = current_blocks[i].compute_entropy()

        if np.absolute(EBt - EIt) <= 1:
            A = np.ones((h,w))
            previous_blocks[i].change_value(current_blocks[i].value)
        else:
            A = np.zeros((h,w))

    return previous_blocks


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # first = None

    initial_time = time.time()
    image_sequence = []
    frame_shape = 0
    f = np.array([])
    while True:
        t, frame = cap.read()
        if t:
            cv2.imshow("current frame", frame)
            image_sequence.append(frame)
            frame_shape = frame.shape
            f = frame
            if time.time() - initial_time > 1 or cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    blocks, width, height =  minimum_indexed_block(image_sequence)
    a = reconstruct_background(blocks, width, height)

    print(a)
    cv2.imshow("hello", a)


    if cv2.waitKey(0) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

    # cap2 = cv2.VideoCapture(0)

    # new_time = time.time()
    #
    # t, frame1 = cap2.read()
    #
    # frame_blocks, w, h = create_blocks(frame1)
    #
    # new_scene = background_update(blocks, frame_blocks)
    #
    # if cv2.waitKey(0) == ord('q'):
    #     cap2.release()
    #     cv2.destroyAllWindows()

    #
    # cap1 = cv2.VideoCapture(0)
    # while True:
    #     h, frame2 = cap1.read()
    #     if h:
    #         cv2.imshow("hello", frame2)
    #         frame_blocks1, w, h = create_blocks(frame2)
    #
    #         new_scene = background_update(new_scene, frame_blocks1)
    #
    #
    #     if cv2.waitKey(0) == ord('q'):
    #         break
    #
    # cap1.release()
    # cv2.destroyAllWindows()
    #
    #
    r, c = blocks[0].value.shape
    new_s = a[r-1:, :]

    grayf = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    e =grayf[:, 639]
    # d = np.array([e])
    cv2.imshow("hi", e)
    if cv2.waitKey(0) == 'q':
        cv2.destroyAllWindows()

    print(grayf.shape)
    print(a.shape)
    print(new_s.shape)

    f = new_s[:, 639]

    cv2.imshow("hi", f)
    if cv2.waitKey(0) == 'q':
        cv2.destroyAllWindows()

    # cap3 = cv2.VideoCapture(0)
    # first = None
    #
    # first_time = time.time()
    # while True:
    #
    #     t, framei = cap3.read()
    #
    #     if t:
    #         # frame = imutils.resize(frame, width=500)
    #         # r, c = blocks[0].value.shape
    #         # h, w = a.shape
    #         # new_back = a[r: h, :]
    #         gray1 = cv2.cvtColor(framei, cv2.COLOR_BGR2GRAY)
    # #
    # #         width, height = gray1.shape
    # #
    # #         width_array = np.array([x for x in range(0, width, round(width/16))])
    # #         height_array  = np.array([y for y in range(0, height, round(height/16))])
    # #
    # #
    # #         gridx, gridy= np.meshgrid(height_array, width_array)
    # #
    # #         print(np.amax(height_array))
    # #         print(np.amax(width_array))
    # #
    # #         frame_height, frame_width = gray1.shape
    # #         print(gray1.shape)
    # #
    # #         border for frame; consider removing later if SAD doesn't work
    # #
    # #         frame = cv2.line(frame, (0, 0), (0, frame_width), (0,0,0), 1)
    # #         frame = cv2.line(frame, (0, 0), (frame_height, 0), (0,0,0), 1)
    # #         frame = cv2.line(frame, (frame_height, frame_width), (0, frame_width), (0,255,0), 1)
    # #         frame = cv2.line(frame, (frame_height, 0), (frame_height, frame_width),  (0,255,0), 1)
    # #
    # #
    # #
    # #         for t in width_array:
    # #             frame = cv2.line(frame, (0, t), (frame_width, t), (255,0, 255), 1)
    # #         for f in height_array:
    # #             frame = cv2.line(frame, (f, 0), (f, frame_height), (255,0,255), 1)
    # #
    #         gray = cv2.GaussianBlur(gray1, (21, 21), 0)
    # #
    # #         if first is None:
    # #             first_time = time.time()
    # #             first = gray
    # #             continue
    # #
    # #         if time.time() - first_time > 1:
    # #             first_time = time.time()
    # #             first = gray
    #
    # #
    # #
    #         frame_difference = cv2.absdiff(a, gray[:, :638])
    #         threshold = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)[1]
    #
    #         threshold = cv2.dilate(threshold, None, iterations = 2)
    #         countours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # #
    #         countours = imutils.grab_contours(countours)
    #
    #         for c in countours:
    #             if cv2.contourArea(c) < 10000:
    #                 continue
    # #
    #             (x, y, w, h) = cv2.boundingRect(c)
    #             frame = cv2.rectangle(framei, (x,y), (x+w, y+h) , (0,255,0), 1)
    # #
    # #
    # #
    #         cv2.imshow("constructionbox", framei)
    # #
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    # cap3.release()
    # cv2.destroyAllWindows()
