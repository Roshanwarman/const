import cv2, numpy as np, os


cap = cv2.VideoCapture(0)

org, font, scale, color, thickness, linetype = (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (234,12,123), 2, cv2.LINE_AA
h,s,v,h1,s1,v1 = 16,0,64,123,111,187
h,s,v,h1,s1,v1 = 0,74,53,68,181,157
data_size = 1000
training_to_test = .75
img_size = 100
height, width = 480,640

#
# lower, upper = np.array([h,s,v]), np.array([h1,s1,v1])
# mask = cv2.inRange(hsv, lower, upper)


def bbox(img):
    try:
        bg = np.zeros((1000,1000), np.uint8)
        bg[250:250+480, 250:250+640] = img
        _, contours, _  = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key = cv2.contourArea)
        rect = cv2.boundingRect(largest_contour)
        circ = cv2.minEnclosingCircle(largest_contour)
        x,y,w,h = rect
        x,y = x+w/2,y+h/2
        x,y = x+250, y+250
        ddd = 200
        return bg[y-ddd:y+ddd, x-ddd:x+ddd]
    except: return img


def largest_contour(contours):
    c = max(contours, key=cv2.contourArea)
    return c[0]


def contour_center(c):
    M = cv2.moments(c)
    try: center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    except: center = 0,0
    return center


def only_color(img, t):
    h,s,v,h1,s1,v1 = t
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h,s,v]), np.array([h1,s1,v1])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((15,15), np.uint)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res, mask
def flatten(dimData, images):
    images = np.array(images)
    images = images.reshape(len(images), dimData)
    images = images.astype('float32')
    images /=255
    return images

images, labels = [],[]

tool_name = ''
patterns = []
tool_num = 0
while True:
    _, img = cap.read()
    cv2.putText(img, 'class', org, font, scale, color, thickness, linetype)
    cv2.putText(img, 'hiiiiiiiii', (50,100), font, scale, color, thickness, linetype)
    cv2.putText(img, tool_name, (50,300), font, 3, (0,0,255), 5, linetype)

    cv2.line(img, (330,240), (310,240), (234,123,234), 3)
    cv2.line(img, (320,250), (320,230), (234,123,234), 3)
    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if k>10: tool_name += chr(k)
    if k == 27: break
    current = 0
    if k == 13:
        while current < data_size:
            _, img = cap.read()
            img, mask = only_color(img, (h,s,v,h1,s1,v1))
            mask = bbox(mask)
            images.append(cv2.resize(mask, (img_size, img_size)))
            labels.append(tool_num)
            current += 1
            cv2.line(img, (330,240), (310,240), (234,123,234), 3)
            cv2.line(img, (320,250), (320,230), (234,123,234), 3)
            cv2.putText(img, 'collecting data', org, font, scale, color, thickness, linetype)
            cv2.putText(img, 'data for'+tool_name+':' + str(current), (50,100), font, scale, color, thickness, linetype)
            cv2.imshow('img', img)
            cv2.imshow('img', mask)
            k = cv2.waitKey(1)
            if k == ord('p'): cv2.waitKey(0)
            if current == data_size:
                patterns.append(tool_name)
                tool_name = ''
                tool_num += 1

                print(tool_num)
                break


to_train= 0
train_images, test_images, train_labels, test_labels = [],[],[],[]
for image, label in zip(images, labels):
    if to_train<3:
        train_images.append(image)
        train_labels.append(label)
        to_train+=1
    else:
        test_images.append(image)
        test_labels.append(label)
        to_train = 0

from keras.utils import to_categorical


dataDim = np.prod(images[0].shape)
train_data  = flatten(dataDim, train_images)
test_data = flatten(dataDim, test_images)

# train_labels = np.array(train_labels)
# test_labels = np.array(test_labels)
# train_labels_one_hot = to_categorical(train_labels)
# test_labels_one_hot = to_categorical(test_labels)



train_labels  = np.array(train_labels)

test_labels =  np.array(test_labels)

trains

classes = np.unique(train_labels)
nClasses  = len(classes)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = (dataDim,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size = 256, epochs=5, verbose=1,
                    validation_data=(test_data, test_labels_one_hot))


[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


def draw_prediction(bg,prediction, motion):
    idxs = [1,2,3,4,5,6,7,8,9]
    for i, pattern, idx in zip(prediction, patterns, idxs):
        text = pattern + ' '+str(round(i,3))
        scale = i*2

        if motion: scale = .4
        if scale<.95: scale = .95
        thickness = 1
        if scale>1.5: thickness = 2
        if scale>1.95: thickness = 4
        scale = scale*.75
        org, font, color = (350, idx*70), cv2.FONT_HERSHEY_SIMPLEX, (0,0,0)
        cv2.putText(bg, text, org, font, scale, color, thickness, cv2.LINE_AA)
    return bg

def draw_bg(prediction):
    motion = False
    bg = np.zeros((1150,1000,3), np.uint8)
    idxs = [1,2,3,4,5,6,7,8,9]
    for i, pattern, idx in zip(prediction, patterns, idxs):
        text = pattern + ' '+str(round(i,3))
        scale = i*2

        if motion: scale = .4
        if scale<.95: scale = .95
        thickness = 1
        if scale>1.5: thickness = 2
        if scale>1.95: thickness = 4
        scale = scale*2
        org, font, color = (200, idx*140), cv2.FONT_HERSHEY_SIMPLEX, (12,234,123)
        cv2.putText(bg, text, org, font, scale, (255,255,255), 1+thickness, cv2.LINE_AA)
    return bg

from keras.models import load_model
dimData = np.prod([img_size, img_size])
while True:
    _, img= cap.read()
    _, mask = only_color(img, (h,s,v,h1,s1,v1))
    mask = bbox(mask)
    mask = cv2.resize(mask, (img_size, img_size))
    cv2.imshow('display', mask)
    mask = mask.reshape(dimData)
    mask = mask.astype('float32')
    mask /=255
    prediction = model.predict(mask.reshape(1,dimData))[0].tolist()
    img = draw_prediction(img, prediction, False)
    display = draw_bg(prediction)

    cv2.imshow('img', img)
    k = cv2.waitKey(10)
    if k == 27: break

cap.release()
cv2.destroyAllWindows()

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = (dataDim,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size = 256, epochs=5, verbose=1,
                    validation_data=(test_data, test_labels_one_hot))


[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


def draw_prediction(bg,prediction, motion):
    idxs = [1,2,3,4,5,6,7,8,9]
    for i, pattern, idx in zip(prediction, patterns, idxs):
        text = pattern + ' '+str(round(i,3))
        scale = i*2

        if motion: scale = .4
        if scale<.95: scale = .95
        thickness = 1
        if scale>1.5: thickness = 2
        if scale>1.95: thickness = 4
        scale = scale*.75
        org, font, color = (350, idx*70), cv2.FONT_HERSHEY_SIMPLEX, (0,0,0)
        cv2.putText(bg, text, org, font, scale, color, thickness, cv2.LINE_AA)
    return bg

def draw_bg(prediction):
    motion = False
    bg = np.zeros((1150,1000,3), np.uint8)
    idxs = [1,2,3,4,5,6,7,8,9]
    for i, pattern, idx in zip(prediction, patterns, idxs):
        text = pattern + ' '+str(round(i,3))
        scale = i*2

        if motion: scale = .4
        if scale<.95: scale = .95
        thickness = 1
        if scale>1.5: thickness = 2
        if scale>1.95: thickness = 4
        scale = scale*2
        org, font, color = (200, idx*140), cv2.FONT_HERSHEY_SIMPLEX, (12,234,123)
        cv2.putText(bg, text, org, font, scale, (255,255,255), 1+thickness, cv2.LINE_AA)
    return bg

from keras.models import load_model
dimData = np.prod([img_size, img_size])
while True:
    _, img= cap.read()
    _, mask = only_color(img, (h,s,v,h1,s1,v1))
    mask = bbox(mask)
    mask = cv2.resize(mask, (img_size, img_size))
    cv2.imshow('display', mask)
    mask = mask.reshape(dimData)
    mask = mask.astype('float32')
    mask /=255
    prediction = model.predict(mask.reshape(1,dimData))[0].tolist()
    img = draw_prediction(img, prediction, False)
    display = draw_bg(prediction)

    cv2.imshow('img', img)
    k = cv2.waitKey(10)
    if k == 27: break

cap.release()
cv2.destroyAllWindows()
