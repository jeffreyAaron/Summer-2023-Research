# python standard libraries
import os
import random
import fnmatch
import datetime
import pickle
import statistics

# data processing
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "%.4f" % x})

import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.max_colwidth', 200)

# tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
print( f'tf.__version__: {tf.__version__}' )
print( f'keras.__version__: {keras.__version__}' )

# sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# imaging
import cv2
from imgaug import augmenters as img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


model_output_dir = "/Users/jeffrey/Downloads/Deep-Pi-Car/SavedModels/"


model = load_model(os.path.join(model_output_dir,'lane_navigation_final.h5'))

def img_preprocess(image):
    height, _, _ = image.shape
    frame_r = cv2.resize(image, (200, 200))
    # blur = cv2.GaussianBlur(frame_r, (11,11), 0)
    hsv = cv2.cvtColor(frame_r, cv2.COLOR_BGR2HSV)

    return hsv /255

to_show = 12
#plt.imshow(img_preprocess(cv2.imread(image_paths[to_show])))
#print(steering_angles[to_show])

def compute_steering_angle(frame):
    preprocessed = img_preprocess(frame)
    X = np.asarray([preprocessed])
    steering_angle = model.predict(X)[0]
    return steering_angle

import numpy as np
import cv2 as cv
import zmq
import random
import time
import numpy as np


snapTaken = False


port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://192.168.1.218:%s" % port)


cap = cv.VideoCapture('http://192.168.1.118:4747/mjpegfeed')

if not cap.isOpened():
 print("Cannot open camera")
 exit()
 

count = 0

socket.send(bytearray([71]))

recentValues = []

steerArray = [0]

while True:
 #msg_bytes = socket.recv()
 
 # Capture frame-by-frame
 ret, frame = cap.read()
 # if frame is read correctly ret is True
 if not ret:
     print("Can't receive frame (stream end?). Exiting ...")
     break

 cv.imshow("frame", img_preprocess(frame))

 count += 1

 if(count > 3):
    count = 0
    angle_to_steer = compute_steering_angle(frame)


    image = frame.copy()
    
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
   

    print(int(angle_to_steer[0]))
    
    # steerArray[0] = int(steerArray[0] * 0.5 + 0.5 * abs(angle_to_steer[0] + 10))

    recentValues.append(steerArray[0])

    if(len(recentValues) > 4):
        recentValues.pop(0)
        steerArray[0] = int(statistics.median(recentValues))



    socket.send(bytearray(steerArray))

 #time.sleep(0.1)
 
 if cv.waitKey(1) == ord('p'):
   socket.send(bytearray([83]))
   break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


# index = 34
# print(compute_steering_angle(cv2.imread(image_paths[index])))
# print(steering_angles[index])
# plt.imshow(img_preprocess(cv2.imread(image_paths[index])))