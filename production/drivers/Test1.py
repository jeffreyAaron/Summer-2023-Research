# general
import numpy as np
import time
import math

# tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# imaging
import cv2 as cv
from imgaug import augmenters as img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# networking
import zmq


port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://192.168.1.218:%s" % port)



def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)




cap = cv.VideoCapture('http://192.168.1.118:4747/mjpegfeed')

while True:
    
    #msg_bytes = socket.recv()

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_r = cv.resize(frame, (200, 200))
    blur = cv.GaussianBlur(frame_r, (11,11), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    lower = np.array([60, 35, 35])
    upper = np.array([180, 180, 180])
    mask = cv.inRange(hsv, lower, upper)

    line_color = (0, 0, 0)

    height, width, _ = blur.shape
    blur = cv.rectangle(blur, (int(width/3*1), int(height/2)), (int(width/3*2), int(height)), line_color, thickness=-1) # fill a black rect
    mask = cv.rectangle(mask, (int(width/3*1), int(height/2)), (int(width/3*2), int(height)), line_color, thickness=-1) # fill a black rect

    canny = cv.Canny(mask, 50, 200, None, 3)

    lines = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 10)



    slopeSum = 0
    slopeCount = 0

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            x1, y1, x2, y2 = l
            cv.line(blur, (x1, y1), (x2, y2), (0,0,255), 3, cv.LINE_AA)
            if(x2 != x1):
                s = (y2-y1)/(x2-x1)
                a = math.atan2((y2-y1), (x2-x1))*180/math.pi
                if a < 0:
                    a += 180

                slopeSum += a
                slopeCount += 1


    if(slopeCount != 0):
        slopeFinal = slopeSum/slopeCount
        print(int(slopeFinal))
        rmin = 65
        rmax = 115

        # for high speed 100
        rmin = 70
        rmax = 110

        # for high speed 150
        # rmin = 65
        # rmax = 115


        print(slopeFinal)
        driveFlip = 180-slopeFinal
        driveFlip = max(rmin, driveFlip)
        driveFlip = min(rmax, driveFlip)

        augmented_angle = translate(driveFlip, rmin, rmax, 0, 170)
        socket.send("G".encode())
        steerArray = [65, int(augmented_angle)]
        socket.send(bytearray(steerArray))



        #cv.imwrite("/Users/jeffrey/Desktop/train_images_custom/frame" + str(int(time.time()*1000)) + "_" + str(augmented_angle) + ".bmp", frame)


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow("frame", blur)
    cv.imshow("canny", canny)
    cv.imshow("mask", mask)


    if cv.waitKey(1) == ord('p'):
        break

socket.send("S".encode())

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


