# python standard libraries
import os
import random
import fnmatch
import datetime
import pickle
import math

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


data_dir = '/Users/jeffrey/Desktop/train_images_new/'
file_list = os.listdir(data_dir)

image_paths = []
images = []
pattern = "*.bmp"
for filename in file_list:
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(data_dir,filename))
        #angle = int(round(float(filename[-8+filename[-9:-4].find('_'):-4])))
        #angle = int(filename[-7:-4])
        #steering_angles.append(angle)



def detect_edges(frame):
    # filter for blue lane lines
    frame = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #plt.imshow(hsv)
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("f", mask)
    plt.imshow(hsv)

    # detect edges
    edges = cv2.Canny(hsv, 200, 400)

    return edges


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        #logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                #logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    #print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_lane(frame):
    edges = detect_edges(frame)
    cropped_edges = edges
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)

    return lane_lines

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):int(height-50),:,:] 
    
    return image


def detect_angle(frame):
    edges = detect_edges(img_preprocess(frame))
    cropped_edges = edges
    line_segments = detect_line_segments(cropped_edges)
    lines = average_slope_intercept(frame, line_segments)

    slope = 1000 #straight
    if len(lines) == 1:
        x1, y1, x2, y2 = lines[0][0]
        if(x2 == x1): 
            slope = 1000
        else:
            slope = (120/(x2-x1))

    slope1 = 1000
    slope2 = 1000

    if len(lines) == 2:
        x1, y1, x2, y2 = lines[0][0]
        if(x2 == x1): 
            slope1 = 1000
        else:
            slope = (120/(x2-x1))
            x1, y1, x2, y2 = lines[1][0]
            if(x2 == x1): 
                slope2 = 1000
            else:
                slope2 = (120/(x2-x1))
                slope = ((slope1+slope2)/2)

        
    print (slope)
    angle = (math.atan(slope) * 180/math.pi + 360 )%360
    if(angle > 180 ):
        angle -= 180
    print(angle)
    return angle


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)


    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def cutImage(image):
    height, _, _ = image.shape
    image = image[0:int(height/2),:,:] 
    return image

# lane_lines_image = display_lines(cv2.imread(image_paths[to_show]), detect_lane(cutImage(cv2.imread(image_paths[to_show]))))

# cv2.imshow("he", lane_lines_image)

# cv2.waitKey(0)

# path_t = "/Users/jeffrey/Desktop/train_images_new/frame_1687758625878.bmp"
# lane_lines_image = display_lines(cv2.imread(path_t), detect_lane(cutImage(cv2.imread(path_t))))

# cv2.imshow("he", lane_lines_image)

# cv2.waitKey(0)

# # Close the window
# cv2.destroyAllWindows()




# START THE REAL CODE

import numpy as np
import cv2 as cv
import zmq
import random
import time
import numpy as np


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

driveAngle = 90

first = True

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://192.168.1.218:%s" % port)


cap = cv.VideoCapture('http://192.168.1.118:4747/mjpegfeed')

if not cap.isOpened():
 print("Cannot open camera")
 exit()
 
 
while True:
    
 #msg_bytes = socket.recv()
 
 # Capture frame-by-frame
 ret, frame = cap.read()
 # if frame is read correctly ret is True
 if not ret:
     print("Can't receive frame (stream end?). Exiting ...")
     break

 cv.imshow("frame", display_lines((frame), detect_lane(frame)))

 angle_to_steer = detect_angle(frame)
 if(abs(angle_to_steer - driveAngle) > 90):
    continue

 driveAngle = (angle_to_steer-driveAngle)*0.1 + driveAngle

 #print(int(driveAngle))


 rmin = 45
 rmax = 135

 driveFlip = driveAngle
 driveFlip = max(rmin, driveFlip)
 driveFlip = min(rmax, driveFlip)

 augmented_angle = translate(driveFlip, rmin, rmax, 0, 170)

 #cv.imwrite("/Users/jeffrey/Desktop/train_images_local/frame" + str(int(time.time()*1000)) + "_" + str(augmented_angle) + ".bmp", frame)


 print(int(augmented_angle))
 
 steerArray = [65, int(augmented_angle)]

 socket.send(bytearray(steerArray))

 #time.sleep(0.05)
 
 if cv.waitKey(1) == ord('p'):
   break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


# index = 34
# print(compute_steering_angle(cv2.imread(image_paths[index])))
# print(steering_angles[index])
# plt.imshow(img_preprocess(cv2.imread(image_paths[index])))