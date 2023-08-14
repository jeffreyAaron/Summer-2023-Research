# general
import numpy as np
import time
import math

# tensorflow
# import tensorflow as tf
# import keras
# from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
# from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
# from keras.optimizers import Adam
# from keras.models import load_model
# from keras.callbacks import ModelCheckpoint

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




#cap = cv.VideoCapture('http://192.168.1.217:4747/mjpegfeed')
cap = cv.VideoCapture(0)


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
    # mask = cv.rotate(mask, cv.ROTATE_90_CLOCKWISE)

    line_color = (0, 0, 0)

    height, width, _ = blur.shape
    # mask = blur[0:int(3*height/4), 0:width]
    # mask = cv.rectangle(blur, (int(width/3*1), int(3*height/4)), (int(width/3*2), int(height)), line_color, thickness=-1) # fill a black rect

    canny = cv.Canny(mask, 50, 200, None, 3)

    lines = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 10)



    slopeSum = 0
    slopeCount = 0

    canvas = np.zeros((height,width,3), np.uint8)

    # Draw the lines
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         l = lines[i][0]
    #         x1, y1, x2, y2 = l
    #         cv.line(canvas, (x1, y1), (x2, y2), (0,0,255), 3, cv.LINE_AA)
    #         if(x2 != x1):
    #             s = (y2-y1)/(x2-x1)
    #             a = math.atan2((y2-y1), (x2-x1))*180/math.pi
    #             if a < 0:
    #                 a += 180

    #             slopeSum += a
    #             slopeCount += 1


    # if(slopeCount != 0):
    #     slopeFinal = slopeSum/slopeCount
    #     print(int(slopeFinal))
    #     rmin = 65
    #     rmax = 115

        # for high speed 100
        #rmin = 70
        #rmax = 110

        # for high speed 150
        # rmin = 65
        # rmax = 115


        # print(slopeFinal)
        #driveFlip = 180-slopeFinal
        # driveFlip = max(rmin, slopeFinal)
        # driveFlip = min(rmax, driveFlip)

        # augmented_angle = translate(slopeFinal, rmin, rmax, 0, 100)
        # augmented_angle = 4.28409 * slopeFinal -316.875
        # # socket.send("G".encode())
        # # augmented_angle = 130
        # print(augmented_angle)
        # steerArray = [int(min(255, max(0, augmented_angle)))]
        # socket.send(bytearray(steerArray))




        #cv.imwrite("/Users/jeffrey/Desktop/train_images_custom/frame" + str(int(time.time()*1000)) + "_" + str(augmented_angle) + ".bmp", frame)


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # LINE FITTING
    coords = np.column_stack(np.where(canny > 200))
    row, col = coords.shape
    
    try:
        x = coords[0:row, :1].squeeze()
        y = coords[0:row, 1:2].squeeze() 
        lspace = np.linspace(0,  mask.shape[0]-1, mask.shape[0])
        power = 1
        coeff = np.polyfit(x, y, power)

        polynomial_line_of_fit = 0
        for i in range(0, power+1):
            if(i == power):
                polynomial_line_of_fit += coeff[i]
            else:
                polynomial_line_of_fit += coeff[i]*(lspace**(power-i))
    # linePlot = plt.axes()
    # linePlot.imshow(mask, aspect='auto') 
    # linePlot.plot(polynomial_line_of_fit, color="green")

    # find the highest point and the lowest point to find slope



        augmented_angle = 3.72 * math.atan(-coeff[0])*180/math.pi + 66.5
        

        verts = np.array(list(zip(polynomial_line_of_fit.astype(int),lspace.astype(int))))
        # blur = cv.rotate(blur, cv.ROTATE_90_CLOCKWISE)
        cv.polylines(blur,[verts],False,(0,0,255),thickness=3)

        center = int(width/2)

        cv.ellipse(blur, (center, height), (20, 20),
           0, 0, 360, (0,0,0), 2)

        cv.line(blur, (center,0), (center,height),(255,0,0),1)
        display_angle = math.atan(-coeff[0])*180/math.pi
        cv.line(blur, (int(center + height*math.tan(display_angle/180*math.pi)),0), (center,height),(255,0,0),1)

        #print(coeff[1])
        #print(coeff[0])
        pt = 0
        highPoint = (int((coeff[0]*pt)+coeff[1]),pt)
        pt = height
        lowPoint = (int((coeff[0]*pt)+coeff[1]),pt)


        # lowPoint = (int((height-coeff[1])/coeff[0]),int(height))

        cv.line(blur, highPoint, lowPoint,(255,0,0),1)

        angleOffset = math.atan((highPoint[0]-center)/height)*180/math.pi

        augmented_angle = 3.72 * angleOffset*1.5 + 66.5
        cv.line(blur, (int(center + height*math.tan(angleOffset/180*math.pi)),0), (center,height),(0,255,0),2)


        # print(augmented_angle)
        steerArray = [int(min(255, max(0, augmented_angle)))]
        # socket.send(bytearray(steerArray))

        cv.putText(blur, "xOff: "+str(int(highPoint[0]-center)), (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(blur, "angle: "+str(int(angleOffset)), (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)



        # blur = cv.rotate(blur, cv.ROTATE_90_COUNTERCLOCKWISE)

    except Exception as e:
        print("no line detected")


    cv.imshow("canny", canny)
    cv.imshow("mask", mask)
    cv.imshow("frame", blur)


    if cv.waitKey(1) == ord('p'):
        break

    
# socket.send("S".encode())

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


