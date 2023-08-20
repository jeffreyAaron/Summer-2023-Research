import cv2 as cv
import math
import time
import numpy as np

# networking
import zmq
port = "5557"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://192.168.1.218:%s" % port)

# COLORS (B, G, R)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)

# Display
size = 400

# Data Vars
accelX = 0
accelY = 0
accelZ = 0

gyroX = 0
gyroY = 0
gyroZ = 0

smoothSpeed = 0

def parseData(dataBytes):
    dataPoints = dataBytes.decode().split(",")
    global gyroX, gyroY, gyroZ, accelX, accelY, accelZ
    gyroX = float(dataPoints[0])
    gyroY = float(dataPoints[1])
    gyroZ = float(dataPoints[2])

    accelX = float(dataPoints[3])
    accelY = float(dataPoints[4])
    accelZ = float(dataPoints[5])

def drawGraphics(frame):
    global smoothSpeed

    w = int(size/2)
    # draw the accel indicator
    cv.ellipse(frame, (w, w),  (w, w), 0, 0, 360, WHITE, thickness=10) # solid circle
    crsH = 3
    cv.line(frame, pt1=(w, 0), pt2=(w, w*2), color=WHITE, thickness=crsH)
    cv.line(frame, pt1=(0, w), pt2=(w*2, w), color=WHITE, thickness=crsH)

    centerX = int(-gyroZ*100) + w

    speed = math.sqrt(accelX**2 + accelY**2)

    smoothSpeed = smoothSpeed + 0.5 * (speed-smoothSpeed)

    centerY = int(-smoothSpeed*100) + w

    l = 20
    crsHS = 6
    cv.line(frame, pt1=(centerX-l, centerY), pt2=(centerX+l, centerY), color=WHITE, thickness=crsHS)
    cv.line(frame, pt1=(centerX, centerY-l), pt2=(centerX, centerY+l), color=WHITE, thickness=crsHS)



while True:
    canvas = np.zeros((size,size,3), np.uint8)

    socket.send(bytes(0)) 
    dataBytes = socket.recv()
    parseData(dataBytes)

    drawGraphics(canvas)
    cv.imshow("Data", canvas)

    if cv.waitKey(1) == ord('p'):
        break

