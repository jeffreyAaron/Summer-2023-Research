import cv2 as cv
import numpy as np
import math

import time

cap = cv.VideoCapture(0)
time.sleep(2) # give time for camera to start

# Constants
IMAGE_WIDTH = 1920 
IMAGE_HEIGHT = 1080

DOWN_SCALE_FACTOR = 4


CENTER_W = int(IMAGE_WIDTH/2)
CENTER_H = int(IMAGE_HEIGHT/2)

# DISPLAY
MAJOR_ARROW_LENGTH = 300
MINOR_ARROW_LENGTH = 100



# COLORS (B, G, R)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)

# global vars
angle_to_turn = 0
speed_of_car = 50

accelX = 0
accelY = 0
accelZ = 0

gyroX = 0
gyroY = 0
gyroZ = 0


# networking
import zmq
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://192.168.1.218:%s" % port)


def processFrame(frame):
    resize = cv.resize(frame, (int(IMAGE_WIDTH/DOWN_SCALE_FACTOR), int(IMAGE_HEIGHT/DOWN_SCALE_FACTOR)))
    blur = cv.GaussianBlur(resize, (51,51), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    lower = np.array([int(217/2-50), 20, 20])
    upper = np.array([int(217/2+30), 225, 225])

    # more restrictive
    # lower = np.array([int(217/2-15), 100, 100])
    # upper = np.array([int(217/2+25), 180, 180])

    mask = cv.inRange(hsv, lower, upper)

    return mask


def getEquationOfPath(frame, power):
    coords = np.column_stack(np.where(frame > 200))*DOWN_SCALE_FACTOR # retrieve all the white pixels
    row, col = coords.shape
    try:
        x = coords[0:row, :1].squeeze()
        y = coords[0:row, 1:2].squeeze() 
        coeff = np.polyfit(x, y, power)


        return coeff
    except Exception as e:
        return None

def path(x0,y0,xf,yf,traj_coeff):
    a = float(traj_coeff[0])
    b = float(traj_coeff[1])
    c = float(traj_coeff[2])
    d = float(traj_coeff[3])

    # A = np.array([
    #     [a*x0**3, b*x0**2, c*x0, d],
    #     [3*a*x0**2, 2*b*x0, c, 0],
    #     [a*xf**3, b*xf**2, c*xf, d],
    #     [3*a*xf**2, 2*b*xf, c, 0]
    # ])
    A = np.array([
        [x0**3, x0**2, x0, 1],
        [3*x0**2, 2*x0, 1, 0],
        [xf**3, xf**2, xf, 1],
        [3*xf**2, 2*xf, 1, 0]
    ])

    b = np.array([
        [float(y0)],
        [0],
        [float(yf)],
        [float(3*a*xf**2 + 2*b*xf + c)]
    ])

    coeff = np.linalg.inv(A)@b 

    return coeff.squeeze()

def evalPolynomial(coeff, power, lspace):
    polynomial_line_of_fit = 0
    for i in range(0, power+1): 
        polynomial_line_of_fit += coeff[i]*(lspace**(power-i))
    return polynomial_line_of_fit

def drawPolynomialEquation(frame, coeff, power, thickness=10, Xrestricted = False, x_start = 0, x_end = 0, color = BLUE, color1 = RED, gradient = False):
    polynomial_line_of_fit = 0
    lspace = np.linspace(0,  frame.shape[0]-1, frame.shape[0])
    if(Xrestricted):
        lspace = np.linspace(int(x_start), int(x_end), int(x_end-x_start))

    polynomial_line_of_fit = evalPolynomial(coeff, power, lspace)
    
    verts = np.array(list(zip(polynomial_line_of_fit.astype(int),lspace.astype(int))))
    
    if(gradient):
        row, col = verts.shape
        curvature = findCurvature(coeff, power, lspace)
        maxCurve = np.max(curvature)
        minCurve = np.min(curvature)
        for i in range(0, row-1):
            
            colorCurve = interpolateColors(minCurve, maxCurve, curvature[i], color, color1)
            cv.line(frame, pt1=verts[i], pt2=verts[i+1], color=colorCurve, thickness=thickness)
        
    else:
        cv.polylines(frame,[verts],False,color,thickness)

def interpolateColors(startRange, endRange, value, minColor, maxColor):
    range = endRange-startRange
    percent = (value-startRange)/range
    return tuple(np.asarray(maxColor)*percent + np.asarray(minColor)*(1-percent)) 


def findCurvature(coeff, power, lspace):
    firstDerivative = 0
    for i in range(0, power+1):
        firstDerivative += (power-i)*coeff[i]*(lspace**(power-i-1))

    secondDerivative = 0
    for i in range(0, power+1):
        secondDerivative += (power-i-1)*(power-i)*coeff[i]*(lspace**(power-i-2))


    curvature = abs(secondDerivative)/((1+firstDerivative**2)**1.5)

    curvature = np.nan_to_num(curvature, nan=0)
    
    return curvature



def drawGraphics(frame):
    # center line and circle
    t = 10
    cv.ellipse(frame, (CENTER_W, IMAGE_HEIGHT),  (MINOR_ARROW_LENGTH, MINOR_ARROW_LENGTH), 0, 0, 360, BLACK, thickness=t)
    cv.ellipse(frame, (CENTER_W, IMAGE_HEIGHT),  (MAJOR_ARROW_LENGTH, MAJOR_ARROW_LENGTH), 0, 0, 360, BLACK, thickness=t)
    cv.line(frame, (CENTER_W, IMAGE_HEIGHT-MINOR_ARROW_LENGTH), (CENTER_W, IMAGE_HEIGHT),  BLACK, thickness=t)
    drawAngleLine(frame, MAJOR_ARROW_LENGTH, angle_to_turn)

    # draw the accel indicator
    w = 200
    cv.ellipse(frame, (w, w),  (w, w), 0, 0, 360, BLACK, thickness=10) # solid circle
    crsH = 3
    cv.line(frame, pt1=(w, 0), pt2=(w, w*2), color=BLACK, thickness=crsH)
    cv.line(frame, pt1=(0, w), pt2=(w*2, w), color=BLACK, thickness=crsH)

    centerX = int(accelX*100) + w
    centerY = int(accelY*100) + w

    l = 20
    crsHS = 6
    cv.line(frame, pt1=(centerX-l, centerY), pt2=(centerX+l, centerY), color=BLACK, thickness=crsHS)
    cv.line(frame, pt1=(centerX, centerY-l), pt2=(centerX, centerY+l), color=BLACK, thickness=crsHS)



def drawAngleLine(frame, radius, angle, x=CENTER_W, y=IMAGE_HEIGHT, color=BLACK, thickness = 10):
    cv.line(frame, (int(x), int(y)), 
        (int(x+radius*math.sin(angle/180*math.pi)), 
        int(y-radius*math.cos(angle/180*math.pi))),
        color, thickness)

def calculateCommandBasedOnSpeed(speed):
    return int(speed_of_car/255*(255-180)+180)

def parseData(dataBytes):
    dataPoints = dataBytes.decode().split(",")
    global gyroX, gyroY, gyroZ, accelX, accelY, accelZ
    gyroX = float(dataPoints[0])
    gyroY = float(dataPoints[1])
    gyroZ = float(dataPoints[2])

    accelX = float(dataPoints[3])
    accelY = float(dataPoints[4])
    accelZ = float(dataPoints[5])

    

i = 0
wantsToStart = False
while True:
    ret, frame = cap.read()
    # frame = cv.imread("./testimage.png")

    # Initial screen
    if(not wantsToStart):
        cv.imshow("frame", frame)
        socket.send(bytearray(0)) # center wheels
        wantsToStart = True
        cv.waitKey(0)
        #cv.imwrite("./testimage.png", frame)


    frameP = processFrame(frame)
    coeff = getEquationOfPath(frameP, power=3)

    print("frame " + str(i))
    i+=1
    if(not (coeff is None)):
        drawPolynomialEquation(frame, coeff, power=3, thickness=25, gradient=True)

        startX = IMAGE_HEIGHT
        startY = CENTER_W

        targetX = int(CENTER_H*0.25)
        targetY = evalPolynomial(coeff, power=3, lspace=np.array([targetX]))[0]

        coeffP = path(startX, startY, targetX, targetY, coeff)
        drawPolynomialEquation(frame, coeffP, power=3, thickness=10, color=BLACK, color1=GRAY, Xrestricted=True, x_start=targetX, x_end=startX, gradient=True)

        # calc angle
        angle_to_turn = -math.atan(((targetY-startY)/(targetX-startX)))*180/math.pi
        augmented_angle = 3.72 * angle_to_turn/2 + 66.5 # calibration

        steerArray = [int(min(180, max(0, augmented_angle))), int(min(181, max(200, 200+15-abs(angle_to_turn))))] # range of values for servo is 0-180
        socket.send(bytearray(steerArray))
        dataBytes = socket.recv()
        parseData(dataBytes)

    
    drawGraphics(frame)
    # if not ret:
    #     print("Can't receive frame (stream end?). Exiting ...")
    #     break

    cv.imshow("frame", frame)
    # cv.imshow("frameP", frameP)

    if cv.waitKey(1) == ord('p'):
        socket.send(bytearray([181])) # stop the car
        break



