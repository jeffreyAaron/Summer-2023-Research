import time
import numpy as np
import zmq
import random
import sys

from RobotDriver import RobotDriver

dataLat = []
dataLon = []

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % port)

robot = RobotDriver(prt="/dev/ttyUSB0")

#dataFile = open("/media/jeffrey/IMG_DATA/angles.txt","w")

#try:
while(True):
    msg_bytes = socket.recv()
    robot.sendBytes(msg_bytes)
    print(msg_bytes)
        #if(len(msg_bytes) == 2):
        #    angle = msg_bytes[1]
        #    print(angle)
        #    dataFile.write(str(angle))
        #    dataFile.write(" " + str(int(time.time()*1000)) + "\n")

#except:
    #dataFile.close()


print("done")


