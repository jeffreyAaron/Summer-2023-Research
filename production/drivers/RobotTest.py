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

robot = RobotDriver(prt="/dev/cu.usbserial-110")

while(True):
    msg = socket.recv()
    robot.setSteering(int(msg))
    print(int(msg))


print("done")


