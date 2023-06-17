import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

while True:
    socket.send(input("angle:").encode())
    socket.send(input("go or st?:").encode())

