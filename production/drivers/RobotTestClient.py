import zmq
import random
import sys
import time
import serial

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://192.168.1.218:%s" % port)

robotSerial = serial.Serial(
            port="/dev/cu.usbserial-110",
            baudrate=115200,
            timeout=1
        )

while True:
    robotSerial.write(str.encode("A"))
    fwd = robotSerial.readline().rstrip()
    steer = robotSerial.readline().rstrip()

    print(fwd)
    print(steer)

    if(steer.isdigit()) :

        fwd = int(fwd)

        if(fwd > 0 ) :
            socket.send("G".encode())

        elif(fwd < 0 ) :
            socket.send("B".encode())
        else :
            socket.send("S".encode())
    

        steer = min([int(steer), 180])

        steerArray = [65, steer]

        socket.send(bytearray(steerArray))
        #socket.recv()
        time.sleep(0.05)

    

