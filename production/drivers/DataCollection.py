import numpy as np
import cv2 as cv
import zmq
import random
import time
import numpy as np

angle = 90

snapTaken = False

from RobotDriver import RobotDriver

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:%s" % port)

robot = RobotDriver(prt="/dev/ttyUSB0")


cap = cv.VideoCapture('http://192.168.224.224:4747/mjpegfeed')

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

 cv.imshow("frame", frame)
 
 #robot.sendBytes(msg_bytes)
 #socket.send("D".encode())
 #print(msg_bytes)
 
 if cv.waitKey(1) == ord('a'):
   angle = max(angle - 1, 0)
   snapTaken = False

 if cv.waitKey(1) == ord('d'):
   angle = max(angle + 1, 0)
   snapTaken = False
   
 if cv.waitKey(1) == ord('e'):
   snapTaken = False
   
 if cv.waitKey(1) == ord('q') and (not snapTaken):
   snapTaken = True
   cv.imwrite('/media/jeffrey/IMG_DATA/frame_' + str(int(time.time()*1000)) + "_" + str(angle) + '.bmp', frame)

   
 print("Angle: " + str(angle))
 
 steerArray = [65, angle]

 robot.sendBytes(bytearray(steerArray))
 
 if cv.waitKey(1) == ord('p'):
   break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

