import numpy as np
import cv2 as cv
import zmq
import random
import time
import numpy as np

from RobotDriver import RobotDriver



cap = cv.VideoCapture('http://192.168.224.224:4747/mjpegfeed')

if not cap.isOpened():
 print("Cannot open camera")
 exit()
 
 
while True:

 # Capture frame-by-frame
 ret, frame = cap.read()
 # if frame is read correctly ret is True
 if not ret:
     print("Can't receive frame (stream end?). Exiting ...")
     break

 #cv.imshow("frame", frame)
 
 #robot.sendBytes(msg_bytes)
 #socket.send("D".encode())
 #print(msg_bytes)
 
 cv.imshow('frame', frame)
 
 print(cv.imwrite('/media/jeffrey/IMG_DATA/frame_' + str(int(time.time()*1000)) + '.bmp', frame))
 print(int(time.time()*1000))
   
 #print("Angle: " + str(angle))
 
 #steerArray = [65, angle]

 #robot.sendBytes(bytearray(steerArray))
 
 if cv.waitKey(1) == ord('p'):
   break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


