import cv2 as cv


cap = cv.VideoCapture('production/drivers/capture/20230712_225238.mp4')

if not cap.isOpened():
 print("Cannot open video")
 exit()

fps = int(cap.get(5))
print("Frame Rate : ",fps,"frames per second")  
 
  # Get frame count
frame_count = cap.get(7)
print("Frame count : ", frame_count)

while True:
    # msg_bytes = socket.recv()

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        continue
        #break

    cv.imshow("frame", frame)
    # the markers
    arucoDict = cv.aruco.getPredefinedDictionary(4)  # 4x4
    arucoParams = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(
        frame)

# # verify *at least* one ArUco marker was detected
# if len(corners) > 0:
#     # flatten the ArUco IDs list
#     ids = ids.flatten()
#     # loop over the detected ArUCo corners
#     for (markerCorner, markerID) in zip(corners, ids):
#         # extract the marker corners (which are always returned in
#         # top-left, top-right, bottom-right, and bottom-left order)
#         corners = markerCorner.reshape((4, 2))
#         (topLeft, topRight, bottomRight, bottomLeft) = corners
#         # convert each of the (x, y)-coordinate pairs to integers
#         topRight = (int(topRight[0]), int(topRight[1]))
#         bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
#         bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
#         topLeft = (int(topLeft[0]), int(topLeft[1]))
#         # draw the bounding box of the ArUCo detection
#         cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
#         cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
#         cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
#         cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
#         # compute and draw the center (x, y)-coordinates of the ArUco
#         # marker
#         cX = int((topLeft[0] + bottomRight[0]) / 2.0)
#         cY = int((topLeft[1] + bottomRight[1]) / 2.0)
#         cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
#         # draw the ArUco marker ID on the image
#         cv2.putText(image, str(markerID),
#             (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
#             0.5, (0, 255, 0), 2)

# cv.imshow("image", image)
