import cv2
import mediapipe as mp
import numpy as np
import glob

chessboardSize = (24,17)
frameSize = (1440,1080)
objpoints = [] 
imgpoints = []

print("calibrating...")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm
images = glob.glob('*.png')
for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
_,cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print("calibrating done.")

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
_, frame = cap.read()
color = (255,255,255)
windowName = "Hand Detector"
windowNameWithRemoval = "Hand detector with background removal"
h, w, c = frame.shape
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    fgmask = fgbg.apply(frame)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    img_width = frame.shape[1]
    img_height = frame.shape[0]

    # Calibrate the image
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (img_width,img_height), 1, (img_width,img_height))
    
    # Without filter
    dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
    x, y, img_width,img_height = roi
    dst = dst[y:y+img_height, x:x+img_width]
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (img_width,img_height), 5)
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, img_width,img_height = roi
    frame = dst[y:y+img_height, x:x+img_width]
    
    # With filter
    dst = cv2.undistort(fgmask, cameraMatrix, dist, None, newCameraMatrix)
    x, y, img_width,img_height = roi
    dst = dst[y:y+img_height, x:x+img_width]
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (img_width,img_height), 5)
    dst = cv2.remap(fgmask, mapx, mapy, cv2.INTER_LINEAR)
    x, y, img_width,img_height = roi
    fgmask = dst[y:y+img_height, x:x+img_width]

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.putText(fgmask,"{} , {}".format(str(x_min),str(y_min)),(x_min,y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
            cv2.putText(fgmask,"{} , {}".format(str(x_max),str(y_max)),(x_max,y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
            cv2.rectangle(fgmask, (x_min, y_min), (x_max, y_max), color, 2)

    cv2.imshow(windowName, frame)
    cv2.imshow(windowNameWithRemoval,fgmask)
    # Code to get outta loop
    k = cv2.waitKey(1)
    if k == 27:
        break
    
cap.release()