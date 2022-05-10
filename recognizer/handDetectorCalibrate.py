import cv2
import mediapipe as mp
import numpy as np
import glob

chessboardSize = (24,17)
frameSize = (1440,1080)
objpoints = [] 
imgpoints = []

def handlePositionPlayer(x_min,y_min,x_max,y_max,img_width,img_height):
  nextState = ""
  playerIsJump = ""
  playerDirection = ""
  
  mean_x = (x_min + x_max) / 2
  mean_y = (y_min + y_max) / 2

  if(mean_y < img_height/2):
    playerIsJump = "is going up"
  elif(mean_y >= img_height/2):
    playerIsJump = "is going down"

  if(mean_x < img_width/3):
    playerDirection = "to left"
  elif(mean_x >=img_width/3 and mean_x <= img_width*2/3):
    playerDirection = "in center"
  elif(mean_x > img_width*2/3):
    playerDirection = "to right"
  nextState = "the player {} and {}".format(playerIsJump,playerDirection)
  return nextState

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
        cv2.waitKey(1000)
_,cameraMatrix2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
cameraMatrix = cameraMatrix2
dist = dist2
rvecs = rvecs2
tvecs = tvecs2
print("calibrating done.")

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
_, frame = cap.read()
color = (0,0,255)
window_name = "Hand Detector"
h, w, c = frame.shape

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    img_width = frame.shape[1]
    img_height = frame.shape[0]

    # Calibrate the image
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (img_width,img_height), 1, (img_width,img_height))
    dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
    x, y, img_width,img_height = roi
    dst = dst[y:y+img_height, x:x+img_width]
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (img_width,img_height), 5)
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, img_width,img_height = roi
    frame = dst[y:y+img_height, x:x+img_width]
    
    lineVerticalOne = [(int(img_width/3),0),(int(img_width/3),img_height)] 
    lineVerticalTwo = [(int(img_width*2/3),0),(int(img_width*2/3),img_height)] 
    lineHorizontalOne = [(0,int(img_height/3*1.5)),(img_width,int(img_height/3*1.5))] 

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
            cv2.line(frame, lineVerticalOne[0], lineVerticalOne[1],color)
            cv2.line(frame, lineVerticalTwo[0], lineVerticalTwo[1],color)
            cv2.line(frame, lineHorizontalOne[0], lineHorizontalOne[1],color)
            cv2.putText(frame,"{} , {}".format(str(x_min),str(y_min)),(x_min,y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
            cv2.putText(frame,"{} , {}".format(str(x_max),str(y_max)),(x_max,y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            playerMovement = handlePositionPlayer(x_min,y_min,x_max,y_max,img_width,img_height)
            print(playerMovement)

    cv2.imshow(window_name, frame)
    # Code to get outta loop
    k = cv2.waitKey(1)
    if k == 27:
        break
    
cap.release()