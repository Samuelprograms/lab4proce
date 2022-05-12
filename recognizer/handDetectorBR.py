import sys
sys.path.append('C:/Users/Wilder Taborda/Desktop/universidad/lab4proce/playerMovement')
from getPlayerMovement import handlePositionPlayer
import numpy as np
import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
windowName = "Hand detector"
windowNameWithRemoval = "Hand detector with background removal"
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
color = (255, 255, 255)
_, frame = cap.read()
h, w, c = frame.shape

while True:
  _,frame = cap.read()
  frame = cv2.flip(frame, 1)
  fgmask = fgbg.apply(frame)
  framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = hands.process(framergb)
  hand_landmarks = result.multi_hand_landmarks

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
      
  cv2.imshow(windowName,frame)
  cv2.imshow(windowNameWithRemoval,fgmask)
  k = cv2.waitKey(30)
  if k == 27:
    break
cap.release()
cv2.destroyAllWindows()