import sys
sys.path.append('C:/Users/Wilder Taborda/Desktop/universidad/lab4proce/camera')
from cameraHandler import getCoefficients,calibrate
import numpy as np
import cv2 as cv
import glob
import os.path
import time

cameraMatrix, dist, rvecs, tvecs,objpoints, imgpoints = getCoefficients()

cap = cv.VideoCapture(0)

while True:
    _,img = cap.read()
    img = cv.flip(img,1)
    h,  w = img.shape[:2]
    img = calibrate(img, cameraMatrix, dist,w,h)
    cv.imshow("img",img)
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()