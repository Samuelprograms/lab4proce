import sys
sys.path.append('C:/Users/Wilder Taborda/Desktop/universidad/lab4proce/camera')
from cameraHandler import getCoefficients,calibrate
import numpy as np
import cv2 as cv
import glob
import os.path
import time

cameraMatrix, dist, rvecs, tvecs,objpoints, imgpoints = getCoefficients()

fps = 10 

cap = cv.VideoCapture(0)
# Read the image of the video capture constantly
while True:
    startTime = time.time()
    # Get the image
    _,img = cap.read()

    # Flip the image
    img = cv.flip(img,1)

    # Get the dimensions of the image
    h,  w = img.shape[:2]

    # Calibrate the image
    img = calibrate(img, cameraMatrix, dist,w,h)

    # Reprojection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    error =  "total error: {}".format(mean_error/len(objpoints))

    # Display the image
    cv.imshow("img",img)

    # Code to get outta loop

    k = cv.waitKey(10)
    operationPerSecond = 1/(time.time() - startTime)
    print(operationPerSecond)
    if k == 27:
        break

cap.release()