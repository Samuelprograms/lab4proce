import sys
sys.path.append('C:/Users/Wilder Taborda/Desktop/universidad/lab4proce/camera')
from cameraHandler import getCoefficients
import numpy as np
import cv2 as cv
import glob
import os.path

cameraMatrix, dist, rvecs, tvecs,objpoints, imgpoints = getCoefficients()

cap = cv.VideoCapture(0)

# Read the image of the video capture constantly
while True:

    # Get the image
    _,img = cap.read()

    # Flip the image
    img = cv.flip(img,1)

    # Get the dimensions of the image
    h,  w = img.shape[:2]

    # Get the camera matrix and distortion coefficients
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    error =  "total error: {}".format(mean_error/len(objpoints))

    # Display the image
    cv.imshow("img",dst)

    # Code to get outta loop
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()