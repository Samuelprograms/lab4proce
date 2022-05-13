import numpy as np
import cv2 as cv
import glob

path = 'C:/Users/Wilder Taborda/Desktop/universidad/lab4proce/camera/images2'

def getCoefficients():
  print("Calibrating...")
  chessboardSize = (3,3)
  frameSize = (1440,1080)
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
  objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
  size_of_chessboard_squares_mm = 20
  objp = objp * size_of_chessboard_squares_mm
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.
  images = glob.glob(path+'/*.jpg')
  for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if ret:
      objpoints.append(objp)
      objpoints.append(objp)
      corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
      imgpoints.append(corners)
      imgpoints.append(corners)
      cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
      # cv.imshow('img',img )
      # cv.waitKey(400)
  cv.destroyAllWindows()
  _,cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
  print("Calibrated!")
  return [cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints]
  
def calibrate(frame, cameraMatrix, dist,img_width,img_height):
  newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (img_width,img_height), 1, (img_width,img_height))
  x, y, w,h = roi
  frameDis = cv.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
  frame = frameDis[y:y+h, x:x+w]
  return frame