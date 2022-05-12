import numpy as np
import cv2 as cv
import glob
path = 'C:/Users/Wilder Taborda/Desktop/universidad/lab4proce/camera'

def getCoefficients():
  chessboardSize = (24,17)
  frameSize = (1440,1080)
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
  objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
  size_of_chessboard_squares_mm = 20
  objp = objp * size_of_chessboard_squares_mm
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.
  images = glob.glob(path+'/*.png')
  for image in images:
      img = cv.imread(image)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
      if ret:
          objpoints.append(objp)
          corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
          imgpoints.append(corners)
          cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
          # cv.imshow('img', img)
          # cv.waitKey(300)
  cv.destroyAllWindows()
  _,cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
  return [cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints]

def calibrate(frame, cameraMatrix, dist,img_width,img_height):
  newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (img_width,img_height), 1, (img_width,img_height))
  dst = cv.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
  x, y, img_width,img_height = roi
  dst = dst[y:y+img_height, x:x+img_width]
  mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (img_width,img_height), 5)
  dst = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
  x, y, img_width,img_height = roi
  frame = dst[y:y+img_height, x:x+img_width]
  return frame