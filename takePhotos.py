import time
import cv2 as cv

cap = cv.VideoCapture(0)
path = 'C:/Users/Wilder Taborda/Desktop/universidad/lab4proce/camera/images2' 

while True:
    _,img = cap.read()
    img = cv.flip(img,1)
    cv.imwrite(path+"/"+str(time.time())+".jpg",img)
    k = cv.waitKey(2000)
    if k == 27:
        break
cap.release()