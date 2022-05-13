import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    _,img = cap.read()
    img = cv.flip(img,1)
    cv.imshow("img",img)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()