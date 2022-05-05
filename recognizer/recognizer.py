import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

color = (0,0,255)

cap = cv2.VideoCapture(0)

def handlePositionPlayer(x,y,w,h,img_width,img_height):
  nextState = ""
  playerIsJump = ""
  playerDirection = ""

  if(y+h/2 < img_height/2):
    playerIsJump = "is going up"
  elif(y+h/2 >= img_height/2):
    playerIsJump = "is going down"

  if(x+w/2 < img_width/3):
    playerDirection = "to left"
  elif(x+w/2 >=img_width/3 and x+w/2 <= img_width*2/3):
    playerDirection = "in center"
  elif(x+w/2 > img_width*2/3):
    playerDirection = "to right"
  nextState = "the player {} and {}".format(playerIsJump,playerDirection)
  return nextState
    

while True:
  # Capture frame-by-frame
  _,img = cap.read()

  # Flip the image
  img = cv2.flip(img,1)

  # Get sizes
  img_width = img.shape[1]
  img_height = img.shape[0]
  
  lineVerticalOne = [(int(img_width/3),0),(int(img_width/3),img_height)] 
  lineVerticalTwo = [(int(img_width*2/3),0),(int(img_width*2/3),img_height)] 
  lineHorizontalOne = [(0,int(img_height/3*1.5)),(img_width,int(img_height/3*1.5))] 

  # Convert to grayscale
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  facesWithoutFilter = face_cascade.detectMultiScale(gray,1.1,4)

  faces = [[]]

  if isinstance(facesWithoutFilter, tuple):
    faces = facesWithoutFilter
  else:
    faces[0] = facesWithoutFilter[0]

  # Draw lines
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
    cv2.line(img, lineVerticalOne[0], lineVerticalOne[1],color)
    cv2.line(img, lineVerticalTwo[0], lineVerticalTwo[1],color)
    cv2.line(img, lineHorizontalOne[0], lineHorizontalOne[1],color)
    cv2.putText(img,"{} , {}".format(str(x),str(y)),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
    cv2.putText(img,"{} , {}".format(str(x+w),str(y+h)),(x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)

    # Get player movement
    playerMovement = handlePositionPlayer(x,y,w,h,img_width,img_height)

    # Show the player movement
    print(playerMovement)

  # Display the resulting frame  
  cv2.imshow("img",img)

  # Wait for 1ms
  k = cv2.waitKey(1)
  
  # Exit if ESC pressed
  if k == 27:
    break
  
cap.release()


  