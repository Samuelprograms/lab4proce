import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()
color = (0,0,255)

window_name = "Hand Detector"

# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

h, w, c = frame.shape

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

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    img_width = frame.shape[1]
    img_height = frame.shape[0]
    
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