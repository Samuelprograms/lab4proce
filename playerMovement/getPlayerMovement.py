def handlePositionPlayer(x_min,y_min,x_max,y_max,img_width,img_height):
  nextState = ""
  playerIsJump = ""
  playerDirection = ""
  mean_x = (x_min + x_max) / 2
  mean_y = (y_min + y_max) / 2
  if(mean_y < img_height/2):
    playerIsJump = 3
  elif(mean_y >= img_height/2):
    playerIsJump = 4
  if(mean_x < img_width/3):
    playerDirection = 1
  elif(mean_x >=img_width/3 and mean_x <= img_width*2/3):
    playerDirection = 0
  elif(mean_x > img_width*2/3):
    playerDirection = 2
  # nextState = "the player {} and {}".format(playerIsJump,playerDirection)
  return playerDirection
