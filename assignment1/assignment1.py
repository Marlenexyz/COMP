import cv2
import time
import numpy as np

# 0
cap = cv2.VideoCapture(0)           #start Camera


while(True):
    start = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #Grey Filter
    _,_,_,maxLoc = cv2.minMaxLoc(gray)
    cv2.circle(gray,maxLoc,10,(0,0,0))

    lower_red = np.array([0, 0, 100])                       #define borders for color red
    upper_red = np.array([50, 50, 255])

    mask = cv2.inRange(frame, lower_red, upper_red)         #create mask
    masked_frame = cv2.bitwise_and(frame,frame,mask=mask)
    masked_grey = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY) 
    _, _, _, maxLocRed = cv2.minMaxLoc(masked_grey)
    cv2.circle(frame, maxLocRed, 10, (0, 0, 255), -1)

    max_brightness = 0
    loc_brightness = (0,0)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            temp_brightness = gray[i,j]
            if temp_brightness > max_brightness:
                max_brightness = temp_brightness
                loc_brightness = (j,i)                                  #position, j and i changed
    cv2.circle(frame, loc_brightness, 20, (0, 0, 0), -1)  

    fps = 1 / (time.time() - start)
    cv2.putText(gray, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    prosTime = time.time() - start
    # cv2.putText(gray, "ProcessingTime: {:.2f}".format(prosTime), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print("ProcessingTime: {:.2f}".format(prosTime))
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    # cv2.imshow('red',red)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
