import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

video_file='./videos/TSH2.mp4'
cap=cv2.VideoCapture(video_file)
if( not cap.isOpened()):
    print("Cannot read the videofile")
i=1
writing=False
while True:
    ret,frame=cap.read()
    if(ret==True):
        if(writing and i%10==0):
            path="images/image_"+str(i)+".png"
            resized=imutils.resize(frame,960,720)
            cv2.imwrite(path,resized)
            print("writing: "+path)
            
        cv2.imshow('writing',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(25) & 0xFF == ord('s'):
            print("start writing ===================== \n")
            writing=True
        elif cv2.waitKey(25) & 0xFF == ord('p'):
            print("pausing ===================== \n")
            writing=False
        i=i+1
    else:
        break


cap.release()
cv2.destroyAllWindows()