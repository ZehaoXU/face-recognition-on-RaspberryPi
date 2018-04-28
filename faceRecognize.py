# -*- coding: utf-8 -*-

# How to run?
# 1. run the trainModel.py once
# 2. plant the model(.yml) to your raspi, and run faceRecognize.py to recognize faces

__author__ = 'Jasper Xu'

import cv2
import os
import time

try: 
    # use ready-made LBPH recognizer in opencv3+
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # initialization
    recognizer.read('D:/SECRET/face/model/trainer.yml')
    haar_dir = 'D:/Anaconda/pkgs/opencv-3.3.1-py36h20b85fd_1/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml'
    lbp_dir = 'D:/Anaconda/pkgs/opencv-3.3.1-py36h20b85fd_1/Library/etc/lbpcascades/lbpcascade_frontalface_improved.xml'
    faceCascade = cv2.CascadeClassifier(haar_dir) # choose a method to detect
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0

    start_time = time.time() 
    fps = 0

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480) 
    
    # map user's name with user's id
    # for example: Jasper->id=1, Tom->id=2, so the list is like following:
    names = ['None', 'Jasper','Tom']

    while True:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize = (50,50))
    
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
            id, match_error = recognizer.predict(gray[y:y+h,x:x+w])
    
            # match_error -> 0 means perfect match
            if (match_error < 100):
                id = names[id]
                confidence = round(100 - match_error)
            else:
                id = 'stranger'
                confidence = round(match_error - 100)
            
            # show confidence level
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence)+'%', (x+5,y+30), font, 1, (255,255,255), 2) 
        
        fps += 1
        d_time = time.time() - start_time
        cv2.putText(img, 'fps: ' + str(fps//d_time), (20,460), font, 1, (255,255,255), 2)

        cv2.imshow('camera',img)
    
        k = cv2.waitKey(10) & 0xff
        if k == ord(' '): # enter space to break
            break

    cam.release()
    cv2.destroyAllWindows()

except :
    print('Error!')