# -*- coding: utf-8 -*-

__author__ = "Jasper Xu"

import cv2
import time

def smile_detect(face_dir, smile_dir):
    #load cascade file for face detection
    faceCascade = cv2.CascadeClassifier(face_dir)
    smileCascade = cv2.CascadeClassifier(smile_dir)
    font = cv2.FONT_HERSHEY_SIMPLEX

    #get video stream
    cam = cv2.VideoCapture(0)
    cam.set(3,640) 
    cam.set(4,480) 

    #show fps
    start_time = time.time() 
    fps = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale (gray, 1.3, 5, minSize = (30,30))
        i = 1

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
            cv2.putText(img, 'Face No.'+str(i), (x,y-5), font, 1, (255,255,255), 2)
            i += 1
            #pick up faces
            gray_face = gray[y:y+h, x:x+w]
            color_face = img[y:y+h, x:x+w]
            #smile detect
            smiles = smileCascade.detectMultiScale (gray_face, 1.3, 15, minSize = (15,15))
            
            for (x1, y1, w1, h1) in smiles:
                cv2.rectangle(color_face, (x1,y1), (x1+w1,y1+h1), (0,0,255), 2)
        
        fps += 1
        d_time = time.time() - start_time
        cv2.putText(img, 'FPS:' + str(fps//d_time), (20,460), font, 1, (255,255,255), 2)

        cv2.imshow('camera',img)
    
        k = cv2.waitKey(10) & 0xff
        if k == ord(' '): # press space to break
            break
    
    cam.release()
    cv2.destroyAllWindows()

#--------------------------------------main---------------------------------------#
if __name__ == '__main__':
    haar_dir = 'D:/Anaconda/pkgs/opencv-3.3.1-py36h20b85fd_1/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml'
    lbp_dir = 'D:/Anaconda/pkgs/opencv-3.3.1-py36h20b85fd_1/Library/etc/lbpcascades/lbpcascade_frontalface_improved.xml'
    smile_dir = 'D:/Anaconda/pkgs/opencv-3.3.1-py36h20b85fd_1/Library/etc/haarcascades/haarcascade_smile.xml'
    smile_detect(haar_dir, smile_dir)