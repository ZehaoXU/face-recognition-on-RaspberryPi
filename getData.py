# -*- coding: utf-8 -*-

__author__ = 'Jasper Xu'

import cv2
from PIL import Image
import os
import numpy as np

haar_dir = 'D:/Anaconda/pkgs/opencv-3.3.1-py36h20b85fd_1/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml'
lbp_dir = 'D:/Anaconda/pkgs/opencv-3.3.1-py36h20b85fd_1/Library/etc/lbpcascades/lbpcascade_frontalface_improved.xml'
data_dir = 'D:/SECRET/face/dataset/'

#-----------------------------------------------------------------------------------#
def get_data():
    '''
    Args:
        none
    Returns:
        data_dir: data dictionary
    '''
    faceCascade = cv2.CascadeClassifier(haar_dir)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get video stream
    cam = cv2.VideoCapture(0)
    cam.set(3,640) 
    cam.set(4,480) 
    # enter a unique id(int) for each person 
    face_id = input('[INFO] Please enter a unique face id: ')
    print('Start to get user\'s samples!')
    print('Please look at the camera and change the angle of your face...')

    i = 0 # count
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize = (30,30))
    
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) 
            cv2.putText(img, 'User : ' + str(face_id), (x,y-5), font, 1, (255,255,255), 2)   
            i += 1
    
            # rename as 'id.count.jpg'
            cv2.imwrite(data_dir + str(face_id) + '.' + str(i) + '.jpg', gray[y:y+h,x:x+w])
    
        cv2.imshow('camera', img)
        # get a sample every 300ms
        k = cv2.waitKey(300) & 0xff 
        # take 50 samples then break, or enter space to break
        if k == ord(' '):
            break
        elif i >= 50: 
            break
    
    print('[INFO] Successful in getting samples of {}\n' .format(face_id))
    cam.release()
    cv2.destroyAllWindows()
    return data_dir

#-----------------------------------------------------------------------------------------#
def read_data(data_dir):
    '''
    Args:
        data_dir: data dictionary
    Returns:
        face_samples: list of images
        labels: list of labels
    '''
    faceCascade = cv2.CascadeClassifier(haar_dir)
    imgPaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]    
    face_samples=[]
    labels = []
    for imgPath in imgPaths:
        # convert to gray-scale
        PIL_img = Image.open(imgPath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        # get label
        label = int(os.path.split(imgPath)[-1].split(".")[0])

        faces = faceCascade.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            face_samples.append(img_numpy[y:y+h,x:x+w])
            labels.append(label)

    return face_samples,labels

#----------------------------------------main---------------------------------------------#
if __name__ == '__main__':
    get_data()