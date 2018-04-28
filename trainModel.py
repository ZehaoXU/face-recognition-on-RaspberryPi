# -*- coding: utf-8 -*-

__author__ = 'Jasper Xu'

import numpy as np
import getData
import cv2
 
try:
    # use read-made LBPH trainer in opencv3+
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # get all users' face samples
    num = input('How many users do you want to recognize? \n')

    for i in range(int(num)):
        print('Now get samples for user No.{} ...' .format(i))
        data_dir = getData.get_data()

    # get all images and labels
    faces,labels = getData.read_data(data_dir)

    print ("[INFO] Start training! Please wait...")
    recognizer.train(faces, np.array(labels))
    # write only work on desktop, not on raspi
    recognizer.write('D:/SECRET/face/model/trainer.yml')

    print("[INFO] Successful in training {} faces".format(len(np.unique(labels))))

except :
    print('Error!')