# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:10:38 2018
SELFIE DETECTOR
@author: Kirill
"""
from os import listdir
import numpy as np
import cv2 as cv
from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Add, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import optimizers


def readImagesFromFolder(path, siz, n):
    l = listdir(path)
    
    if n:
        X = np.zeros((n,siz[0],siz[1],3), dtype=np.uint8)
    else:
        X = np.zeros((len(l),siz[0],siz[1],3), dtype=np.uint8)
    
    i = 0
    for im_name in l:
        
        I_temp = image.load_img(path + im_name, target_size=siz)
        I_temp = image.img_to_array(I_temp)
        X[i] = I_temp
#        Y_train[i] = np.array([[x1],[y1],[x2],[y2]])
        i += 1
    
    return X
    
    
    

X_all = np.zeros()

model = Sequential()
model.add(Flatten())
model.add(Dense(1), activation='sigmoid')

model.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit()