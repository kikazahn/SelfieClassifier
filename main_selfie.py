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
from keras.models import Sequential
from keras.layers import Add, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau
from tensorflow import set_random_seed
#import time
#from skimage.io import imread
#from skimage.transform import resize
from keras.utils import Sequence

set_random_seed(591)
np.random.seed(999)

#
#class MY_Generator(Sequence):
#
#    def __init__(self, image_filenames, labels, batch_size):
#        self.image_filenames, self.labels = image_filenames, labels
#        self.batch_size = batch_size
#
#    def __len__(self):
#        return np.int8(np.ceil((len(self.image_filenames) / self.batch_size)))
#
#    def __getitem__(self, idx):
#        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
#        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
#        
#        X_batch = np.zeros((self.batch_size, 224,224,3))
#        
#        for i in range(self.batch_size):
#            X_batch[i] = image.img_to_array(image.load_img(batch_x[i], target_size=(224,224)))
#        
#        return X_batch, np.array(batch_y)
    
    
    
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

def readImagesFromFolder(path, siz, n):
    l = listdir(path)
    
    if n:
        X = np.zeros((n,siz[0],siz[1],3), dtype=np.uint8)
    else:
        X = np.zeros((len(l),siz[0],siz[1],3), dtype=np.uint8)
    
    for i in range(n):
        
        I_temp = image.load_img(path + l[i], target_size=siz)
        I_temp = image.img_to_array(I_temp)
        X[i] = I_temp
#        Y_train[i] = np.array([[x1],[y1],[x2],[y2]])
    return X


model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in model.layers:
    layer.trainable = False

model2 = Sequential()

model2.add(model)
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.layers[0].trainable = False

model2.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

siz = (224,224)
#n = 1500
#X_all = np.zeros((2*n,siz[0],siz[1],3),dtype=np.uint8)
#X_all[:n] = readImagesFromFolder('D:\MTS_WORK\SelfieDetector\SelfieDataset\images\\', siz, n)
#X_all[n:] = readImagesFromFolder('D:\MTS_WORK\SelfieDetector\Flicker8k_Dataset\\', siz, n)
#X_all = X_all/255
#Y_all = np.zeros((2*n,1),dtype=np.uint8)
#Y_all[:n] = 1
#
#permut = np.random.permutation(np.arange(2*n))
#d = np.int(2*n*0.9)
#
#X_train = X_all[permut[:d]]
#X_test = X_all[permut[d:]]
#Y_train = Y_all[permut[:d]]
#Y_test = Y_all[permut[d:]]
# ---
#model = Sequential()
#model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))
#
#model.compile(optimizer=optimizers.Adam(lr=1e-4),
#              loss='binary_crossentropy',
#              metrics=['acc'])

#model2.fit(X_train,Y_train, batch_size = 100, epochs = 12, verbose = 1,
#           callbacks=[learning_rate_reduction], validation_data = (X_test,Y_test))

#dir_selfies = 'D:\\MTS_WORK\SelfieDetector\Datasets\SelfieDataset\images\\'
#dir_not_selfies = 'D:\\MTS_WORK\SelfieDetector\Datasets\Flicker8k_Dataset\\'
#list_selfies = listdir(dir_selfies)
#list_not_selfies = listdir(dir_not_selfies)
#
#for i in range(8000):
#    list_selfies[i] = dir_selfies + list_selfies[i]
#    list_not_selfies[i] = dir_not_selfies + list_not_selfies[i]
#
#all_filenames = list_selfies[:8000]
#all_filenames += list_not_selfies[:8000]
#
#all_labels = np.zeros((16000,1),dtype=np.uint8)
#all_labels[:8000] = 1
#
#permut = np.random.permutation(np.arange(16000))
#
#training_filenames = list(np.zeros((15500,1)))
#training_labels = np.zeros((15500,1))
#test_filenames = list(np.zeros((500,1)))
#test_labels = np.zeros((500,1),dtype=np.uint8)
#
#for i in range(len(permut)-500):
#    training_filenames[i] = all_filenames[permut[i]]
#    training_labels[i] = all_labels[permut[i]]
#    
#X_test = np.zeros((500,224,224,3))
#for i in range(500):
#    test_filenames[i] = all_filenames[permut[i+15500]]
#    I_temp = image.load_img(all_filenames[i], target_size=siz)
#    I_temp = image.img_to_array(I_temp)
#    X_test[i] = I_temp
#    
#    test_labels[i] = all_labels[permut[i+15500]]
    
    
    
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'D:\MTS_WORK\SelfieDetector\Datasets\Train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'D:\MTS_WORK\SelfieDetector\Datasets\Validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

model2.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=500)




#d = np.int(2*n*0.9)
#batch_size = 128
#
#my_training_batch_generator = MY_Generator(training_filenames, training_labels, batch_size)
#
#model2.fit_generator(generator=my_training_batch_generator, epochs = 12, verbose = 1,
#           callbacks=[learning_rate_reduction], validation_data = (X_test, test_labels))


#test_path = 'D:\\Users\Kirill\Pictures\VK\\'
#
##def predictSelfie(model2,test_path,siz):
#    l = listdir(test_path)
#    
#    i=0
#    for i in range(len(l)):
#        print(test_path + l[i])
#        I = image.load_img(test_path + l[i], target_size=siz)
#        I = image.img_to_array(I)/255
#        plt.imshow(I)
#        I = np.expand_dims(I,axis=0)
#        print(model2.predict(I))
#        i += 1
        

