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
#import matplotlib.pyplot as plt
#from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ReduceLROnPlateau
from tensorflow import set_random_seed
#import time
#from skimage.io import imread
#from skimage.transform import resize
from keras.utils import Sequence

#def readImagesFromFolder(path, siz, n):
#    l = listdir(path)
#    
#    if n:
#        X = np.zeros((n,siz[0],siz[1],3), dtype=np.uint8)
#    else:
#        X = np.zeros((len(l),siz[0],siz[1],3), dtype=np.uint8)
#    
#    for i in range(n):
#        
#        I_temp = image.load_img(path + l[i], target_size=siz)
#        I_temp = image.img_to_array(I_temp)
#        X[i] = I_temp
##        Y_train[i] = np.array([[x1],[y1],[x2],[y2]])
#    return X

set_random_seed(591)
np.random.seed(999)    

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

siz = (229,229)

#model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
#model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(229,229,3))
#model = Xception(weights='imagenet', include_top=False, input_shape=(229,229,3))

#for layer in model.layers:
#    layer.trainable = False

model2 = Sequential()

model2.add(model)
model2.add(Flatten())
model2.add(Dropout(0.2))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.layers[0].trainable = False

model2.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.15,
        zoom_range=0.15,
        rotation_range=15,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'D:\MTS_WORK\SelfieDetector\Datasets\Train',
        target_size=siz,
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'D:\MTS_WORK\SelfieDetector\Datasets\Validation',
        target_size=siz,
        batch_size=32,
        class_mode='binary')

history = model2.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=10,
        verbose = 1,
        callbacks=[learning_rate_reduction],
        validation_data=validation_generator,
        validation_steps=100)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()