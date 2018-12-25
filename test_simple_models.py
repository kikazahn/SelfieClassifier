# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 15:07:32 2018

@author: Kirill
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:10:38 2018
SELFIE DETECTOR
@author: Kirill
"""

import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Add, Dense, Dropout, Activation, Flatten, Input, InputLayer
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from tensorflow import set_random_seed

set_random_seed(591)
np.random.seed(999)    

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

CNN_num = 2 # 1 - My CNN; 2 - LogRegression

if CNN_num == 1:
    siz = (128,128)
    # myCNN
    model = Sequential()
    model.add(Conv2D(32, padding='same', kernel_size=(3, 3),
                     activation='relu',input_shape=(siz[0],siz[1],3)))
                     
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    
elif CNN_num == 2:
    # Log Regression
    siz = (32,32)
    model = Sequential()
    model.add(InputLayer((siz[0],siz[1],3)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

model.summary()    

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.15,
        #zoom_range=0.15,
        #rotation_range=15,
        #horizontal_flip=True
        )

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

history = model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=1,
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