# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:10:38 2018
SELFIE DETECTOR
@author: Kirill
"""
#from os import listdir
#import cv2 as cv
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D #Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from tensorflow import set_random_seed
import selfiePredictor
import logging
#from keras.applications.resnet50 import preprocess_input, decode_predictions

set_random_seed(591)
np.random.seed(999) 

# === loggers initialization ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(message)s')
file_handler = logging.FileHandler('Experiment/Training.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger2 = logging.getLogger('No time')
logger2.setLevel(logging.INFO)
formatter2 = logging.Formatter('%(message)s')
file_handler2 = logging.FileHandler('Experiment/Training.log')
file_handler2.setFormatter(formatter2)
logger2.addHandler(file_handler2)

# === Constants ===
batch_sz = 32
epochs = 10
steps_per_epoch = 10
val_steps = 5

dense_num = 128
out_num = 1

train_folder = 'D:\MTS_WORK\SelfieDetector\Datasets\Train'
validation_folder = 'D:\MTS_WORK\SelfieDetector\Datasets\Validation'

model_evaluation = True

net = 'Xception'

logger.info('TRAINING SESSION STARTED. {}\n'.format(net))
logger2.info('Training folder: {}'.format(train_folder))
logger2.info('Validation folder: {}\n'.format(validation_folder))

logger2.info('Parameters | Batch size: {}\n\
           | Epochs: {}\n\
           | Steps per epoch: {}\n\
           | Validation steps: {}\n\
           | Dense neurons: {}\n\
           | Out neurons: {}\n'.format(batch_sz,epochs,steps_per_epoch,val_steps,dense_num,out_num))

# === Model constructor ===
if net == 'VGG16':
    from keras.applications.vgg16 import VGG16
    siz = (224,224)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

elif net == 'ResNet50':
    from keras.applications.resnet50 import ResNet50
    siz = (224,224)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    
elif net == 'InceptionResNetV2':
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    siz = (229,229)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(229,229,3))
    
elif net == 'Xception':
    from keras.applications.xception import Xception
    siz = (229,229)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(229,229,3))
    
elif net == 'InceptionV3':
    from keras.applications.inception_v3 import InceptionV3
    siz = (299,299)
    base_model = InceptionV3(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
#K.set_learning_phase(1)
x = GlobalAveragePooling2D()(x)
x = Dense(dense_num, activation='relu')(x)
model_out = Dense(out_num, activation='sigmoid')(x)
model = Model(inputs = base_model.input, outputs=model_out)

# === Another method ===
#model = Sequential()
#model.add(base_model)
#model.add(Flatten())
#model.add(Dropout(0.15))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.layers[0].trainable = False

# === Directories ===
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.15,
        zoom_range=0.15,
        rotation_range=15,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=siz,
        batch_size=batch_sz,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_folder,
        target_size=siz,
        batch_size=batch_sz,
        class_mode='binary')

# === Callbacks ===
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

# === Compile and run training ===
model.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose = 1,
        callbacks=[learning_rate_reduction],
        validation_data=validation_generator,
        validation_steps=val_steps)

# === Logging and Visualization ===
# Plot training & validation accuracy values
H = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
H.savefig(fname='Experiment\model_accuracy.png')

# Plot training & validation loss values
H = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
H.savefig(fname='Experiment\model_loss.png')

plot_model(base_model, to_file='Experiment/last_model.png')
with open('Experiment/last_model_summary.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
logger2.info('Accuracy  : {}'.format(np.round(history.history['acc'],4)))
logger2.info('Val_acc   : {}'.format(np.round(history.history['val_acc'],4)))
logger2.info('Loss      : {}'.format(np.round(history.history['loss'],4)))
logger2.info('Val_loss  : {}'.format(np.round(history.history['val_loss'],4)))

# ===== Model evaluation =====
if model_evaluation:
    csv = 'D:\\MTS_WORK\\SelfieDetector\\test_dataset_my_photos.csv'
    selfiePredictor.main(model,csv,siz)
