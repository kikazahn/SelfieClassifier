# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:23:39 2018
PREDICT FROM CSV
@author: Kirill
"""

import pandas as pd
import numpy as np
import keras
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

path = 'D:\\MTS_WORK\\SelfieDetector\\test_dataset_my_photos.csv'
#model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\Xception_256_1.h5')
#model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\ResNet50_512_1.h5')
#model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\InceptionResNetV2_256_1.h5')
model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\myCNN_1.h5')

df = pd.read_csv(path,sep=',')

siz = (229,229)

preds = []
wrong_i = []

j = 0
for i in df['name']:
    I_temp = image.load_img(i, target_size=siz)
    I_temp = image.img_to_array(I_temp)/255
    I_temp_exp = np.expand_dims(I_temp,axis=0)
    
    score = model.predict(I_temp_exp)
    preds.append(score[0][0])
    
    if np.round(score) != df['label'][j]:
        wrong_i.append(j)
    j += 1
    
threshold = 0.5
y_true = df['label']
y_pred = [[k > threshold] for k in preds]
cm = confusion_matrix(y_true, y_pred)
print(cm)

report = classification_report(y_true, y_pred, target_names=('not selfies','selfies'))
print(report)

#for k in wrong_i:
#    plt.figure()
#    print(preds[k])
#    plt.imshow(cv.imread(df['name'][k])/255)