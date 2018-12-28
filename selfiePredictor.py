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
#import matplotlib.pyplot as plt
import logging

# === loggers initialization ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(message)s')
file_handler = logging.FileHandler('Experiment/Training.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger2 = logging.getLogger('Predictor')
logger2.setLevel(logging.INFO)
formatter2 = logging.Formatter('%(message)s')
file_handler2 = logging.FileHandler('Experiment/Training.log')
file_handler2.setFormatter(formatter2)
logger2.addHandler(file_handler2)

def main(model, df, siz):
    logger.info('MODEL EVALUATION STARTED')
    
    # Load model and data if needed
    if model.__class__.__name__ == 'str':
        logger2.info('Model path: {}'.format(df))
        model = keras.models.load_model(model)
    
    if df.__class__.__name__ == 'str':
        logger2.info('Test path: {}'.format(df))
        df = pd.read_csv(df,sep=',')
    
    # Make predictions
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
    
    # Prediction processing and logging
    threshold = 0.5
    y_true = df['label']
    y_pred = [[k > threshold] for k in preds]
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    logger2.info('Confusion matrix:\n {}'.format(cm))
    
    report = classification_report(y_true, y_pred, target_names=('not selfies','selfies'))
    print(report)
    logger2.info('Accuracy report:\n {}'.format(report))
    
    #for k in wrong_i:
    #    plt.figure()
    #    print(preds[k])
    #    plt.imshow(cv.imread(df['name'][k])/255)


if __name__ == '__main__':
    csv = 'D:\\MTS_WORK\\SelfieDetector\\test_dataset_my_photos.csv'
    #model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\Xception_256_1.h5')
    #model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\ResNet50_512_1.h5')
    #model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\InceptionResNetV2_256_1.h5')
    model = keras.models.load_model('D:\MTS_WORK\SelfieDetector\Saved Models\InceptionV3_1024_1.h5')
    siz = (229,229)
    main(model, csv, siz)