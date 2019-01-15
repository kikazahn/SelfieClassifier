# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 11:55:05 2018
LABELLER
@author: Kirill
"""
#ksjkhksgf
import pandas as pd
import numpy as np
from os import listdir

test_dir_selfies = 'D:\\MTS_WORK\\SelfieDetector\\Datasets\\TestSet2\\Selfies\\'
test_dir_not_selfies = 'D:\\MTS_WORK\\SelfieDetector\\Datasets\\TestSet2\\NotSelfies\\'

test_list_selfies = listdir(test_dir_selfies)
test_list_not_selfies = listdir(test_dir_not_selfies)
        
labels = np.ones(len(test_list_selfies),dtype=np.uint8).tolist()
labels.extend(np.zeros(len(test_list_not_selfies),dtype=np.uint8))
    
df = pd.DataFrame({'name': [test_dir_selfies+text for text in test_list_selfies]+[test_dir_not_selfies+text for text in test_list_not_selfies],
     'label': labels})

df.to_csv('test_dataset_my_photos.csv',sep=',',index=False)