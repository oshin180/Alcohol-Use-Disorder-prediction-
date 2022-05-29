# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:48:09 2020

@author: User
"""

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

targets = pd.read_excel('F:/AUD files for LSTM/hilbert and wavelet train/labels for wav-hil train.xlsx')
y = targets.iloc[:, 0].values
#changing y to numbers
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

kk=3

ran=list(range(0,kk))
random.shuffle(ran)

y_final=[None]*kk
seq_f=[None]*kk
sequences = list()
path = 'F:/AUD files for LSTM/hilbert and wavelet train/Data/Data__'

for i in range(1,kk+1):          #i=1
    file_path = path + str(i) + '.xlsx'    #take Data__1
    print(file_path)
    df = pd.read_excel(file_path, header=0)   #read Data__1
    values = df.values
    sequences.append(values) #Data__1 added
    s=ran[i-1]               #s=1
    seq_f[s]=sequences[i-1]  # seq 1--sequences 0--ran 
    y_final[s]=y[i-1]