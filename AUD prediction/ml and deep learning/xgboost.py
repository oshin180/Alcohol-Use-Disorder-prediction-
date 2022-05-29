# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 06:14:34 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_excel('DTAB.xlsx')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

#label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

#train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from xgboost importXGBClassifier