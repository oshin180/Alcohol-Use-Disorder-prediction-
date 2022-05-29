# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 07:39:42 2020

@author: User
"""
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler


import keras
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score


from keras.wrappers.scikit_learn import KerasClassifier


#import the dataset and divide them to dependent and independent variables
targets = pd.read_excel('F:/AUD files for LSTM/MTech project final susma/real time series/Test/labels test.xlsx')
y = targets.iloc[:, 0].values
#changing y to numbers
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

kk=len(y)

ran=list(range(0,kk))
random.shuffle(ran)

y_final=[None]*kk
seq_f=[None]*kk
sequences = list()
path = 'F:/AUD files for LSTM/MTech project final susma/real time series/Test/Data__'

for i in range(1,kk+1):
    file_path = path + str(i) + '.xlsx'
    print(file_path)
    df = pd.read_excel(file_path, header=0)
    values = df.values
    sc=MinMaxScaler(feature_range=(0,1))
    valuess=sc.fit_transform(values)
    sequences.append(valuess) 
    s=ran[i-1]
    seq_f[s]=sequences[i-1]
    y_final[s]=y[i-1]
 
#store train results after the for loop 

train=[None]*kk
train_y=[None]*kk
train=seq_f
train_y=y_final


test=[None]*kk
test_y=[None]*kk
test=seq_f
test_y=y_final

X_train, X_test, y_train, y_test=train_test_split(train,train_y,test_size=0.3,random_state=0)
X_testmod,X_val,y_testmod,y_val=train_test_split(X_test,y_test,test_size=0.5,random_state=0)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_val = np.array(X_val)
y_val = np.array(y_val)

X_testmod = np.array(X_testmod)
y_testmod = np.array(y_testmod)





#NOTE: NN=Neural Network;   ANN=Artificial Neural Network

#building the NN


#initialise the ANN
model = Sequential()
model.add(LSTM(300, input_shape=(256,10)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

adam = Adam(lr=0.001)
chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=256, callbacks=[chk], validation_data=(X_val,y_val))

#loading the model and checking accuracy on the test data
model = load_model('best_model.pkl')

y_pred = model.predict_classes(X_testmod)
accuracy_score(y_testmod, y_pred)
cm=confusion_matrix(y_testmod,y_pred)
#RESULTS: 
#we get an accuracy of 54.8%. this is very bad. hence we will next perform 10 cross validation and check for improvements.





def build_classifier():
    classifier=Sequential()
    classifier.add(LSTM(200, return_sequences=True,input_shape=(256,10)))
    classifier.add(Dropout(0.2))
    classifier.add(LSTM(200, return_sequences=True))
    classifier.add(Dropout(0.2))
    classifier.add(LSTM(200))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(1,activation='sigmoid'))
    adam = Adam(lr=0.001)  
    classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    classifier.fit(X_train,y_train,validation_data=(X_val,y_val))
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=100,nb_epoch=20)

#contains 10 accuracies returned by k fold cross validation. The highest accuracy obtained is 83.3 and the lowest obtained is 41.6. We see that the variance is about 20%. This is huge!!!
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=3,n_jobs=1)

#the mean of the accuracies obtained is 68.4%
mean=accuracies.mean()

#variance obtained is 14%
variance=accuracies.std()

#RESULT ANALYSIS: there are many reasons for having a huge variance:
#1. the dataset is small. If the dataset could include the no-match data, the prediction results have chances of improvement. Therefor we can expand our dataset.
#2. increasing the number of epochs, batch_size can show considerable improvement. We can therefore perform GridSearch and obtain the necessary parameters to improve the accuracy without overfitting
#3. lot of information was lost during the processing by converting a time-series signal into the frequency domain. to get better results we must extract more features from the brain signal. Hence wavelet transform will improve the predictions drastically.







keras.backend.clear_session()