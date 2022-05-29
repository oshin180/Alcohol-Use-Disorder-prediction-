# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:05:56 2020

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


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

#--------do this for train and test set folders-----------------------------------------------


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





#-----------end of doing this for train and test set-------------------------

train.extend(test)
train_y.extend(test_y)
#-----------------------------------
train=np.array(train)
test=np.array(test)
#-------------------------------------
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


def build_classifier():
    
    classifier=Sequential()
    #Adding 1st LSTM layer and some dropout regularization
    classifier.add(LSTM(1000, return_sequences=True,input_shape=(256,10)))
    classifier.add(Dropout(0.2))
    #Adding 2nd LSTM layer and some dropout regularization
    classifier.add(LSTM(1000, return_sequences=True))
    classifier.add(Dropout(0.2))
    #Adding 3rd LSTM layer and some dropout regularization
    #classifier.add(LSTM(100, return_sequences=True))
    #classifier.add(Dropout(0.4))
    #Adding 4th LSTM layer with droput
    classifier.add(LSTM(1000))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(1,activation='sigmoid'))
    #compiling the rnn
    adam = Adam(lr=0.001)   #it is an optimizer
    #chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
    classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

my_classifier=KerasClassifier(build_fn=build_classifier,epochs=30,batch_size=32)
#history=classifier.fit(X_train,y_train,epochs=100,batch_size=36,callbacks=[chk], validation_data=(X_val,y_val))
history=my_classifier.fit(X_train,y_train,validation_data=(X_val,y_val))



perm=PermutationImportance(my_classifier,random_state=1).fit(X_val,y_val)
eli5.show_weights(perm,feature_names=X_val.columns.tolist())








#classifier = load_model('best_model.pkl')

y_pred = my_classifier.predict(X_testmod)
accuracy_score(y_testmod, y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()



keras.backend.clear_session()