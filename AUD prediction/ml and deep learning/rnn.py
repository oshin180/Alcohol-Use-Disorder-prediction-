# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:02:58 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('AUD for RNN.xlsx')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 64].values

#normalization of X
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X_scaled=sc.fit_transform(X)

w=X_scaled.shape[0]/5  #5 is the timestep
w=int(w)
#reshape
X_scaled_1 = np.reshape(X_scaled, (31,5,X_scaled.shape[1]))


#changing y to numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

x=y.shape[0]/5 
x=int(x)
y_reshaped=np.reshape(y, (x,5,1))




#train, test, split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_scaled_1,y,test_size=0.2,random_state=0)
X_train,X_test=train_test_split(X_scaled_1,test_size=0.2,random_state=0)


#building rnn
#importing keras libraries
import keras                              #keras will build ANN based on tensorflow
from keras.models import Sequential       #used to initialize ANN
from keras.layers.core import Dense       #used to create layers in NN
#---from keras.layers.core import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam

#initializing RNN as sequence of layers
classifier=Sequential()
#Adding 1st LSTM layer and some dropout regularization
classifier.add(LSTM(50, return_sequences=True,input_shape=(5,64)))
classifier.add(Dropout(0.2))
#Adding 2nd LSTM layer and some dropout regularization
classifier.add(LSTM(50, return_sequences=True,))
classifier.add(Dropout(0.2))
#Adding 3rd LSTM layer and some dropout regularization
classifier.add(LSTM(50, return_sequences=True))
classifier.add(Dropout(0.2))
#Adding 4th LSTM layer with droput
classifier.add(LSTM(50))
classifier.add(Dropout(0.2))


#---classifier.add(Flatten())

#Adding the output layer
classifier.add(Dense(1,activation='sigmoid'))

#compiling the rnn
adam = Adam(lr=0.001)   #it is an optimizer
classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

#fitting rnn to the train set
classifier.fit(X_train,y_train,epochs=100,batch_size=10)



from sklearn.metrics import accuracy_score
y_pred = classifier.predict_classes(X_test)
accuracy_score(y_test, y_pred)


keras.backend.clear_session()
