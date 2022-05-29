# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 07:39:42 2020

@author: User
"""
# DEEP LEARNING FOR DTAB code
#import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset and divide them to dependent and independent variables
df = pd.read_excel('F:/AUD files for LSTM\MTech project final susma/statistical coeff/wave/combined train and test dataset wave.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, 40].values

#label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)


#feature scaling
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


#train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)





#NOTE: NN=Neural Network;   ANN=Artificial Neural Network

#building the NN
#install keras, tensorflow and theano libraries
#import keras and related libraries
import keras                              #keras will build ANN based on tensorflow
from keras.models import Sequential       #used to initialize ANN
from keras.layers.core import Dense       #used to create layers in NN

#initialise the ANN
classifier=Sequential()                              #defines ANN as sequence of layers

#we will add the input layer and the 1st hidden layer
#here the .add adds different layers
#output_dim=number of nodes in the hidden layer. It is calculated as the average(number of input layers+number of output layers)
#'init' specifies the type of weights to be assigned. 
#the rectifier function is used as the activation function for the hidden layer
#input_dim=number of nodes in the input layer. In our case there are 5 input nodes, i.e., alpha, beta, gamma, delta, theta
classifier.add(Dense(output_dim=21,init='uniform',activation='relu',input_dim=40))

#now add another hidden layer
classifier.add(Dense(output_dim=21,init='uniform',activation='relu'))

#add the output layer
#output_dim=outcome for the output layer. Here it is set to 1 because the outcome is binary, i.e., alcoholic or control
#the activation function used on the output layer is sigmoid. This is to get the probability of the outcome
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


#compile the ANN
#we will use Stochastic Gradient Descent to the entire ANN to find the optimal set of weights. There are many types of descent. we have used 'adam' algorithm as our stochastic gradient descent. 
#this 'adam' algorithm is based on loss function. For example, in simple linear regression the loss function is the sum of the difference between the real value and predicted value. For stochastic gradient descent, the the loss function is the logarithmic loss given by a complex formula. P.S: I will attach the screenshot of the formula seperately
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fit ANN to the training set
#NOTE: there is no standard rule to select batch_size or epochs. Generally the batch_size=10 and epochs=100
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#predict the test set result
#y_pred shows the probability of the outcome being control or alcoholic
y_pred=classifier.predict(X_test)

#it means that if y_pred>=0.5, the subject is control. Else he is an alcoholic
#we get the y_pred output as true and false
y_pred=(y_pred>=0.50)

#the y_pred values are converted to 0 and 1
y_pred = np.multiply(y_pred, 1)

#implement the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#RESULTS: 
#we get an accuracy of 54.8%. this is very bad. hence we will next perform 10 cross validation and check for improvements.



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=21,init='uniform',activation='relu',input_dim=40))
    classifier.add(Dense(output_dim=21,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    classifier.fit(X_train,y_train,batch_size=10,epochs=100)
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)

#contains 10 accuracies returned by k fold cross validation. The highest accuracy obtained is 83.3 and the lowest obtained is 41.6. We see that the variance is about 20%. This is huge!!!
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=1)

#the mean of the accuracies obtained is 68.4%
mean=accuracies.mean()

#variance obtained is 14%
variance=accuracies.std()

#RESULT ANALYSIS: there are many reasons for having a huge variance:
#1. the dataset is small. If the dataset could include the no-match data, the prediction results have chances of improvement. Therefor we can expand our dataset.
#2. increasing the number of epochs, batch_size can show considerable improvement. We can therefore perform GridSearch and obtain the necessary parameters to improve the accuracy without overfitting
#3. lot of information was lost during the processing by converting a time-series signal into the frequency domain. to get better results we must extract more features from the brain signal. Hence wavelet transform will improve the predictions drastically.