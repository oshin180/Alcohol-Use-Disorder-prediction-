import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel('F:/AUD files for LSTM/MTech project final susma/statistical coeff/wave/combined train and test dataset wave.xlsx')



from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
lab_enc.fit(df['labels'])
variable = lab_enc.transform(df['labels'])
df['labels'] = variable



colname=['Gamma_power',	'Gamma_mean',	'Gamma_variance',	'Gamma_std-deviation',	'Gamma_max-amp',	'Gamma_min-amp',	'Gamma_kurtosis',	'Gamma_skewness',	'Beta_power',	'Beta_mean',	'Beta_variance',	'Beta_std-deviation',	'Beta_max-amp',	'Beta_min-amp',	'Beta_kurtosis',	'Beta_skewness',	'Alpha_power',	'Alpha_mean',	'Alpha_variance',	'Alpha_std-deviation',	'Alpha_max-amp',	'Alpha_min-amp',	'Alpha_kurtosis',	'Alpha_skewness',	'Theta_power',	'Theta_mean',	'Theta_variance',	'Theta_std-deviation',	'Theta_max-amp',	'Theta_min-amp',	'Theta_kurtosis',	'Theta_skewness',	'Delta_power',	'Delta_mean',	'Delta_variance',	'Delta_std-deviation',	'Delta_max-amp',	'Delta_min-amp',	'Delta_kurtosis',	'Delta_skewness',	'labels']
dataname=pd.DataFrame(df, columns=colname)


corr = dataname.corr().round(2)

#remove features having correlation>0.9
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = dataname.columns[columns]
dataname = dataname[selected_columns]

#additions
dataname=dataname.drop(columns=['skewness_gamma'])
selected_columns=selected_columns.drop(['skewness_gamma'])
X=df.iloc[:,:-1].values
y=df.iloc[:,40].values


selected_columns = selected_columns[:-1].values

import statsmodels.api as sm


def backwardElimination(xx, Y, sl, columns):
    numVars = len(xx[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, xx).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    xx = np.delete(xx, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return xx, columns
SL = 0.05
kk=28
data_modeled, selected_columns = backwardElimination(dataname.iloc[:,:-1].values, dataname.iloc[:,kk].values, SL, selected_columns)

#moving result to  a new dataframe
result = pd.DataFrame()
result['labels'] = dataname.iloc[:,kk]

#Creating a dataframe with the columns selected using the p-value and correlation
data = pd.DataFrame(data = data_modeled, columns = selected_columns)

X = data.values
y = result.values

#feature scaling
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


#train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_std,y,test_size=0.2,random_state=0)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)


c, r = y_train.shape
y_train = y_train.reshape(c,)
c, r = y_test.shape
y_test = y_test.reshape(c,)


#feature scaling required LR
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train, y_train)

#feature scaling required knn
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

#feature scaling required SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

#feature scaling required kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#feature scaling required Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


#feature scaling not required DT
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#feature scaling required Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)



from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_pred,y_test)


#k cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
accuracies=cross_val_score(estimator=classifier,X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()


#grid search 

#for LR
param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
             }
             
             
#for random forest
param_grid = {
                 'n_estimators': [5, 10, 15, 20],
                 'max_depth': [2, 5, 7, 9]
             }            
from sklearn.model_selection import GridSearchCV

grid_clf = GridSearchCV(classifier, param_grid, cv=10)
grid_clf.fit(X_train, y_train)
accuracy=grid_clf.cv_results_

accuracies=cross_val_score(estimator=grid_clf,X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()


#for naive bayes
param_grid = {
                 
             }            
from sklearn.model_selection import GridSearchCV

grid_clf = GridSearchCV(classifier, param_grid, cv=10)
grid_clf.fit(X_train, y_train)
accuracy=grid_clf.cv_results_

accuracies=cross_val_score(estimator=grid_clf,X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()


#for svm and kernel svm
param_grid = {
               'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1] 
             }   
             
#for knn
param_grid = {
              'n_neighbors': [1,35,1], 'weights':["uniform", "distance"]
              }
             

#for DT
param_grid = {
              'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)
              }              


#perform grid search              
from sklearn.model_selection import GridSearchCV

grid_clf = GridSearchCV(classifier, param_grid, cv=10)
grid_clf.fit(X_train, y_train)
accuracy=grid_clf.cv_results_

accuracies=cross_val_score(estimator=grid_clf,X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()




# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classification (Training set)')
plt.xlabel('delta')
plt.ylabel('alpha')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classification (Test set)')
plt.xlabel('delta')
plt.ylabel('alpha')
plt.legend()
plt.show()
