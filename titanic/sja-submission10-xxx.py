# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:05:24 2020

@author: sjuan
https://www.kaggle.com/mdmahmudferdous/titanic-survivor-prediction-0-804-top-8
https://www.kaggle.com/dantefilu/keras-neural-network-a-hitchhiker-s-guide-to-nn
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
training_set = pd.read_csv('train.csv')
X_train = training_set.iloc[:, [False,False,True,False,True,True,True,True,False,True,False,True]].values
y_train = training_set.iloc[:, 1].values
test_set=pd.read_csv('test.csv')
X_test=test_set.iloc[:,[False,True,False,True,True,True,True,False,True,False,True]].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues=SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
missingvalues = missingvalues.fit(X_train[:, 2:3])
X_train[:, 2:3]=missingvalues.transform(X_train[:, 2:3])
y_train = y_train[~pd.isnull(np.array(X_train , dtype=object)).any(axis=1)]
X_train = X_train[~pd.isnull(np.array(X_train , dtype=object)).any(axis=1)]

missingvalues=SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
missingvalues = missingvalues.fit(X_test[:, [2,5]])
X_test[:, [2,5]]=missingvalues.transform(X_test[:, [2,5]])
X_test = X_test[~pd.isnull(np.array(X_test , dtype=object)).any(axis=1)]
#One improvement could be to do the average of the fare and age according to their name title and/or class of fare/value
#Another alternative could be to make different groups per age range and then convert it to a categorical variable (rather than numeric)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,6])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train), dtype=np.float)
X_test = np.array(ct.fit_transform(X_test), dtype=np.float)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train XGBoost on the training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

"""
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
from sklearn.svm import SVC
classifier = SVC(kernel= 'rbf', random_state=0)
classifier.fit(X_train,y_train)
"""

# Predicting the Test set results
y_pred=classifier.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()))
print("Standar Deviation: {:.2f} %".format(accuracies.std()*100))

"""
#Apliying GridSearch to find the best model and the best paramenters
from sklearn.model_selection import GridSearchCV
parameters=[{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
            {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)
grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

classifier_best=SVC(C=best_parameters['C'],gamma=best_parameters['gamma'],kernel=best_parameters['kernel'])
classifier_best.fit(X_train,y_train)
y_pred=classifier_best.predict(X_test)
"""

# Generating the output file
output=np.append((test_set.iloc[:,0:1].values),y_pred.reshape(y_pred.shape[0],1),axis=1)
outputdf=pd.DataFrame(data=output,columns=['PassengerId','Survived'])
outputdf=outputdf.set_index('PassengerId')
outputdf.to_csv('sja-submission1-xxx.csv')
