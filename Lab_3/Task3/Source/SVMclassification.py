#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:28:39 2017

@author: bhavyateja
"""

# Importing the libraries

import pandas as pd

# Importing the dataset

from sklearn.datasets import load_digits
dataset = load_digits()
X = dataset.data
y = dataset.target

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set

from sklearn.svm import SVC
Linearclassifier = SVC(kernel = 'linear', random_state = 10)
RBFclassifier = SVC(kernel = 'rbf', random_state = 10)
Linearclassifier.fit(X_train, y_train)
RBFclassifier.fit(X_train, y_train)

# Predicting the Test set results

y_predLinear = Linearclassifier.predict(X_test)
y_predRBF = RBFclassifier.predict(X_test)

# Calculating the Accuracy of the Model

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predLinear)
accuracy_score(y_test,y_predRBF)