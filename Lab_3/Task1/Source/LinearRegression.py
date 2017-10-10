#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:33:17 2017

@author: bhavyateja
"""

# Simple Linear Regression

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('../Dataset/Movie_dataset.csv')
dataset['Gross'] = dataset['Domestic Gross($M)'] + dataset['Worldwide Gross($M)']
X = dataset.iloc[:, 5].values
y = dataset.iloc[:,7].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)

# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results

y_pred = regressor.predict(X_test)

# Visualising the Training set results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Movie Budget vs Gross Income (Training set)')
plt.xlabel('Move Budget ($M)')
plt.ylabel('Gross Income ($M)')
plt.show()

# Visualising the Test set results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Movie Budget vs Gross Income (Test set)')
plt.xlabel('Move Budget ($M)')
plt.ylabel('Gross Income ($M)')
plt.show()