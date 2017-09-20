# Using Numpy creating a random vector of size 15 and replace the maximum value by 100

# Importing the libraries

import numpy as np

# Creating a vector with 15 random values

vector = np.random.rand(15)

# Printing the vector

print('The Vector with random values \n')

print(vector)

# Getting the maximum value of the vector and replacing that value with 100

vector[vector.argmax()] = 100

# Printing the new vector

print('The Vector after replacing the maximum value with 100 \n ')

print(vector)
