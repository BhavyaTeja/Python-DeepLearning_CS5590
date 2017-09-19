#Importing the libraries

import numpy as np

vector = np.random.rand(15)

print(vector)

vector[vector.argmax()] = 100

print(vector)
