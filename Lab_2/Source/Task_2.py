"""
Python program to generate a dictionary that contains (k, k*k).

And printing the dictionary that is generated including both 1 and k.

"""

# Taking the input

num = int(input("Input a number "))

# Initialising the dictionary

dictionary = dict()

# Computing the k*k using a for loop

for k in range(1,num+1):
    dictionary[k] = k*k

# Printing the whole dictionary

print('Dictionary with (k, k*k) is ')

print(dictionary)

