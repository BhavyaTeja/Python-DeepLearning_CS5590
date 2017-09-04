# Program to check if a string contains all letters of the alphabet

# Importing the library

import string

# Taking the input string

input_string = input("Enter a string: ")

# Creating a set of lowercase alphabets

alphabet = set(string.ascii_lowercase)

# Condition to check if all the 26 characters are present in the entered string

if set(input_string.lower()) >= alphabet:
    print("\n The string '%s' contains all the characters." %(input_string))
else:
    print("\n The string '%s' doesn't contain all the characters." %(input_string))