"""
Python program that accepts a sentence as input and remove duplicate words.

Sort them alphanumerically and print it.

"""
# Taking the sentence as input

Input_Sentence = input('Enter any sentence: ')

# Splitting the words

words = Input_Sentence.split()

# converting all the strings to lowercase

words = [element.lower() for element in words]

# Taking the words as a set to remove the duplicate words

words = set(words)

# Now taking the set of words as list

word_list = list(words)

# Sorting the words alphanumerically

word_list = sorted(word_list)

# Printing the sorted words

print(' '.join(word for word in word_list))


