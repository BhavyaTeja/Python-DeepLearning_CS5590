# Reading the text file and calculating the frequency of the words

# Wordcount to take the list of words and their respective frequencies

WordCount = {}

# Opening the file in read-write mode

with open('TextFile.txt', 'r+') as txtfile:
    for line in txtfile:                                            # Splitting the file into lines
        for word in line.split():                                   # Splitting the lines into words
            if word not in WordCount:                               # Condition to check if the word is present or not
                WordCount[word] = 1                                 # Calculating the frequency of the word
            else:
                WordCount[word] += 1                                # Incrementing the frequency if the word is already present
    for k, v in WordCount.items():                                  # Printing the words and their frequency of occurences
        print(k, v)
txtfile.close()