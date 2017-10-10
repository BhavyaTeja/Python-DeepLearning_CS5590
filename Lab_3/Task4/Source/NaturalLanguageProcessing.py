# Importing the Libraries

import re, collections
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import sys

# Reading the Text file


with open('/Users/bhavyateja/Github_Projects/Python-DeepLearning_CS5590/Lab_3/Task4/Text/TextFile.txt', 'r') as myfile:
    File = myfile.read().replace('\n', '')


def tokens(text):
    # Get all words from the corpus
    return re.findall('[a-z]+', text.lower())

Words = tokens(File)

# Implementing the word and sentence tokenization to extract the words in a sentence

File = File.decode('utf-8')
Sentences = sent_tokenize(File)

print '\nThe Sentences in the File are: \n', Sentences

WordsInSentences = [word_tokenize(sent) for sent in Sentences]

print '\nThe Words in the Sentences are: \n', WordsInSentences

# Implementation of the Lemmatization

lemmatizer = WordNetLemmatizer()
LemmatizedWords = [lemmatizer.lemmatize(word) for word in Words]
print'\nThe Lemmatized words are: \n', LemmatizedWords

# Removing the Verbs using POS

WordsV = pos_tag(Words)

WordsNoV = [word for (word, tag) in WordsV if not tag.startswith('V')]

# Calculating the word frequency

WordCount = collections.Counter(WordsNoV)

Top5Words = WordCount.most_common(5)

TopWords = [word for (word, freq) in Top5Words]

print '\nThe Top 5 Words with high frequency of occurrence: \n', str(TopWords)

# Sentences with the Top5Words

TopSentences = []
for s in Sentences:
    q = word_tokenize(s)
    for a in q:
        if a in TopWords:
            TopSentences.append(s)
            break

print '\nThe Sentences with most frequently occurred words:\n', TopSentences

# Concatenating the Sentences & Summarization of the Text

Text = ''.join(TopSentences)

print '\nThe Summary of the Text File after Natural Language Processing: \n', Text
