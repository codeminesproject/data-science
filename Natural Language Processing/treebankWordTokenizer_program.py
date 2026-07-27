
from nltk.tokenize import TreebankWordTokenizer

corpus = "At our institution, we are committed to delivering high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."
print(corpus)

# convert documents into words

# step 1 -  create object of TreebankWordTokenizer
tokenize = TreebankWordTokenizer()

#  step 2 - convert documents into words
words = tokenize.tokenize(corpus)
print(words)