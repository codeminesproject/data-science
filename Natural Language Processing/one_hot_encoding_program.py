# pip install scikit-learn

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from nltk import word_tokenize

sentence = "Welcome to CodeMines Computer"

# convert sentences into token
words = word_tokenize(sentence)

# Convert list into a 2D NumPy array
words_array = np.array(words).reshape(-1, 1)

print(words_array)

# create object of one hot encoder
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(words_array)

print(encoded)