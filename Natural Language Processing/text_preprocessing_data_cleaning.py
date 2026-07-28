
from nltk import sent_tokenize,word_tokenize
import re

corpus = "At our institution,123 we are committed to delivering %&^ high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."
print(corpus)

# convert copus into lower case

corpus = corpus.lower()
print(corpus)

print("*****************************************************************")

# convert corpus into documents (sentences) and perform data cleaning like remove special character from senteces
sentences=sent_tokenize(corpus)

sentences_list = []

for sentence in sentences:
    # remove all special character
    sentence_without_special_character = re.sub(r"[^a-zA-Z0-9\s]"," ",sentence)

    # Remove all numbers
    sentence_without_number = re.sub(r'\d+', '', sentence_without_special_character)

    # Remove all extra spaces
    clean_sentence = re.sub(r'\s+', ' ', sentence_without_number).strip()

    sentences_list.append(clean_sentence)

print(sentences_list)

print("*****************************************************************")

words_list = []

for sentence in sentences_list:
    words = word_tokenize(sentence)
    words_list.append(words)

print(words_list)