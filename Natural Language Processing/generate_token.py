
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import re


corpus = "At our institution, we are at institution we are committed to delivering high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."

# convert corpus into lower case (it is a part of data cleaning technique)
corpus = corpus.lower()
print(corpus)

print("**************************************************")

# conver corpus into documets (list of sentences)

sentences_list = sent_tokenize(corpus)

sentences = []

for sentence in sentences_list:
    # remove special characters from sentence
    sentence_without_special_character = re.sub(r"[^a-zA-Z0-9\s]"," ",sentence)

    # remove extra space
    final_sentence = re.sub(r"\s+"," ",sentence_without_special_character)

    sentences.append(final_sentence)


print(sentences)

print("**************************************************")

# get all stopwords

stop_words = stopwords.words("english")

print("**************************************************")

# convert sentence into words list

words_list = []

for sentence in sentences:
    words = word_tokenize(sentence)

    # create list of unique words from sentence
    unique_words = []
    for word in words:
        if word not in unique_words:
            if word not in stop_words:
                unique_words.append(word)
    words_list.append(unique_words)

print(words_list)


