
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

corpus = "At our institution, we are committed to delivering high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."

# convert corpus into sentences
sentences = sent_tokenize(corpus)
print(sentences)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

for sentence in sentences:
    words = word_tokenize(sentence)
    words_without_stopwords = []
    for word in words:
        if word not in stopwords.words('english'):
            words_without_stopwords.append(word)
    print(nltk.pos_tag(words_without_stopwords))



