
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

corpus = "At our institution, we are committed to delivering high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."

# convert corpus into sentences
sentences = sent_tokenize(corpus)
print(sentences)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

stemming = PorterStemmer()
sentences_without_stopwords = []

for sentence in sentences:
    words = word_tokenize(sentence)
    sentence_without_stopwords = ""
    for word in words:
        if word not in stopwords.words('english'):
            sentence_without_stopwords += (stemming.stem(word) +" ")
    sentences_without_stopwords.append(sentence_without_stopwords)

print(sentences_without_stopwords)

