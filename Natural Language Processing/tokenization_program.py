
from nltk.tokenize import sent_tokenize,word_tokenize

corpus = "At our institution, we are committed to delivering high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."
print(corpus)

# convert corpus into documents (sentences)
sentences=sent_tokenize(corpus)
print(sentences)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

# convert documents into words
words = word_tokenize(corpus)
print(words)


