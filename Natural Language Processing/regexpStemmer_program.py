
from nltk.tokenize import word_tokenize
from nltk.stem import RegexpStemmer

corpus = "At our institution, we are committed to delivering high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."

# convert documents into words
words = word_tokenize(corpus)
print(words)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")


# create object of RegexpStemmer

stemming = RegexpStemmer('ing$|able$', min=4)
for word in words:
    print(f"{word} ------> {stemming.stem(word)}")