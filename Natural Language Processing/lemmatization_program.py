
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

corpus = "At our institution, we are committed to delivering high-quality education in programming and technology. Our courses are designed to provide comprehensive knowledge and practical skills in various programming languages and technologies. We pride ourselves on offering a supportive learning environment where students can thrive and reach their full potential."

# convert documents into words
words = word_tokenize(corpus)
print(words)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

'''
POS- Noun-n
verb-v
adjective-a
adverb-r
'''

# create object of WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
for word in words:
    print(f"{word} ------> {lemmatizer.lemmatize(word,pos='v')}")

print(lemmatizer.lemmatize("fairly",pos='v'))
print(lemmatizer.lemmatize("sportingly",pos='v'))
print(lemmatizer.lemmatize("goes",pos='v'))