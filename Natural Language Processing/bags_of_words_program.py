from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "He is a good boy",
    "She is a good girl",
    "Boy and girl are good"
]

cv = CountVectorizer()

X = cv.fit_transform(corpus)

print("Vocabulary:")
print(cv.vocabulary_)

print("Feature Names:")
print(cv.get_feature_names_out())

print("Bag of Words Matrix:")
print(X.toarray())