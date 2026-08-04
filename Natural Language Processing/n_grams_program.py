import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Download stopwords (Run only once)
nltk.download('stopwords')

# ---------------------------------------------
# Step 1 : Original Sentences
# ---------------------------------------------
sentences = [
    "The food is good",
    "The food is not good"
]

print("Original Sentences")
for i, sentence in enumerate(sentences, start=1):
    print(f"S{i}: {sentence}")

# ---------------------------------------------
# Step 2 : Remove Stop Words using NLTK
# ---------------------------------------------
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

processed_sentences = []

for sentence in sentences:

    words = sentence.lower().split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    processed_sentence = " ".join(filtered_words)
    processed_sentences.append(processed_sentence)

print("\nAfter Removing Stop Words")
for i, sentence in enumerate(processed_sentences, start=1):
    print(f"S{i}: {sentence}")

# ---------------------------------------------
# Step 3 : Unigram
# ---------------------------------------------
print("\n========== UNIGRAM ==========")

cv = CountVectorizer(ngram_range=(1,1))

X = cv.fit_transform(processed_sentences)

print("Vocabulary:")
print(cv.get_feature_names_out())

print("\nFeature Matrix")
print(X.toarray())

# ---------------------------------------------
# Step 4 : Bigram
# ---------------------------------------------
print("\n========== BIGRAM ==========")

cv = CountVectorizer(ngram_range=(1,2))

X = cv.fit_transform(processed_sentences)

print("Vocabulary:")
print(cv.get_feature_names_out())

print("\nFeature Matrix")
print(X.toarray())

# ---------------------------------------------
# Step 5 : Trigram
# ---------------------------------------------
print("\n========== TRIGRAM ==========")

cv = CountVectorizer(ngram_range=(1,3))

X = cv.fit_transform(processed_sentences)

print("Vocabulary:")
print(cv.get_feature_names_out())

print("\nFeature Matrix")
print(X.toarray())