from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Documents
documents = [
    "good boy",
    "good girl",
    "boy girl good"
]

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(use_idf=True, norm=None)

# Fit and Transform
tfidf_matrix = vectorizer.fit_transform(documents)

print("Vocabulary")
print(vectorizer.get_feature_names_out())

# Convert to DataFrame
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=["S1", "S2", "S3"]
)

print("\nTF-IDF Matrix")
print(df.round(4))