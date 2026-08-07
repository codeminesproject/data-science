
"""
===========================================================
Word2Vec Complete Implementation using Gensim
===========================================================

pip install gensim
===========================================================
"""

# ---------------------------------------------------------
# Step 1: Import Library
# ---------------------------------------------------------

from gensim.models import Word2Vec

# ---------------------------------------------------------
# Step 2: Create Training Dataset
# ---------------------------------------------------------

sentences = [
    ["the", "king", "is", "a", "strong", "man"],
    ["the", "queen", "is", "a", "wise", "woman"],
    ["the", "boy", "is", "a", "young", "man"],
    ["the", "girl", "is", "a", "young", "woman"],
    ["prince", "is", "a", "young", "king"],
    ["princess", "is", "a", "young", "queen"],
    ["man", "is", "strong"],
    ["woman", "is", "beautiful"]
]

print("=" * 60)
print("Training Dataset")
print("=" * 60)

for sentence in sentences:
    print(" ".join(sentence))

# ---------------------------------------------------------
# Step 3: Train Word2Vec Model
# ---------------------------------------------------------

model = Word2Vec(
    sentences=sentences,
    vector_size=5,     # Size of each word vector
    window=5,            # Context window size
    min_count=1,         # Keep every word
    workers=4,           # Number of CPU cores
    sg=0                 # 0 = CBOW, 1 = Skip-Gram
)

print("\nModel Training Completed Successfully.")

# ---------------------------------------------------------
# Step 4: Display Vocabulary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Vocabulary")
print("=" * 60)

print(model.wv.index_to_key)

# ---------------------------------------------------------
# Step 5: Display Word Vector
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Word Vector of 'king'")
print("=" * 60)

print(model.wv["king"])

# ---------------------------------------------------------
# Step 6: Vector Shape
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Vector Shape")
print("=" * 60)

print(model.wv["king"].shape)

# ---------------------------------------------------------
# Step 7: Similarity Between Two Words
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Similarity Between 'king' and 'queen'")
print("=" * 60)

similarity = model.wv.similarity("king", "queen")

print(similarity)

# ---------------------------------------------------------
# Step 8: Find Similar Words
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Most Similar Words to 'king'")
print("=" * 60)

similar_words = model.wv.most_similar("king")

for word, score in similar_words:
    print(f"{word:12} {score:.4f}")

# ---------------------------------------------------------
# Step 9: Odd Word
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Odd Word")
print("=" * 60)

odd_word = model.wv.doesnt_match(
    ["king", "queen", "man", "apple"]
)

print(odd_word)

# ---------------------------------------------------------
# Step 10: Word Analogy
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Word Analogy")
print("=" * 60)

result = model.wv.most_similar(
    positive=["queen", "man"],
    negative=["woman"],
    topn=1
)

print(result)

# ---------------------------------------------------------
# Step 11: Save Model
# ---------------------------------------------------------

model.save("word2vec.model")

print("\nModel saved successfully as word2vec.model")

# ---------------------------------------------------------
# Step 12: Load Model
# ---------------------------------------------------------

loaded_model = Word2Vec.load("word2vec.model")

print("\nModel loaded successfully.")

# ---------------------------------------------------------
# Step 13: Check Loaded Model
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Word Vector from Loaded Model")
print("=" * 60)

print(loaded_model.wv["king"])

# ---------------------------------------------------------
# Step 14: Display Complete Word Embedding Matrix
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Word Embeddings")
print("=" * 60)

for word in loaded_model.wv.index_to_key:

    print(f"\nWord : {word}")

    print(loaded_model.wv[word])

# ---------------------------------------------------------
# Step 15: Print All Information Together
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print("Vocabulary Size :", len(loaded_model.wv.index_to_key))
print("Embedding Size  :", loaded_model.vector_size)
print("Window Size     :", loaded_model.window)
print("Algorithm       :", "CBOW" if loaded_model.sg == 0 else "Skip-Gram")

print("\nProgram Executed Successfully.")
