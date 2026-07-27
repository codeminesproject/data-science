
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

# create object of PorterStemmer

porterstemmer = PorterStemmer()
print(porterstemmer.stem("fairly"))
print(porterstemmer.stem("sportingly"))
print(porterstemmer.stem("goes"))

print("--------------------------------------------------------------------")

# create object of SnowballStemmer

snowballsstemmer = SnowballStemmer('english')
print(snowballsstemmer.stem("fairly"))
print(snowballsstemmer.stem("sportingly"))
print(snowballsstemmer.stem("goes"))

