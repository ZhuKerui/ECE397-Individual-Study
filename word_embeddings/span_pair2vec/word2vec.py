import sys
import re
# import nltk
from gensim.models import Word2Vec

text = []
with open(sys.argv[1]) as corpus:
    for line in corpus:
        line = line.lower()
        line = re.sub('[^a-zA-Z]', ' ', line)
        line = re.sub(r'\s+', ' ', line)
        text.append(line)

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(text)):
    text[i] = [w for w in text[i].split() if w not in stopwords.words('english')]

wv = Word2Vec(text, size=300, min_count=10, window=7, workers=10)
wv.save(sys.argv[2])
# print(text)