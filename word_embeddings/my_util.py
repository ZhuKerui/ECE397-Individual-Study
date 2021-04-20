from typing import Iterable, List
import numpy as np
from heapq import nlargest
from nltk import WordNetLemmatizer, pos_tag, word_tokenize

wnl = WordNetLemmatizer()

def ugly_normalize(vecs:np.ndarray):
    normalizers = np.sqrt((vecs * vecs).sum(axis=1))
    normalizers[normalizers==0]=1
    return (vecs.T / normalizers).T

def simple_normalize(vec:np.ndarray):
    normalizer = np.sqrt(np.matmul(vec, vec))
    if normalizer == 0:
        normalizer = 1
    return vec / normalizer

def ntopidx(n, score:Iterable):
    s = nlargest(n, zip(np.arange(len(score)), score), key = lambda x: x[1])
    return [item[0] for item in s]

def lemmatize_all(sentence:str):
    return ' '.join((wnl.lemmatize(word, pos='n') if tag.startswith('NN') else word for word, tag in pos_tag(word_tokenize(sentence))))

def my_read(file_name:str):
    return open(file_name, 'r').read().split('\n')

def my_write(file_name:str, content:List[str]):
    with open(file_name, 'w') as f_out:
        f_out.write('\n'.join(content))

def phrase_normalize(sent:str):
    return ' '.join([wnl.lemmatize(word, pos='n') if tag.startswith('NN') else word for word, tag in pos_tag(word_tokenize(sent)) if not tag.startswith('RB') and not tag.startswith('DT')])