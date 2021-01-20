import sys
sys.path.append('..')
import numpy as np

from my_keywords import Keyword_Vocab
from my_multithread import multithread_wrapper

class Dep_Based_Embed_Generator(Keyword_Vocab):

    def __init__(self, keyword_vocab:Keyword_Vocab):
        self.ignore_dep = set(['punct', 'dep', 'punctI', 'depI'])
        self.itos = keyword_vocab.itos
        self.stoi = keyword_vocab.stoi
        self.vectors = keyword_vocab.vectors

    def __extract_context(self, line:str):
        if not line:
            return None
        triplets = self.find_keyword_context_dependency(line)
        context_buffer = []
        for keyword, context, relation in triplets:
            if relation not in self.ignore_dep:
                context_buffer.append('%s %s_%s\n' % (keyword.text, relation, context.text))
        if context_buffer:
            return ''.join(context_buffer)
        else:
            return None

    def extract_context(self, freq:int, input_file:str, output_file:str, thread_num:int=1):
        multithread_wrapper(self.__extract_context, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num)

    def get_vectors(self, keywords)->np.ndarray:
        if not isinstance(keywords, list):
            keywords = [keywords]
        idxs = np.array([self.stoi[kw] for kw in keywords])
        return self.vectors[idxs]

    def get_similarity(self, kw1:str, kw2:str):
        idx_1, idx_2 = self.stoi[kw1], self.stoi[kw2]
        if idx_1 and idx_2:
            return self.vectors[idx_1].dot(self.vectors[idx_2])
        else:
            return None

if __name__ == '__main__':
    kv = Keyword_Vocab()
    kv.load_vocab(sys.argv[1])
    dg = Dep_Based_Embed_Generator(kv)
    dg.extract_context(80, sys.argv[2], sys.argv[3], 30)