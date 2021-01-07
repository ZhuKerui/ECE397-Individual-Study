import csv
import io
import heapq
import numpy as np

from my_keywords import Vocab_Base, Keyword_Vocab
from my_multithread import multithread_wrapper


class Pair_Embed(Vocab_Base):
    def __init__(self, keyword_vocab:Keyword_Vocab, word_list:list=['<unk>'], vectors: np.ndarray=None):
        super().__init__(word_list=word_list, vectors=vectors)
        self.keyword_vocab = keyword_vocab

    def generate_pair_vocab(self, pair_file, min_count:int=1, min_npmi:float=-1):
        with io.open(pair_file, 'r', encoding='utf-8') as pair_f:
            csv_r = csv.reader(pair_f)
            new_vocab = []
            for row in csv_r:
                if int(row[1]) >= min_count and float(row[2]) >= min_npmi:
                    new_vocab.append(row[0])
                    word0, word1 = row[0].split('__')
                    new_vocab.append('%s__%s' % (word1, word0))
            super().__init__(word_list=['<unk>'] + new_vocab)

    def __extract_context(self, line:str):
        if not line:
            return None
        kw2ctx = {}
        triplets = self.keyword_vocab.find_keyword_context_dependency(line)
        for keyword, context, relation in triplets:
            if keyword.text not in kw2ctx:
                kw2ctx[keyword.text] = set()
            kw2ctx[keyword.text].add('%s_%s' % (relation, context.text))
        if len(kw2ctx) <= 1:
            return None
        kws = list(kw2ctx.keys())
        str_buffer = []
        for i in range(len(kws)-1):
            for j in range(i+1, len(kws)):
                pair_1 = '%s__%s' % (kws[i], kws[j])
                pair_2 = '%s__%s' % (kws[j], kws[i])
                if pair_1 in self.stoi:
                    for ctx in kw2ctx[kws[i]]:
                        str_buffer.append('%s h_%s\n' % (pair_1, ctx))
                        str_buffer.append('%s t_%s\n' % (pair_2, ctx))
                    for ctx in kw2ctx[kws[j]]:
                        str_buffer.append('%s t_%s\n' % (pair_1, ctx))
                        str_buffer.append('%s h_%s\n' % (pair_2, ctx))

        return ''.join(str_buffer)

    def extract_context(self, freq:int, input_file:str, output_file:str, thread_num:int=1):
        multithread_wrapper(self.__extract_context, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num)
            
    def get_vectors(self, pairs)->np.ndarray:
        if not isinstance(pairs, list):
            pairs = [pairs]
        idxs = np.array([self.stoi[pair] for pair in pairs])
        return self.vectors[idxs]

    # def find_similar_pairs(self, cw, kw, n):
    #     pair = cw + '__' + kw
    #     if pair not in self.vocab:
    #         print('%s does not exist' % (pair))
    #         return None
    #     pairs, vecs = self.get_co_occur_pairs(cw)
    #     pairs = [item.split('__')[1] for item in pairs]
    #     pair_vec = self.wvecs[self.vocab2i[pair]]
    #     similarity_vec = vecs.dot(pair_vec)
    #     result = heapq.nlargest(n, zip(similarity_vec, pairs), key=lambda x: x[0])
    #     return_pairs = [item[1] for item in result]
    #     return return_pairs
