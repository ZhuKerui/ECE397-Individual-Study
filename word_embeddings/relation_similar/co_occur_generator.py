import io
from sklearn.metrics import silhouette_score
import numpy as np
import csv
import math
from collections import Counter
import sys
sys.path.append('..')

from relation_similar.vdbscan import do_cluster
from my_keywords import Keyword_Vocab
from my_multithread import multithread_wrapper

def dbscan_cluster(vecs:np.ndarray, vocab:list, k:int=3):
    label = do_cluster(vecs, k)
    if all(label == -1):
        score = -1
    else:
        score = silhouette_score(vecs, label, metric='cosine')
    cluster_num = max(label) + 1
    clusters = []
    for i in range(cluster_num + 1):
        clusters.append(set())
    for word_idx, cluster_id in enumerate(label):
        clusters[cluster_id].add(vocab[word_idx])
    return score, clusters

def nmpi_analysis(counter:dict, output_file:str):
    print('Start NPMI analysis ...')
    Z = 0.
    word_freq = {}
    for pair, freq in counter.items():
        word0, word1 = pair.split('__')
        if word0 in word_freq.keys():
            word_freq[word0] += freq
        else:
            word_freq[word0] = freq
        if word1 in word_freq.keys():
            word_freq[word1] += freq
        else:
            word_freq[word1] = freq
        Z += 2 * freq

    with io.open(output_file, 'w', encoding='utf-8') as dump_file:
        csv_w = csv.writer(dump_file)
        for pair, freq in counter.items():
            word0, word1 = pair.split('__')
            npmi = -math.log((2 * Z * freq) / (word_freq[word0] * word_freq[word1])) / math.log(2 * freq / Z)
            csv_w.writerow([pair, freq, '%.2f' % npmi])
    print('NPMI analysis is done.')

class Co_Occur_Generator:
    def __init__(self, keyword_vocab:Keyword_Vocab):
        self.keyword_vocab = keyword_vocab

    def __extract_co_occur(self, line:str):
        if not line:
            return None
        words = line.strip().split()
        co_occur_set = set()
        for word in words:
            if word in self.keyword_vocab.stoi:
                co_occur_set.add(word)
        if len(co_occur_set) > 1:
            co_occur_list = list(co_occur_set)
            co_occur_list.sort()
            pair_list = []
            for i in range(len(co_occur_list)-1):
                for j in range(i + 1, len(co_occur_list)):
                    pair_list.append('%s__%s' % (co_occur_list[i], co_occur_list[j]))
            return ' '.join(pair_list) + '\n'
        else:
            return None

    def extract_co_occur(self, freq:int, input_file:str, output_file:str, thread_num:int=1):
        def count_pair(line_output_file, output_file):
            print('Start counting ...')
            pair_counter = Counter()
            with io.open(line_output_file, 'r', encoding='utf-8') as line_output:
                for line in line_output:
                    pair_counter.update(Counter(line.strip().split()))
            print('Counting is done.')
            nmpi_analysis(pair_counter, output_file)
        multithread_wrapper(self.__extract_co_occur, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num, post_operation=count_pair)

    def extract_semantic_related(self, dep_context_file:str, output_file:str):
        pair_dict = {}
        with io.open(dep_context_file, 'r', encoding='utf-8') as load_file:
            for idx, line in enumerate(load_file):
                if not line:
                    continue
                word0, word1 = line.strip().split()
                word1 = word1.split('_', 1)[1]
                if word1 in self.keyword_vocab.stoi:
                    pair = ('%s__%s' % (word0, word1)) if word0 < word1 else ('%s__%s' % (word1, word0))
                    if pair not in pair_dict:
                        pair_dict[pair] = 1
                    else:
                        pair_dict[pair] += 1

                if idx % 100000 == 0:
                    print(idx)
        
        nmpi_analysis(pair_dict, output_file)
                    
    def load_pairs(self, pair_file):
        self.pairs = {}
        self.related = {}
        print('Start loading pairs...')
        with io.open(pair_file, 'r', encoding='utf-8') as load_file:
            csv_r = csv.reader(load_file)
            for row in csv_r:
                pair = row[0]
                freq = int(row[1])
                npmi = float(row[2])
                self.pairs[pair] = {'freq':freq, 'npmi':npmi}
                word0, word1 = pair.split('__')
                if word0 not in self.related.keys():
                    self.related[word0] = set()
                if word1 not in self.related.keys():
                    self.related[word1] = set()
                self.related[word0].add(word1)
                self.related[word1].add(word0)

        self._pairs_save = self.pairs

    def get_related(self, keyword, min_count:int=1, min_npmi:float=-1.0):
        if keyword not in self.related:
            return None
        pairs = ((keyword + '__' + w if keyword < w else w + '__' + keyword) for w in self.related[keyword])
        return [w for pair, w in zip(pairs, self.related[keyword]) if self.pairs[pair]['freq'] >= min_count and self.pairs[pair]['npmi'] >= min_npmi]

if __name__ == '__main__':
    kv = Keyword_Vocab()
    kv.load_vocab(sys.argv[1])
    cog = Co_Occur_Generator(keyword_vocab=kv)
    cog.extract_co_occur(80, sys.argv[2], sys.argv[3], thread_num=30)