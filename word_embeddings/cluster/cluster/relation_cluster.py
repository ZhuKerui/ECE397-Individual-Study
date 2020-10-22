import numpy as np
import json
import io



import heapq
import numpy as np
from numpy.lib.function_base import copy

def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

def simple_normalize(vec):
   return vec / (np.sqrt(np.matmul(vec, vec)))


class cluster:
    def __init__(self, topic_similar_words=None, function_similar_words=None) -> None:
        self._t_vecs = None
        self._t_vocab = None
        self._t_w2i = None
        self._f_vecs = None
        self._f_vocab = None
        self._f_w2i = None

        if topic_similar_words is not None:
            self.load_topic_similar(topic_similar_words)

        if function_similar_words is not None:
            self.load_function_similar(function_similar_words)

    def load_topic_similar(self, topic_similar_words):
        self._t_vecs = np.load(topic_similar_words+'.npy')
        self._t_vocab = io.open(topic_similar_words+'.vocab').read().split()
        self._t_w2i = {w:i for i,w in enumerate(self._t_vocab)}

    def load_function_similar(self, function_similar_words):
        self._f_vecs = np.load(function_similar_words+'.npy')
        self._f_vocab = io.open(function_similar_words+'.vocab').read().split()
        self._f_w2i = {w:i for i,w in enumerate(self._f_vocab)}

    def get_topic_related(self, central_word, threshold):
        if central_word in self._t_w2i.keys():
            return self._get_similar(self._t_vecs[self._t_w2i[central_word]], self._t_vecs, self._t_vocab, threshold)
        else:
            return None

    def get_function_similar_cluster(self, vocab, threshold):
        vocab_ = (word for word in vocab if word in self._f_w2i.keys())
        vecs_ = (self._f_vecs[self._f_w2i[word]] for word in vocab if word in self._f_w2i.keys())
        return self._get_cluster(vecs_, vocab_, threshold)

    def _get_cluster(self, vecs, vocab, threshold):
        cluster_num = 0
        cluster_center_vecs = None
        cluster_set = []
        w2c = {}

        for word, vec in zip(vocab, vecs):
            if cluster_num == 0:
                cluster_set.append([vec.copy(), set()])
                w2c[word] = cluster_num
                cluster_set[cluster_num][1].add(word)
                cluster_num += 1
                cluster_center_vecs = np.array([vec])
            else:
                mat_result = np.matmul(cluster_center_vecs, vec)
                closest_idx = np.argmax(mat_result)
                if mat_result[closest_idx] >= threshold:
                    cluster_set[closest_idx][0] += vec
                    cluster_set[closest_idx][1].add(word)
                    w2c[word] = closest_idx
                    cluster_center_vecs[closest_idx] = simple_normalize(cluster_set[closest_idx][0] / len(cluster_set[closest_idx][1]))
                else:
                    cluster_set.append([vec.copy(), set()])
                    w2c[word] = cluster_num
                    cluster_set[cluster_num][1].add(word)
                    cluster_num += 1
                    cluster_center_vecs = np.concatenate(cluster_center_vecs, np.array([vec]))

        cluster_set_ = [group[1] for group in cluster_set]
        return (cluster_num, cluster_set_, cluster_center_vecs, w2c)

    def _get_similar(self, vec, vecs, vocab, threshold):
        mul_result = np.matmul(vecs, vec)
        ret = []
        for val, word in zip(mul_result, vocab):
            if val >= threshold:
                ret.append((val, word))
        return ret