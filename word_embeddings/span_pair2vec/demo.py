from span_pair2vec.play import Play
import numpy as np
from heapq import nlargest

def ugly_normalize(vecs:np.ndarray):
    normalizers = np.sqrt((vecs * vecs).sum(axis=1))
    normalizers[normalizers==0]=1
    return (vecs.T / normalizers).T

class Demo:
    def __init__(self, model_file, model_config_file, relation_file):
        self.model = Play(model_file, model_config_file)
        self.rel_str = open(relation_file).read().splitlines()
        self.relation_representation = ugly_normalize(self.model.get_relation([line.split() for line in self.rel_str]).numpy())

    def find_closest_relation(self, subject:list, object:list, num:int, return_score:bool=False):
        predicted_representation = self.model.get_prediction(subject, object).numpy()
        score = np.matmul(ugly_normalize(predicted_representation), self.relation_representation.T)[0]
        x = nlargest(num, zip(score, np.arange(len(score))), key=lambda x: x[0])
        filtered_idx = [x_[1] for x_ in x]
        result = [self.rel_str[i] for i in filtered_idx]
        if not return_score:
            return result
        else:
            return result, [score[i] for i in filtered_idx]

    def cal_score(self, subject:list, object:list, relation:list):
        normalize_rel = []
        for rel in relation:
            rel = rel.split()
            rel += ['<pad>'] * (6 - len(rel))
            normalize_rel.append(rel)
        relation_representation = self.model.get_relation(normalize_rel).numpy()
        predicted_representation = self.model.get_prediction(subject, object).numpy()
        score = np.matmul(ugly_normalize(predicted_representation), ugly_normalize(relation_representation).T)[0]
        return score