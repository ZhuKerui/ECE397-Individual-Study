from tools.TextProcessing import nlp, find_dependency_path_from_tree
from typing import List
import pandas as pd
import networkx as nx

class Dataset_Generator:
    def __init__(self, entity_list:List[str], co_occur_list:List[List[int]], pair_graph:nx.Graph, kw_dist_max:int=6):
        self.keyword_list = entity_list
        self.pair_graph = pair_graph
        self.kw_dist_max = kw_dist_max
        self.co_occur_list = co_occur_list
        self.line_record = []

    def line_operation(self, line:str):
        line_id, sent = line.split(':', 1)
        tokens = sent.split()
        kws = self.co_occur_list[int(line_id)-1]
        kws = [idx for idx in kws if tokens.count(self.keyword_list[idx]) == 1]
        if len(kws) <= 1:
            return
        pairs = [(self.keyword_list[kws[i]], self.keyword_list[kws[j]]) for i in range(len(kws)-1) for j in range(i+1, len(kws)) if self.pair_graph.has_edge(kws[i], kws[j])]
        pairs = [pair for pair in pairs if abs(tokens.index(pair[0]) - tokens.index(pair[1])) <= self.kw_dist_max]
        if not pairs:
            return
        doc = nlp(sent)
        for pair in pairs:
            self.line_record.append((find_dependency_path_from_tree(doc, pair[0], pair[1]), pair[0], pair[1]))
            self.line_record.append((find_dependency_path_from_tree(doc, pair[1], pair[0]), pair[1], pair[0]))

def dataset_generator_post_operation(result:list):
    print('End')
    ret = []
    for obj in result:
        ret = ret.append(obj.line_record)
    return pd.DataFrame(ret, columns=['path', 'subj', 'obj'])

