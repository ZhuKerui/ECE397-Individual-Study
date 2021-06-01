from numpy.lib.function_base import append
from tools.TextProcessing import nlp, find_dependency_path_from_tree
from typing import List
import pandas as pd
import networkx as nx

class Dataset_Generator:
    def __init__(self, entity_list:List[str], co_occur_list:List[List[int]], pair_graph:nx.Graph, kw_dist_max:int=6, sent_length_max:int=32):
        self.keyword_list = entity_list
        self.pair_graph = pair_graph
        self.kw_dist_max = kw_dist_max
        self.sent_length_max = sent_length_max
        self.co_occur_list = co_occur_list
        self.line_record = []

    def line_operation(self, line:str):
        line_id, sent = line.split(':', 1)
        tokens = sent.split()
        if len(tokens) > self.sent_length_max:
            return
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
            path = find_dependency_path_from_tree(doc, pair[0], pair[1])
            if path:
                self.line_record.append((path, pair[0], pair[1]))
            else:
                print("%s %s %s" % (line_id, pair[0], pair[1]))
                continue
            self.line_record.append((find_dependency_path_from_tree(doc, pair[1], pair[0]), pair[1], pair[0]))

def dataset_generator_post_operation(result:list):
    print('End')
    ret = []
    for obj in result:
        ret = ret.append(obj.line_record)
    return pd.DataFrame(ret, columns=['path', 'subj', 'obj'])


class TriEntities:
    def __init__(self, entity_list:List[str], co_occur_list:List[List[int]], pair_graph:nx.Graph, kw_dist_max:int=6, sent_length_max:int=32):
        self.keyword_list = entity_list
        self.pair_graph = pair_graph
        self.kw_dist_max = kw_dist_max
        self.sent_length_max = sent_length_max
        self.co_occur_list = co_occur_list
        self.line_record = []

    def line_operation(self, line:str):
        line_id, sent = line.split(':', 1)
        tokens = sent.split()
        if len(tokens) > self.sent_length_max:
            return
        kws = self.co_occur_list[int(line_id)-1]
        kws = [idx for idx in kws if tokens.count(self.keyword_list[idx]) == 1]
        if len(kws) < 3:
            return
        pairs = [(self.keyword_list[kws[i]], self.keyword_list[kws[j]]) for i in range(len(kws)-1) for j in range(i+1, len(kws)) if self.pair_graph.has_edge(kws[i], kws[j])]
        pairs = [pair for pair in pairs if abs(tokens.index(pair[0]) - tokens.index(pair[1])) <= self.kw_dist_max]
        if len(pairs) < 2:
            return
        kws = [self.keyword_list[idx] for idx in kws]
        for kw in kws:
            pair_idx = [1 if kw in pair else 0 for pair in pairs]
            if sum(pair_idx) < 2:
                continue
            kw_set = set()
            sub_pairs = [pair for i, pair in enumerate(pairs) if pair_idx[i] == 1]
            for pair in sub_pairs:
                kw_set.update(pair)
            kw_set.remove(kw)
            self.line_record.append((sent, kw, ','.join(kw_set)))
        # doc = nlp(sent)
        # tokens = [token.text for token in doc]
        # subj_test = False
        # subj_text = ''
        # obj_test = False
        # obj_text = ''
        # for kw in kws:
        #     if doc[tokens.index(kw)].dep_ == 'nsubj':
        #         subj_test = True
        #         subj_text = kw
        #     elif doc[tokens.index(kw)].dep_ == 'dobj':
        #         obj_test = True
        #         obj_text = kw
        
        # for pair in pairs:
        #     path = find_dependency_path_from_tree(doc, pair[0], pair[1])
        #     if path:
        #         self.line_record.append((path, pair[0], pair[1]))
        #     else:
        #         print("%s %s %s" % (line_id, pair[0], pair[1]))
        #         continue
            # self.line_record.append((find_dependency_path_from_tree(doc, pair[1], pair[0]), pair[1], pair[0]))