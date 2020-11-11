import csv
import json
import io
import requests
from wikidata.client import Client
from wikidata.entity import Entity
from node2vec import Node2Vec
import networkx as nx
import numpy as np

class Commons:
    P_instance_of = 'P31'
    P_subclass_of = 'P279'
    P_part_of = 'P361'
    P_facet_of = 'P1269'
    P_manifestation_of = 'P1557'
    Properties = [P_instance_of, P_subclass_of, P_part_of, P_facet_of, P_manifestation_of]

def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

def simple_normalize(vec):
    normalizer = np.sqrt(np.matmul(vec, vec))
    if normalizer == 0:
        normalizer = 1
    return vec / normalizer

def find_id(keyword_file, output_file, error_log):
    with io.open(keyword_file, 'r', encoding='utf-8') as load_file:
        with io.open(output_file, 'w', encoding='utf-8') as dump_file:
            with io.open(error_log, 'w', encoding='utf-8') as log_file:
                f_csv = csv.writer(dump_file)
                url = 'https://www.wikidata.org/w/api.php'
                params = {'action':'wbsearchentities','format':'json','language':'en'}
                cnt = 0
                for word in load_file:
                    word = word.strip()
                    params['search'] = word
                    r = requests.get(url, params=params)
                    if r.json()['search']:
                        id = r.json()['search'][0]['id']
                        label = r.json()['search'][0]['label']
                        f_csv.writerow([word, id, label])
                    else:
                        log_file.write(word + '\n')
                    cnt += 1
                    if cnt % 100 == 0:
                        print(cnt)

def find_related(keyword_csv, related_set):
    with io.open(keyword_csv, 'r', encoding='utf-8') as load_file:
        with io.open(related_set, 'w', encoding='utf-8') as dump_file:
            load_csv = csv.reader(load_file)
            client = Client()
            instance_of = Entity(Commons.P_instance_of, client)
            subclass_of = Entity(Commons.P_subclass_of, client)
            part_of = Entity(Commons.P_part_of, client)
            facet_of = Entity(Commons.P_facet_of, client)
            manifestation_of = Entity(Commons.P_manifestation_of, client)
            myDict = {}
            cnt = 0
            for row in load_csv:
                id = row[1]
                entity = client.get(id, load=True)
                myDict[id] = {}
                myDict[id][Commons.P_instance_of] = [item.id for item in entity.getlist(instance_of)]
                myDict[id][Commons.P_subclass_of] = [item.id for item in entity.getlist(subclass_of)]
                myDict[id][Commons.P_part_of] = [item.id for item in entity.getlist(part_of)]
                myDict[id][Commons.P_facet_of] = [item.id for item in entity.getlist(facet_of)]
                myDict[id][Commons.P_manifestation_of] = [item.id for item in entity.getlist(manifestation_of)]
                cnt += 1
                if cnt % 100 == 0:
                    print(cnt)
            json.dump(myDict, dump_file)

class Graph_Embed:
    def preprocess_raw_data(self, f_triple:str, f_json:str, f_edgelist:str) -> None:
        q_list = []
        q2w = {}

        with io.open(f_triple, 'r', encoding='utf-8') as triple_file:
            triple_csv = csv.reader(triple_file)
            for row in triple_csv:
                word = row[0].replace(' ', '_')
                q_list.append(row[1].strip('Q'))
                q2w[row[1].strip('Q')] = word

        self._g = nx.Graph()
        q_set = set(q_list)
        with io.open(f_json, 'r', encoding='utf-8') as json_file:
            relations = json.load(json_file)
            for q in relations.keys():
                q_num = q.strip('Q')
                if q_num not in q_set:
                    continue
                for property in Commons.Properties:
                    if relations[q][property]:
                        for targetQ in relations[q][property]:
                            targetQ_num = targetQ.strip('Q')
                            if targetQ_num in q_set:
                                self._g.add_edge(q_num, targetQ_num)
        self._q = list(self._g.nodes())
        self._w = [q2w[q] for q in self._q]
        self._q2i = {q:i for i, q in enumerate(self._q)}
        self._w2i = {w:i for i, w in enumerate(self._w)}
        self._w2q = {self._w[i]:self._q[i] for i in range(len(self._q))}
        nx.write_edgelist(self._g, f_edgelist, delimiter=' ', data=False)

    def vecs2nps(self, f_emb, f_vecs):
        fh=io.open(f_emb, 'r', encoding='utf-8')
        foutname=f_vecs
        first=fh.readline()
        size=list(map(int,first.strip().split()))

        self._vecs=np.zeros((len(self._q),size[1]),float)

        for line in fh:
            line = line.strip().split()
            try:
                idx = self._q2i[line[0]]
                self._vecs[idx,] = np.array(list(map(float,line[1:])))
            except Exception as e:
                print(e.args)

        np.save(foutname+".npy",self._vecs)
        with io.open(foutname+".json", "w", encoding='utf-8') as outf:
            json.dump(self._w2q, outf)
        with io.open(foutname+".vocab", "w", encoding='utf-8') as outf:
            outf.write(" ".join(self._w))

    def load_nps(self, f_vecs):
        self._vecs = np.load(f_vecs+'.npy')
        self._w = io.open(f_vecs+'.vocab').read().split()
        self._w2q = json.load(io.open(f_vecs+'.json'))
        self._q = [self._w2q[w] for w in self._w]
        self._w2i = {w:i for i,w in enumerate(self._w)}
        self._q2i = {q:i for i,q in enumerate(self._q)}

    def is_word_exist(self, word):
        return word in self._w

    def is_q_exist(self, q):
        return q in self._q

    def get_similarity(self, word1, word2):
        if word1 not in self._w:
            print('"' + word1 + '" is not in the vocabulary')
            return 0
        elif word2 not in self._w:
            print('"' + word2 + '" is not in the vocabulary')
            return 0
        vec1 = simple_normalize(self._vecs[self._w2i[word1]])
        vec2 = simple_normalize(self._vecs[self._w2i[word2]])
        return vec1.dot(vec2)
        
    def get_related(self, central_word, threshold, similarity=False):
        if central_word in self._w:
            return self._get_similar(simple_normalize(self._vecs[self._w2i[central_word]]), ugly_normalize(self._vecs), self._w, threshold, similarity)
        else:
            return None

    def _get_similar(self, vec, vecs, vocab, threshold, similarity):
        mul_result = np.matmul(vecs, vec)
        ret = []
        for val, word in zip(mul_result, vocab):
            if val >= threshold:
                if similarity:
                    ret.append((val, word))
                else:
                    ret.append(word)
        return ret
