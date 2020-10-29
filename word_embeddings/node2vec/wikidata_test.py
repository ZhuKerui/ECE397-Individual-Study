import csv
import json
import io
import requests
from wikidata.client import Client
from wikidata.entity import Entity
from node2vec import Node2Vec
import networkx as nx

class Commons:
    P_instance_of = 'P31'
    P_subclass_of = 'P279'
    P_part_of = 'P361'
    P_facet_of = 'P1269'
    P_manifestation_of = 'P1557'
    Properties = [P_instance_of, P_subclass_of, P_part_of, P_facet_of, P_manifestation_of]

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

# instance of(P31), subclass of(P297), part of(P361), facet of(P1269), manifestation of(P1557)
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

class GraphEmbed:
    def __init__(self, f_triple:str, f_json:str) -> None:
        self._w2i = {}
        self._q2i = {}
        self._vocab = []
        self._q = []
        with io.open(f_triple, 'r', encoding='utf-8') as triple_file:
            triple_csv = csv.reader(triple_file)
            for i, row in enumerate(triple_csv):
                self._w2i[row[0]] = i
                self._q2i[row[1]] = i
                self._vocab.append(row[0])
                self._q.append(row[1])
        self._q_set = set(self._q)
        self._g = nx.Graph()
        self._g.add_nodes_from(range(len(self._q)))
        with io.open(f_json, 'r', encoding='utf-8') as json_file:
            self._relations = json.load(json_file)
            for q in self._relations.keys():
                nodeId = self._q2i[q]
                for property in Commons.Properties:
                    if self._relations[q][property]:
                        for targetQ in self._relations[q][property]:
                            if targetQ in self._q_set:
                                self._g.add_edge(nodeId, self._q2i[targetQ])
        self._node2vec = Node2Vec(self._g, dimensions=64, walk_length=30, num_walks=200, workers=4)
        self._node2vec.fit(window=10, min_count=1, batch_words=4)
    
    def get_similar(self, word):
        idx = self._w2i[word]
        ret_idx = self._node2vec.wv.most_similar(idx)
        return self._vocab[int(ret_idx)]

    def save_embed(self, output_file):
        self._node2vec.wv.save_word2vec_format(output_file)