import json
import io
import spacy
import re
import numpy as np
import csv
import math

from dep_generator import *

def nmpi_analysis(co_occur_file, related_pair_file):
    Z = 0.
    word_freq = {}
    pair_freq = {}
    with io.open(co_occur_file, 'r', encoding='utf-8') as load_file:
        csv_r = csv.reader(load_file)
        for row in csv_r:
            words = row[0:2]
            words.sort()
            word0 = words[0]
            word1 = words[1]
            pair = word0 + '__' + word1
            freq = int(row[2])
            if word0 in word_freq.keys():
                word_freq[word0] += freq
            else:
                word_freq[word0] = freq
            if word1 in word_freq.keys():
                word_freq[word1] += freq
            else:
                word_freq[word1] = freq
            pair_freq[pair] = freq
            Z += 2 * freq
        Z = float(Z)
    with io.open(related_pair_file, 'w', encoding='utf-8') as dump_file:
        csv_w = csv.writer(dump_file)
        for pair, freq in pair_freq.items():
            word0, word1 = pair.split('__')
            npmi = -math.log((2 * Z * pair_freq[pair]) / (word_freq[word0] * word_freq[word1])) / math.log(2 * pair_freq[pair] / Z)
            csv_w.writerow([word0, word1, '%.2f' % npmi])

def generate_keyword_set_from_wv(wv_file, keyword_file):
    with io.open(wv_file, 'r', encoding='utf-8') as load_file:
        with io.open(keyword_file, 'w', encoding='utf-8') as dump_file:
            for line in load_file:
                dump_file.write(line.split()[0] + '\n')

class Co_Occur_Generator(Dep_Based_Embed_Generator):
    def extract_co_occur(self, reformed_file:str, co_occur_output_file:str):
        if self.keywords is None:
            print('Keywords are not loaded, please use "build_word_tree(input_txt, dump_file)" or  "load_word_tree(json_file)" to load keywords')
            return
        pair_dict = {}
        with io.open(reformed_file, 'r', encoding='utf-8') as sents:
            cnt = 0
            for sent in sents:
                words = sent.strip().split()
                co_occur_set = set()
                for word in words:
                    if word in self.keywords:
                        co_occur_set.add(word)
                if len(co_occur_set) > 1:
                    co_occur_list = list(co_occur_set)
                    co_occur_list.sort()
                    for i in range(len(co_occur_list)-1):
                        for j in range(i + 1, len(co_occur_list)):
                            word_1 = co_occur_list[i]
                            word_2 = co_occur_list[j]
                            sub_dict = pair_dict.get(word_1)
                            if sub_dict is None:
                                pair_dict[word_1] = {word_2 : 1}
                            else:
                                if word_2 in sub_dict.keys():
                                    sub_dict[word_2] += 1
                                else:
                                    sub_dict[word_2] = 1
                cnt += 1
                if cnt % 10000 == 0:
                    print(cnt)
        with io.open(co_occur_output_file, 'w', encoding='utf-8') as output:
            csv_f = csv.writer(output)
            for k1, sub_dict in pair_dict.items():
                for k2, freq in sub_dict.items():
                    csv_f.writerow([k1, k2, freq])

    def extract_semantic_related(self, reformed_file:str, co_occur_output_file:str):
        if self.keywords is None:
            print('Keywords are not loaded, please use "build_word_tree(input_txt, dump_file)" or  "load_word_tree(json_file)" to load keywords')
            return
        pair_dict = {}
        with io.open(reformed_file, 'r', encoding='utf-8') as load_file:
            cnt = 0
            for line in load_file:
                cnt += 1
                if not line:
                    continue
                doc = nlp(line)
                for word in doc:
                    if word.text.lower() not in self.keywords:
                        continue
                    word_txt = word.text.lower()
                    for child in word.children:
                        if child.text in self.keywords:
                            pair = [word_txt, child.text.lower()]
                            pair.sort()
                            word_1 = pair[0]
                            word_2 = pair[1]
                            sub_dict = pair_dict.get(word_1)
                            if sub_dict is None:
                                pair_dict[word_1] = {word_2 : 1}
                            else:
                                if word_2 in sub_dict.keys():
                                    sub_dict[word_2] += 1
                                else:
                                    sub_dict[word_2] = 1

                if cnt % 1000 == 0:
                    print(cnt)
        
        with io.open(co_occur_output_file, 'w', encoding='utf-8') as output:
            csv_f = csv.writer(output)
            for k1, sub_dict in pair_dict.items():
                for k2, freq in sub_dict.items():
                    csv_f.writerow([k1, k2, freq])
