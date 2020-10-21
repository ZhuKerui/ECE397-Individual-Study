import csv
import io
import json
import spacy
from nltk.tokenize import word_tokenize
import re
from spacy_conll import init_parser
import numpy as np
import sys

nlp = spacy.load('en_core_web_sm')

# with io.open('../../dataset/word2vec/mag_cs_keywords.csv', 'r', encoding='utf-8') as load_file:
#     csv_reader = csv.reader(load_file)
#     with io.open('../../dataset/word2vec/keyword.txt', 'w', encoding='utf-8') as dump_file:
#         for row in csv_reader:
#             dump_file.write(row[1])
#             dump_file.write('\n')



# with io.open('../../dataset/filtered_arxiv.json', 'r', encoding='utf-8') as load_file:
#     with io.open('../../dataset/sent.txt', 'w', encoding='utf-8') as dump_file:
#         cnt = 0
#         for line in load_file:
#             temp = line.strip().split(':')
#             if temp[0].strip() == '"abstract"':
#                 cnt += 1
#                 abstract_str = ':'.join(temp[1:]).strip().replace('\n', ' ')
#                 latex_str = re.search(r'\$.*?\$', abstract_str)
#                 while latex_str:
#                     abstract_str = abstract_str.replace(latex_str.group(), '')
#                     latex_str = re.search(r'\$.*?\$', abstract_str)
#                 doc = nlp(abstract_str)
#                 for sentence in doc.sents:
#                     dump_file.write(str(sentence) + '\n')
#                 if cnt % 1000 == 0:
#                     print(cnt)

with io.open('../../dataset/word2vec/sent.txt', 'r', encoding='utf-8') as load_file:
    with io.open('../../dataset/word2vec/sent2.txt', 'w', encoding='utf-8') as dump_file:
        cnt = 0
        not_finish = False
        for line in load_file:
            line = line.strip()
            if line[-1] == '.':
                if cnt == 0:
                    dump_file.write(line)
                elif not_finish:
                    dump_file.write(' '+line)
                    not_finish = False
                else:
                    dump_file.write('\n'+line)
                cnt += 1
            else:
                dump_file.write(' '+line)
                not_finish = True
            if cnt % 10000 == 0:
                print(cnt)
