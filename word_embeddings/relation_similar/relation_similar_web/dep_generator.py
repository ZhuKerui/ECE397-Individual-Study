import json
import io
import spacy
from nltk.tokenize import word_tokenize
import re
import numpy as np
import sys

nlp = spacy.load('en_core_web_sm')

def extract_sent(json_file, store_file, start_line:int=0, end_line:int=None):
    with io.open(json_file, 'r', encoding='utf-8') as load_file:
        with io.open(store_file, 'w', encoding='utf-8') as output:
            idx = -1
            for idx, line in enumerate(load_file):
                if idx < start_line:
                    continue
                jsonObj = json.loads(line)
                para = jsonObj['abstract'].strip().replace('\n', ' ')
                latex_str = re.search(r'\$.*?\$', para)
                while latex_str:
                    para = para.replace(latex_str.group(), '')
                    latex_str = re.search(r'\$.*?\$', para)
                doc = nlp(para)
                for sentence in doc.sents:
                    output.write(str(sentence) + '\n')

                if end_line is not None and idx >= end_line - 1:
                    break
                if idx % 1000 == 0:
                    print(idx)
            print(idx)

def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

def simple_normalize(vec):
    normalizer = np.sqrt(np.matmul(vec, vec))
    if normalizer == 0:
        normalizer = 1
    return vec / normalizer


def tailor(output_file, input_prefix, input_posfix, num):
    with io.open(output_file, 'w', encoding='utf-8') as dump_file:
        for i in range(num):
            with io.open(input_prefix + str(i) + input_posfix, 'r', encoding='utf-8') as load_file:
                for line in load_file:
                    if line.strip():
                        dump_file.write(line)

class Dep_Based_Embed_Generator:

    def build_word_tree(self, input_txt, dump_file):
        self.MyTree = {}
        self.keywords = set()
        cnt = 0
        with io.open(input_txt, 'r', encoding='utf-8') as load_file:
            for word in load_file:
                cnt += 1

                word = word.strip()
                # Directly add the '_' connected keyword
                self.keywords.add(word.replace(' ', '_').replace('-', '_'))

                # Insert the keyword to the tree structure
                if ' ' not in word and '-' not in word:
                    # If the word is an atomic word instead of a phrase
                    if word not in self.MyTree.keys():
                        # If this is the first time that this word is inserted to the tree
                        self.MyTree[word] = {"":""}
                    elif "" not in self.MyTree[word].keys():
                        # If the word has been inserted but is viewed as an atomic word the first time
                        self.MyTree[word][""] = ""
                    # If the word has already been inserted as an atomic word, then we do nothing
                else:
                    # If the word is an phrase
                    phrase = re.split('[ -]', word)
                    length = len(phrase)
                    fw = phrase[0]
                    if fw not in self.MyTree.keys():
                        self.MyTree[fw] = {}
                    temp_dict = self.MyTree.copy()
                    parent_node = fw
                    for i in range(1, length):
                        if phrase[i]:
                            sw = phrase[i]
                            if sw not in temp_dict[parent_node].keys():
                                # The second word is inserted to as the child of parent node the first time
                                temp_dict[parent_node][sw] = {}
                            if i == length - 1:
                                # If the second word is the last word in the phrase
                                if "" not in temp_dict[parent_node][sw].keys():
                                    temp_dict[parent_node][sw][""] = ""
                            else:
                                # If the second word is not the last word in the phrase
                                temp_dict = temp_dict[parent_node].copy()
                                parent_node = sw
                if cnt % 1000 == 0:
                    print(cnt)
            print('Building word tree is accomplished with {:d} words added'.format(cnt))
        with io.open(dump_file, 'w', encoding='utf-8') as output_file:
            json.dump(self.MyTree, output_file)

    def load_word_tree(self, json_file):
        with io.open(json_file, 'r', encoding='utf-8') as load_file:
            self.MyTree = eval(load_file.readline())
            self.keywords = set(self.__read_word_tree('', self.MyTree))

    def __read_word_tree(self, head:str, node:dict):
        # print(head)
        for key in node.keys():
            if key:
                for child in self.__read_word_tree(key, node[key]):
                    if head:
                        yield head + '_' + child
                    else:
                        yield child
            else:
                yield head

    def process_sent(self, sent):
        if self.MyTree is None:
            print("You haven't load the keywords yet, please use build_word_tree(input_txt, dump_file) or load_word_tree(json_file) to load the keywords")
            return
        if not sent:
            return ''
        word_tokens = word_tokenize(sent.lower().replace('-', ' '))
        reformed_sent = []
        i = 0
        while i < len(word_tokens):
            if word_tokens[i] not in self.MyTree.keys():
                # If the word is not part of a key word directly add the word to the reformed_sent list
                reformed_sent.append(word_tokens[i])
                i += 1

            else:
                # If the word is part of a key word
                phrase_buf = []
                phrase_wait_buf = []
                tail_buf = []
                it = self.MyTree
                while i < len(word_tokens) and word_tokens[i] in it.keys():
                    # Add the word to the wait list
                    phrase_wait_buf.append(word_tokens[i])
                    tail_buf.append(word_tokens[i])
                    if "" in it[word_tokens[i]].keys():
                        # If the word could be the last word of a keyword, update the phrase buffer to be the same with wait buffer
                        phrase_buf = phrase_wait_buf.copy()
                        tail_buf = []
                    # Go down the tree to the next child
                    it = it[word_tokens[i]]
                    i += 1
                # Change the keyword into one uniformed word and add it to the reformed_sent
                if phrase_buf:
                    reformed_sent.append('_'.join(phrase_buf))
                reformed_sent += tail_buf
        return ' '.join(reformed_sent)

    def process_sents(self, sent_file, reformed_sent_file, start_line:int=0, end_line:int=None):
        with io.open(reformed_sent_file, 'w', encoding='utf-8') as output_file:
            with io.open(sent_file, 'r', encoding='utf-8') as load_file:
                idx = -1
                for idx, line in enumerate(load_file):
                    if idx < start_line:
                        continue
                    sent = line.strip()
                    if sent:
                        reformed = self.process_sent(sent)
                        if reformed:
                            output_file.write(reformed)
                            output_file.write('\n')
                    if end_line is not None and idx >= end_line - 1:
                        break
                    if idx % 1000 == 0:
                        print(idx)
                print('Process sentences accomplished with {:d} lines processed'.format(1 + idx - start_line))

    def extract_context(self, reformed_file:str, context_file:str, start_line:int=0, end_line:int=None):
        if self.keywords is None:
            print("You haven't load the keywords yet, please use build_word_tree(input_txt, dump_file) or load_word_tree(json_file) to load the keywords")
            return
        with io.open(context_file, 'w', encoding='utf-8') as output_file:
            with io.open(reformed_file, 'r', encoding='utf-8') as load_file:
                idx = -1
                for idx, line in enumerate(load_file):
                    if idx < start_line:
                        continue
                    if not line:
                        continue
                    doc = nlp(line)
                    for word in doc:
                        if word.text not in self.keywords:
                            continue
                        word_txt = word.text.lower()
                        for child in word.children:
                            if child.dep_ == 'prep':
                                relation = ''
                                child_txt = ''
                                for grand_child in child.children:
                                    if grand_child.dep_ == 'pobj':
                                        relation = 'prep_' + child.text.lower()
                                        child_txt = grand_child.text.lower()
                                if not relation:
                                    continue
                            # elif child.dep_ in self.bad_deps:
                                # continue
                            else:
                                relation = child.dep_
                                child_txt = child.text.lower()
                            output_file.write(' '.join((word_txt, '_'.join((relation, child_txt)))) + '\n')

                        # if word.head.text and word.dep_ not in self.bad_deps:
                        output_file.write(' '.join((word_txt, 'I_'.join((word.dep_, word.head.text.lower())))) + '\n')
                    if end_line is not None and idx >= end_line - 1:
                        break
                    if idx % 1000 == 0:
                        print(idx)
                print('Extract context accomplished with {:d} lines processed'.format(1 + idx - start_line))

    def extract_word_vector(self, origin_file, output_file):
        if self.keywords is None:
            print("You haven't load the keywords yet, please use build_word_tree(input_txt, dump_file) or load_word_tree(json_file) to load the keywords")
            return
        fh=open(origin_file)
        first=fh.readline()
        wvecs=[]
        vocab=[]
        for line in fh:
            line = line.strip().split()
            if line[0] in self.keywords:
                vocab.append(line[0])
                wvecs.append(np.array(list(map(float,line[1:]))))

        self.vocab = vocab
        self.wvecs = np.array(wvecs, float)
        self.n_wvecs = ugly_normalize(self.wvecs)
        self.vocab2i = {word:i for i, word in enumerate(self.vocab)}
        np.save(output_file+".npy",self.wvecs)
        with open(output_file+".vocab","w") as outf:
            outf.write(" ".join(self.vocab))
            
    def load_word_vector(self, load_file):
        self.vocab = io.open(load_file + '.vocab', 'r', encoding='utf-8').readline().split()
        self.wvecs = np.load(load_file + '.npy')
        self.n_wvecs = ugly_normalize(self.wvecs)
        self.vocab2i = {word:i for i, word in enumerate(self.vocab)}

    def get_similarity(self, kw1, kw2):
        if kw1 in self.vocab and kw2 in self.vocab:
            return self.n_wvecs[self.vocab2i[kw1]].dot(self.n_wvecs[self.vocab2i[kw2]])
        else:
            return None