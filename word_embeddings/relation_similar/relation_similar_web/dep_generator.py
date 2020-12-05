import json
import io
import spacy
import numpy as np
from relation_similar_web.my_multithread import *

nlp = spacy.load('en_core_web_sm')

def extract_sent(line):
    if not line:
        return None
    jsonObj = json.loads(line)
    para = jsonObj['abstract'].strip().replace('\n', ' ').replace('$', '').replace('--', ', ')
    return para + '\n'

def extract_sent_2(line):
    if line.find('"abstract": "') == 0:
        para = line.split(':', 1)[1].strip(' ",')
        para = para.replace('\\n', ' ').replace('$', '').replace('--', ', ')
        return para + '\n'

def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

def simple_normalize(vec):
    normalizer = np.sqrt(np.matmul(vec, vec))
    if normalizer == 0:
        normalizer = 1
    return vec / normalizer

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

                if '-' in word:
                    phrase = [w.text for w in nlp(word)]
                else:
                    phrase = word.split()
                if len(phrase) == 1:
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
            print('Building word tree is accomplished with %d words added' % (cnt))
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
                    if head and head != '-':
                        yield head + '_' + child
                    else:
                        yield child
            else:
                yield head

    def _process_sent(self, sent):
        if not sent:
            return None
        word_tokens = [w.text for w in nlp(sent.lower())]
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
                    reformed_sent.append('_'.join(phrase_buf).replace('_-_', '_'))
                reformed_sent += tail_buf
        return ' '.join(reformed_sent).replace(' - ', '-')

    def extract_context(self, line):
        if not line:
            return None
        doc = nlp(line)
        context_buffer = []
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
                else:
                    relation = child.dep_
                    child_txt = child.text.lower()
                context_buffer.append(' '.join((word_txt, '_'.join((relation, child_txt)))) + '\n')

            context_buffer.append(' '.join((word_txt, 'I_'.join((word.dep_, word.head.text.lower())))) + '\n')
        if context_buffer:
            return ''.join(context_buffer)
        else:
            return None

    def extract_word_vector(self, origin_file, output_file):
        fh=open(origin_file)
        first=fh.readline()
        line_num, vec_len = first.strip().split(' ')
        line_num = int(line_num)
        vec_len = int(vec_len)
        wvecs=[]
        vocab=[]
        vocab2i = {}
        for i in range(line_num):
            line = fh.readline().strip().split()
            if len(line) == vec_len + 1:
                vocab.append(line[0])
                wvecs.append(np.array(list(map(float,line[1:]))))
                vocab2i[line[0]] = i
            else:
                print('Broken file')
                return

        self.vocab = vocab
        self.wvecs = ugly_normalize(np.array(wvecs, float))
        self.vocab2i = vocab2i
        np.save(output_file+".npy",self.wvecs)
        with open(output_file+".vocab","w") as outf:
            outf.write(" ".join(self.vocab))
            
    def load_word_vector(self, load_file):
        self.vocab = io.open(load_file + '.vocab', 'r', encoding='utf-8').readline().split()
        self.wvecs = np.load(load_file + '.npy')
        self.vocab2i = {word:i for i, word in enumerate(self.vocab)}

    def get_similarity(self, kw1, kw2):
        if kw1 in self.vocab and kw2 in self.vocab:
            return self.wvecs[self.vocab2i[kw1]].dot(self.wvecs[self.vocab2i[kw2]])
        else:
            return None
