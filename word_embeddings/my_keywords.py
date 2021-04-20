import numpy as np
import json
import io
import spacy
from collections import defaultdict, Counter
import sys
sys.path.append('..')
from my_multithread import MultiThreading
from my_util import ugly_normalize, simple_normalize
nlp = spacy.load('en_core_web_sm')

class Keyword_Base:

    def build_word_tree(self, input_txt:str, dump_file:str):
        self.MyTree = {}
        self.keywords = set()
        cnt = 0
        with io.open(input_txt, 'r', encoding='utf-8') as load_file:
            for word in load_file:
                cnt += 1

                word = word.strip()
                # Directly add the '_' connected keyword
                normalized_word = word.replace('-', ' - ').replace(' ', '_')
                self.keywords.add(normalized_word)

                # Insert the keyword to the tree structure

                phrase = normalized_word.split('_')
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
                # if cnt % 1000 == 0:
                #     print(cnt)
            print('Building word tree is accomplished with %d words added' % (cnt))
        with io.open(dump_file, 'w', encoding='utf-8') as output_file:
            json.dump(self.MyTree, output_file)

    def __read_word_tree(self, head:str, node:dict):
        for key in node.keys():
            if key:
                for child in self.__read_word_tree(key, node[key]):
                    if head:
                        yield head + '_' + child
                    else:
                        yield child
            else:
                yield head

    def load_word_tree(self, json_file:str):
        with io.open(json_file, 'r', encoding='utf-8') as load_file:
            self.MyTree = eval(load_file.readline())
            self.keywords = set(self.__read_word_tree('', self.MyTree))

    def __process_sent(self, line:str):
        if not line:
            return None
        sent_list = []
        for sent in nlp(line.lower()).sents:
            word_tokens = [w.text for w in sent]
            reformed_sent = []
            i = 0
            keyword_num = 0
            while i < len(word_tokens):
                if word_tokens[i] not in self.MyTree.keys():
                    # If the word is not part of a key word directly add the word to the reformed_sent list
                    if word_tokens[i] != ' ':
                        reformed_sent.append(word_tokens[i])
                    i += 1

                else:
                    # If the word is part of a key word
                    phrase_buf = []
                    phrase_wait_buf = []
                    tail_buf = []
                    it = self.MyTree
                    while i < len(word_tokens) and word_tokens[i] in it.keys():
                        if word_tokens[i] != ' ':
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
                        keyword_num += 1
                    reformed_sent += tail_buf
            if keyword_num > 0:
                sent_list.append(' '.join(reformed_sent) + '\n')
        return ''.join(sent_list)

    def process_sent(self, freq:int, input_file:str, output_file:str, thread_num:int=1):
        multithreading = MultiThreading()
        multithreading.run(self.__process_sent, input_file=input_file, output_file=output_file, thread_num=thread_num)


class Vocab_Base:
    def __init__(self, word_list:list=['<unk>'], vectors:np.ndarray=None):
        self.itos = word_list
        self.stoi = defaultdict(lambda: 0)
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
        self.vectors = vectors if vectors is not None and len(vectors) == len(self.itos) else None

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def read_embedding_file(self, source_file:str, save_file:str=None):
        with io.open(source_file, 'r', encoding='utf-8') as fh:
            first=fh.readline()
            line_num, vec_len = first.strip().split(' ')
            line_num = int(line_num)
            vec_len = int(vec_len)
            vectors = np.zeros((len(self.itos), vec_len), dtype=np.float)
            try:
                for i in range(line_num):
                    line = fh.readline().strip().split()
                    idx = self.stoi[line[0]]
                    if idx != 0 or line[0] == self.itos[0]:
                        vectors[idx] = np.array(list(map(float,line[1:])))
            except:
                print('Broken file')
                return

            self.vectors = ugly_normalize(np.array(vectors, float))
            if save_file is not None:
                self.save_vocab(save_file+'.vocab')
                self.save_vector(save_file+'.npy')
    
    def save_vocab(self, save_file:str):
        with io.open(save_file, 'w', encoding='utf-8') as save_f:
            save_f.write('\n'.join(self.itos))
    
    def save_vector(self, save_file:str):
        if self.vectors is not None:
            np.save(save_file, self.vectors)

    def load_vocab(self, load_file:str):
        self.itos = io.open(load_file, 'r', encoding='utf-8').read().split('\n')
        self.stoi = defaultdict(lambda: 0)
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def load_vector(self, load_file:str):
        vectors = np.load(load_file)
        if vectors is not None and len(vectors) == len(self.itos):
            self.vectors = vectors
        else:
            print('vector file does not meet requirement')


class Keyword_Vocab(Vocab_Base):
    def __init__(self, word_list:list=['<unk>'], vectors:np.ndarray=None):
        super().__init__(word_list=word_list, vectors=vectors)

    def find_keyword_tokens(self, sent:str):
        ret = [word_token for word_token in nlp(sent.lower()) if self.stoi[word_token.text] != 0]
        return ret

    def find_keyword_context_dependency(self, sent:str):
        keywords = self.find_keyword_tokens(sent)
        triplets = []
        for keyword in keywords:
            for child in keyword.children:
                child_token = None
                if child.dep_ == 'prep':
                    relation = 'prep_' + child.text
                    for grand_child in child.children:
                        if grand_child.dep_ == 'pobj':
                            child_token = grand_child
                            break
                else:
                    relation = child.dep_
                    child_token = child
                if child_token is not None:
                    triplets.append((keyword, child_token, relation))
            triplets.append((keyword, keyword.head, keyword.dep_ + 'I'))
        return triplets


class Vocab_Generator(Keyword_Base):

    def build_vocab(self, corpus_file, vocab_file, special_key=['<unk>'], special_ctx=['<unk>'], thr=10):
        ctx_counter = Counter()
        key_counter = Counter()
        with open(corpus_file, mode='r', encoding='utf-8') as f:
            for i_line, line in enumerate(f):
                words = line.strip().split()
                ctx_counter.update(Counter(words))
                key_counter.update(Counter([word for word in words if word in self.keywords]))
                if i_line % 1000000 == 0:
                    print(i_line)

        words_and_frequencies = sorted(ctx_counter.items(), key=lambda tup: tup[1], reverse=True)
        ctx_selected = []
        for word, freq in words_and_frequencies:
            if freq < thr:
                break
            ctx_selected.append(word)
        words_and_frequencies = sorted(key_counter.items(), key=lambda tup: tup[1], reverse=True)
        key_selected = []
        for word, freq in words_and_frequencies:
            if freq < thr:
                break
            key_selected.append(word)
        
        key_vocab = Keyword_Vocab(special_key + key_selected)
        ctx_vocab = Vocab_Base(special_ctx + ctx_selected)
        key_vocab.save_vocab(vocab_file + '_key.vocab')
        ctx_vocab.save_vocab(vocab_file + '_ctx.vocab')
        return key_vocab, ctx_vocab

