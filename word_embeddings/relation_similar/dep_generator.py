import numpy as np

from my_keywords import *

def extract_sent_from_big(freq:int, input_file:str, output_file:str, thread_num:int=1):
    def extract_sent(line:str):
        if not line:
            return None
        jsonObj = json.loads(line)
        para = jsonObj['abstract'].strip().replace('\n', ' ').replace('$', '').replace('--', ', ').replace('-', ' - ')
        return para + '\n'
    multithread_wrapper(extract_sent, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num)

def extract_sent_from_small(freq:int, input_file:str, output_file:str, thread_num:int=1):
    def extract_sent(line:str):
        if line.find('"abstract": "') == 0:
            para = line.split(':', 1)[1].strip(' ",')
            para = para.replace('\\n', ' ').replace('$', '').replace('--', ', ').replace('-', ' - ')
            return para + '\n'
    multithread_wrapper(extract_sent, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num)

def ugly_normalize(vecs:np.ndarray):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

def simple_normalize(vec:np.ndarray):
    normalizer = np.sqrt(np.matmul(vec, vec))
    if normalizer == 0:
        normalizer = 1
    return vec / normalizer

class Dep_Based_Embed_Generator(Keywords):

    ignore_dep = set(['punct', 'dep'])

    def __extract_context(self, line:str):
        if not line:
            return None
        doc = nlp(line)
        context_buffer = []
        for word in doc:
            if word.text not in self.keywords:
                continue
            word_txt = word.text.lower()
            for child in word.children:
                if child.dep_ in self.ignore_dep:
                    continue
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

            if word.dep_ not in self.ignore_dep:
                context_buffer.append(' '.join((word_txt, 'I_'.join((word.dep_, word.head.text.lower())))) + '\n')
        if context_buffer:
            return ''.join(context_buffer)
        else:
            return None

    def extract_context(self, freq:int, input_file:str, output_file:str, thread_num:int=1):
        multithread_wrapper(self.__extract_context, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num)

    def extract_word_vector(self, origin_file:str, output_file:str):
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
            
    def load_word_vector(self, load_file:str):
        self.vocab = io.open(load_file + '.vocab', 'r', encoding='utf-8').readline().split()
        self.wvecs = np.load(load_file + '.npy')
        self.vocab2i = {word:i for i, word in enumerate(self.vocab)}

    def get_similarity(self, kw1:str, kw2:str):
        if kw1 in self.vocab and kw2 in self.vocab:
            return self.wvecs[self.vocab2i[kw1]].dot(self.wvecs[self.vocab2i[kw2]])
        else:
            return None
