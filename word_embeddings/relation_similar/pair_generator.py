import torch
from pyhocon import ConfigFactory
import sys
sys.path.append('..')
import os

# print(os.getcwd())

from pair2vec.model import MLP, Pair2Vec
from pair2vec.representation import SpanRepresentation
from my_keywords import *
from pair2vec.vocab import Vocab

class Pair_Generator:
    def __init__(self, keyword_vocab:Keyword_Vocab, context_vocab:Vocab_Base, win=8, margin=1):
        self.win = win
        self.margin = margin
        self.ignored_pos = set(['PUNCT', 'DET'])
        self.unk, self.pad, self.x_placeholder, self.y_placeholder = '<unk>', '<pad>', '<X>', '<Y>'
        self.key_vocab = keyword_vocab
        self.ctx_vocab = context_vocab
        self.model = None

    def __extract_context(self, line):
        doc = nlp(line)
        sent = []
        sent_ids = []
        for token in doc:
            ctx_id = self.ctx_vocab.stoi[token.text]
            if ctx_id != 0:
                sent.append(token.text)
                sent_ids.append(ctx_id)
        kw_idx = [idx for idx, word in enumerate(sent) if self.key_vocab.stoi[word] != 0]
        if len(kw_idx) <= 1:
            return None
        context = []
        for ix in range(len(kw_idx)-1):
            for iy in range(ix + 1, len(kw_idx)):
                x_id, y_id = kw_idx[ix], kw_idx[iy]
                interval = y_id - x_id
                if interval >= self.win:
                    break
                int_context = [self.key_vocab.stoi[sent[x_id]], self.key_vocab.stoi[sent[y_id]]] + [self.ctx_vocab.stoi[self.x_placeholder]] + sent_ids[x_id+1:y_id] + [self.ctx_vocab.stoi[self.y_placeholder]]
                int_context += [self.ctx_vocab.stoi[self.pad]] * (self.win + 2 - len(int_context))
                context.append(' '.join(map(str, int_context)))
                context.append('\n')
        return ''.join(context)

    def context_to_npy(self, context_file, npy_file):
        with io.open(context_file, 'r', encoding='utf-8') as f_ctx:
            npy_list = []
            block_id = 0
            for i_line, line in enumerate(f_ctx):
                new_npy_line = np.array(list(map(int, line.strip().split())), dtype=np.int32)
                npy_list.append(new_npy_line)
                if (i_line+1) % 1000000 == 0:
                    np.save(npy_file+str(block_id)+'.npy', np.array(npy_list, dtype=np.int32))
                    block_id += 1
                    npy_list = []
            if len(npy_list) > 0:
                np.save(npy_file+str(block_id)+'.npy', np.array(npy_list, dtype=np.int32))

    def extract_context(self, freq: int, input_file: str, output_file: str, thread_num: int):
        multithread_wrapper(self.__extract_context, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num, post_operation=self.context_to_npy)

    def translate_keyword(self, item):
        return ' '.join([self.key_vocab.itos[word] for word in item])

    def translate_context(self, item):
        return ' '.join([self.ctx_vocab.itos[word] for word in item])
    
    def translate_contexts(self, items):
        ret = []
        for item in items:
            ret.append(self.translate_context(item))
        return '\n'.join(ret)

    def translate_triple(self, item):
        return ' '.join([self.translate_keyword(item[:2]), self.translate_context(item[2:])])

    def translate_text(self, sent:str):
        text = self.__extract_context(sent)
        context_txt = text.strip().split('\n')
        context = [list(map(int, line.split())) for line in context_txt]
        for item in context:
            print(self.translate_triple(item))

    def load_inference_model(self, model_file:str, config_file:str):
        pair2vec_config = ConfigFactory.parse_file(config_file)
        rel_vocab = Vocab(self.ctx_vocab.itos, specials=[])
        arg_vocab = Vocab(self.key_vocab.itos, specials=[])
        self.model = Pair2Vec(pair2vec_config, rel_vocab=rel_vocab, arg_vocab=arg_vocab)
        state_dict = torch.load(model_file, map_location='cpu')['state_dict']
        self.model.load_state_dict(state_dict=state_dict)

    def get_vectors(self, keyword_pair):
        if not isinstance(keyword_pair, list):
            keyword_pair = [keyword_pair]
        
        idx_batch = [[self.key_vocab.stoi[kw1], self.key_vocab.stoi[kw2]] for kw1, kw2 in keyword_pair]
        idx_batch = np.array(idx_batch, dtype=np.int)
        predicted_relations = self.get_predict_vectors(idx_batch)
        return ugly_normalize(predicted_relations)

    def get_predict_vectors(self, sub_obj_pair:np.ndarray):
        sub_obj_pair_torch = torch.from_numpy(sub_obj_pair)
        sub, obj = sub_obj_pair_torch[:, 0], sub_obj_pair_torch[:, 1]
        sub = sub.reshape(-1)
        obj = obj.reshape(-1)
        subjects, objects = self.model.to_tensors((sub, obj))
        embedded_subjects = self.model.represent_left_argument(subjects.type(torch.LongTensor))
        embedded_objects = self.model.represent_right_argument(objects.type(torch.LongTensor))
        predicted_relations = self.model.predict_relations(embedded_subjects, embedded_objects).detach().numpy()
        return predicted_relations

    def get_relation_vectors(self, relation):
        relation_torch = torch.from_numpy(relation).type(torch.LongTensor)
        relation_torch, dump = self.model.to_tensors((relation_torch, torch.zeros(5)))
        observed_relations = self.model.represent_relations(relation_torch).detach().numpy()
        return observed_relations

    def context_text_to_npy(self, context_text_file, npy_file):
        with io.open(context_text_file, 'r', encoding='utf-8') as f_ctx_txt:
            npy_list = []
            for line in f_ctx_txt:
                new_npy_line = np.array([self.ctx_vocab.stoi[token] for token in line.strip().split(' ')], dtype=np.int32)
                npy_list.append(new_npy_line)
            np.save(npy_file, np.array(npy_list, dtype=np.int32))
            
if __name__ == '__main__':
    kv = Keyword_Vocab()
    cv = Vocab_Base()
    key_vocab_file = sys.argv[1]
    ctx_vocab_file = sys.argv[2]
    kv.load_vocab(key_vocab_file)
    cv.load_vocab(ctx_vocab_file)
    pg = Pair_Generator(kv, cv)
    # pg.load_inference_model('../../../dataset/outputs/pair2vec/triplets_1_big_model.pt')
    pg.extract_context(80, sys.argv[3], sys.argv[4], 30)