import torch
from pyhocon import ConfigFactory

from pair2vec.model import MLP, SpanRepresentation
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
        self.vectors = None

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

    # def __extract_context(self, line):
    #     doc = nlp(line)
    #     keywords = [token for token in doc if self.key_vocab.stoi[token.text] != 0]
    #     if len(keywords) <= 1:
    #         return None
    #     ids = np.arange(len(doc))
    #     mask = np.zeros(len(doc), dtype=np.bool)
    #     kw_mask = np.zeros(len(doc), dtype=np.bool)
    #     for kw in keywords:
    #         kw_mask[kw.i] = True
    #         word = kw
    #         while True:
    #             if word.pos_ not in self.ignored_pos:
    #                 mask[word.i] = self.ctx_vocab.stoi[word.text] != 0
    #             # mask[word.i] = self.ctx_vocab.stoi[word.text] != 0
    #             if word.dep_ == 'ROOT':
    #                 break
    #             word = word.head
    #     filtered_ids = ids[mask]
    #     filtered_kw_mask = kw_mask[mask]
    #     filtered_sent = [doc[i].text for i in filtered_ids]
    #     # if len(filtered_sent) < self.win:
    #     #     append_num = self.win - len(filtered_sent)
    #     #     filtered_sent += [self.pad] * append_num
    #     #     filtered_kw_mask = np.append(filtered_kw_mask, [False] * append_num)
    #     valid_ids = [self.ctx_vocab.stoi[word] for word in filtered_sent]
    #     context = []
    #     kw_ids = np.arange(len(filtered_kw_mask))[filtered_kw_mask]
    #     for ix in range(len(kw_ids)-1):
    #         for iy in range(ix + 1, len(kw_ids)):
    #             x_id, y_id = kw_ids[ix], kw_ids[iy]
    #             interval = y_id - x_id
    #             if interval >= self.win:
    #                 break
    #             available_margin = self.win - (interval + 1)
    #             start = max(0, x_id - min(available_margin, self.margin))
    #             end = min(y_id + self.margin, start + self.win - 1)
    #             int_context = [self.key_vocab.stoi[filtered_sent[x_id]], self.key_vocab.stoi[filtered_sent[y_id]]] + valid_ids[start:x_id] + [self.ctx_vocab.stoi[self.x_placeholder]] + valid_ids[x_id+1:y_id] + [self.ctx_vocab.stoi[self.y_placeholder]] + valid_ids[y_id+1:end+1]
    #             int_context += [self.ctx_vocab.stoi[self.pad]] * (self.win + 2 - len(int_context))
    #             context.append(' '.join(map(str, int_context)))
    #             context.append('\n')

    #     # win_idx = np.arange(self.win)
    #     # for start_i in range(0, len(filtered_sent) - self.win + 1):
    #     #     sub_kw_mask = filtered_kw_mask[start_i:start_i+self.win]
    #     #     if sub_kw_mask.sum() <= 1:
    #     #         continue
    #     #     sub_sent = filtered_sent[start_i:start_i+self.win]
    #     #     sub_valid_ids = valid_ids[start_i:start_i+self.win]
    #     #     keyword_pairs = []
    #     #     temp_idx = win_idx[sub_kw_mask]
    #     #     for head in range(len(temp_idx)-1):
    #     #         for tail in range(head+1, len(temp_idx)):
    #     #             keyword_pairs.append([temp_idx[head], temp_idx[tail]])

    #     #     for pair in keyword_pairs:
    #     #         int_context = [self.key_vocab.stoi[sub_sent[pair[0]]], self.key_vocab.stoi[sub_sent[pair[1]]]] + sub_valid_ids[:pair[0]] + [self.ctx_vocab.stoi[self.x_placeholder]] + sub_valid_ids[pair[0]+1:pair[1]] + [self.ctx_vocab.stoi[self.y_placeholder]] + sub_valid_ids[pair[1]+1:]
    #     #         context.append(' '.join(map(str, int_context)))
    #     #         context.append('\n')
    #     return ''.join(context)

    def extract_context(self, freq: int, input_file: str, output_file: str, thread_num: int):
        def context_to_npy(context_file, npy_file):
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
        multithread_wrapper(self.__extract_context, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num, post_operation=context_to_npy)

    def translate_keyword(self, item):
        return ' '.join([self.key_vocab.itos[word] for word in item])

    def translate_context(self, item):
        return ' '.join([self.ctx_vocab.itos[word] for word in item])

    def translate_triple(self, item):
        return ' '.join([self.translate_keyword(item[:2]), self.translate_context(item[2:])])

    def translate_text(self, sent:str):
        text = self.__extract_context(sent)
        context_txt = text.strip().split('\n')
        context = [list(map(int, line.split())) for line in context_txt]
        for item in context:
            print(self.translate_triple(item))

    def load_inference_model(self, model_file:str):
        model = torch.load(model_file, map_location='cpu')
        mlp_config = ConfigFactory.parse_file('pair2vec/mlp_config.json')
        self.mlp = MLP(mlp_config)
        mlp_state_dict = {
            'mlp.1.weight': model['state_dict']['predict_relations.mlp.1.weight'],
            'mlp.1.bias': model['state_dict']['predict_relations.mlp.1.bias'],
            'mlp.4.weight': model['state_dict']['predict_relations.mlp.4.weight'],
            'mlp.4.bias': model['state_dict']['predict_relations.mlp.4.bias'],
            'mlp.7.weight': model['state_dict']['predict_relations.mlp.7.weight'],
            'mlp.7.bias': model['state_dict']['predict_relations.mlp.7.bias'],
            'mlp.10.weight': model['state_dict']['predict_relations.mlp.10.weight'],
            'mlp.10.bias': model['state_dict']['predict_relations.mlp.10.bias']
        }
        self.mlp.load_state_dict(state_dict=mlp_state_dict)
        self.vectors = model['state_dict']['represent_arguments.weight']

        rep_config = ConfigFactory.parse_file('pair2vec/rep_config.json')
        rel_vocab = Vocab(self.ctx_vocab.itos, specials=['<unk>', '<pad>', '<X>', '<Y>'])
        self.rep = SpanRepresentation(rep_config, rep_config.d_rels, rel_vocab)
        rep_state_dict = {
            'embedding.weight' : model['state_dict']['represent_relations.embedding.weight'],
            'contextualizer.rnn.weight_ih_l0' : model['state_dict']['represent_relations.contextualizer.rnn.weight_ih_l0'],
            'contextualizer.rnn.weight_hh_l0' : model['state_dict']['represent_relations.contextualizer.rnn.weight_hh_l0'],
            'contextualizer.rnn.bias_ih_l0' : model['state_dict']['represent_relations.contextualizer.rnn.bias_ih_l0'],
            'contextualizer.rnn.bias_hh_l0' : model['state_dict']['represent_relations.contextualizer.rnn.bias_hh_l0'],
            'contextualizer.rnn.weight_ih_l0_reverse' : model['state_dict']['represent_relations.contextualizer.rnn.weight_ih_l0_reverse'],
            'contextualizer.rnn.weight_hh_l0_reverse' : model['state_dict']['represent_relations.contextualizer.rnn.weight_hh_l0_reverse'],
            'contextualizer.rnn.bias_ih_l0_reverse' : model['state_dict']['represent_relations.contextualizer.rnn.bias_ih_l0_reverse'],
            'contextualizer.rnn.bias_hh_l0_reverse' : model['state_dict']['represent_relations.contextualizer.rnn.bias_hh_l0_reverse'],
            'head_attention.1.weight' : model['state_dict']['represent_relations.head_attention.1.weight'],
            'head_attention.1.bias' : model['state_dict']['represent_relations.head_attention.1.bias'],
            'head_transform.1.weight' : model['state_dict']['represent_relations.head_transform.1.weight'],
            'head_transform.1.bias' : model['state_dict']['represent_relations.head_transform.1.bias']
        }
        self.rep.load_state_dict(state_dict=rep_state_dict)

    def get_vectors(self, keyword_pair):
        if not isinstance(keyword_pair, list):
            keyword_pair = [keyword_pair]
        
        idx_batch = [[self.key_vocab.stoi[kw1], self.key_vocab.stoi[kw2]] for kw1, kw2 in keyword_pair]
        idx_batch = np.array(idx_batch, dtype=np.int)
        idx_batch_torch = torch.from_numpy(idx_batch)
        obj_idx_batch = idx_batch_torch[:, 1]
        sub_idx_batch = idx_batch_torch[:, 0]
        obj_vecs = self.vectors[obj_idx_batch]
        sub_vecs = self.vectors[sub_idx_batch]
        vectors = self.mlp(sub_vecs, obj_vecs).detach().numpy()
        return ugly_normalize(vectors)