from relation_similar_web.dep_generator import *
from collections import Counter
from relation_similar_web.vocab import Vocab

class Pair_Generator(Dep_Based_Embed_Generator):
    def __init__(self, win=8):
        super().__init__()
        self.win = win
        self.ignored_pos = set(['PUNCT', 'DET'])
        self.unk, self.pad, self.x_placeholder, self.y_placeholder = '<unk>', '<pad>', '<X>', '<Y>'

    def __gen_keyword_pairs(self, keywords):
        ret = []
        for head in range(len(keywords)-1):
            for tail in range(head+1, len(keywords)):
                ret.append([keywords[head], keywords[tail]])
        return ret

    def extract_context(self, line):
        doc = nlp(line)
        keywords = [token for token in doc if self.key_vocab.stoi[token.text] != 0]
        if len(keywords) <= 1:
            return None
        ids = np.arange(len(doc))
        mask = np.zeros(len(doc), dtype=np.bool)
        kw_mask = np.zeros(len(doc), dtype=np.bool)
        for kw in keywords:
            kw_mask[kw.i] = True
            word = kw
            while True:
                if word.pos_ not in self.ignored_pos:
                    mask[word.i] = self.ctx_vocab.stoi[word.text] != 0
                if word.dep_ == 'ROOT':
                    break
                word = word.head
        filtered_ids = ids[mask]
        filtered_kw_mask = kw_mask[mask]
        filtered_sent = [doc[i].text for i in filtered_ids]
        if len(filtered_sent) < self.win:
            append_num = self.win - len(filtered_sent)
            filtered_sent += [self.pad] * append_num
            filtered_kw_mask = np.append(filtered_kw_mask, [False] * append_num)
        valid_ids = [self.ctx_vocab.stoi[word] for word in filtered_sent]
        context = []
        win_idx = np.arange(self.win)
        for start_i in range(0, len(filtered_sent) - self.win + 1):
            sub_kw_mask = filtered_kw_mask[start_i:start_i+self.win]
            if sub_kw_mask.sum() <= 1:
                continue
            sub_sent = filtered_sent[start_i:start_i+self.win]
            sub_valid_ids = valid_ids[start_i:start_i+self.win]
            keyword_pairs = self.__gen_keyword_pairs(win_idx[sub_kw_mask])
            for pair in keyword_pairs:
                context.append(' '.join([self.key_vocab.stoi[sub_sent[pair[0]]], self.key_vocab.stoi[sub_sent[pair[1]]]] + sub_valid_ids[:pair[0]] + [self.x_placeholder] + sub_valid_ids[pair[0]+1:pair[1]] + [self.y_placeholder] + sub_valid_ids[pair[1]+1:]))
                context.append('\n')
        return ''.join(context)

    def read_vocab_from_file(self, vocab_file):
        tokens = None
        with open(vocab_file+'_k') as f:
            text = f.read()
            tokens = text.rstrip().split('\n')
        self.key_vocab = Vocab(tokens, specials=[])

        with open(vocab_file+'_c') as f:
            text = f.read()
            tokens = text.rstrip().split('\n')
        self.ctx_vocab = Vocab(tokens, specials=['<unk>', '<pad>', '<X>', '<Y>'])

    def build_vocab(self, corpus_file, vocab_file, thr=30):
        ctx_counter = Counter()
        key_counter = Counter()
        with open(corpus_file, mode='r', encoding='utf-8') as f:
            for i_line, line in enumerate(f):
                words = line.strip().split()
                ctx_counter.update(Counter(words))
                key_counter.update(Counter([word for word in words if word in self.keywords]))
                if i_line % 1000000 == 0:
                    print(i_line)

        words_and_frequencies = sorted(ctx_counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        ctx_selected = []
        for word, freq in words_and_frequencies:
            if freq < thr:
                break
            ctx_selected.append(word)
        words_and_frequencies = sorted(key_counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        key_selected = []
        for word, freq in words_and_frequencies:
            if freq < thr:
                break
            key_selected.append(word)
        
        with io.open(vocab_file+'_k', 'w', encoding='utf-8') as f:
            f.write('\n'.join(key_selected))
        with io.open(vocab_file+'_c', 'w', encoding='utf-8') as f:
            f.write('\n'.join(ctx_selected))

        self.key_vocab = Vocab(key_selected, specials=[])
        self.ctx_vocab = Vocab(ctx_selected, specials=['<unk>', '<pad>', '<X>', '<Y>'])