from relation_similar_web.dep_generator import *

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
        keywords = []
        for token in doc:
            if token.text in self.keywords:
                keywords.append(token)
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
                    mask[word.i] = True
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
        context = []
        win_idx = np.arange(self.win)
        for start_i in range(0, len(filtered_sent) - self.win + 1):
            sub_kw_mask = filtered_kw_mask[start_i:start_i+self.win]
            if sub_kw_mask.sum() <= 1:
                continue
            sub_sent = filtered_sent[start_i:start_i+self.win]
            keyword_pairs = self.__gen_keyword_pairs(win_idx[sub_kw_mask])
            for pair in keyword_pairs:
                context.append(' '.join([sub_sent[pair[0]], sub_sent[pair[1]]] + sub_sent[:pair[0]] + [self.x_placeholder] + sub_sent[pair[0]+1:pair[1]] + [self.y_placeholder] + sub_sent[pair[1]+1:]))
                context.append('\n')
        return ''.join(context)
