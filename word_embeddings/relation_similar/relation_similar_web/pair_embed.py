from relation_similar_web.dep_generator import *
from relation_similar_web.vdbscan import do_cluster
import heapq

class Pair_Embed(Dep_Based_Embed_Generator):
    def extract_context(self, line):
        if not line:
            return None
        doc = nlp(line)
        kw2ctx = {}
        for word in doc:
            if word.text not in self.keywords:
                continue
            word_txt = word.text
            if word_txt not in kw2ctx:
                kw2ctx[word_txt] = set()
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
                kw2ctx[word_txt].add(relation + '_' + child_txt)

            kw2ctx[word_txt].add(word.dep_ + 'I_' + word.head.text)
        if len(kw2ctx) <= 1:
            return None
        kws = list(kw2ctx.keys())
        str_buffer = []
        for i in range(len(kws)-1):
            for j in range(i+1, len(kws)):
                pair_1 = kws[i] + '__' + kws[j]
                pair_2 = kws[j] + '__' + kws[i]
                for ctx in kw2ctx[kws[i]]:
                    str_buffer.append(pair_1 + ' h_' + ctx + '\n')
                    str_buffer.append(pair_2 + ' t_' + ctx + '\n')
                for ctx in kw2ctx[kws[j]]:
                    str_buffer.append(pair_1 + ' t_' + ctx + '\n')
                    str_buffer.append(pair_2 + ' h_' + ctx + '\n')

        return ''.join(str_buffer)

    def extract_word_vector(self, load_file, output_file):
        Dep_Based_Embed_Generator.extract_word_vector(self, load_file, output_file)
        w2set = {}
        for word, idx in self.vocab2i.items():
            cw = word.split('__')[0]
            if cw not in w2set:
                w2set[cw] = []
            w2set[cw].append(idx)
        self.w2set = {}
        for cw, idxs in w2set.items():
            self.w2set[cw] = np.array(idxs)
        with io.open(output_file+'.json', 'w', encoding='utf-8') as f_output:
            json.dump(self.w2set, f_output)
            
    def load_word_vector(self, load_file):
        Dep_Based_Embed_Generator.load_word_vector(self, load_file)
        self.w2set = json.load(io.open(load_file + '.json', 'r', encoding='utf-8'))

    def get_co_occur_pairs(self, cw):
        try:
            self.w2set
        except NameError:
            print('Pairs are not loaded')
            return None
        if cw not in self.w2set:
            print('%s does not exist' % (cw))
            return None
        idxs = self.w2set[cw]
        pair_vecs = self.wvecs[idxs]
        pairs = [self.vocab[i] for i in idxs]
        return (pairs, pair_vecs)
        
    def dbscan_cluster(self, cw, k:int=3):
        try:
            self.w2set
        except NameError:
            print('Pairs are not loaded')
            return None
        if cw not in self.w2set:
            print('%s does not exist' % (cw))
            return None
        pairs, vecs = self.get_co_occur_pairs(cw)
        pairs = [item.split('__')[1] for item in pairs]
        label = do_cluster(vecs, k)
        if all(label == -1):
            score = -1
        else:
            score = silhouette_score(vecs, label, metric='cosine')
        clusters = []
        cluster_num = max(label) + 1
        for i in range(cluster_num + 1):
            clusters.append(set())
        for word_idx, cluster_id in enumerate(label):
            clusters[cluster_id].add(pairs[word_idx])
        return score, clusters

    def find_similar_pairs(self, pair, n):
        try:
            self.vocab
        except NameError:
            print('Pairs are not loaded')
            return None
        if pair not in self.vocab:
            print('%s does not exist' % (pair))
            return None
        pair_vec = self.wvecs[self.vocab2i[pair]]
        similarity_vec = self.wvecs.dot(pair_vec)
        result = heapq.nlargest(n, zip(similarity_vec, self.vocab), key=lambda x: x[0])
        return_pairs = [item[1].split('__')[1] for item in result]
        return return_pairs