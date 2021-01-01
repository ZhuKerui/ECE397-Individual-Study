from dep_generator import *

class Sent2KW(Dep_Based_Embed_Generator):
    def find_kw(self, sent):
        word_tokens = [w.text for w in nlp(sent.lower())]
        keywords = []
        for idx, word_token in enumerate(word_tokens):
            if word_token in self.keywords:
                keywords.append(idx)
        return keywords

    def find_semantic_related_kw(self, sent):
        doc = nlp(sent.lower())
        ret = []
        for word in doc:
            if word.text not in self.keywords:
                continue
            for child in word.children:
                child_id = -1
                if child.dep_ == 'prep':
                    relation = 'prep_' + child.text
                    for grand_child in child.children:
                        if grand_child.dep_ == 'pobj':
                            if grand_child.text in self.keywords:
                                child_id = grand_child.i
                            break
                else:
                    relation = child.dep_
                    if child.text in self.keywords:
                        child_id = child.i
                if child_id >= 0:
                    ret.append([word.i, child_id, relation])
        return ret

    def register_files(self, reformed_file, relation_record_file):
        self.reformed_file = reformed_file
        self.relation_record_file = relation_record_file

    def extract_relation(self, start_line:int=0, lines:int=0):
        with io.open(self.reformed_file, 'r', encoding='utf-8') as load_file:
            self.relation_record = {}
            for idx, sent in enumerate(load_file):
                if idx < start_line:
                    continue
                semantic_related_list = self.find_semantic_related_kw(sent)
                if not semantic_related_list:
                    continue
                for triple in semantic_related_list:
                    relation = triple[2]
                    if relation not in self.relation_record.keys():
                        self.relation_record[relation] = [(idx, triple[0], triple[1])]
                    else:
                        self.relation_record[relation].append((idx, triple[0], triple[1]))
                cnt = idx - start_line
                if cnt >= lines - 1:
                    break
                if cnt % 100 == 0:
                    print('processed %.2f' %(float(cnt) * 100 / lines))

            with io.open(self.relation_record_file, 'w', encoding='utf-8') as dump_file:
                json.dump(self.relation_record, dump_file)

    def load_relation(self):
        self.relation_record = json.load(io.open(self.relation_record_file, 'r', encoding='utf-8'))

    def merge_relation(self, new_file):
        new_json = json.load(io.open(new_file, 'r', encoding='utf-8'))
        for key, value in new_json.items():
            if key in self.relation_record.keys():
                self.relation_record[key] += value
            else:
                self.relation_record[key] = value

    def get_sent_by_relation(self, relation, count):
        if relation not in self.relation_record.keys():
            return None
        with io.open(self.reformed_file, 'r', encoding='utf-8') as load_file:
            if count > len(self.relation_record[relation]):
                sent_list = self.relation_record[relation]
                count = len(self.relation_record[relation])
            else:
                sent_list = self.relation_record[relation][0:count]
            cnt = 0
            sent_id = -1
            sent = ''
            ret = []
            while cnt < count:
                item = sent_list[cnt]
                if sent_id < item[0]:
                    sent_id += 1
                    sent = load_file.readline().strip()
                elif sent_id == item[0]:
                    ret.append([sent, item[1], item[2], relation])
                    cnt += 1
                else:
                    # This should never happen
                    cnt += 1
            return ret