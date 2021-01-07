from my_keywords import *

class Sent2KW(Keyword_Base):
    def find_kw_idx(self, sent:str):
        keywords = self.find_keyword_tokens(sent)
        kw_idx = [keyword.i for keyword in keywords]
        return kw_idx

    def find_semantic_related_kw_idx(self, sent):
        triplets = self.find_keyword_context_dependency(sent)
        ret = [(keyword.i, child.i, relation) for keyword, child, relation in triplets if child.text in self.keywords]
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