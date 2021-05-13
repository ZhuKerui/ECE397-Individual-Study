from tools.BasicUtils import my_read

class EntityOccurrence:
    def __init__(self, entity_file:str):
        self.keyword_list = my_read(entity_file)
        self.entity_dict = {word : i for i, word in enumerate(self.keyword_list)}
        self.line_record = [set() for i in range(len(self.keyword_list))]

    def line_operation(self, line:str):
        line_idx, sent = line.split(':', 1)
        tokens = sent.split()
        for token in tokens:
            if token in self.entity_dict:
                self.line_record[self.entity_dict[token]].add(int(line_idx) - 1)
