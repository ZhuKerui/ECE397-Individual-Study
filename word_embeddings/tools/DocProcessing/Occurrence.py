import json
from typing import Dict
from tools.TextProcessing import sent_lemmatize

class Occurrence:
    def __init__(self, wordtree_file:str, keyword_file:str):
        self.wordtree = json.load(open(wordtree_file, 'r'))
        self.keyword_list = open(keyword_file, 'r').read().strip().split('\n')
        self.keywords_dict = {word : i for i, word in enumerate(self.keyword_list)}
        self.line_record = [set() for i in range(len(self.keyword_list))]

    def line_operation(self, line:str):
        line_idx, sent = line.split(':', 1)
        reformed_sent = sent_lemmatize(sent)
        i = 0
        while i < len(reformed_sent):
            if reformed_sent[i] in self.wordtree: # If the word is the start word of a keyword
                phrase_buf = []
                it = self.wordtree
                j = i
                while j < len(reformed_sent) and reformed_sent[j] in it:
                    # Add the word to the wait list
                    phrase_buf.append(reformed_sent[j])
                    if "" in it[reformed_sent[j]]: # If the word could be the last word of a keyword, update the list
                        self.line_record[self.keywords_dict[' '.join(phrase_buf).replace(' - ', '-')]].add(int(line_idx) - 1)
                    # Go down the tree to the next child
                    it = it[reformed_sent[j]]
                    j += 1
            i += 1

def occurrence_post_operation(result:list):
    if not result:
        return {}
    line_records = [obj.line_record for obj in result]
    keyword_list = result[0].keyword_list
    keyword_num = len(keyword_list)
    occur_list = [set() for i in range(keyword_num)]
    for line_record in line_records:
        occur_list = [occur_list[i] | line_record[i] for i in range(keyword_num)]
    occur_dict = {keyword_list[i] : occur_list[i] for i in range(keyword_num)}
    return occur_dict

def occurrence_dump(output_file:str, occur_dict:Dict[str, set]):
    with open(output_file, 'w') as f_out:
        json.dump({key:list(val) for key, val in occur_dict.items()}, f_out)

def occurrence_load(occur_file:str):
    with open(occur_file) as f_in:
        temp_dict = json.load(f_in)
        return {str(key):set(val) for key, val in temp_dict.items()}
