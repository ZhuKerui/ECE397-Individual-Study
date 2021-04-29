import json
from typing import Dict, List
from nltk import WordNetLemmatizer, pos_tag, word_tokenize

wnl = WordNetLemmatizer()

def build_word_tree(input_txt:str, dump_file:str):
    MyTree = {}
    keywords = set()
    cnt = 0
    with open(input_txt, 'r', encoding='utf-8') as load_file:
        for word in load_file:
            cnt += 1

            word = word.strip()
            # Directly add the '_' connected keyword
            normalized_word = word.replace('-', ' - ').replace(' ', '_')
            keywords.add(normalized_word)

            # Insert the keyword to the tree structure

            phrase = normalized_word.split('_')
            if len(phrase) == 1:
                # If the word is an atomic word instead of a phrase
                if word not in MyTree.keys():
                    # If this is the first time that this word is inserted to the tree
                    MyTree[word] = {"":""}
                elif "" not in MyTree[word].keys():
                    # If the word has been inserted but is viewed as an atomic word the first time
                    MyTree[word][""] = ""
                # If the word has already been inserted as an atomic word, then we do nothing
            else:
                # If the word is an phrase
                length = len(phrase)
                fw = phrase[0]
                if fw not in MyTree.keys():
                    MyTree[fw] = {}
                temp_dict = MyTree.copy()
                parent_node = fw
                for i in range(1, length):
                    if phrase[i]:
                        sw = phrase[i]
                        if sw not in temp_dict[parent_node].keys():
                            # The second word is inserted to as the child of parent node the first time
                            temp_dict[parent_node][sw] = {}
                        if i == length - 1:
                            # If the second word is the last word in the phrase
                            if "" not in temp_dict[parent_node][sw].keys():
                                temp_dict[parent_node][sw][""] = ""
                        else:
                            # If the second word is not the last word in the phrase
                            temp_dict = temp_dict[parent_node].copy()
                            parent_node = sw
            # if cnt % 1000 == 0:
            #     print(cnt)
        print('Building word tree is accomplished with %d words added' % (cnt))
    with open(dump_file, 'w', encoding='utf-8') as output_file:
        json.dump(MyTree, output_file)
        
def sent_lemmatize(sentence:str):
    return [str(wnl.lemmatize(word, pos='n') if tag.startswith('NN') else word) for word, tag in pos_tag(word_tokenize(sentence))]

class Occurance:
    def __init__(self, wordtree_file:str, keyword_file:str):
        self.test = False
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
                        self.test = True
                    # Go down the tree to the next child
                    it = it[reformed_sent[j]]
                    j += 1
            i += 1

def occurance_post_operation(result:List[Occurance]):
    if not result:
        return {}
    line_records = [obj.line_record for obj in result if obj.test]
    keyword_list = result[0].keyword_list
    keyword_num = len(keyword_list)
    occur_list = [set() for i in range(keyword_num)]
    for line_record in line_records:
        occur_list = [occur_list[i] | line_record[i] for i in range(keyword_num)]
    occur_dict = {keyword_list[i] : occur_list[i] for i in range(keyword_num)}
    return occur_dict


def occurance_dump(output_file:str, occur_dict:Dict[str, set]):
    with open(output_file, 'w') as f_out:
        json.dump({key:list(val) for key, val in occur_dict.items()}, f_out)

def occurance_load(occur_file:str):
    with open(occur_file) as f_in:
        temp_dict = json.load(f_in)
        return {str(key):set(val) for key, val in temp_dict.items()}
