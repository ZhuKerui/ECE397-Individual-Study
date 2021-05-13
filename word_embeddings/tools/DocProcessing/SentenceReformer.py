import json
from tools.TextProcessing import sent_lemmatize

class SentenceReformer:
    
    def __init__(self, json_file:str):
        self.MyTree = json.load(open(json_file, 'r'))
        self.line_record = []

    def line_operation(self, line:str):
        if not line:
            return
        word_tokens = sent_lemmatize(line)
        reformed_sent = []
        i = 0
        while i < len(word_tokens):
            if word_tokens[i] not in self.MyTree.keys():
                # If the word is not part of a key word directly add the word to the reformed_sent list
                reformed_sent.append(word_tokens[i])
                i += 1

            else:
                # If the word is part of a key word
                phrase_buf = []
                phrase_wait_buf = []
                tail_buf = []
                it = self.MyTree
                while i < len(word_tokens) and word_tokens[i] in it.keys():
                    # Add the word to the wait list
                    phrase_wait_buf.append(word_tokens[i])
                    tail_buf.append(word_tokens[i])
                    if "" in it[word_tokens[i]].keys():
                        # If the word could be the last word of a keyword, update the phrase buffer to be the same with wait buffer
                        phrase_buf = phrase_wait_buf.copy()
                        tail_buf = []
                    # Go down the tree to the next child
                    it = it[word_tokens[i]]
                    i += 1
                # Change the keyword into one uniformed word and add it to the reformed_sent
                if phrase_buf:
                    reformed_sent.append('_'.join(phrase_buf))
                reformed_sent += tail_buf
        self.line_record.append(' '.join(reformed_sent))
