# python occurance.py [wordtree_file] [keyword_file] [sentence_file] [occurance_file]

import sys
import json
from nltk import WordNetLemmatizer, pos_tag, word_tokenize

wordtree = json.load(open(sys.argv[1], 'r'))
keyword_list = open(sys.argv[2], 'r').read().strip().split('\n')
keywords_dict = {word : i for i, word in enumerate(keyword_list)}
line_record = [set() for i in range(len(keyword_list))]

with open(sys.argv[3], 'r') as f_in:
    wnl = WordNetLemmatizer()
    for line_idx, line in enumerate(f_in):
        reformed_sent = [wnl.lemmatize(word, pos='n') if tag.startswith('NN') else word for word, tag in pos_tag(word_tokenize(line))]
        i = 0
        while i < len(reformed_sent):
            if reformed_sent[i] in wordtree: # If the word is the start word of a keyword
                phrase_buf = []
                it = wordtree
                j = i
                while j < len(reformed_sent) and reformed_sent[j] in it:
                    # Add the word to the wait list
                    phrase_buf.append(reformed_sent[j])
                    if "" in it[reformed_sent[j]]: # If the word could be the last word of a keyword, update the list
                        line_record[keywords_dict[' '.join(phrase_buf).replace(' - ', '-')]].add(line_idx)
                    # Go down the tree to the next child
                    it = it[reformed_sent[j]]
                    j += 1
            i += 1
            
with open(sys.argv[4], 'w') as f_out:
    content = ['%s:%s' % (keyword_list[i], ' '.join(map(str, line_record[i]))) for i in range(len(keyword_list))]
    f_out.write('\n'.join(content))