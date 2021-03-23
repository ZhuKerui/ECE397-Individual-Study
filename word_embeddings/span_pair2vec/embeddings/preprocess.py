import io
import sys
import numpy as np
from collections import Counter
from vocab import Vocab

with io.open(sys.argv[1], 'r', encoding='utf-8') as f_in:
    content = []
    corpus = Counter()
    for line in f_in:
        element = line.lower().strip().split(';')
        if len(element) != 3:
            continue
        arg1, rel, arg2 = (span.strip().split() for span in element)
        if len(arg1) > 6 or len(arg2) > 6 or len(rel) > 6:
            continue
        corpus.update(arg1 + arg2 + rel)
        content.append((arg1, arg2, rel))

    data_text = []
    data_idx = []
    min_count = int(sys.argv[3])
    filtered_words = [item[0] for item in corpus.items() if item[1] >= min_count]
    vocab = Vocab(filtered_words, specials=['<unk>', '<pad>'])
    for spans in content:
        temp = []
        for span in spans:
            temp_span = [token for token in span if vocab.stoi[token] != 0]
            temp += temp_span
            temp += ['<pad>'] * (6 - len(temp_span))
        data_text.append(' '.join(temp))
        data_idx.append([vocab.stoi[token] for token in temp])
                
    with io.open(sys.argv[2], 'w', encoding='utf-8') as vocab_file:
        vocab_file.write('\n'.join(filtered_words))
    with io.open(sys.argv[4], 'w', encoding='utf-8') as data_file:
        data_file.write('\n'.join(data_text))
    file_num = int((len(data_idx) + 499999) / 500000)
    for i in range(file_num):
        np.save('%s%d' % (sys.argv[5], i), data_idx[i*500000:(i+1)*500000])