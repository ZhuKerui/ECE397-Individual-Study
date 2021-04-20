#! /bin/bash

# Assume keywords.txt, big_arxiv.json and small_arxiv.json files 
# are already in the data/raw_data folder

# Filter the keyword
python3 helper.py filter_keyword ./data/raw_data/keywords.txt ./data/corpus/keyword_f.txt
python3 helper.py generate_word_tree ./data/corpus/keyword_f.txt ./data/corpus/wordtree.json

# Extract the sentences from the arxiv.json files
python3 helper.py extract_sent_from_small ./data/raw_data/small_arxiv.json ./data/corpus/small_sent.txt
python3 helper.py extract_sent_from_big ./data/raw_data/big_arxiv.json ./data/corpus/big_sent.txt
# python3 helper.py reform_sent ./data/corpus/wordtree.json ./data/corpus/small_sent.txt ./data/corpus/small_reform.txt
# python3 helper.py reform_sent ./data/corpus/wordtree.json ./data/corpus/big_sent.txt ./data/corpus/big_reform.txt
python3 helper.py build_test_corpus ./data/corpus/small_reform.txt ./data/test_corpus/test_reform.txt
python3 helper.py generate_vocab ./data/corpus/wordtree.json ./data/corpus/big_reform.txt ./data/corpus/big
python3 helper.py filter_ctx ./data/corpus/big_ctx.vocab ./data/corpus/big_ctx.vocab
# python3 helper.py generate_vocab ./data/corpus/wordtree.json ./data/corpus/small_reform.txt ./data/corpus/small
# python3 helper.py generate_vocab ./data/corpus/wordtree.json ./data/test_corpus/test_reform.txt ./data/test_corpus/test