import io
import os
import sys
import json
from my_multithread import multithread_wrapper
from my_keywords import Keyword_Base, Vocab_Generator

def build_test_corpus(original_file:str, output_file:str):
    with io.open(output_file, 'w', encoding='utf-8') as test_f:
        with io.open(original_file, 'r', encoding='utf-8') as load_f:
            for i, line in enumerate(load_f):
                if i >= 10000:
                    break
                test_f.write(line)

def count_line(file_name:str):
    with io.open(file_name, 'r', encoding='utf-8') as load_file:
        cnt = -1
        for cnt, line in enumerate(load_file):
            pass
        print(cnt+1)

def filter_keyword(original_file:str, output_file:str):
    filtered_words = set(['can', 'it', 'work', 'in', 'parts', 'a','b','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    with io.open(original_file, 'r', encoding='utf-8') as load_file:
        with io.open(output_file, 'w', encoding='utf-8') as dump_file:
            print('filtered words:')
            for line in load_file:
                if line.strip() in filtered_words:
                    print(line)
                else:
                    dump_file.write(line)

def extract_sent_from_big(freq:int, input_file:str, output_file:str, thread_num:int=1):
    def extract_sent(line:str):
        if not line:
            return None
        jsonObj = json.loads(line)
        para = jsonObj['abstract'].strip().replace('\n', ' ').replace('$', '').replace('--', ', ').replace('-', ' - ')
        return para + '\n'
    multithread_wrapper(extract_sent, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num)

def extract_sent_from_small(freq:int, input_file:str, output_file:str, thread_num:int=1):
    def extract_sent(line:str):
        if line.find('"abstract": "') == 0:
            para = line.split(':', 1)[1].strip(' ",')
            para = para.replace('\\n', ' ').replace('$', '').replace('--', ', ').replace('-', ' - ')
            return para + '\n'
    multithread_wrapper(extract_sent, freq=freq, input_file=input_file, output_file=output_file, thread_num=thread_num)

def generate_word_tree(keyword_file:str, wordtree_file:str):
    kb = Keyword_Base()
    kb.build_word_tree(keyword_file, wordtree_file)

def reform_sent(wordtree_file:str, original_file:str, reformed_file:str):
    kb = Keyword_Base()
    kb.load_word_tree(wordtree_file)
    kb.process_sent(100, original_file, reformed_file, thread_num=25)

def generate_vocab(wordtree_file, reform_file, vocab_file):
    vg = Vocab_Generator()
    vg.load_word_tree(wordtree_file)
    key_vocab, ctx_vocab = vg.build_vocab(corpus_file=reform_file, vocab_file=vocab_file, special_key=['<unk>'], special_ctx=['<unk>', '<pad>', '<X>', '<Y>'], thr=30)

if __name__ == '__main__':
    if sys.argv[1] == 'build_test_corpus':
        build_test_corpus(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'count_line':
        count_line(sys.argv[2])
    elif sys.argv[1] == 'filter_keyword':
        filter_keyword(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'extract_sent_from_big':
        extract_sent_from_big(freq=80, input_file=sys.argv[2], output_file=sys.argv[3], thread_num=20)
    elif sys.argv[1] == 'extract_sent_from_small':
        extract_sent_from_small(freq=80, input_file=sys.argv[2], output_file=sys.argv[3], thread_num=20)
    elif sys.argv[1] == 'generate_word_tree':
        generate_word_tree(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'reform_sent':
        reform_sent(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'generate_vocab':
        generate_vocab(sys.argv[2], sys.argv[3], sys.argv[4])