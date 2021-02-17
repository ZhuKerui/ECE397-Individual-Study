import io
import sys
import json
from my_multithread import multithread_wrapper
from my_keywords import Keyword_Base, Vocab_Generator, Vocab_Base, Keyword_Vocab
import numpy as np
from relation_similar.pair_generator import Pair_Generator

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
    filtered_words = set(['can', 'it', 'work', 'in', 'parts', 'its', 'a','b','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    with io.open(original_file, 'r', encoding='utf-8') as load_file:
        with io.open(output_file, 'w', encoding='utf-8') as dump_file:
            print('filtered words:')
            for line in load_file:
                if line.strip() in filtered_words:
                    print(line)
                else:
                    dump_file.write(line)

def filter_ctx(original_file:str, output_file:str):
    filtered_chars = ['{', '}', '(', ')', '\\','[',']','~', '=', '^', '/', '|']
    filtered_words = set(['-', '`'])
    good_words = []
    with io.open(original_file, 'r', encoding='utf-8') as load_file:
        print('filtered words:')
        for line in load_file:
            if line.strip() in filtered_words:
                print(line)
                continue
            accepted = True
            for char in filtered_chars:
                if char in line:
                    print(line)
                    accepted = False
                    break
            if accepted:
                good_words.append(line)
    with io.open(output_file, 'w', encoding='utf-8') as dump_file:
        dump_file.write(''.join(good_words))

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

def manage_triplets():
    triplet_dir = './data/outputs/pair2vec/big_triplet'
    triplets = np.load('%s%d.npy' % (triplet_dir, 0))
    for i in range(1, 68):
        triplets = np.append(triplets, np.load('%s%d.npy' % (triplet_dir, i)), axis=0)
    np.save('%s_all.npy' % (triplet_dir), triplets)
    key, ctx = triplets[:, :2], triplets[:, 2:]
    ctx_uni, ctx_cnt = np.unique(ctx, return_counts=True, axis=0)
    ctx_cnt = np.hstack((ctx_uni, ctx_cnt.reshape(-1,1)))
    ctx_cnt = np.array(sorted(ctx_cnt, key=lambda x:x[-1]))
    np.save(triplet_dir+'_ctx.npy', ctx_cnt)

def extract_relation(relation_file, output_file, min_count):
    ctx = np.load(relation_file)
    ctx_uni, ctx_cnt = ctx[:, :-1], ctx[:, -1]
    filtered_ctx = ctx_uni[ctx_cnt >= min_count]
    np.save(output_file, filtered_ctx)
    print(filtered_ctx.shape)

def extract_relation_by_pattern(relation_file, output_file):
    ctx = np.load(relation_file)
    kv = Keyword_Vocab()
    cv = Vocab_Base()
    key_vocab_file = './data/corpus/big_key.vocab'
    ctx_vocab_file = './data/corpus/big_ctx.vocab'
    kv.load_vocab(key_vocab_file)
    cv.load_vocab(ctx_vocab_file)
    pg = Pair_Generator(kv, cv)
    pattern1 = np.array([pg.ctx_vocab.stoi['<X>'], pg.ctx_vocab.stoi['is'], pg.ctx_vocab.stoi['a']])
    pattern2 = np.array([pg.ctx_vocab.stoi['<X>'], pg.ctx_vocab.stoi['is'], pg.ctx_vocab.stoi['an']])
    start_word = ctx[:, :3]
    idx1 = (start_word == pattern1).all(axis=1)
    idx2 = (start_word == pattern2).all(axis=1)
    new_ctx = np.vstack([ctx[idx1], ctx[idx2]])
    np.save(output_file+'.npy', new_ctx)
    with io.open(output_file+'.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write(pg.translate_contexts(new_ctx))

def extract_relation_by_pattern_2(relation_file, output_file):
    ctx = np.load(relation_file)
    kv = Keyword_Vocab()
    cv = Vocab_Base()
    key_vocab_file = './data/corpus/big_key.vocab'
    ctx_vocab_file = './data/corpus/big_ctx.vocab'
    kv.load_vocab(key_vocab_file)
    cv.load_vocab(ctx_vocab_file)
    pg = Pair_Generator(kv, cv)
    pattern1 = np.array([pg.ctx_vocab.stoi['<X>'], pg.ctx_vocab.stoi['is']])
    start_word = ctx[:, :2]
    idx1 = (start_word == pattern1).all(axis=1)
    new_ctx = ctx[idx1]
    idx = np.argwhere(new_ctx == pg.ctx_vocab.stoi['<Y>']).T
    of_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['of']
    on_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['on']
    in_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['in']
    at_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['at']
    for_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['for']
    to_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['to']
    with_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['with']
    about_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['about']
    against_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['against']
    by_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['by']
    above_idx = new_ctx[idx[0], idx[1]-1] == pg.ctx_vocab.stoi['above']
    new_ctx = np.vstack((new_ctx[of_idx], new_ctx[on_idx], new_ctx[in_idx], new_ctx[at_idx], new_ctx[for_idx], new_ctx[to_idx], new_ctx[with_idx], new_ctx[about_idx], new_ctx[against_idx], new_ctx[by_idx], new_ctx[above_idx]))
    np.save(output_file+'.npy', new_ctx)
    with io.open(output_file+'.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write(pg.translate_contexts(new_ctx))

if __name__ == '__main__':
    if sys.argv[1] == 'build_test_corpus':
        build_test_corpus(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'count_line':
        count_line(sys.argv[2])
    elif sys.argv[1] == 'filter_keyword':
        filter_keyword(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'filter_ctx':
        filter_ctx(sys.argv[2], sys.argv[3])
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
    elif sys.argv[1] == 'manage_triplets':
        manage_triplets()
    elif sys.argv[1] == 'extract_relation':
        extract_relation(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == 'extract_relation_by_pattern':
        extract_relation_by_pattern(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'extract_relation_by_pattern_2':
        extract_relation_by_pattern_2(sys.argv[2], sys.argv[3])