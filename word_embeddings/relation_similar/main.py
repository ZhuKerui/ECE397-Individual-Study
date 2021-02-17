import sys
sys.path.append('..')

from relation_similar.pair_generator import Pair_Generator
from my_keywords import Keyword_Vocab, Vocab_Base
import numpy as np
kv = Keyword_Vocab()
cv = Vocab_Base()
key_vocab_file = './data/corpus/big_key.vocab'
ctx_vocab_file = './data/corpus/big_ctx.vocab'
kv.load_vocab(key_vocab_file)
cv.load_vocab(ctx_vocab_file)
pg = Pair_Generator(kv, cv)
pg.load_inference_model('./data/outputs/pair2vec/best_big_model.pt', './pair2vec/pair2vec_train.json')
ctx = np.load('./data/outputs/pair2vec/ctx_filtered.npy')
ctx_uni, ctx_cnt = ctx[:, :-1], ctx[:, -1]
b = pg.get_relation_vectors(ctx_uni)
np.save('./data/outputs/pair2vec/ctx_filtered_emb.npy', b)