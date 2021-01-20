# reformed_file = '../../../dataset/corpus/big_reformed'
# vocab_file = '../../../dataset/test_corpus/big'
# key_vocab_file = vocab_file + '_key'
# dep_train_file = vocab_file + '_dep_train'
# dep_pair_train_file = vocab_file + '_dep_pair_train'
# pair_vocab_file = vocab_file + '_pair_key'

# ----------------------------------- Load Pair2vec Model -----------------------------------
# from pair_generator import Pair_Generator
# from my_keywords import Keyword_Vocab, Vocab_Base
# kv = Keyword_Vocab()
# cv = Vocab_Base()
# key_vocab_file = '../../../dataset/corpus/big_key'
# ctx_vocab_file = '../../../dataset/corpus/big_ctx'
# kv.load_vocab(key_vocab_file)
# cv.load_vocab(ctx_vocab_file)
# pg = Pair_Generator(kv, cv)
# pg.load_inference_model('../../../dataset/outputs/pair2vec/triplets_1_big_model.pt')
# pg.extract_context(80, '../../../dataset/corpus/big_reformed', '../../../dataset/outputs/pair2vec/triplets_1/big_triplet', 30)

# import numpy as np
# triplet_dir = '../../../dataset/outputs/pair2vec/triplets_1/big_triplet'
# triplets = np.load('%s%d.npy' % (triplet_dir, 0))
# for i in range(1, 65):
#     triplets = np.append(triplets, np.load('%s%d.npy' % (triplet_dir, i)), axis=0)
# np.save('%s_all.npy' % (triplet_dir), triplets)



from pair_generator import Pair_Generator
from my_keywords import Keyword_Vocab, Vocab_Base
import numpy as np
kv = Keyword_Vocab()
cv = Vocab_Base()
key_vocab_file = '../../../dataset/corpus/big_key'
ctx_vocab_file = '../../../dataset/corpus/big_ctx'
kv.load_vocab(key_vocab_file)
cv.load_vocab(ctx_vocab_file)
pg = Pair_Generator(kv, cv)
pg.load_inference_model('../../../dataset/outputs/pair2vec/triplets_1_big_model.pt')
ctx_unique = np.load('../../../dataset/outputs/pair2vec/triplets_1/ctx_unique.npy')
ctx, ctx_cnt = ctx_unique[:, :-1], ctx_unique[:, -1]
sub_ctx = ctx[:1000]
# sub_ctx = pg.rep.