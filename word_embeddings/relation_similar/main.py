reformed_file = '../../../dataset/corpus/big_reformed'
vocab_file = '../../../dataset/test_corpus/big'
key_vocab_file = vocab_file + '_key'
dep_train_file = vocab_file + '_dep_train'
dep_pair_train_file = vocab_file + '_dep_pair_train'
pair_vocab_file = vocab_file + '_pair_key'

# ----------------------------------- Generate Keyword Vocab and Context Vocab -----------------------------------
# from my_keywords import Vocab_Generator
# vg = Vocab_Generator()
# vg.load_word_tree('../../../dataset/corpus/wordtree.json')
# key_vocab, ctx_vocab = vg.build_vocab(corpus_file=reformed_file, vocab_file=vocab_file, special_key=['<unk>'], special_ctx=['<unk>', '<pad>', '<X>', '<Y>'], thr=30)

# ----------------------------------- Generate Dependency Based embedding training data -----------------------------------
# from dep_generator import Dep_Based_Embed_Generator, Keyword_Vocab
# kv = Keyword_Vocab()
# kv.load_vocab(key_vocab_file)
# dg = Dep_Based_Embed_Generator(kv)
# dg.extract_context(80, reformed_file, dep_train_file, thread_num=30)

# ----------------------------------- Generate Co-occur pairs -----------------------------------
# from co_occur_generator import Co_Occur_Generator
# from my_keywords import Keyword_Vocab
# kv = Keyword_Vocab()
# kv.load_vocab(key_vocab_file)
# cog = Co_Occur_Generator(keyword_vocab=kv)
# cog.extract_co_occur(80, reformed_file, dep_pair_train_file, thread_num=30)

# ----------------------------------- Generate Pair embedding training data -----------------------------------
# from pair_embed import Pair_Embed
# from my_keywords import Keyword_Vocab
# kv = Keyword_Vocab()
# kv.load_vocab(key_vocab_file)
# pe = Pair_Embed(kv)
# pe.generate_pair_vocab(dep_pair_train_file+'.csv', min_count=5, min_npmi=0.15)
# pe.save_vocab(pair_vocab_file)
# pe.extract_context(80, reformed_file, dep_pair_train_file, thread_num=30)

# ----------------------------------- Load Pair2vec Model -----------------------------------
from pair_generator import Pair_Generator
from my_keywords import Keyword_Vocab, Vocab_Base
kv = Keyword_Vocab()
cv = Vocab_Base()
key_vocab_file = '../../../dataset/corpus/big_key'
ctx_vocab_file = '../../../dataset/corpus/big_ctx'
kv.load_vocab(key_vocab_file)
cv.load_vocab(ctx_vocab_file)
pg = Pair_Generator(kv, cv)
pg.load_inference_model('../../../dataset/outputs/pair2vec/test_model.pt')