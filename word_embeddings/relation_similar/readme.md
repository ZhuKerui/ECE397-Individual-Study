from dep_generator import *
d = Dep_Based_Embed_Generator()
d.build_word_tree('../../../dataset/corpus/keyword_f.txt', '../../../dataset/corpus/wordtree.json') / d.load_word_tree('../../../dataset/corpus/wordtree.json')
extract_sent_from_small(80, '../../../dataset/raw_data/small_arxiv.json', '../../../dataset/corpus/small_sent.txt', 10)
extract_sent_from_big(80, '../../../dataset/raw_data/big_arxiv.json', '../../../dataset/corpus/big_sent.txt', 10)
d.extract_context(80, '../../../dataset/corpus/small_sent.txt', '../../../dataset/outputs/small_ctx.txt', 10)