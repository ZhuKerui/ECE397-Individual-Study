from co_occur_generator import *

c = Co_Occur_Generator()
c.load_word_tree('../../../dataset/relation_similar/wordtree.json')
c.load_word_vector('vecs')
c.load_pairs('../../../dataset/relation_similar/co_occur.csv')
c.filter(min_count=3, min_npmi=0.1)
# print(c.dbscan_cluster('data_structure', 0.50, min_samples=3))