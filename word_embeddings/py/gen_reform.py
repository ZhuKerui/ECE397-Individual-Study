# python gen_reform.py wordtree_file sentence_file reformed_file
import sys

sys.path.append('..')
from tools.BasicUtils import MultiProcessing, my_read, my_write
from tools.DocProcessing.SentenceReformer import SentenceReformer
p = MultiProcessing()
reformed_sents = p.run(lambda: SentenceReformer(sys.argv[1]), my_read(sys.argv[2]), thread_num=20)
my_write(sys.argv[3], reformed_sents)