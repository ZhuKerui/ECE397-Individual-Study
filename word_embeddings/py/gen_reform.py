# python gen_reform.py wordtree_file sentence_file reformed_file
import sys

sys.path.append('..')
from tools.BasicUtils import MultiProcessing, my_read, my_write
from tools.TextProcessing import Entity_Reformer
p = MultiProcessing()
reformed_sents = p.run(lambda: Entity_Reformer(sys.argv[1]), my_read(sys.argv[2]), thread_num=20)
my_write(sys.argv[3], reformed_sents)