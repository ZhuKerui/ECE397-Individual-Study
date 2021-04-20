# python gen_occur.py wordtree_file keyword_file sentence_file occur_file
import sys

sys.path.append('..')
from my_util import my_read, my_write, MultiProcessing
from my_occurance import Occurance, occurance_dump, occurance_load, occurance_post_operation

# Generate occurance file
p = MultiProcessing()
occur_dict = p.run(lambda: Occurance(sys.argv[1], sys.argv[2]), open(sys.argv[3]).readlines(), 8, occurance_post_operation)
occurance_dump(sys.argv[4], occur_dict)